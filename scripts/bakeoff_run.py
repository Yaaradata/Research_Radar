#!/usr/bin/env python3
"""Run classification bake-off candidates on the sampled papers.

By default prints cost estimate only — does NOT call paid APIs.
Pass --allow-paid to execute scoring (pass 1, self-consistency pass 2, batch-B).
Pass --probe-gates to test structured-output eligibility per candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    BAKEOFF_PROMPT_VERSION,
    estimate_full_bakeoff_cost,
    load_bakeoff_config,
    load_eligible_papers,
    new_run_id,
    persist_sample,
    score_papers_batch,
    select_stratified_sample,
    shuffle_batch_arrangement,
    test_structured_output_gate,
    vocabulary_from_conn,
)
from research_radar.llm_batch import random_batches
from research_radar.pipeline import connect
from research_radar.semantic_scoring import create_llm_client, require_api_key, require_scoring_enabled


def _load_sample(conn, run_id: str | None, seed: int | None) -> tuple[list[dict], str]:
    if run_id:
        row = conn.execute(
            "SELECT sample_seed FROM research_radar.bakeoff_runs WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if not row:
            raise SystemExit(f"run_id {run_id} not found in bakeoff_runs")
        seed = int(row["sample_seed"])
    papers = load_eligible_papers(conn)
    sample, _ = select_stratified_sample(papers, seed=seed)
    if not run_id:
        run_id = str(new_run_id())
        persist_sample(conn, UUID(run_id), seed, sample, BAKEOFF_PROMPT_VERSION)
    return sample, run_id


def _store_results(conn, run_id, candidate, pass_index, batch_arrangement, call_results):
    for cr in call_results:
        parsed = cr.parsed or {}
        conn.execute(
            """
            INSERT INTO research_radar.bakeoff_results (
                run_id, candidate_id, model_name, content_id, pass_index, batch_arrangement,
                domain, subdomains, application_domains, primary_audience, ai_relevance,
                json_valid_first_try, retries, tokens_in, tokens_out, cost_usd, latency_ms, raw_response
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s::jsonb, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (run_id, candidate_id, content_id, pass_index, batch_arrangement)
            DO UPDATE SET
                domain = EXCLUDED.domain,
                subdomains = EXCLUDED.subdomains,
                application_domains = EXCLUDED.application_domains,
                primary_audience = EXCLUDED.primary_audience,
                ai_relevance = EXCLUDED.ai_relevance,
                json_valid_first_try = EXCLUDED.json_valid_first_try,
                retries = EXCLUDED.retries,
                tokens_in = EXCLUDED.tokens_in,
                tokens_out = EXCLUDED.tokens_out,
                cost_usd = EXCLUDED.cost_usd,
                latency_ms = EXCLUDED.latency_ms,
                raw_response = EXCLUDED.raw_response
            """,
            (
                run_id,
                candidate.id,
                candidate.model,
                cr.content_id,
                pass_index,
                batch_arrangement,
                parsed.get("domain"),
                parsed.get("subdomains"),
                json.dumps(parsed.get("application_domains") or []),
                parsed.get("primary_audience"),
                parsed.get("ai_relevance"),
                cr.json_valid_first_try,
                cr.retries,
                cr.tokens_in,
                cr.tokens_out,
                cr.cost_usd,
                cr.latency_ms,
                cr.raw_response,
            ),
        )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Classification bake-off runner")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-paid", action="store_true", help="Execute paid API calls")
    parser.add_argument("--probe-gates", action="store_true", help="Test structured-output gate per candidate")
    parser.add_argument("--estimate-only", action="store_true", help="Print cost estimate and exit")
    args = parser.parse_args()

    config = load_bakeoff_config()
    with connect() as conn:
        vocab = vocabulary_from_conn(conn)

    estimate = estimate_full_bakeoff_cost(config, vocab)
    print("ESTIMATED BAKE-OFF COST")
    print(json.dumps(estimate, indent=2))
    print(
        f"\nTotal: ${estimate['total_estimated_cost_usd']:.2f} for "
        f"{estimate['sample_size']} papers × {estimate['candidates']} candidates × "
        f"{estimate['passes_per_candidate']} passes"
    )

    if args.probe_gates:
        require_scoring_enabled()
        require_api_key()
        client = create_llm_client()
        print("\nSTRUCTURED-OUTPUT GATE PROBE")
        for cand in config.candidates:
            result = test_structured_output_gate(cand, vocab, client=client)
            status = "PASS" if result["passed"] else "DISQUALIFIED"
            print(f"  {cand.id} ({cand.model}): {status}")
            if result["error"]:
                print(f"    error: {result['error'][:200]}")

    if args.estimate_only or not args.allow_paid:
        if not args.allow_paid:
            print("\nNo API calls made (pass --allow-paid to run scoring).")
        return 0

    require_scoring_enabled()
    require_api_key()
    client = create_llm_client()

    with connect() as conn:
        seed = args.seed if args.seed is not None else config.sample_seed
        sample, run_id = _load_sample(conn, args.run_id, seed)
        print(f"\nRunning bake-off run_id={run_id} sample_size={len(sample)}")

        disqualified = set()
        for cand in config.candidates:
            gate = test_structured_output_gate(cand, vocab, client=client)
            if not gate["passed"]:
                print(f"DISQUALIFIED {cand.id}: structured-output gate failed")
                disqualified.add(cand.id)

        for cand in config.candidates:
            if cand.id in disqualified:
                continue
            print(f"Candidate {cand.id} ({cand.model})...")
            batches_a = random_batches(sample, config.batch_size)
            for batch in batches_a:
                results = score_papers_batch(batch, cand, vocab, client=client, batch_arrangement="A")
                _store_results(conn, run_id, cand, 1, "A", results)
            # Self-consistency pass 2
            batches_p2 = random_batches(sample, config.batch_size)
            for batch in batches_p2:
                results = score_papers_batch(batch, cand, vocab, client=client, batch_arrangement="A")
                _store_results(conn, run_id, cand, 2, "A", results)
            # Batch arrangement B
            shuffled = shuffle_batch_arrangement(sample, config.sample_seed + hash(cand.id) % 10000)
            batches_b = random_batches(shuffled, config.batch_size)
            for batch in batches_b:
                results = score_papers_batch(batch, cand, vocab, client=client, batch_arrangement="B")
                _store_results(conn, run_id, cand, 1, "B", results)
            time.sleep(0.5)

    print(f"Done. run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
