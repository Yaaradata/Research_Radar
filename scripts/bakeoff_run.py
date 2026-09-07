#!/usr/bin/env python3
"""Run non-baseline bake-off candidates on the sampled papers (2 passes each).

Default: print cost estimate and OpenRouter model resolution — no API calls.
Pass --allow-paid to execute scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    analyze_raw_classify_response,
    classify_batch_with_retries,
    estimate_bakeoff_run_cost,
    insert_bakeoff_result,
    load_bakeoff_config,
    load_sample_papers,
    verify_openrouter_models,
)
from research_radar.llm_batch import random_batches
from research_radar.pipeline import connect
from research_radar.semantic_scoring import create_llm_client, require_api_key, require_scoring_enabled


def _verify_models(config) -> dict[str, bool]:
    models = [c.model for c in config.candidates if c.id != config.baseline_candidate_id]
    resolved = verify_openrouter_models(models)
    print("OpenRouter model resolution:")
    for model, ok in sorted(resolved.items()):
        print(f"  {'OK' if ok else 'MISSING':7} {model}")
    missing = [m for m, ok in resolved.items() if not ok]
    if missing:
        print(f"WARNING: {len(missing)} model(s) not found on OpenRouter", file=sys.stderr)
    return resolved


def _run_pass(conn, run_id, papers, candidate, pass_index: int, batch_seed: int, batch_size: int, client) -> None:
    import random

    rng = random.Random(batch_seed)
    shuffled = list(papers)
    rng.shuffle(shuffled)
    batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]

    for batch in batches:
        validations, retries, _, tin, tout, latency = classify_batch_with_retries(
            batch, candidate=candidate, client=client
        )
        per_paper_tin = tin // max(1, len(batch))
        per_paper_tout = tout // max(1, len(batch))
        per_latency = latency // max(1, len(batch))
        for paper in batch:
            cid = int(paper["content_id"])
            v = validations.get(cid)
            if v is None:
                v = analyze_raw_classify_response("", {cid})[cid]
            insert_bakeoff_result(
                conn,
                run_id=run_id,
                candidate=candidate,
                content_id=cid,
                pass_index=pass_index,
                validation=v,
                retries=retries,
                tokens_in=per_paper_tin,
                tokens_out=per_paper_tout,
                latency_ms=per_latency,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run classification bake-off candidates")
    parser.add_argument("--run-id", required=True, help="bakeoff_runs.run_id from bakeoff_sample.py")
    parser.add_argument("--allow-paid", action="store_true", help="Authorise paid OpenRouter calls")
    parser.add_argument("--passes", type=int, default=2, help="Passes per candidate (default 2)")
    parser.add_argument("--estimate-only", action="store_true", help="Print cost estimate and exit")
    args = parser.parse_args()

    config = load_bakeoff_config()
    run_id = UUID(args.run_id)
    resolved = _verify_models(config)

    with connect() as conn:
        papers = load_sample_papers(conn, run_id)
        if not papers:
            print(f"No sample papers found for run_id={run_id}", file=sys.stderr)
            return 1
        estimate = estimate_bakeoff_run_cost(config, len(papers), n_passes=args.passes)
        print("\nCost estimate (non-baseline candidates only):")
        print(json.dumps(estimate, indent=2))

        if args.estimate_only or not args.allow_paid:
            print("\nNo API calls made. Re-run with --allow-paid to execute.")
            return 0

        missing_models = [
            c for c in config.candidates
            if c.id != config.baseline_candidate_id and not resolved.get(c.model, False)
        ]
        if missing_models:
            print("Aborting: unresolved models on OpenRouter:", [c.model for c in missing_models], file=sys.stderr)
            return 1

        require_scoring_enabled()
        require_api_key()
        client = create_llm_client()

        for candidate in config.candidates:
            if candidate.id == config.baseline_candidate_id:
                print(f"Skipping baseline {candidate.id} (already imported)")
                continue
            for pass_index in range(1, args.passes + 1):
                print(f"Running {candidate.id} pass {pass_index} ({len(papers)} papers)...")
                batch_seed = config.sample_seed + pass_index * 1000 + hash(candidate.id) % 10000
                _run_pass(conn, run_id, papers, candidate, pass_index, batch_seed, config.batch_size, client)
            conn.commit()
            print(f"Completed {candidate.id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
