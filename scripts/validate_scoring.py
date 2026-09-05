#!/usr/bin/env python3
"""Scoring v2 validation harness (scoring-v2 brief §5; tiering brief §4).

Five subcommands, each writing a markdown report to reports/:

    --test compression        v1 vs v2 spread/tie-blocks. Diagnostic only.
    --test batch-stability    single-call vs two random batch arrangements.
                               PAID (~100-300 quality-scorer calls).
    --test affiliation-leak   blind vs affiliation-injected quality prompt.
                               PAID (~200 quality-scorer calls).
    --test human-agreement    v1 vs v2 against research_radar.editorial_reviews.
                               Free — reads only stored data. Requires the
                               human review to already exist (see
                               scripts/export_review_set.py).
    --test tier-recall         What fraction of pass 2's true top 50 survives
                               a pass-1 (screen) gate, at several gate widths.
                               PAID (~2x sample-size calls). Required before
                               trusting GATE_PERCENTILE for a real run — see
                               run_tier_recall's docstring.

Standard deviation alone does not prove better scoring — a random scorer has
excellent spread and a useless ranking. Only human-agreement can show scoring
actually improved; the other four are diagnostics / architecture checks.

batch-stability, affiliation-leak and tier-recall make real OpenRouter calls
and are therefore gated the same way as paid pipeline stages: pass
--allow-paid to actually spend, or --dry-run to estimate cost with zero calls.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

from research_radar import final_score as fs
from research_radar import semantic_scoring as ss
from research_radar.llm_batch import call_chat_completion, random_batches
from research_radar.pipeline import connect

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
PAID_TESTS = {"batch-stability", "affiliation-leak", "tier-recall"}


def _write_report(name: str, content: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{name}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation with no scipy dependency (none in requirements.txt)."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _percentiles(vals: list[float]) -> dict[str, float]:
    vals = sorted(vals)

    def pct(q):
        return vals[min(len(vals) - 1, int(len(vals) * q))]

    return {"p50": pct(0.50), "p75": pct(0.75), "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99)}


def _tie_block_at_cut(vals: list[float], cut: int) -> int:
    """Size of the tie-block straddling the Top-N cutoff value."""
    vals = sorted(vals, reverse=True)
    if len(vals) < cut:
        return len(vals)
    cutoff_value = vals[cut - 1]
    return sum(1 for v in vals if v == cutoff_value)


# ===========================================================================
# --test compression
# ===========================================================================


def run_compression(conn, *, top: int = 20) -> str:
    v1_rows = fs.load_scored_papers(conn, prompt_version="research-semantic-v1")
    v2_rows = fs.load_scored_papers(conn, prompt_version="research-semantic-v2")
    common = {r["content_id"] for r in v1_rows} & {r["content_id"] for r in v2_rows}

    _, w1 = fs.resolve_profile("radar-v1")
    _, w2 = fs.resolve_profile("radar-v2")

    def composites(rows, weights):
        out = []
        for r in rows:
            if r["content_id"] not in common:
                continue
            try:
                out.append(fs.compute_semantic_core({k: r.get(k) for k in fs.DIMENSIONS}, weights))
            except ValueError:
                continue
        return out

    v1_scores = composites(v1_rows, w1)
    v2_scores = composites(v2_rows, w2)

    lines = [
        "# Scoring v2 validation — compression (v1 vs v2)",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Papers scored under both prompt_versions:** {len(common)}  ",
        "",
        "Diagnostic only — spread alone does not prove better ranking (see human-agreement).",
        "",
        "| version | n | mean | stdev | p50 | p75 | p90 | p95 | p99 | tie-block @ Top-%d |" % top,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, vals in (("v1 (radar-v1)", v1_scores), ("v2 (radar-v2)", v2_scores)):
        if len(vals) < 2:
            lines.append(f"| {label} | {len(vals)} | insufficient data (<2) | | | | | | | |")
            continue
        p = _percentiles(vals)
        tie = _tie_block_at_cut(vals, top)
        lines.append(
            f"| {label} | {len(vals)} | {st.mean(vals):.2f} | {st.stdev(vals):.2f} | "
            f"{p['p50']:.2f} | {p['p75']:.2f} | {p['p90']:.2f} | {p['p95']:.2f} | {p['p99']:.2f} | {tie} |"
        )
    return "\n".join(lines) + "\n"


# ===========================================================================
# --test batch-stability
# ===========================================================================


def _score_batches_v2(batches: list[list[dict]], *, client) -> dict[int, dict]:
    """Real quality-scorer calls. Never persisted — in-memory only."""
    out: dict[int, dict] = {}
    total = len(batches)
    for i, batch in enumerate(batches, 1):
        ids = [p["content_id"] for p in batch]
        print(
            f"[pass2 quality] batch {i}/{total} papers={len(batch)} ids={ids}",
            flush=True,
        )
        result = ss.call_quality_batch_with_retry(batch, client=client)
        out.update(result["results"])
        missing = {p["content_id"] for p in batch} - set(result["results"])
        for pid in missing:
            print(f"[pass2 quality] retry single id={pid}", flush=True)
            paper = next(p for p in batch if p["content_id"] == pid)
            single = ss.call_quality_batch_with_retry([paper], client=client)
            out.update(single["results"])
    return out


def _estimate_batch_stability_cost(sample: list[dict]) -> float:
    n = len(sample)
    single_batches = [[p] for p in sample]
    arr_batches = random_batches(sample, ss.QUALITY_BATCH_SIZE)
    est_in = sum(ss.estimate_quality_prompt_tokens(b) for b in single_batches) + 2 * sum(
        ss.estimate_quality_prompt_tokens(b) for b in arr_batches
    )
    est_out = 220 * n * 3
    return ss.estimate_cost_usd(est_in, est_out)


def run_batch_stability(conn, *, n: int = 100, dry_run: bool = False, client=None) -> str:
    candidates = ss.load_quality_candidates(conn, limit=None)
    if len(candidates) < n:
        raise SystemExit(f"!! batch-stability needs {n} eligible ENTITY_RESOLVED papers, only {len(candidates)} available")
    sample = random.sample(candidates, n)

    if dry_run:
        cost = _estimate_batch_stability_cost(sample)
        return (
            "# Scoring v2 validation — batch-stability DRY RUN\n\n"
            f"n={n}  approx_cost_usd={cost:.4f}  "
            f"(single={n} calls, arrangement A={-(-n // ss.QUALITY_BATCH_SIZE)} calls, "
            f"arrangement B={-(-n // ss.QUALITY_BATCH_SIZE)} calls)\n"
        )

    client = client or ss.create_llm_client()
    single_batches = [[p] for p in sample]
    arrangement_a = random_batches(sample, ss.QUALITY_BATCH_SIZE)
    arrangement_b = random_batches(sample, ss.QUALITY_BATCH_SIZE)

    single = _score_batches_v2(single_batches, client=client)
    arr_a = _score_batches_v2(arrangement_a, client=client)
    arr_b = _score_batches_v2(arrangement_b, client=client)

    weights = fs.WEIGHT_PROFILES["radar-v2"]
    single_c = {cid: fs.compute_semantic_core(d, weights) for cid, d in single.items()}
    a_c = {cid: fs.compute_semantic_core(d, weights) for cid, d in arr_a.items()}
    b_c = {cid: fs.compute_semantic_core(d, weights) for cid, d in arr_b.items()}
    common_ids = sorted(set(single_c) & set(a_c) & set(b_c))

    def compare(label, ca, cb):
        xs = [ca[i] for i in common_ids]
        ys = [cb[i] for i in common_ids]
        rho = _spearman(xs, ys)
        diffs = [a - b for a, b in zip(xs, ys)]
        mean_shift = sum(diffs) / len(diffs)
        mean_abs = sum(abs(d) for d in diffs) / len(diffs)
        passed = rho > 0.90 and abs(mean_shift) < 0.20 and mean_abs < 0.40
        return {
            "label": label, "n": len(common_ids), "spearman": rho,
            "mean_shift": mean_shift, "mean_abs_diff": mean_abs, "pass": passed,
        }

    results = [
        compare("single vs A", single_c, a_c),
        compare("single vs B", single_c, b_c),
        compare("A vs B", a_c, b_c),
    ]

    lines = [
        "# Scoring v2 validation — batch-stability",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**n:** {len(common_ids)} papers scored single / arrangement A / arrangement B  ",
        f"**Pass condition:** Spearman > 0.90, |mean shift| < 0.20, mean abs diff < 0.40, all three comparisons  ",
        "",
        "| comparison | n | spearman | mean shift | mean abs diff | pass |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['spearman']:.3f} | {r['mean_shift']:+.3f} | "
            f"{r['mean_abs_diff']:.3f} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    overall = all(r["pass"] for r in results)
    lines += ["", f"**Overall: {'PASS' if overall else 'FAIL'}**"]
    return "\n".join(lines) + "\n"


# ===========================================================================
# --test affiliation-leak
# ===========================================================================


def _load_affiliation_text_map(conn, content_ids: list[int]) -> dict[int, list]:
    if not content_ids:
        return {}
    rows = conn.execute(
        "SELECT content_id, affiliation_text FROM research_radar.paper_metadata WHERE content_id = ANY(%s)",
        (list(content_ids),),
    ).fetchall()
    return {r["content_id"]: (r["affiliation_text"] or []) for r in rows}


def _load_watchlisted_content_ids(conn, content_ids: list[int]) -> set[int]:
    if not content_ids:
        return set()
    rows = conn.execute(
        """
        SELECT DISTINCT content_id
        FROM research_radar.content_organisations
        WHERE content_id = ANY(%s)
          AND relationship_type = 'paper_author_affiliation'
          AND current_affiliation = FALSE
        """,
        (list(content_ids),),
    ).fetchall()
    return {r["content_id"] for r in rows}


def _build_leaky_quality_prompt(paper: dict) -> str:
    """DELIBERATE guarantee violation for measurement only — never used for real
    scoring. Appends an AFFILIATIONS section to the otherwise-blind paper block."""
    block = ss.build_quality_paper_block(paper)
    aff = paper.get("affiliation_text") or []
    aff_text = "; ".join(str(a) for a in aff if a) or "not available"
    return block + f"AFFILIATIONS: {aff_text}\n"


def _score_single_blind(paper: dict, client) -> dict | None:
    result = ss.call_quality_batch_with_retry([paper], client=client)
    return result["results"].get(paper["content_id"])


def _score_single_leaky(paper: dict, client) -> dict | None:
    prompt = _build_leaky_quality_prompt(paper)
    resp = call_chat_completion(
        client,
        model=ss.resolve_model_name(),
        system_prompt=ss.QUALITY_SYSTEM_PROMPT,
        user_prompt=prompt,
        reasoning_effort=ss.QUALITY_REASONING_EFFORT,
        temperature=0.2,
        max_retries=ss.QUALITY_MAX_RETRIES,
        request_sleep=ss.QUALITY_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "quality_assessment",
                "strict": True,
                "schema": ss.QUALITY_RESPONSE_SCHEMA,
            },
        },
    )
    parsed, _ = ss.parse_quality_batch(resp["text"], {paper["content_id"]})
    return parsed.get(paper["content_id"])


def run_affiliation_leak(conn, *, n: int = 100, dry_run: bool = False, client=None) -> str:
    candidates = ss.load_quality_candidates(conn, limit=None)
    ids = [p["content_id"] for p in candidates]
    aff_map = _load_affiliation_text_map(conn, ids)
    eligible = [p for p in candidates if aff_map.get(p["content_id"])]
    if len(eligible) < n:
        raise SystemExit(
            f"!! affiliation-leak needs {n} papers with non-empty affiliation_text, only {len(eligible)} available"
        )
    sample = random.sample(eligible, n)
    for p in sample:
        p["affiliation_text"] = aff_map[p["content_id"]]

    if dry_run:
        est_in = sum(len(ss.QUALITY_SYSTEM_PROMPT) + len(ss.build_quality_paper_block(p)) for p in sample)
        est_in += sum(len(ss.QUALITY_SYSTEM_PROMPT) + len(_build_leaky_quality_prompt(p)) for p in sample)
        est_in //= 4
        est_out = 220 * n * 2
        cost = ss.estimate_cost_usd(est_in, est_out)
        return f"# Scoring v2 validation — affiliation-leak DRY RUN\n\nn={n}  approx_cost_usd={cost:.4f}  ({2 * n} calls)\n"

    watchlisted = _load_watchlisted_content_ids(conn, [p["content_id"] for p in sample])
    client = client or ss.create_llm_client()
    weights = fs.WEIGHT_PROFILES["radar-v2"]

    rows = []
    for p in sample:
        blind = _score_single_blind(p, client)
        leaky = _score_single_leaky(p, client)
        if blind is None or leaky is None:
            continue
        rows.append(
            {
                "content_id": p["content_id"],
                "watchlisted": p["content_id"] in watchlisted,
                "delta": fs.compute_semantic_core(leaky, weights) - fs.compute_semantic_core(blind, weights),
            }
        )

    watchlist_deltas = [r["delta"] for r in rows if r["watchlisted"]]
    other_deltas = [r["delta"] for r in rows if not r["watchlisted"]]

    lines = [
        "# Scoring v2 validation — affiliation-leak",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**n scored (blind + leaky pair):** {len(rows)}  ",
        "",
        "delta = leaky_quality - blind_quality (positive = affiliation leak inflated the score). "
        "A small gap means the two-call architecture was cheap insurance; a large gap means it was necessary.",
        "",
        "| group | n | mean delta | stdev |",
        "|---|---|---|---|",
    ]
    for label, vals in (("watchlisted org", watchlist_deltas), ("no watchlisted org", other_deltas)):
        if len(vals) < 2:
            lines.append(f"| {label} | {len(vals)} | insufficient data (<2) | |")
            continue
        lines.append(f"| {label} | {len(vals)} | {st.mean(vals):+.3f} | {st.stdev(vals):.3f} |")
    return "\n".join(lines) + "\n"


# ===========================================================================
# --test human-agreement
# ===========================================================================

VERDICT_VALUE = {"STRONG": 2.0, "BORDERLINE": 1.0, "WEAK": 0.0}


def _load_human_scores(conn, *, score_version: str | None) -> dict[int, float]:
    """One human_score per content_id, averaged across reviewers.

    editorial_reviews.score_version records which calibration round the paper
    was drawn under (export_review_set.py stratifies by v1); it is NOT which
    ranking is being judged — the same review set is compared against both v1
    and v2 orderings below. Pass score_version=None to use every stored review.
    """
    if score_version:
        rows = conn.execute(
            "SELECT content_id, verdict FROM research_radar.editorial_reviews WHERE score_version = %s",
            (score_version,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT content_id, verdict FROM research_radar.editorial_reviews").fetchall()
    by_id: dict[int, list[float]] = {}
    for r in rows:
        by_id.setdefault(r["content_id"], []).append(VERDICT_VALUE[r["verdict"]])
    return {cid: sum(vals) / len(vals) for cid, vals in by_id.items()}


def _load_model_ranking(conn, *, score_version: str) -> dict[int, float]:
    rows = conn.execute(
        "SELECT content_id, final_score FROM research_radar.content_final_scores WHERE score_version = %s",
        (score_version,),
    ).fetchall()
    return {r["content_id"]: float(r["final_score"]) for r in rows}


def _top_n_ids(scores: dict[int, float], n: int) -> set[int]:
    return {cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]}


def _agreement_for_version(conn, *, score_version: str, human_scores: dict[int, float], top: int) -> dict:
    model_scores = _load_model_ranking(conn, score_version=score_version)
    common = sorted(set(model_scores) & set(human_scores))
    if len(common) < 2:
        return {"score_version": score_version, "n": len(common), "precision_at_top": None, "recall_at_2x_top": None, "spearman": None}

    human_top = _top_n_ids(human_scores, top)
    model_top = _top_n_ids(model_scores, top)
    precision = len(model_top & human_top) / len(model_top) if model_top else None

    human_top2 = _top_n_ids(human_scores, 2 * top)
    model_top2 = _top_n_ids(model_scores, 2 * top)
    recall = len(model_top2 & human_top2) / len(human_top2) if human_top2 else None

    xs = [model_scores[c] for c in common]
    ys = [human_scores[c] for c in common]
    rho = _spearman(xs, ys)

    return {
        "score_version": score_version, "n": len(common),
        "precision_at_top": precision, "recall_at_2x_top": recall, "spearman": rho,
    }


def run_human_agreement(conn, *, top: int = 20, review_score_version: str | None = None) -> str:
    human_scores = _load_human_scores(conn, score_version=review_score_version)
    if not human_scores:
        raise SystemExit(
            "!! No rows in research_radar.editorial_reviews. Run scripts/export_review_set.py, "
            "get it human-reviewed, and load the verdicts before running --test human-agreement."
        )

    v1 = _agreement_for_version(conn, score_version="radar-v1", human_scores=human_scores, top=top)
    v2 = _agreement_for_version(conn, score_version="radar-v2", human_scores=human_scores, top=top)

    lines = [
        "# Scoring v2 validation — human-agreement",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Human-reviewed papers:** {len(human_scores)}  ",
        "",
        "This is the only test that can show scoring actually improved.",
        "",
        f"| version | n (model ∩ human) | Top-{top} precision | Top-{2*top} recall | Spearman vs human |",
        "|---|---|---|---|---|",
    ]
    for r in (v1, v2):
        prec = f"{r['precision_at_top']:.2f}" if r["precision_at_top"] is not None else "—"
        rec = f"{r['recall_at_2x_top']:.2f}" if r["recall_at_2x_top"] is not None else "—"
        rho = f"{r['spearman']:.3f}" if r["spearman"] is not None else "—"
        lines.append(f"| {r['score_version']} | {r['n']} | {prec} | {rec} | {rho} |")
    return "\n".join(lines) + "\n"


# ===========================================================================
# --test tier-recall (tiering brief §4) — the ONLY thing standing between the
# gate and silently discarding excellent papers because a cheap model ranked
# them at the 24th percentile. GATE_PERCENTILE must be set from this test's
# recommendation, never chosen from a cost target.
# ===========================================================================

TIER_RECALL_GATE_VALUES = (10, 15, 20, 25, 30, 35)
TIER_RECALL_TRUE_TOP_N = 50
TIER_RECALL_MIN_RECALL = 0.95


def _load_paper_fields(conn, content_ids: list[int]) -> dict[int, dict]:
    if not content_ids:
        return {}
    rows = conn.execute(
        """
        SELECT ci.id AS content_id, ci.title,
               COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
               COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.id = ANY(%s)
        """,
        (content_ids,),
    ).fetchall()
    return {r["content_id"]: dict(r) for r in rows}


def _score_batches_screen(batches: list[list[dict]], *, client) -> dict[int, dict]:
    out: dict[int, dict] = {}
    total = len(batches)
    for i, batch in enumerate(batches, 1):
        ids = [p["content_id"] for p in batch]
        print(
            f"[pass1 screen] batch {i}/{total} papers={len(batch)} ids={ids}",
            flush=True,
        )
        result = ss.call_screen_batch_with_retry(batch, client=client)
        out.update(result["results"])
        missing = {p["content_id"] for p in batch} - set(result["results"])
        for pid in missing:
            print(f"[pass1 screen] retry single id={pid}", flush=True)
            paper = next(p for p in batch if p["content_id"] == pid)
            single = ss.call_screen_batch_with_retry([paper], client=client)
            out.update(single["results"])
    return out


def _estimate_tier_recall_cost(sample: list[dict]) -> float:
    n = len(sample)
    p2_batches = random_batches(sample, ss.QUALITY_BATCH_SIZE)
    p1_batches = random_batches(sample, ss.SCREEN_BATCH_SIZE)
    est_in = sum(ss.estimate_quality_prompt_tokens(b) for b in p2_batches) + sum(
        ss.estimate_screen_prompt_tokens(b) for b in p1_batches
    )
    est_out = 220 * n + 40 * n
    return ss.estimate_cost_usd(est_in, est_out)


def run_tier_recall(conn, *, n: int = 400, dry_run: bool = False, client=None) -> str:
    """
    1. Random sample of n papers at ENTITY_RESOLVED (literal brief wording —
       NOT the widened ENTITY_RESOLVED/SCORED/CANDIDATE production filter;
       see the SystemExit message below and the deliverable's flagged-concerns
       section for why that might not be what was intended).
    2. Score all n with pass 2 (full rubric) — ground truth.
    3. Score the same n with pass 1 (screen).
    4. For each gate value in TIER_RECALL_GATE_VALUES, report what fraction of
       pass 2's true top 50 (by radar-v2 composite quality) survives inside
       pass 1's top X% (by mean of the four screen dimensions, ai_relevance
       <= 3 excluded — same rule stage_semantic_score_v2 uses for real).
    """
    rows = conn.execute(
        "SELECT id AS content_id FROM research_radar.content_items WHERE status = 'ENTITY_RESOLVED'"
    ).fetchall()
    ids = [r["content_id"] for r in rows]
    if len(ids) < n:
        raise SystemExit(
            f"!! tier-recall needs {n} papers at status='ENTITY_RESOLVED', only {len(ids)} available. "
            f"Note: production scoring selects ENTITY_RESOLVED+SCORED+CANDIDATE (corrected in an "
            f"earlier round of this brief), but this test's OWN sampling frame is ENTITY_RESOLVED "
            f"only, per the tiering brief's literal wording — flagged as a possible inconsistency, "
            f"not silently widened here."
        )
    sample_ids = random.sample(ids, n)
    fields = _load_paper_fields(conn, sample_ids)
    sample = [fields[i] for i in sample_ids if i in fields]

    if dry_run:
        cost = _estimate_tier_recall_cost(sample)
        return (
            "# Scoring v2 validation — tier-recall DRY RUN\n\n"
            f"n={len(sample)}  approx_cost_usd={cost:.4f}  "
            f"(pass2 calls={-(-len(sample) // ss.QUALITY_BATCH_SIZE)}, "
            f"pass1 calls={-(-len(sample) // ss.SCREEN_BATCH_SIZE)})\n"
        )

    client = client or ss.create_llm_client()

    p2_batches = random_batches(sample, ss.QUALITY_BATCH_SIZE)
    print(
        f"Starting pass2 quality: {len(sample)} papers in {len(p2_batches)} batches "
        f"(batch_size={ss.QUALITY_BATCH_SIZE})",
        flush=True,
    )
    p2_results = _score_batches_v2(p2_batches, client=client)
    print(f"Pass2 done: scored={len(p2_results)}", flush=True)

    p1_batches = random_batches(sample, ss.SCREEN_BATCH_SIZE)
    print(
        f"Starting pass1 screen: {len(sample)} papers in {len(p1_batches)} batches "
        f"(batch_size={ss.SCREEN_BATCH_SIZE})",
        flush=True,
    )
    p1_results = _score_batches_screen(p1_batches, client=client)
    print(f"Pass1 done: screened={len(p1_results)}", flush=True)

    weights = fs.WEIGHT_PROFILES["radar-v2"]
    ground_truth = {cid: fs.compute_semantic_core(d, weights) for cid, d in p2_results.items()}
    true_top_ids = set(
        sorted(ground_truth, key=lambda c: (-ground_truth[c], c))[:TIER_RECALL_TRUE_TOP_N]
    )

    screen_ranked: list[tuple[int, float]] = []
    for cid, d in p1_results.items():
        ai_rel = d.get("ai_relevance")
        if ai_rel is None or ai_rel <= ss.SCREEN_AI_RELEVANCE_FLOOR:
            continue
        vals = [d.get(k) for k in ss.SCREEN_NUMERIC_FIELDS]
        if any(v is None for v in vals):
            continue
        screen_ranked.append((cid, sum(vals) / len(vals)))
    screen_ranked.sort(key=lambda x: (-x[1], x[0]))
    screen_pool_n = len(screen_ranked)

    gate_rows = []
    recommended = None
    for pct in TIER_RECALL_GATE_VALUES:
        k = math.ceil(screen_pool_n * pct / 100.0)
        top_k_ids = {cid for cid, _ in screen_ranked[:k]}
        hits = len(true_top_ids & top_k_ids)
        recall = (hits / len(true_top_ids)) if true_top_ids else 0.0
        gate_rows.append((pct, k, hits, recall))
        if recommended is None and recall >= TIER_RECALL_MIN_RECALL:
            recommended = pct

    lines = [
        "# Scoring v2 validation — tier-recall",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**n:** {len(sample)}  **scored by pass 2:** {len(ground_truth)}  "
        f"**scored by pass 1:** {len(p1_results)}  **screen pool (ai_relevance>3):** {screen_pool_n}  ",
        f"**True top {TIER_RECALL_TRUE_TOP_N}:** by pass-2 radar-v2 composite quality (technical_significance, "
        f"apparent_novelty, practical_applicability, professional_value, learning_value)  ",
        "",
        "| gate % | gated n | hits (of true top 50) | recall |",
        "|---|---|---|---|",
    ]
    for pct, k, hits, recall in gate_rows:
        lines.append(f"| {pct} | {k} | {hits} | {recall:.1%} |")
    lines.append("")
    if recommended is not None:
        lines.append(f"RECOMMENDED GATE_PERCENTILE = {recommended}")
    else:
        lines.append(
            f"WARNING: no gate value in {TIER_RECALL_GATE_VALUES} reached >= "
            f"{TIER_RECALL_MIN_RECALL:.0%} recall. The screen model "
            f"({ss.resolve_screen_model()}) is not good enough to gate on at this "
            f"quality bar — try a different SCREEN_MODEL before trusting any gate "
            f"percentage from this run."
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Scoring v2 validation harness")
    ap.add_argument(
        "--test",
        required=True,
        choices=["compression", "batch-stability", "affiliation-leak", "human-agreement", "tier-recall"],
    )
    ap.add_argument(
        "--n",
        "--sample",  # tiering brief §4 spells it --sample; same option, both names accepted
        dest="n",
        type=int,
        default=100,
        help="Sample size for batch-stability / affiliation-leak / tier-recall",
    )
    ap.add_argument("--top", type=int, default=20, help="Top-N cutoff for compression tie-block / human-agreement precision")
    ap.add_argument("--allow-paid", action="store_true", help="Required for batch-stability / affiliation-leak (live OpenRouter calls)")
    ap.add_argument("--dry-run", action="store_true", help="batch-stability / affiliation-leak: estimate cost only, zero calls")
    args = ap.parse_args()

    if args.test in PAID_TESTS and not args.dry_run and not args.allow_paid:
        raise SystemExit(
            f"!! --test {args.test} makes paid OpenRouter calls. "
            f"Re-run with --allow-paid to authorise spend, or --dry-run to estimate only."
        )

    with connect() as conn:
        if args.test == "compression":
            report = run_compression(conn, top=args.top)
            path = _write_report("compression", report)
        elif args.test == "batch-stability":
            report = run_batch_stability(conn, n=args.n, dry_run=args.dry_run)
            path = _write_report("batch_stability", report)
        elif args.test == "affiliation-leak":
            report = run_affiliation_leak(conn, n=args.n, dry_run=args.dry_run)
            path = _write_report("affiliation_leak", report)
        elif args.test == "human-agreement":
            report = run_human_agreement(conn, top=args.top)
            path = _write_report("human_agreement", report)
        elif args.test == "tier-recall":
            report = run_tier_recall(conn, n=args.n, dry_run=args.dry_run)
            path = _write_report("tier_recall", report)
        else:
            raise SystemExit(f"Unknown --test {args.test}")

    print(report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
