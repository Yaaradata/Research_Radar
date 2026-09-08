#!/usr/bin/env python3
"""Write bakeoff-decision report with recommendation template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    compute_candidate_metrics,
    load_bakeoff_config,
    load_human_labels,
    load_results_for_run,
    self_consistency_rate,
)
from research_radar.pipeline import connect

REPORTS_DIR = ROOT / "reports"
REMAINING_PAPERS = 91_000


def build_decision_report(run_id: UUID, results: list[dict], labels: list[dict], config) -> str:
    candidates = sorted({r["candidate_id"] for r in results})
    cand_cfg = {c.id: c for c in config.candidates}
    lines = [
        f"# Bake-off decision — `{run_id}`",
        "",
        "## Comparison table",
        "",
        "| candidate | accuracy | force_fit_rate | schema_valid_rate | self_consistency | cost_per_1k | disqualified |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    best = None
    for cid in candidates:
        rows_p1 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 1]
        rows_p2 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 2]
        cfg = cand_cfg.get(cid)
        m = compute_candidate_metrics(
            rows_p1,
            labels,
            input_cost_per_million=cfg.input_cost_per_million if cfg else 1.0,
            output_cost_per_million=cfg.output_cost_per_million if cfg else 5.0,
        )
        p1 = {int(r["content_id"]): r for r in rows_p1}
        p2 = {int(r["content_id"]): r for r in rows_p2}
        sc = self_consistency_rate(p1, p2) if p2 else None
        disqualified = m.get("schema_valid_rate", 0) < 0.95
        lines.append(
            f"| {cid} | {m.get('accuracy', '—')} | {m.get('force_fit_rate', '—')} | "
            f"{m.get('schema_valid_rate', '—')} | {round(sc, 4) if sc is not None else '—'} | "
            f"{m.get('cost_per_1000', '—')} | {'yes' if disqualified else 'no'} |"
        )
        if not disqualified and m.get("accuracy") is not None:
            if best is None or (m["accuracy"], -(m.get("force_fit_rate") or 0)) > (
                best[1].get("accuracy", 0),
                -(best[1].get("force_fit_rate") or 0),
            ):
                best = (cid, m, cfg)

    lines.extend(["", "## Recommendation", ""])
    if best:
        cid, m, cfg = best
        cost_1k = m.get("cost_per_1000") or 0
        projected = round(cost_1k * REMAINING_PAPERS / 1000.0, 2)
        baseline_cfg = cand_cfg.get(config.baseline_candidate_id)
        baseline_rows = [
            r
            for r in results
            if r["candidate_id"] == config.baseline_candidate_id and int(r["pass_index"]) == 1
        ]
        baseline_cost = (
            compute_candidate_metrics(
                baseline_rows,
                [],
                input_cost_per_million=baseline_cfg.input_cost_per_million,
                output_cost_per_million=baseline_cfg.output_cost_per_million,
            ).get("cost_per_1000")
            if baseline_cfg and baseline_rows
            else None
        )
        lines.extend(
            [
                f"**Recommended model for classify/screen:** `{cfg.model}` (`{cid}`)",
                "",
                f"- Accuracy vs human labels: **{m.get('accuracy')}**",
                f"- Force-fit rate: **{m.get('force_fit_rate')}**",
                f"- Schema valid rate: **{m.get('schema_valid_rate')}**",
                f"- Cost per 1,000 papers (measured tokens): **${cost_1k:.4f}**",
                f"- Haiku baseline cost per 1,000: **${baseline_cost:.4f}**" if baseline_cost else "- Haiku baseline cost: n/a",
                f"- Projected cost for ~{REMAINING_PAPERS:,} remaining papers: **${projected:,.2f}**",
                "",
            ]
        )
    else:
        lines.append("_Import human labels and re-run after bakeoff_run completes to populate recommendation._")

    lines.extend(
        [
            "## Disqualifications",
            "",
            "Candidates with schema_valid_rate < 0.95 are flagged disqualified in the table above.",
            "",
            "## Open questions for human decision",
            "",
            "- Confirm human label coverage is sufficient before trusting accuracy.",
            "- Review force-fit examples in `bakeoff-guidance-{run_id}.md` before prompt changes.",
            "- Validate OpenRouter model ids remain available at execution time.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write bake-off decision report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_id = UUID(args.run_id)
    config = load_bakeoff_config()
    out = Path(args.out) if args.out else REPORTS_DIR / f"bakeoff-decision-{run_id}.md"

    with connect() as conn:
        results = load_results_for_run(conn, run_id)
        labels = load_human_labels(conn, run_id)

    report = build_decision_report(run_id, results, labels, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
