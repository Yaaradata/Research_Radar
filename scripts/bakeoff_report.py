#!/usr/bin/env python3
"""Compute bake-off metrics and write reports/bakeoff-{run_id}.md."""

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


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(run_id: UUID, results: list[dict], labels: list[dict], config) -> str:
    candidates = sorted({r["candidate_id"] for r in results})
    cand_cfg = {c.id: c for c in config.candidates}
    lines = [
        f"# Classification bake-off report — `{run_id}`",
        "",
        "## Comparison",
        "",
        "| candidate | accuracy | general_method_rate | force_fit_rate | exclusivity | invalid_rate | valid_json_rate | self_consistency | cost_per_1k | mean_latency_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    per_candidate: dict[str, dict] = {}
    for cid in candidates:
        rows_p1 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 1]
        rows_p2 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 2]
        cfg = cand_cfg.get(cid)
        metrics = compute_candidate_metrics(
            rows_p1,
            labels,
            input_cost_per_million=cfg.input_cost_per_million if cfg else 1.0,
            output_cost_per_million=cfg.output_cost_per_million if cfg else 5.0,
        )
        p1 = {int(r["content_id"]): r for r in rows_p1}
        p2 = {int(r["content_id"]): r for r in rows_p2}
        sc = self_consistency_rate(p1, p2) if p2 else None
        metrics["self_consistency"] = round(sc, 4) if sc is not None else None
        per_candidate[cid] = metrics
        note = ""
        if cfg is not None and not getattr(cfg, "comparable", True):
            note = " ⚠️ non-comparable (reasoning-class latency; excluded from decision)"
        lines.append(
            f"| {cid}{note} | {_fmt(metrics.get('accuracy'))} | {_fmt(metrics.get('general_method_rate'))} | "
            f"{_fmt(metrics.get('force_fit_rate'))} | {metrics.get('exclusivity_violations', 0)} | "
            f"{_fmt(metrics.get('invalid_rate'))} | {_fmt(metrics.get('valid_json_rate'))} | "
            f"{_fmt(metrics.get('self_consistency'))} | {_fmt(metrics.get('cost_per_1000'))} | "
            f"{_fmt(metrics.get('mean_latency_ms'))} |"
        )

    lines.extend(["", "## Per-candidate detail", ""])
    for cid, metrics in per_candidate.items():
        lines.append(f"### {cid}")
        lines.append("")
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if not labels:
        lines.extend(
            [
                "> No human labels imported yet — accuracy and force_fit_rate require bakeoff_labels.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate bake-off metrics report")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default=None, help="Output path (default reports/bakeoff-{run_id}.md)")
    args = parser.parse_args()

    run_id = UUID(args.run_id)
    config = load_bakeoff_config()
    out_path = Path(args.out) if args.out else REPORTS_DIR / f"bakeoff-{run_id}.md"

    with connect() as conn:
        results = load_results_for_run(conn, run_id)
        labels = load_human_labels(conn, run_id)

    if not results:
        print(f"No bakeoff_results for run_id={run_id}", file=sys.stderr)
        return 1

    report = build_report(run_id, results, labels, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
