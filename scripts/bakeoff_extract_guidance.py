#!/usr/bin/env python3
"""Extract prompt-guidance patterns from human labelling disagreements."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import is_force_fit, is_general_method, load_human_labels  # noqa: E402
from research_radar.pipeline import connect

REPORTS_DIR = ROOT / "reports"


def build_guidance(run_id: UUID, labels: list[dict], results: list[dict]) -> str:
    by_paper: dict[int, list[dict]] = defaultdict(list)
    for row in labels:
        by_paper[int(row["content_id"])].append(row)

    model_p1 = {
        int(r["content_id"]): r
        for r in results
        if int(r.get("pass_index") or 1) == 1
    }

    patterns: dict[str, list[str]] = defaultdict(list)

    for cid, lab_rows in by_paper.items():
        if len(lab_rows) < 1:
            continue
        human_domains = [r.get("application_domain") for r in lab_rows]
        human_general_votes = [is_general_method(d) for d in human_domains if d is not None]
        if not human_general_votes:
            continue
        human_general = sum(human_general_votes) >= len(human_general_votes) / 2
        model = model_p1.get(cid)
        if model and is_force_fit(model.get("application_domain"), human_general):
            reason = " | ".join(
                str(r.get("reasoning") or "").strip() for r in lab_rows if r.get("reasoning")
            )
            patterns["force_fit_model_sector_human_general"].append(
                f"content_id={cid}: model={model.get('application_domain')} human_general={human_general}. {reason}"
            )
        domain_sets = {tuple(sorted(d or [])) for d in human_domains if d}
        if len(domain_sets) > 1:
            patterns["human_disagreement_on_domain"].append(
                f"content_id={cid}: labeller domains differ {domain_sets}"
            )
        for row in lab_rows:
            if row.get("reasoning"):
                patterns["human_reasoning_samples"].append(
                    f"content_id={cid} [{row.get('labeller')}]: {row['reasoning']}"
                )

    lines = [
        f"# Bake-off prompt guidance — `{run_id}`",
        "",
        "Candidate prompt clarifications derived from human labels. **Report only** — a human decides what enters the next prompt version.",
        "",
    ]
    for pattern, examples in patterns.items():
        lines.append(f"## {pattern}")
        lines.append("")
        for ex in examples[:25]:
            lines.append(f"- {ex}")
        if len(examples) > 25:
            lines.append(f"- … and {len(examples) - 25} more")
        lines.append("")
    if not patterns:
        lines.append("_No labelled disagreements imported yet._")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract bake-off prompt guidance from labels")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_id = UUID(args.run_id)
    out = Path(args.out) if args.out else REPORTS_DIR / f"bakeoff-guidance-{run_id}.md"

    with connect() as conn:
        labels = load_human_labels(conn, run_id)
        from research_radar.bakeoff import load_results_for_run

        results = load_results_for_run(conn, run_id)

    guidance = build_guidance(run_id, labels, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(guidance, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
