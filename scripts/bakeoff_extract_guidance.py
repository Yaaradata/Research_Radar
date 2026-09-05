#!/usr/bin/env python3
"""Extract prompt-guidance candidates from human bake-off disagreement reasoning.

Report only — a human decides what goes into the next prompt version.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import group_disagreements_by_pattern  # noqa: E402
from research_radar.pipeline import connect

REPORTS_DIR = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract bake-off prompt guidance from human labels")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT bl.content_id, bl.labeller, bl.domain, bl.application_domains,
                   bl.is_general_method, bl.reasoning, ci.title
            FROM research_radar.bakeoff_labels bl
            JOIN research_radar.content_items ci ON ci.id = bl.content_id
            WHERE bl.run_id = %s AND bl.reasoning IS NOT NULL AND TRIM(bl.reasoning) <> ''
            """,
            (args.run_id,),
        ).fetchall()
        disagreements = [dict(r) for r in rows]

    buckets = group_disagreements_by_pattern(disagreements)
    lines = [
        "# Bake-off prompt guidance extraction",
        "",
        f"**run_id:** `{args.run_id}`  ",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Labelled disagreements with reasoning:** {len(disagreements)}  ",
        "",
        "This report groups human reasoning by failure pattern. Do not apply automatically.",
        "",
    ]

    for bucket, items in buckets.items():
        lines.append(f"## {bucket.replace('_', ' ').title()} ({len(items)})")
        lines.append("")
        by_theme: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            key = (item.get("reasoning") or "")[:80].strip()
            by_theme[key].append(item)
        for theme, group in sorted(by_theme.items(), key=lambda x: -len(x[1]))[:15]:
            lines.append(f"### Pattern ({len(group)} papers)")
            lines.append(f"> {theme}...")
            sample = group[0]
            lines.append(f"- Example: `{sample.get('title', '')[:70]}` (content_id={sample['content_id']})")
            if sample.get("is_general_method") is not None:
                lines.append(f"- Human general-method: {sample['is_general_method']}")
            lines.append("")
        lines.append("**Candidate prompt clarification:**")
        if bucket == "force_fitting":
            lines.append(
                "- Reinforce that methods papers without a named sector must receive "
                "`general-method` alone; constructing a plausible deployment chain is an error."
            )
        elif bucket == "domain_mismatch":
            lines.append("- Clarify domain vs subdomain boundaries with one more in-prompt example.")
        elif bucket == "audience_mismatch":
            lines.append("- Distinguish practitioner vs technical_leadership with a counterexample.")
        else:
            lines.append("- Review individual reasoning snippets above for recurring phrasing.")
        lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"bakeoff-guidance-{args.run_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
