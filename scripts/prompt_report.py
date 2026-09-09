#!/usr/bin/env python3
"""Free read-only report: papers per prompt_version per stage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.pipeline import connect  # noqa: E402
from research_radar.prompt_registry import get_prompt, papers_per_prompt_version  # noqa: E402


def main() -> int:
    with connect() as conn:
        print("Registered prompts (active preferred via get_prompt):")
        for kind in (
            "paper_scoring",
            "screen",
            "classify",
            "independence",
            "topics",
            "policy",
            "embedding_policy",
            "clustering_policy",
        ):
            row = get_prompt(conn, kind)
            if row:
                print(f"  {kind}/{row['version']} sha={row['body_sha256'][:12]}… model={row.get('model_name')}")
            else:
                print(f"  {kind}: (none registered)")

        print("\nPapers / assessments by prompt_version:")
        print(f"{'stage':<16}{'prompt_version':<32}{'n':>8}")
        for r in papers_per_prompt_version(conn):
            print(f"{r['stage']:<16}{r['prompt_version']:<32}{r['n']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
