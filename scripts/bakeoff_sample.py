#!/usr/bin/env python3
"""Draw a stratified 400-paper sample for the classification bake-off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    BAKEOFF_PROMPT_VERSION,
    load_bakeoff_config,
    new_run_id,
    persist_sample,
    select_stratified_sample,
    load_eligible_papers,
)
from research_radar.pipeline import connect


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a stratified bake-off sample")
    parser.add_argument("--seed", type=int, default=None, help="Override sample seed from config")
    parser.add_argument("--run-id", type=str, default=None, help="Reuse or specify run UUID")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not write DB row")
    args = parser.parse_args()

    config = load_bakeoff_config()
    seed = args.seed if args.seed is not None else config.sample_seed
    run_id = args.run_id or str(new_run_id())

    with connect() as conn:
        papers = load_eligible_papers(conn)
        sample, counts = select_stratified_sample(papers, seed=seed)
        total = sum(counts.values())
        print(f"Eligible pool: {len(papers)}")
        print(f"Sample seed: {seed}")
        print(f"Stratum counts: {json.dumps(counts)}")
        print(f"Total selected: {total}")
        if total < config.sample_size:
            print(
                f"WARNING: requested {config.sample_size} but only {total} papers matched strata",
                file=sys.stderr,
            )
        if args.dry_run:
            return 0
        persist_sample(conn, run_id, seed, sample, BAKEOFF_PROMPT_VERSION)
        print(f"run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
