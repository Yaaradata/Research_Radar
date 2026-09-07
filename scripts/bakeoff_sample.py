#!/usr/bin/env python3
"""Draw a stratified 400-paper sample from Haiku-classified papers and import baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    CLASSIFY_PROMPT_VERSION,
    import_haiku_baseline,
    load_baseline_classified_papers,
    load_bakeoff_config,
    new_run_id,
    persist_bakeoff_run,
    scaled_stratum_targets,
    select_stratified_sample,
)
from research_radar.pipeline import connect


def main() -> int:
    parser = argparse.ArgumentParser(description="Create stratified bake-off sample and import Haiku baseline")
    parser.add_argument("--seed", type=int, default=None, help="Override sample seed from config")
    parser.add_argument("--run-id", type=str, default=None, help="Specify run UUID")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size from config (scales stratum targets; default 400)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not write DB")
    args = parser.parse_args()

    config = load_bakeoff_config()
    seed = args.seed if args.seed is not None else config.sample_seed
    sample_size = args.sample_size if args.sample_size is not None else config.sample_size
    targets = scaled_stratum_targets(sample_size)
    run_id = UUID(args.run_id) if args.run_id else new_run_id()

    with connect() as conn:
        pool = load_baseline_classified_papers(conn, config)
        sample, counts = select_stratified_sample(pool, seed=seed, targets=targets)
        total = len(sample)
        print(f"Baseline pool ({config.baseline_date_from}..{config.baseline_date_until}): {len(pool)}")
        print(f"Sample seed: {seed}")
        print(f"Sample size target: {sample_size}")
        print(f"Stratum targets: {json.dumps(targets)}")
        print(f"Stratum counts: {json.dumps(counts)}")
        print(f"Total selected: {total}")
        if total < sample_size:
            print(
                f"WARNING: target {sample_size} but only {total} papers sampled",
                file=sys.stderr,
            )
        if args.dry_run:
            return 0
        persist_bakeoff_run(
            conn,
            run_id,
            seed=seed,
            sample_size=total,
            prompt_version=CLASSIFY_PROMPT_VERSION,
        )
        imported = import_haiku_baseline(conn, run_id, sample, candidate_id=config.baseline_candidate_id)
        conn.commit()
        print(f"run_id={run_id}")
        print(f"imported_haiku_baseline={imported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
