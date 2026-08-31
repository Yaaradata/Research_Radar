#!/usr/bin/env python3
"""Export a 150-paper editorial calibration set for human review.

50 top / 50 middle / 50 bottom, stratified by v1's semantic_core (radar-v1
weighted mean), so the sample spans the whole quality range rather than
clustering near the old cut line. Title + abstract only, verdict column left
blank — no scores are exported, so the reviewer is not anchored to how the
model already ranked the paper (brief §5).

Usage:
    python3 scripts/export_review_set.py --out reports/editorial_review_set.csv

The reviewer fills in the `verdict` column (STRONG / BORDERLINE / WEAK) and
the CSV is then loaded into research_radar.editorial_reviews for
validate_scoring.py --test human-agreement to read.
"""

from __future__ import annotations

import argparse
import csv
import random

from research_radar import final_score as fs
from research_radar.pipeline import connect

PER_GROUP = 50
SEED = 20260831


def select_calibration_set(conn, *, per_group: int = PER_GROUP, seed: int = SEED) -> list[dict]:
    rows = fs.load_scored_papers(conn, prompt_version="research-semantic-v1")
    _, weights = fs.resolve_profile("radar-v1")

    scored = []
    for r in rows:
        try:
            core = fs.compute_semantic_core({k: r.get(k) for k in fs.DIMENSIONS}, weights)
        except ValueError:
            continue
        scored.append({**r, "v1_semantic_core": core})

    scored.sort(key=lambda r: (-r["v1_semantic_core"], r["content_id"]))
    n = len(scored)
    if n < 3 * per_group:
        raise SystemExit(
            f"!! Need at least {3 * per_group} v1-scored papers to stratify top/middle/bottom, only {n} available"
        )

    top = scored[:per_group]
    bottom = scored[-per_group:]
    middle_pool = scored[per_group : n - per_group]
    rng = random.Random(seed)
    middle = rng.sample(middle_pool, min(per_group, len(middle_pool)))

    ids = [r["content_id"] for r in (top + middle + bottom)]
    abstracts = _load_abstracts(conn, ids)

    selected = []
    for group, papers in (("top", top), ("middle", middle), ("bottom", bottom)):
        for r in papers:
            selected.append(
                {
                    "content_id": r["content_id"],
                    "group": group,
                    "title": r["title"],
                    "abstract": abstracts.get(r["content_id"], ""),
                }
            )
    return selected


def _load_abstracts(conn, content_ids: list[int]) -> dict[int, str]:
    if not content_ids:
        return {}
    rows = conn.execute(
        """
        SELECT ci.id AS content_id, COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.id = ANY(%s)
        """,
        (content_ids,),
    ).fetchall()
    return {r["content_id"]: r["abstract"] for r in rows}


def write_csv(rows: list[dict], out_path: str, *, shuffle: bool = True, seed: int = SEED):
    ordered = list(rows)
    if shuffle:
        # Reviewer sees top/middle/bottom interleaved, not in blocks — the
        # group label stays in the file for later analysis but should not
        # anchor the reviewer while they work through it in file order.
        random.Random(seed + 1).shuffle(ordered)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["content_id", "group", "title", "abstract", "verdict", "note"])
        writer.writeheader()
        for r in ordered:
            writer.writerow(
                {
                    "content_id": r["content_id"],
                    "group": r["group"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "verdict": "",
                    "note": "",
                }
            )


def main():
    ap = argparse.ArgumentParser(description="Export editorial calibration set (title+abstract only, no scores)")
    ap.add_argument("--out", default="reports/editorial_review_set.csv", help="Output CSV path")
    ap.add_argument("--per-group", type=int, default=PER_GROUP, help="Papers per top/middle/bottom group (default 50)")
    args = ap.parse_args()

    with connect() as conn:
        rows = select_calibration_set(conn, per_group=args.per_group)
    write_csv(rows, args.out)
    print(f"Wrote {args.out} ({len(rows)} papers: {args.per_group} top / {args.per_group} middle / {args.per_group} bottom)")


if __name__ == "__main__":
    main()
