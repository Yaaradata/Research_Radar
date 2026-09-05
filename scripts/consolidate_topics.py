#!/usr/bin/env python3
"""Report-only vocabulary hygiene check for level-3 (open-vocabulary) topics.

Two kinds of candidate for a human to merge into an alias, never merged
automatically:

  - low usage: usage_count <= --max-usage (default 2) — likely one-off
    phrasings that should have resolved to an existing topic.
  - near-duplicates: pairs of level-3 topics whose canonical_name strings are
    similar (character-trigram Jaccard) above --min-similarity, e.g.
    "ai-text-detection" / "synthetic-text-detection".

No pg_trgm dependency — trigram similarity is computed in Python so this
script has no DDL prerequisite beyond 012/013 having been applied.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/consolidate_topics.py
    PYTHONPATH=src .venv/bin/python scripts/consolidate_topics.py --min-similarity 0.4 --max-usage 1
"""

from __future__ import annotations

import argparse

from research_radar.pipeline import connect


def trigrams(s: str) -> set[str]:
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def trigram_similarity(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def load_level3_topics(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT topic_id, canonical_name, usage_count, aliases, origin
        FROM research_radar.topics
        WHERE level = 'topic'
        ORDER BY usage_count DESC, canonical_name
        """
    ).fetchall()
    return [dict(r) for r in rows]


def find_low_usage(topics: list[dict], max_usage: int) -> list[dict]:
    return [t for t in topics if t["usage_count"] <= max_usage]


def find_near_duplicates(topics: list[dict], min_similarity: float) -> list[tuple[dict, dict, float]]:
    pairs = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            a, b = topics[i], topics[j]
            sim = trigram_similarity(a["canonical_name"], b["canonical_name"])
            if sim >= min_similarity:
                pairs.append((a, b, sim))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def render_report(topics: list[dict], low_usage: list[dict], duplicates: list[tuple[dict, dict, float]]) -> str:
    lines = [
        f"CONSOLIDATE TOPICS REPORT  ({len(topics)} level-3 topics total)",
        "Report only — nothing below has been merged. Add a merged name to the",
        "survivor's `aliases` array and re-point/delete the loser by hand.",
        "",
        f"LOW USAGE ({len(low_usage)} candidates)",
    ]
    for t in low_usage:
        lines.append(f"  usage={t['usage_count']:<3} origin={t['origin']:<4} {t['canonical_name']}")
    lines.append("")
    lines.append(f"NEAR-DUPLICATES ({len(duplicates)} pairs)")
    for a, b, sim in duplicates:
        lines.append(
            f"  sim={sim:.2f}  {a['canonical_name']} (usage={a['usage_count']})  <->  "
            f"{b['canonical_name']} (usage={b['usage_count']})"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Report level-3 topic merge candidates. Never merges automatically.")
    ap.add_argument("--max-usage", type=int, default=2, help="Flag topics with usage_count <= this (default 2)")
    ap.add_argument("--min-similarity", type=float, default=0.5, help="Trigram Jaccard threshold for near-duplicates (default 0.5)")
    args = ap.parse_args()

    with connect() as conn:
        topics = load_level3_topics(conn)

    low_usage = find_low_usage(topics, args.max_usage)
    duplicates = find_near_duplicates(topics, args.min_similarity)
    print(render_report(topics, low_usage, duplicates))


if __name__ == "__main__":
    main()
