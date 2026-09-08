#!/usr/bin/env python3
"""Report which sql/ migrations appear applied versus present (read-only, free).

Probes each migration's key object via research_radar.schema_preflight.
Does not run DDL. Does not call paid APIs.

Usage:
  PYTHONPATH=src python scripts/check_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_radar.pipeline import connect  # noqa: E402
from research_radar.schema_preflight import (  # noqa: E402
    MIGRATION_CHECKS,
    STAGE_REQUIRED_MIGRATIONS,
    report_all_migrations,
)


def main() -> int:
    sql_dir = ROOT / "sql"
    present = sorted(p.name for p in sql_dir.glob("*.sql"))
    registered = {Path(c.migration).name for c in MIGRATION_CHECKS}

    print("Research Radar schema check (read-only)\n")
    print(f"sql/ files present: {len(present)}")
    print(f"probes registered:  {len(MIGRATION_CHECKS)}\n")

    unregistered = [n for n in present if n not in registered]
    if unregistered:
        print("Present in sql/ but no probe registered:")
        for n in unregistered:
            print(f"  ? {n}")
        print()

    with connect() as conn:
        rows = report_all_migrations(conn)

    missing = []
    print(f"{'STATUS':<8} {'MIGRATION':<42} OBJECT")
    print("-" * 90)
    for row in rows:
        status = "OK" if row["applied"] else "MISSING"
        if not row["applied"]:
            missing.append(row)
        print(f"{status:<8} {row['migration']:<42} {row['object']}")

    print("\nStage prerequisites:")
    for stage, migs in STAGE_REQUIRED_MIGRATIONS.items():
        print(f"  {stage}: {', '.join(migs)}")

    print()
    if missing:
        print(f"Unapplied against live DB ({len(missing)}):")
        for row in missing:
            print(f"  - {row['migration']}  ({row['object']})")
        return 1

    print("All probed migrations appear applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
