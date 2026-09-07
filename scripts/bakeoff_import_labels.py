#!/usr/bin/env python3
"""Import completed labelling workbook into bakeoff_labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openpyxl import load_workbook

from research_radar.bakeoff import LABELLERS  # noqa: E402
from research_radar.pipeline import connect


def _parse_domains(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _import_sheet(conn, run_id: UUID, ws, labellers: tuple[str, ...]) -> tuple[int, int]:
    headers = [c.value for c in ws[1]]
    inserted = 0
    skipped = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            skipped += 1
            continue
        content_id = int(row[0])
        for i, labeller in enumerate(labellers):
            base = 3 + i * 4
            app_dom = _parse_domains(row[base] if base < len(row) else None)
            audience = _parse_domains(row[base + 1] if base + 1 < len(row) else None)
            paper_kind = row[base + 2] if base + 2 < len(row) else None
            reasoning = row[base + 3] if base + 3 < len(row) else None
            if not any([app_dom, audience, paper_kind, reasoning]):
                continue
            conn.execute(
                """
                INSERT INTO research_radar.bakeoff_labels (
                    content_id, run_id, labeller,
                    application_domain, audience_relevance, paper_kind, reasoning
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_id, run_id, labeller) DO UPDATE SET
                    application_domain = EXCLUDED.application_domain,
                    audience_relevance = EXCLUDED.audience_relevance,
                    paper_kind = EXCLUDED.paper_kind,
                    reasoning = EXCLUDED.reasoning,
                    labelled_at = NOW()
                """,
                (
                    content_id,
                    str(run_id),
                    labeller,
                    app_dom,
                    audience,
                    str(paper_kind).strip() if paper_kind else None,
                    str(reasoning).strip() if reasoning else None,
                ),
            )
            inserted += 1
    return inserted, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Import bake-off labels from xlsx")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workbook", required=True, help="Path to completed .xlsx")
    args = parser.parse_args()

    run_id = UUID(args.run_id)
    wb = load_workbook(args.workbook, read_only=True, data_only=True)

    total_inserted = 0
    total_skipped = 0
    with connect() as conn:
        for sheet_name in ("Disagreements", "Agreement control"):
            if sheet_name not in wb.sheetnames:
                continue
            ins, sk = _import_sheet(conn, run_id, wb[sheet_name], LABELLERS)
            total_inserted += ins
            total_skipped += sk
        conn.commit()

    print(f"Imported {total_inserted} label rows ({total_skipped} blank rows skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
