#!/usr/bin/env python3
"""Restore archived papers from S3 via the manifest table."""

from __future__ import annotations

import io
import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.arxiv_backfill import record_to_item, upsert_backfill_paper_metadata
from research_radar.pipeline import connect, detect_source_type, normalize_url, set_status, upsert_item

PROTECTED_STATUSES = frozenset({"SCORED", "CANDIDATE", "ENTITY_RESOLVED", "ENRICHED", "RELEVANT"})


def _download_object(bucket: str, key: str) -> bytes:
    import boto3

    client = boto3.client("s3")
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _iter_jsonl_gzip(data: bytes):
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as gz:
        for line in gz:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rejected_to_item(rec: dict) -> dict:
    url = rec.get("canonical_url") or ""
    categories = rec.get("categories") or []
    return {
        "source": "restore",
        "source_external_id": str(rec.get("content_id") or rec.get("arxiv_id") or url),
        "source_type": detect_source_type(url, "", categories),
        "canonical_url": normalize_url(url),
        "title": rec.get("title") or "(untitled)",
        "summary": rec.get("abstract") or "",
        "authors_raw": [],
        "categories_raw": categories,
        "inoreader_tags": [],
        "raw_metadata": rec,
    }


def _restore_record(conn, rec: dict, *, into_status: str, dry_run: bool) -> str:
    if rec.get("arxiv_id") or rec.get("identifier"):
        item = record_to_item(rec) if rec.get("arxiv_id") else _rejected_to_item(rec)
    else:
        item = _rejected_to_item(rec)

    if dry_run:
        return "would_restore"

    content_id, is_new = upsert_item(conn, item)
    if rec.get("arxiv_id"):
        upsert_backfill_paper_metadata(conn, content_id, rec)

    row = conn.execute(
        "SELECT status FROM research_radar.content_items WHERE id=%s",
        (content_id,),
    ).fetchone()
    current = row["status"] if row else None
    if is_new or current not in PROTECTED_STATUSES:
        set_status(conn, content_id, into_status)
        return "restored"
    return "skipped_protected"


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore papers from S3 archives")
    parser.add_argument("--kind", choices=["raw", "rejected"], default="rejected")
    parser.add_argument("--from", dest="date_from", default=None)
    parser.add_argument("--until", dest="date_until", default=None)
    parser.add_argument("--archive-id", type=int, default=None)
    parser.add_argument("--into-status", default="INGESTED")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with connect() as conn:
        if args.archive_id is not None:
            rows = conn.execute(
                "SELECT * FROM research_radar.s3_archives WHERE archive_id = %s",
                (args.archive_id,),
            ).fetchall()
        else:
            filters = ["kind = %s"]
            params: list = [args.kind]
            if args.date_from:
                filters.append("created_at::date >= %s::date")
                params.append(args.date_from)
            if args.date_until:
                filters.append("created_at::date <= %s::date")
                params.append(args.date_until)
            rows = conn.execute(
                f"SELECT * FROM research_radar.s3_archives WHERE {' AND '.join(filters)} ORDER BY archive_id",
                params,
            ).fetchall()

        if not rows:
            print("No matching archives in manifest")
            return 1

        counts = {"restored": 0, "skipped_protected": 0, "would_restore": 0}
        for row in rows:
            print(f"archive_id={row['archive_id']} s3://{row['s3_bucket']}/{row['s3_key']} records={row['record_count']}")
            if args.dry_run and args.archive_id is None:
                counts["would_restore"] += int(row["record_count"])
                continue
            data = _download_object(row["s3_bucket"], row["s3_key"])
            for rec in _iter_jsonl_gzip(data):
                outcome = _restore_record(conn, rec, into_status=args.into_status, dry_run=args.dry_run)
                counts[outcome] = counts.get(outcome, 0) + 1
        if not args.dry_run:
            conn.commit()
        print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
