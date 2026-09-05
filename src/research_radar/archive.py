"""S3 archive for raw pulls and rejected papers — best-effort, batched, gzipped."""

from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import UUID

log = logging.getLogger("research-radar")

S3_ARCHIVE_ENABLED = os.getenv("S3_ARCHIVE_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("RESEARCH_RADAR_S3_BUCKET", "")
S3_PREFIX = os.getenv("RESEARCH_RADAR_S3_PREFIX", "research-radar")

REJECTED_RECORD_FIELDS = (
    "content_id",
    "arxiv_id",
    "canonical_url",
    "title",
    "abstract",
    "categories",
    "relevance_score",
    "primary_topic",
    "rejection_reason",
    "relevance_version",
    "rejected_at",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _s3_key(kind: str, stage: str, run_id: UUID, when: datetime | None = None) -> str:
    when = when or _utc_now()
    y, m, d = when.year, f"{when.month:02d}", f"{when.day:02d}"
    if kind == "raw":
        return f"{S3_PREFIX}/raw/{stage}/{y}/{m}/{d}/{run_id}.jsonl.gz"
    return f"{S3_PREFIX}/rejected/{stage}/{y}/{m}/{d}/{run_id}.jsonl.gz"


def _iter_records(records: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for rec in records:
        if rec is not None:
            yield rec


def _write_jsonl_gzip(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as gz:
        for rec in _iter_records(records):
            gz.write(json.dumps(rec, default=str) + "\n")
            count += 1
    return count, path.stat().st_size


def _upload_file(local_path: Path, bucket: str, key: str) -> None:
    import boto3

    client = boto3.client("s3")
    with open(local_path, "rb") as fh:
        client.put_object(Bucket=bucket, Key=key, Body=fh, ContentType="application/gzip")


def _write_manifest(
    conn,
    *,
    run_id: UUID,
    kind: str,
    stage: str,
    bucket: str,
    key: str,
    record_count: int,
    bytes_written: int,
    window_from=None,
    window_until=None,
) -> None:
    conn.execute(
        """
        INSERT INTO research_radar.s3_archives (
            run_id, kind, stage, s3_bucket, s3_key, record_count, bytes_written,
            window_from, window_until
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (s3_bucket, s3_key) DO UPDATE SET
            record_count = EXCLUDED.record_count,
            bytes_written = EXCLUDED.bytes_written,
            created_at = NOW()
        """,
        (
            str(run_id),
            kind,
            stage,
            bucket,
            key,
            record_count,
            bytes_written,
            window_from,
            window_until,
        ),
    )


def _archive_records(
    conn,
    *,
    run_id: UUID,
    kind: str,
    stage: str,
    records: Iterable[dict[str, Any]],
    window_from=None,
    window_until=None,
) -> dict[str, Any] | None:
    key = _s3_key(kind, stage, run_id)
    if not S3_ARCHIVE_ENABLED:
        count = sum(1 for _ in _iter_records(records))
        log.info(
            "S3 archive disabled: would write kind=%s stage=%s records=%d key=s3://%s/%s",
            kind,
            stage,
            count,
            S3_BUCKET or "(unset)",
            key,
        )
        return None
    if not S3_BUCKET:
        log.warning("S3_ARCHIVE_ENABLED but RESEARCH_RADAR_S3_BUCKET is unset; skipping archive")
        return None

    with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        record_count, nbytes = _write_jsonl_gzip(tmp_path, records)
        if record_count == 0:
            log.info("S3 archive skip empty kind=%s stage=%s run_id=%s", kind, stage, run_id)
            return None
        try:
            _upload_file(tmp_path, S3_BUCKET, key)
        except Exception as exc:
            log.warning(
                "S3 archive upload failed kind=%s stage=%s key=%s: %s",
                kind,
                stage,
                key,
                exc,
            )
            return None
        if conn is not None:
            _write_manifest(
                conn,
                run_id=run_id,
                kind=kind,
                stage=stage,
                bucket=S3_BUCKET,
                key=key,
                record_count=record_count,
                bytes_written=nbytes,
                window_from=window_from,
                window_until=window_until,
            )
        log.info(
            "S3 archived kind=%s stage=%s records=%d bytes=%d s3://%s/%s",
            kind,
            stage,
            record_count,
            nbytes,
            S3_BUCKET,
            key,
        )
        return {"bucket": S3_BUCKET, "key": key, "record_count": record_count, "bytes_written": nbytes}
    finally:
        tmp_path.unlink(missing_ok=True)


def archive_raw(
    conn,
    run_id: UUID,
    source: str,
    records: Iterable[dict[str, Any]],
    *,
    window_from=None,
    window_until=None,
) -> dict[str, Any] | None:
    """Archive source payloads before any database write."""
    return _archive_records(
        conn,
        run_id=run_id,
        kind="raw",
        stage=source,
        records=records,
        window_from=window_from,
        window_until=window_until,
    )


def archive_rejected(
    conn,
    run_id: UUID,
    stage: str,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Archive rejection payloads before status is set to REJECTED."""
    return _archive_records(conn, run_id=run_id, kind="rejected", stage=stage, records=records)


def build_rejected_record(
    *,
    content_id: int,
    canonical_url: str,
    title: str,
    abstract: str,
    categories,
    relevance_score: float,
    primary_topic: str | None,
    rejection_reason: str,
    relevance_version: str,
    arxiv_id: str | None = None,
    rejected_at: datetime | None = None,
) -> dict[str, Any]:
    if isinstance(categories, str):
        try:
            categories = json.loads(categories)
        except json.JSONDecodeError:
            categories = [categories]
    return {
        "content_id": content_id,
        "arxiv_id": arxiv_id,
        "canonical_url": canonical_url,
        "title": title,
        "abstract": abstract or "",
        "categories": categories or [],
        "relevance_score": relevance_score,
        "primary_topic": primary_topic,
        "rejection_reason": rejection_reason,
        "relevance_version": relevance_version,
        "rejected_at": (rejected_at or _utc_now()).isoformat(),
    }


class RawArchiveWriter:
    """Stream records to a temp gzip file; upload once on finish()."""

    def __init__(self, run_id: UUID, source: str):
        self.run_id = run_id
        self.source = source
        self._path: Path | None = None
        self._gz = None
        self.count = 0

    def __enter__(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False)
        self._path = Path(tmp.name)
        tmp.close()
        self._gz = gzip.open(self._path, "wt", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self._gz is None:
            raise RuntimeError("RawArchiveWriter not opened")
        self._gz.write(json.dumps(record, default=str) + "\n")
        self.count += 1

    def finish(self, conn, *, window_from=None, window_until=None) -> dict[str, Any] | None:
        if self._gz is not None:
            self._gz.close()
            self._gz = None
        if self._path is None or self.count == 0:
            if self._path:
                self._path.unlink(missing_ok=True)
            return None
        key = _s3_key("raw", self.source, self.run_id)
        if not S3_ARCHIVE_ENABLED:
            log.info(
                "S3 archive disabled: would write kind=raw stage=%s records=%d key=s3://%s/%s",
                self.source,
                self.count,
                S3_BUCKET or "(unset)",
                key,
            )
            self._path.unlink(missing_ok=True)
            return None
        if not S3_BUCKET:
            log.warning("S3_ARCHIVE_ENABLED but RESEARCH_RADAR_S3_BUCKET is unset; skipping archive")
            self._path.unlink(missing_ok=True)
            return None
        nbytes = self._path.stat().st_size
        try:
            _upload_file(self._path, S3_BUCKET, key)
        except Exception as exc:
            log.warning("S3 raw archive upload failed key=%s: %s", key, exc)
            self._path.unlink(missing_ok=True)
            return None
        if conn is not None:
            _write_manifest(
                conn,
                run_id=self.run_id,
                kind="raw",
                stage=self.source,
                bucket=S3_BUCKET,
                key=key,
                record_count=self.count,
                bytes_written=nbytes,
                window_from=window_from,
                window_until=window_until,
            )
        log.info(
            "S3 archived kind=raw stage=%s records=%d bytes=%d s3://%s/%s",
            self.source,
            self.count,
            nbytes,
            S3_BUCKET,
            key,
        )
        self._path.unlink(missing_ok=True)
        return {"bucket": S3_BUCKET, "key": key, "record_count": self.count, "bytes_written": nbytes}

    def __exit__(self, *args):
        if self._gz is not None:
            self._gz.close()
            self._gz = None


def estimate_storage_cost(
    *,
    corpus_items: int,
    avg_raw_bytes: int = 3500,
    reject_fraction: float = 0.15,
    pulls_per_year: int = 365,
    s3_price_per_gb_month: float = 0.023,
) -> dict[str, float]:
    """Rough S3 Standard storage estimate for gzipped jsonl (~5:1)."""
    compressed_factor = 5.0
    corpus_gb = (corpus_items * avg_raw_bytes / compressed_factor) / (1024**3)
    annual_pull_gb = (pulls_per_year * 2500 * avg_raw_bytes / compressed_factor) / (1024**3)
    reject_gb = corpus_gb * reject_fraction
    total_gb = corpus_gb + annual_pull_gb + reject_gb
    monthly_usd = total_gb * s3_price_per_gb_month
    return {
        "corpus_gb": round(corpus_gb, 3),
        "annual_pull_gb": round(annual_pull_gb, 3),
        "reject_gb": round(reject_gb, 3),
        "total_gb": round(total_gb, 3),
        "monthly_usd": round(monthly_usd, 4),
    }
