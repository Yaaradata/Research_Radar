"""Tests for S3 archive layer and relevance versioning."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from research_radar import archive as arch
from research_radar.archive import RawArchiveWriter, archive_raw, archive_rejected, build_rejected_record


def test_archive_disabled_logs_without_error():
    with patch.object(arch, "S3_ARCHIVE_ENABLED", False):
        result = archive_raw(None, uuid4(), "inoreader", [{"id": 1}])
    assert result is None


def test_archive_upload_failure_does_not_raise():
    run_id = uuid4()
    with patch.object(arch, "S3_ARCHIVE_ENABLED", True), patch.object(arch, "S3_BUCKET", "bucket"), patch.object(
        arch, "_write_jsonl_gzip", return_value=(1, 100)
    ), patch.object(arch, "_upload_file", side_effect=RuntimeError("s3 down")):
        result = archive_rejected(MagicMock(), run_id, "relevance", [{"content_id": 1}])
    assert result is None


def test_manifest_written_only_on_successful_upload():
    conn = MagicMock()
    run_id = uuid4()
    with patch.object(arch, "S3_ARCHIVE_ENABLED", True), patch.object(arch, "S3_BUCKET", "bucket"), patch.object(
        arch, "_write_jsonl_gzip", return_value=(2, 200)
    ), patch.object(arch, "_upload_file"):
        archive_raw(conn, run_id, "inoreader", [{"a": 1}, {"b": 2}])
    assert conn.execute.called


def test_raw_archive_writer_streams_without_loading_all():
    run_id = uuid4()
    with patch.object(arch, "S3_ARCHIVE_ENABLED", False):
        writer = RawArchiveWriter(run_id, "arxiv_oai")
        writer.__enter__()
        writer.write({"arxiv_id": "1234.5678"})
        writer.write({"arxiv_id": "2345.6789"})
        result = writer.finish(None)
    assert result is None
    assert writer.count == 2


def test_build_rejected_record_fields():
    rec = build_rejected_record(
        content_id=1,
        canonical_url="https://arxiv.org/abs/1234.5678",
        title="T",
        abstract="A",
        categories=["cs.AI"],
        relevance_score=2.0,
        primary_topic=None,
        rejection_reason="low",
        relevance_version="relevance-v1",
        arxiv_id="1234.5678",
    )
    assert rec["relevance_version"] == "relevance-v1"
    assert "rejected_at" in rec


def test_stage_relevance_archives_before_reject(monkeypatch):
    from research_radar import pipeline as pl

    calls = {"archive": 0, "reject": 0}
    order = []

    def fake_archive(conn, run_id, stage, records):
        calls["archive"] += 1
        order.append("archive")
        list(records)
        return None

    def fake_set_status(conn, content_id, status):
        if status == "REJECTED":
            calls["reject"] += 1
            order.append("reject")

    monkeypatch.setattr("research_radar.archive.archive_rejected", fake_archive)
    monkeypatch.setattr(pl, "set_status", fake_set_status)
    monkeypatch.setattr(pl, "store_relevance", lambda *a, **k: None)
    monkeypatch.setattr(pl, "set_relevance_version", lambda *a, **k: None)
    monkeypatch.setattr(pl, "event", lambda *a, **k: None)
    monkeypatch.setattr(pl, "bump", lambda *a, **k: None)
    monkeypatch.setattr(pl, "MIN_AI_RELEVANCE", 5.0)
    monkeypatch.setattr(
        pl,
        "score_relevance",
        lambda *a, **k: (2.0, None, [], "low signal"),
    )

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        {
            "id": 9,
            "title": "x",
            "summary": "y",
            "categories_raw": [],
            "source_type": "other",
            "canonical_url": "https://example.com/x",
            "arxiv_id": None,
            "abstract": "y",
        }
    ]
    pl.stage_relevance(conn, uuid4(), limit=1)
    assert calls["archive"] == 1
    assert calls["reject"] == 1
    assert order.index("archive") < order.index("reject")


def test_stage_relevance_skips_rejected_without_reprocess_flag(monkeypatch):
    from research_radar import pipeline as pl

    conn = MagicMock()
    pl.stage_relevance(conn, uuid4(), limit=1, reprocess_version=None)
    sql = conn.execute.call_args[0][0]
    assert "REJECTED" not in sql


def test_stage_relevance_includes_rejected_with_reprocess_flag(monkeypatch):
    from research_radar import pipeline as pl

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    pl.stage_relevance(conn, uuid4(), limit=1, reprocess_version="relevance-v1")
    sql = conn.execute.call_args[0][0]
    assert "REJECTED" in sql


def test_restore_skips_scored_papers():
    from pathlib import Path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "restore_from_s3",
        Path(__file__).resolve().parents[1] / "scripts" / "restore_from_s3.py",
    )
    restore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restore)

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"status": "SCORED"}
    with patch.object(restore, "upsert_item", return_value=(42, False)):
        outcome = restore._restore_record(
            conn, {"canonical_url": "https://arxiv.org/abs/1"}, into_status="INGESTED", dry_run=False
        )
    assert outcome == "skipped_protected"


def test_select_gated_content_ids_excludes_ai_relevance_from_mean():
    from research_radar.semantic_scoring import SCREEN_RANKING_FIELDS, select_gated_content_ids

    assert "ai_relevance" not in SCREEN_RANKING_FIELDS
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        {
            "content_id": 1,
            "ai_relevance": 10.0,
            "technical_significance": 4.0,
            "apparent_novelty": 4.0,
            "evidence_strength": 4.0,
        },
        {
            "content_id": 2,
            "ai_relevance": 8.0,
            "technical_significance": 9.0,
            "apparent_novelty": 9.0,
            "evidence_strength": 9.0,
        },
    ]
    with patch("research_radar.semantic_scoring.GATE_PERCENTILE", 100):
        ids = select_gated_content_ids(conn)
    assert ids[0] == 2


def test_independence_rejects_invalid_paper_kind():
    from research_radar.independence import IndependenceParseError, parse_independence_batch

    text = json.dumps(
        {"papers": [{"paper_id": 1, "status": "independent", "reason": "ok", "paper_kind": "not_a_real_kind"}]}
    )
    with pytest.raises(IndependenceParseError):
        parse_independence_batch(text, {1})
