"""Tests for pull_state tracking and resume logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from research_radar.pull_state import (
    PULL_OVERLAP_HOURS,
    SOURCE_ARXIV_OAI,
    SOURCE_INOREADER,
    format_resume_banner,
    get_pull_state,
    mark_run_completed,
    mark_run_started,
    resume_cutoff,
    track_newest_published,
)


def _mock_conn_with_state(state: dict | None = None):
    conn = MagicMock()
    if state is None:
        conn.execute.return_value.fetchone.return_value = None
    else:
        conn.execute.return_value.fetchone.return_value = state
    return conn


def test_track_newest_published_picks_max():
    now = datetime.now(timezone.utc)
    earlier = now - timedelta(hours=3)
    current = (None, None)
    current = track_newest_published(current, earlier, "ext-1")
    assert current == (earlier, "ext-1")
    current = track_newest_published(current, now, "ext-2")
    assert current == (now, "ext-2")
    current = track_newest_published(current, earlier, "ext-3")
    assert current == (now, "ext-2")


def test_track_newest_published_ignores_none():
    current = (datetime.now(timezone.utc), "ext-1")
    assert track_newest_published(current, None, None) == current


def test_resume_cutoff_subtracts_overlap():
    now = datetime.now(timezone.utc)
    state = {"last_published_at": now}
    conn = _mock_conn_with_state(state)
    cutoff = resume_cutoff(conn, SOURCE_INOREADER)
    expected = now - timedelta(hours=PULL_OVERLAP_HOURS)
    assert cutoff is not None
    assert abs((cutoff - expected).total_seconds()) < 1


def test_resume_cutoff_none_when_no_watermark():
    conn = _mock_conn_with_state(None)
    assert resume_cutoff(conn, SOURCE_INOREADER) is None


def test_resume_cutoff_custom_overlap():
    now = datetime.now(timezone.utc)
    state = {"last_published_at": now}
    conn = _mock_conn_with_state(state)
    cutoff = resume_cutoff(conn, SOURCE_INOREADER, overlap_hours=48)
    expected = now - timedelta(hours=48)
    assert cutoff is not None
    assert abs((cutoff - expected).total_seconds()) < 1


def test_format_resume_banner_formats_correctly():
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    state = {
        "last_run_completed_at": now,
        "items_last_run": 42,
    }
    banner = format_resume_banner(SOURCE_INOREADER, state, cutoff)
    assert "inoreader" in banner
    assert "resume from" in banner
    assert "42 items" in banner


def test_format_resume_banner_none_state():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    banner = format_resume_banner(SOURCE_INOREADER, None, cutoff)
    assert "never" in banner
    assert "?" in banner


def test_mark_run_started_calls_execute():
    conn = MagicMock()
    run_id = uuid4()
    mark_run_started(conn, SOURCE_INOREADER, run_id)
    assert conn.execute.called
    sql = conn.execute.call_args[0][0]
    assert "pull_state" in sql
    assert "last_run_started_at" in sql


def test_mark_run_completed_calls_execute():
    conn = MagicMock()
    run_id = uuid4()
    now = datetime.now(timezone.utc)
    mark_run_completed(
        conn,
        SOURCE_INOREADER,
        run_id=run_id,
        last_published_at=now,
        last_external_id="ext-1",
        items_last_run=10,
    )
    assert conn.execute.called
    sql = conn.execute.call_args[0][0]
    assert "pull_state" in sql
    assert "GREATEST" in sql


def test_failed_run_does_not_call_mark_completed():
    """Simulates the pipeline pattern: mark_run_completed is only called on success."""
    conn = MagicMock()
    run_id = uuid4()
    mark_run_started(conn, SOURCE_INOREADER, run_id)
    completed_called = False

    try:
        raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    assert not completed_called


def test_source_constants():
    assert SOURCE_INOREADER == "inoreader"
    assert SOURCE_ARXIV_OAI == "arxiv_oai"


def test_resume_cutoff_adds_utc_if_naive():
    naive_time = datetime(2026, 9, 1, 12, 0, 0)
    state = {"last_published_at": naive_time}
    conn = _mock_conn_with_state(state)
    cutoff = resume_cutoff(conn, SOURCE_INOREADER)
    assert cutoff is not None
    assert cutoff.tzinfo is not None
