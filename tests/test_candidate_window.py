"""Tests for published_at window filters on paid scoring candidate loaders."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from research_radar import classify as cl
from research_radar import independence as ind
from research_radar import semantic_scoring as ss
from research_radar import topics as tp
from research_radar.candidate_window import (
    format_published_window,
    merge_window_summary,
    published_at_sql_filters,
)


def test_published_at_sql_filters_none_when_both_omitted():
    sql, params = published_at_sql_filters(None, None)
    assert sql == ""
    assert params == []


def test_published_at_sql_filters_since_only():
    sql, params = published_at_sql_filters("2026-09-04", None)
    assert "ci.published_at >= %s::date" in sql
    assert "INTERVAL" not in sql
    assert params == ["2026-09-04"]


def test_published_at_sql_filters_until_is_inclusive_via_next_day():
    sql, params = published_at_sql_filters(None, "2026-09-04")
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params == ["2026-09-04"]


def test_published_at_sql_filters_both_bounds():
    sql, params = published_at_sql_filters(date(2026, 9, 1), date(2026, 9, 4))
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params == ["2026-09-01", "2026-09-04"]


def test_format_published_window_no_filter_label():
    out = format_published_window(None, None)
    assert out["published_window"] == "all dates (no published_at filter)"
    assert out["published_since"] is None
    assert out["published_until"] is None


def test_merge_window_summary_adds_window_fields():
    summary = merge_window_summary({"requested": 3}, "2026-09-04", "2026-09-05")
    assert summary["requested"] == 3
    assert summary["published_since"] == "2026-09-04"
    assert summary["published_until"] == "2026-09-05"
    assert "until=2026-09-05 (inclusive)" in summary["published_window"]


def _capture_execute(conn):
    calls = []

    def _execute(sql, params=None):
        calls.append((sql, params))
        result = MagicMock()
        result.fetchall.return_value = []
        return result

    conn.execute.side_effect = _execute
    return calls


def test_load_quality_candidates_applies_published_window():
    conn = MagicMock()
    calls = _capture_execute(conn)
    ss.load_quality_candidates(conn, since="2026-09-04", until="2026-09-05")
    sql, params = calls[0]
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert "2026-09-04" in params
    assert "2026-09-05" in params


def test_load_quality_candidates_omits_window_when_unset():
    conn = MagicMock()
    calls = _capture_execute(conn)
    ss.load_quality_candidates(conn)
    sql, _params = calls[0]
    assert "published_at" not in sql


def test_load_topics_candidates_applies_published_window():
    conn = MagicMock()
    calls = _capture_execute(conn)
    tp.load_topics_candidates(conn, since="2026-09-01", until="2026-09-02")
    sql, params = calls[0]
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert "2026-09-01" in params
    assert "2026-09-02" in params


def test_load_independence_candidates_applies_published_window():
    conn = MagicMock()
    calls = _capture_execute(conn)
    ind.load_independence_candidates(conn, since="2026-09-03", until="2026-09-03")
    sql, params = calls[0]
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params[0] == ind.QUALITY_PROMPT_VERSION
    assert "2026-09-03" in params
    assert params[-1] == 10_000


@pytest.mark.parametrize(
    "stage_fn,module_path",
    [
        (ss.stage_screen, "research_radar.semantic_scoring.load_quality_candidates"),
        (cl.stage_classify, "research_radar.semantic_scoring.load_quality_candidates"),
        (ind.stage_independence, "research_radar.independence.load_independence_candidates"),
        (tp.stage_topics, "research_radar.topics.load_topics_candidates"),
    ],
)
def test_empty_window_exits_cleanly(stage_fn, module_path, capsys):
    conn = MagicMock()
    with patch(module_path, return_value=[]):
        if stage_fn is tp.stage_topics:
            vocab = {
                "domains_list": ["Other"],
                "subdomains_by_domain": {"Other": []},
                "applications_list": [],
            }
            with patch("research_radar.topics.load_topic_vocabulary", return_value=vocab):
                stats = stage_fn(
                    conn,
                    "run-empty",
                    dry_run=True,
                    since="2099-01-01",
                    until="2099-01-01",
                )
        elif stage_fn is ss.stage_screen:
            with patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
                "research_radar.semantic_scoring.require_api_key"
            ):
                stats = stage_fn(
                    conn,
                    "run-empty",
                    dry_run=True,
                    since="2099-01-01",
                    until="2099-01-01",
                )
        elif stage_fn is cl.stage_classify:
            with patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
                "research_radar.semantic_scoring.require_api_key"
            ):
                stats = stage_fn(
                    conn,
                    "run-empty",
                    dry_run=True,
                    since="2099-01-01",
                    until="2099-01-01",
                )
        else:
            stats = stage_fn(
                conn,
                "run-empty",
                dry_run=True,
                since="2099-01-01",
                until="2099-01-01",
            )

    assert stats.requested == 0
    out = capsys.readouterr().out
    assert "published_window:" in out
    assert "candidates: 0" in out
    assert "nothing to do" in out


def test_screen_dry_run_prints_window_and_candidate_count(capsys):
    conn = MagicMock()
    papers = [{"content_id": 1, "title": "t", "categories": [], "abstract": "a"}]
    with patch("research_radar.semantic_scoring.load_quality_candidates", return_value=papers), patch(
        "research_radar.semantic_scoring.screen_assessment_exists", return_value=False
    ), patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
        "research_radar.semantic_scoring.require_api_key"
    ):
        ss.stage_screen(
            conn,
            "run-dry",
            dry_run=True,
            since="2026-09-04",
            until="2026-09-05",
        )
    out = capsys.readouterr().out
    assert "published_window:" in out
    assert "candidates: 1" in out
    assert "since=2026-09-04" in out
    assert "until=2026-09-05 (inclusive)" in out
