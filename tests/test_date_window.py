"""Tests for published_at date-window filtering on paid scoring stages."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from research_radar import classify as cl
from research_radar import independence as ind
from research_radar import semantic_scoring as ss
from research_radar import topics as tp
from research_radar.candidate_window import (
    format_run_line,
    format_window_label,
    merge_window_summary,
    published_at_sql_filters,
)


def test_published_at_sql_filters_none_when_both_omitted():
    sql, params = published_at_sql_filters(None, None)
    assert sql == ""
    assert params == []


def test_published_at_sql_filters_from_only():
    sql, params = published_at_sql_filters("2026-09-04", None)
    assert "ci.published_at >= %s::date" in sql
    assert "INTERVAL" not in sql
    assert params == ["2026-09-04"]


def test_published_at_sql_filters_until_is_inclusive_via_next_day():
    sql, params = published_at_sql_filters(None, "2026-09-05")
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params == ["2026-09-05"]


def test_published_at_sql_filters_both_bounds():
    sql, params = published_at_sql_filters(date(2026, 9, 2), date(2026, 9, 5))
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params == ["2026-09-02", "2026-09-05"]


def test_format_window_label_all_when_unset():
    assert format_window_label(None, None) == "all"


def test_format_window_label_open_ended_bounds():
    assert format_window_label("2026-09-02", None) == "2026-09-02.."
    assert format_window_label(None, "2026-09-05") == "..2026-09-05"
    assert format_window_label("2026-09-02", "2026-09-05") == "2026-09-02..2026-09-05"


def test_format_run_line_matches_required_shape():
    line = format_run_line(
        "classify",
        date(2026, 9, 2),
        date(2026, 9, 5),
        candidates=412,
        batches=28,
        est_cost=1.48,
    )
    assert line == "classify: window=2026-09-02..2026-09-05 candidates=412 batches=28 est_cost=$1.48"


def test_merge_window_summary_includes_window():
    summary = merge_window_summary({"requested": 3}, "2026-09-04", "2026-09-05")
    assert summary["window"] == "2026-09-04..2026-09-05"
    assert summary["requested"] == 3


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
    ss.load_quality_candidates(conn, date_from="2026-09-04", date_until="2026-09-05")
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
    tp.load_topics_candidates(conn, date_from="2026-09-01", date_until="2026-09-02")
    sql, params = calls[0]
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert "2026-09-01" in params
    assert "2026-09-02" in params


def test_load_independence_candidates_applies_published_window():
    conn = MagicMock()
    calls = _capture_execute(conn)
    ind.load_independence_candidates(conn, date_from="2026-09-03", date_until="2026-09-03")
    sql, params = calls[0]
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql
    assert params[0] == ind.QUALITY_PROMPT_VERSION
    assert "2026-09-03" in params


def test_until_includes_late_same_day_timestamp_in_sql():
    """Upper bound is < next day, so 23:00 on the until date is included."""
    sql, params = published_at_sql_filters(None, "2026-09-05")
    assert params == ["2026-09-05"]
    assert "+ INTERVAL '1 day'" in sql
    late = datetime(2026, 9, 5, 23, 0, tzinfo=timezone.utc)
    early_next = datetime(2026, 9, 6, 0, 0, tzinfo=timezone.utc)
    assert late.date() == date(2026, 9, 5)
    assert late < datetime(2026, 9, 6, tzinfo=timezone.utc)
    assert early_next >= datetime(2026, 9, 6, tzinfo=timezone.utc)


def test_select_gated_content_ids_ranks_within_window_not_corpus_intersection():
    """Window filter must be applied before percentile ranking."""
    rows = [
        {"content_id": 1, "ai_relevance": 8.0, "technical_significance": 10.0, "apparent_novelty": 10.0, "evidence_strength": 10.0},
        {"content_id": 2, "ai_relevance": 8.0, "technical_significance": 9.0, "apparent_novelty": 9.0, "evidence_strength": 9.0},
        {"content_id": 3, "ai_relevance": 8.0, "technical_significance": 5.0, "apparent_novelty": 5.0, "evidence_strength": 5.0},
        {"content_id": 4, "ai_relevance": 8.0, "technical_significance": 4.0, "apparent_novelty": 4.0, "evidence_strength": 4.0},
    ]
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows[:2]

    with patch.object(ss, "GATE_PERCENTILE", 50.0):
        in_window = ss.select_gated_content_ids(
            conn, gate_percentile=50.0, date_from="2026-09-02", date_until="2026-09-05"
        )
        conn.execute.return_value.fetchall.return_value = rows
        corpus_wide = ss.select_gated_content_ids(conn, gate_percentile=50.0)

    assert in_window == [1]
    assert corpus_wide == [1, 2]
    assert in_window != corpus_wide


def test_select_gated_content_ids_sql_joins_content_items_for_window():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    ss.select_gated_content_ids(conn, date_from="2026-09-02", date_until="2026-09-05")
    sql = conn.execute.call_args[0][0]
    assert "JOIN research_radar.content_items ci" in sql
    assert "ci.published_at >= %s::date" in sql
    assert "ci.published_at < (%s::date + INTERVAL '1 day')" in sql


@pytest.mark.parametrize(
    "stage_fn,module_path,stage_name",
    [
        (ss.stage_screen, "research_radar.semantic_scoring.load_quality_candidates", "screen"),
        (cl.stage_classify, "research_radar.semantic_scoring.load_quality_candidates", "classify"),
        (ind.stage_independence, "research_radar.independence.load_independence_candidates", "independence"),
        (tp.stage_topics, "research_radar.topics.load_topics_candidates", "topics"),
    ],
)
def test_empty_window_exits_cleanly(stage_fn, module_path, stage_name, capsys):
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
                    date_from="2099-01-01",
                    date_until="2099-01-01",
                )
        elif stage_fn in (ss.stage_screen, cl.stage_classify):
            with patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
                "research_radar.semantic_scoring.require_api_key"
            ):
                stats = stage_fn(
                    conn,
                    "run-empty",
                    dry_run=True,
                    date_from="2099-01-01",
                    date_until="2099-01-01",
                )
        else:
            stats = stage_fn(
                conn,
                "run-empty",
                dry_run=True,
                date_from="2099-01-01",
                date_until="2099-01-01",
            )

    assert stats.requested == 0
    out = capsys.readouterr().out
    assert f"{stage_name}: window=2099-01-01..2099-01-01 candidates=0" in out


def test_classify_dry_run_prints_resolved_window(capsys):
    conn = MagicMock()
    papers = [{"content_id": 1, "title": "t", "categories": [], "abstract": "a"}]
    with patch("research_radar.classify.load_quality_candidates", return_value=papers), patch(
        "research_radar.classify.classification_exists", return_value=False
    ), patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
        "research_radar.semantic_scoring.require_api_key"
    ):
        cl.stage_classify(
            conn,
            "run-dry",
            dry_run=True,
            date_from="2026-09-02",
            date_until="2026-09-05",
        )
    out = capsys.readouterr().out
    assert out.startswith("classify: window=2026-09-02..2026-09-05 candidates=1")


def test_classify_dry_run_window_all_when_unset(capsys):
    conn = MagicMock()
    papers = [{"content_id": 1, "title": "t", "categories": [], "abstract": "a"}]
    with patch("research_radar.classify.load_quality_candidates", return_value=papers), patch(
        "research_radar.classify.classification_exists", return_value=False
    ), patch("research_radar.semantic_scoring.require_scoring_enabled"), patch(
        "research_radar.semantic_scoring.require_api_key"
    ):
        cl.stage_classify(conn, "run-dry", dry_run=True)
    out = capsys.readouterr().out
    assert out.startswith("classify: window=all candidates=1")
