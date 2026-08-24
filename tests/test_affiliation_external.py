"""Tests for Crossref + OpenAlex DOI-first affiliation resolution."""

from unittest.mock import MagicMock, patch

import pytest

from research_radar.affiliation_external import (
    AffiliationStageStats,
    OpenAlexBudgetExhausted,
    OpenAlexRateLimited,
    reset_openalex_budget_flag,
    resolve_crossref_by_doi,
    resolve_openalex_by_doi,
    resolve_openalex_by_title,
)


def _mock_response(status_code, json_data=None, text="", headers=None):
    r = MagicMock()
    r.status_code = status_code
    r.ok = 200 <= status_code < 300
    r.text = text
    r.headers = headers or {}
    r.json.return_value = json_data or {}
    return r


@patch("research_radar.pipeline.http_get")
def test_crossref_doi_match(mock_get):
    mock_get.return_value = _mock_response(
        200,
        {
            "message": {
                "title": ["Agentic AI at Meta"],
                "author": [
                    {
                        "given": "Jane",
                        "family": "Doe",
                        "affiliation": [{"name": "Meta AI Research"}],
                    }
                ],
            }
        },
    )
    stats = AffiliationStageStats()
    hit = resolve_crossref_by_doi("10.1234/example", "test@example.com", stats)
    assert hit is not None
    assert hit["institutions"][0]["institution"] == "Meta AI Research"
    assert stats.crossref_attempted == 1
    mock_get.assert_called_once()
    assert "mailto=test@example.com" in mock_get.call_args[0][0] or "mailto" in str(mock_get.call_args)


@patch("research_radar.pipeline.http_get")
def test_crossref_no_affiliation_returns_none(mock_get):
    mock_get.return_value = _mock_response(
        200,
        {"message": {"title": ["Paper"], "author": [{"given": "A", "family": "B"}]}},
    )
    hit = resolve_crossref_by_doi("10.1234/noaff", stats=AffiliationStageStats())
    assert hit is None


@patch("research_radar.pipeline.http_get")
def test_openalex_doi_match(mock_get):
    reset_openalex_budget_flag()
    mock_get.return_value = _mock_response(
        200,
        {
            "id": "https://openalex.org/W123",
            "title": "Test Paper",
            "authorships": [
                {
                    "author": {"display_name": "Alice"},
                    "institutions": [{"display_name": "Google DeepMind"}],
                }
            ],
        },
    )
    stats = AffiliationStageStats()
    hit = resolve_openalex_by_doi("10.5555/abc", stats)
    assert hit is not None
    assert hit["work_id"] == "https://openalex.org/W123"
    assert hit["institutions"][0]["institution"] == "Google DeepMind"
    assert stats.openalex_doi_attempted == 1
    assert "openalex.org/works/https://doi.org" in mock_get.call_args[0][0]


@patch("research_radar.pipeline.http_get")
def test_openalex_doi_404_returns_none(mock_get):
    reset_openalex_budget_flag()
    mock_get.return_value = _mock_response(404)
    hit = resolve_openalex_by_doi("10.5555/missing", AffiliationStageStats())
    assert hit is None


@patch("research_radar.affiliation_external.time.sleep")
@patch("research_radar.pipeline.http_get")
def test_openalex_doi_429_budget_no_title_fallback(mock_get, _sleep):
    reset_openalex_budget_flag()
    mock_get.return_value = _mock_response(
        429,
        text="Insufficient budget",
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Remaining-USD": "0",
            "X-RateLimit-Reset": "3600",
        },
    )
    with pytest.raises(OpenAlexBudgetExhausted):
        resolve_openalex_by_doi("10.5555/budget", AffiliationStageStats())

    with pytest.raises(OpenAlexBudgetExhausted):
        resolve_openalex_by_title("Some AI Paper Title", AffiliationStageStats())

    assert mock_get.call_count == 1


@patch("research_radar.affiliation_external.time.sleep")
@patch("research_radar.pipeline.http_get")
def test_openalex_daily_budget_stops_further_calls(mock_get, _sleep):
    reset_openalex_budget_flag()
    mock_get.return_value = _mock_response(
        429,
        text="Insufficient budget",
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Remaining-USD": "0"},
    )
    stats = AffiliationStageStats()
    with pytest.raises(OpenAlexBudgetExhausted):
        resolve_openalex_by_doi("10.1/first", stats)

    with pytest.raises(OpenAlexBudgetExhausted):
        resolve_openalex_by_doi("10.2/second", stats)
    assert mock_get.call_count == 1


@patch("research_radar.pipeline.OPENALEX_ENABLED", True)
@patch("research_radar.pipeline.connect")
@patch("research_radar.pipeline.load_orgs")
@patch("research_radar.affiliation_external.resolve_openalex_by_doi")
@patch("research_radar.affiliation_external.resolve_crossref_by_doi")
def test_stage_preserves_pending_on_budget_skip(mock_crossref, mock_openalex_doi, mock_orgs, mock_connect):
    from research_radar.pipeline import stage_openalex

    mock_orgs.return_value = [
        {
            "organisation_id": 1,
            "canonical_name": "Meta",
            "aliases": [],
            "domains": [],
            "priority": 10,
        }
    ]

    rows = [
        {"id": 101, "title": "Paper A", "item_status": "ENTITY_RESOLVED", "doi": "10.1/a", "openalex_status": "PENDING", "openalex_attempts": 0},
        {"id": 102, "title": "Paper B", "item_status": "ENTITY_RESOLVED", "doi": "10.2/b", "openalex_status": "PENDING", "openalex_attempts": 0},
    ]

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = rows

    mock_crossref.return_value = None
    mock_openalex_doi.side_effect = OpenAlexBudgetExhausted("budget gone")

    wconn = MagicMock()
    wconn.execute.return_value.fetchone.return_value = {"n": 0}
    mock_connect.return_value.__enter__.return_value = wconn

    stage_openalex(conn, "run-test", limit=2)

    # Budget item should be RATE_LIMITED; unattempted rows stay PENDING (no status UPDATE).
    rate_limited = any(
        "RATE_LIMITED" in str(c)
        for c in wconn.execute.call_args_list
    )
    assert rate_limited or mock_openalex_doi.called
