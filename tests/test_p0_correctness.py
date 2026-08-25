"""P0 correctness tests for Research Radar pipeline."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from research_radar.affiliation_external import (
    AffiliationStageStats,
    OPENALEX_COST_DOI_USD,
    OPENALEX_COST_TITLE_SEARCH_USD,
    OpenAlexBudgetExhausted,
    reset_openalex_budget_flag,
    resolve_openalex_by_doi,
)
from research_radar.pipeline import (
    effective_item_date,
    extract_arxiv_id,
    inoreader_item_to_canonical,
    intrinsic_scores,
    normalize_url,
    orgs_from_text,
    parse_unix_microseconds,
    parse_unix_milliseconds,
    parse_unix_seconds,
    reprocess_orgs_local,
    stage_ingest,
    run_stage,
    timestamps_from_inoreader_raw,
)


def test_inoreader_ot_uses_microseconds():
    fixed_cutoff = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    expected_usec = int(fixed_cutoff.timestamp() * 1_000_000)

    with patch("research_radar.pipeline.lookback_cutoff", return_value=fixed_cutoff):
        with patch("research_radar.pipeline.INOREADER_FIXTURE", ""):
            with patch("research_radar.pipeline.INOREADER_ACCESS_TOKEN", "token"):
                with patch("research_radar.pipeline.http_get") as mock_get:
                    mock_get.return_value = MagicMock(
                        ok=True,
                        status_code=200,
                        json=lambda: {"items": []},
                    )
                    from research_radar.pipeline import fetch_inoreader_items

                    fetch_inoreader_items()
                    params = mock_get.call_args.kwargs.get("params") or mock_get.call_args[1].get("params")
                    assert params["ot"] == expected_usec


def test_parse_unix_microseconds():
    dt = parse_unix_microseconds(1_700_000_000_000_000)
    assert dt == datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)


def test_parse_unix_milliseconds():
    dt = parse_unix_milliseconds(1_700_000_000_000)
    assert dt == datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)


def test_parse_unix_seconds_published():
    dt = parse_unix_seconds(1_700_000_000)
    assert dt == datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)


def test_published_at_independent_of_source_seen_at():
    item = {
        "alternate": [{"href": "https://arxiv.org/abs/2608.02345"}],
        "origin": {"title": "feed"},
        "published": 1_700_000_000,
        "timestampUsec": 1_700_000_100_000_000,
        "updated": 1_700_000_200,
        "title": "Test Paper",
    }
    canonical = inoreader_item_to_canonical(item)
    assert canonical["published_at"] == parse_unix_seconds(1_700_000_000)
    assert canonical["source_seen_at"] == parse_unix_microseconds(1_700_000_100_000_000)
    assert canonical["published_at"] != canonical["source_seen_at"]


def test_missing_published_stays_null_with_source_seen():
    item = {
        "alternate": [{"href": "https://arxiv.org/abs/2608.02345"}],
        "origin": {"title": "feed"},
        "timestampUsec": 1_700_000_100_000_000,
        "title": "No published field",
    }
    canonical = inoreader_item_to_canonical(item)
    assert canonical["published_at"] is None
    assert canonical["source_seen_at"] is not None


def test_effective_date_for_lookback_only():
    pub = parse_unix_seconds(1_700_000_000)
    seen = parse_unix_microseconds(1_700_000_100_000_000)
    assert effective_item_date(pub, seen) == pub
    assert effective_item_date(None, seen) == seen


@pytest.mark.parametrize(
    "status",
    ["CANDIDATE", "SCORED", "REJECTED", "ENTITY_RESOLVED", "ENRICHED"],
)
def test_duplicate_ingest_preserves_status(status):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"status": status}
    item = {
        "source_type": "arxiv",
        "source": "inoreader",
        "canonical_url": "https://arxiv.org/abs/2608.02345",
        "title": "Duplicate Paper",
        "summary": "",
        "authors_raw": [],
        "categories_raw": [],
        "inoreader_tags": [],
        "published_at": None,
        "source_seen_at": datetime.now(timezone.utc),
        "updated_at": None,
        "raw_metadata": {},
    }
    with patch("research_radar.pipeline.fetch_inoreader_items", return_value=[item]):
        with patch("research_radar.pipeline.upsert_item", return_value=(42, False)):
            with patch("research_radar.pipeline.set_status") as mock_set:
                with patch("research_radar.pipeline.event") as mock_event:
                    stage_ingest(conn, "run-test")
                    mock_set.assert_not_called()
                    mock_event.assert_called_once()
                    assert mock_event.call_args[0][4] == "ingest_duplicate_preserved"
                    assert mock_event.call_args[0][6]["existing_status"] == status


def test_arxiv_v1_v2_same_canonical_url():
    u1 = normalize_url("https://arxiv.org/pdf/2608.02345v1")
    u2 = normalize_url("https://arxiv.org/pdf/2608.02345v2")
    assert u1 == u2 == "https://arxiv.org/abs/2608.02345"
    assert extract_arxiv_id(u1) == ("2608.02345", None)


def _org(name, aliases=None):
    return {
        "organisation_id": 1,
        "canonical_name": name,
        "aliases": aliases or [],
        "domains": [],
        "priority": 8,
    }


def test_org_meta_matches_meta_ai_not_metadata():
    meta = _org("Meta", ["Meta AI"])
    assert orgs_from_text("Researchers at Meta AI built a model", [meta])
    assert not orgs_from_text("This paper discusses metadata extraction", [meta])


def test_org_mit_matches_csail():
    mit = _org("Massachusetts Institute of Technology", ["MIT", "MIT CSAIL"])
    hits = orgs_from_text("Affiliation: MIT CSAIL", [mit])
    assert hits and hits[0][1] in {"MIT", "MIT CSAIL", "Massachusetts Institute of Technology"}


def test_org_ibm_matches_research():
    ibm = _org("IBM Research", ["IBM"])
    hits = orgs_from_text("IBM Research, Yorktown Heights", [ibm])
    assert hits


def test_openalex_doi_cost_is_zero():
    assert OPENALEX_COST_DOI_USD == 0.0
    assert OPENALEX_COST_TITLE_SEARCH_USD == 0.001


@patch("research_radar.pipeline.http_get")
def test_openalex_doi_zero_estimated_cost(mock_get):
    reset_openalex_budget_flag()
    mock_get.return_value = MagicMock(
        status_code=200,
        ok=True,
        headers={},
        json=lambda: {
            "id": "https://openalex.org/W1",
            "authorships": [{"author": {"display_name": "A"}, "institutions": [{"display_name": "MIT"}]}],
        },
    )
    stats = AffiliationStageStats()
    resolve_openalex_by_doi("10.1/test", stats)
    assert stats.openalex_estimated_cost_usd == 0.0


@patch("research_radar.affiliation_external.time.sleep")
@patch("research_radar.pipeline.http_get")
def test_openalex_budget_stops_further_http(mock_get, _sleep):
    reset_openalex_budget_flag()
    mock_get.return_value = MagicMock(
        status_code=429,
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
@patch("research_radar.pipeline.OPENALEX_TITLE_SEARCH_ENABLED", False)
@patch("research_radar.pipeline.connect")
@patch("research_radar.pipeline.load_orgs")
def test_no_doi_excluded_when_title_search_disabled(mock_orgs, mock_connect):
    from research_radar.pipeline import stage_openalex

    mock_orgs.return_value = [_org("Meta")]
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []

    stage_openalex(conn, "run-test", limit=10)

    sql = conn.execute.call_args[0][0]
    assert "pm.doi IS NOT NULL" in sql


def test_all_stage_calls_stages_in_order():
    call_order = []
    with patch("research_radar.pipeline.connect") as mock_connect:
        conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = conn
        with patch("research_radar.pipeline.start_run", return_value="run-id"):
            with patch("research_radar.pipeline.finish_run"):
                with patch("research_radar.pipeline.print_status_counts"):
                    with patch("research_radar.pipeline.stage_ingest", side_effect=lambda *a, **k: call_order.append("ingest")):
                        with patch("research_radar.pipeline.stage_relevance", side_effect=lambda *a, **k: call_order.append("relevance")):
                            with patch("research_radar.pipeline.stage_enrich", side_effect=lambda *a, **k: call_order.append("enrich")):
                                with patch("research_radar.pipeline.stage_entities", side_effect=lambda *a, **k: call_order.append("entities")):
                                    with patch("research_radar.pipeline.stage_score", side_effect=lambda *a, **k: call_order.append("score")):
                                        run_stage("all")
    assert call_order == [
        "ingest",
        "relevance",
        "enrich",
        "entities",
        "score",
    ]


def test_scoring_provenance_structure():
    s, provenance = intrinsic_scores(
        7.0,
        "Benchmark evaluation of agent systems",
        "We present experiments on deployment and production workflows.",
        [8],
        [],
        include_person_signal=False,
    )
    assert provenance["method"] == "deterministic_heuristic"
    assert provenance["version"] == "deterministic-v0.2"
    assert provenance["person_weight_enabled"] is False
    assert "weights" in provenance
    assert provenance["dimensions"]["technical_significance"]["matched_rules"]
    assert provenance["dimensions"]["novelty"]["novelty_method"] == "fixed_proxy"
    assert provenance["intrinsic_candidate_score"] == s["intrinsic_candidate_score"]
    assert provenance["industry_relevance"]["status"] == "not_yet_semantically_scored"


def test_industry_relevance_not_falsely_scored():
    _, provenance = intrinsic_scores(6.0, "AI paper", "abstract", [], [], include_person_signal=False)
    assert provenance["industry_relevance"]["status"] == "not_yet_semantically_scored"
    assert "score" not in provenance["industry_relevance"]


def test_timestamps_from_raw_metadata():
    raw = {
        "published": 1_700_000_000,
        "timestampUsec": 1_700_000_100_000_000,
        "updated": 1_700_000_200,
    }
    pub, seen, upd = timestamps_from_inoreader_raw(raw)
    assert pub == parse_unix_seconds(1_700_000_000)
    assert seen == parse_unix_microseconds(1_700_000_100_000_000)
    assert upd == parse_unix_seconds(1_700_000_200)
    assert pub != seen


def test_reprocess_orgs_local_preserves_external_evidence():
    conn = MagicMock()
    orgs = [_org("Meta", ["Meta AI"])]
    p = {
        "emails": [],
        "affiliation_text": ["Meta AI Research Lab"],
        "evidence_url": "https://arxiv.org/abs/123",
    }
    with patch("research_radar.pipeline.delete_local_org_evidence", return_value=2) as mock_del:
        with patch("research_radar.pipeline.resolve_orgs_local", return_value=1) as mock_resolve:
            with patch("research_radar.pipeline.set_openalex_status") as mock_oa:
                added, deleted = reprocess_orgs_local(conn, 99, p, orgs)
    assert added == 1
    assert deleted == 2
    mock_del.assert_called_once_with(conn, 99)
    mock_resolve.assert_called_once()
    mock_oa.assert_called_once_with(conn, 99, "NOT_NEEDED", error=None)


def test_paid_stages_require_explicit_authorisation():
    import pytest as _pytest

    from research_radar.pipeline import PaidStageNotAuthorised, run_stage

    for stage in ("affiliation-gpt", "semantic-score"):
        with _pytest.raises(PaidStageNotAuthorised):
            run_stage(stage)


def test_dry_run_does_not_require_paid_authorisation():
    """--dry-run must pass the guard without ever opening a DB connection."""
    from unittest.mock import patch

    import pytest as _pytest

    from research_radar.pipeline import run_stage

    with patch(
        "research_radar.pipeline.connect",
        side_effect=RuntimeError("sentinel: guard passed"),
    ):
        for stage in ("affiliation-gpt", "semantic-score"):
            with _pytest.raises(RuntimeError, match="sentinel"):
                run_stage(stage, dry_run=True)


def test_all_stage_does_not_call_paid_affiliation():
    """`all` is the cron path and must never invoke a paid stage."""
    import inspect

    from research_radar import pipeline

    src = inspect.getsource(pipeline.run_stage)
    all_block = src.split('elif stage == "all":')[1].split("else:")[0]
    assert "stage_affiliation_gpt" not in all_block
    assert "stage_semantic_score" not in all_block
