"""Unit tests for the arXiv OAI-PMH backfill harvester (fixture XML; no network, no DB)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from research_radar import arxiv_backfill as ab
from research_radar.pipeline import normalize_url

OAI_ENVELOPE_OPEN = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">\n'
    "  <responseDate>2026-08-31T00:00:00Z</responseDate>\n"
    '  <ListRecords>\n'
)
OAI_ENVELOPE_CLOSE = "  </ListRecords>\n</OAI-PMH>\n"

# Verified real example from the brief: created 2012, revised (and datestamped)
# inside a January 2026 window.
RECORD_1211_3613 = """
<record>
  <header>
    <identifier>oai:arXiv.org:1211.3613</identifier>
    <datestamp>2026-01-05</datestamp>
    <setSpec>cs:cs:NA</setSpec>
  </header>
  <metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>1211.3613</id>
      <created>2012-11-15</created>
      <updated>2026-01-05</updated>
      <authors><author><keyname>Zlotnik</keyname><forenames>Alexander</forenames></author></authors>
      <title>A Numerical Method Example</title>
      <categories>cs.NA math.NA</categories>
      <abstract>  An example   abstract   with odd   whitespace. </abstract>
    </arXiv>
  </metadata>
</record>
"""

RECORD_MATH_ONLY = """
<record>
  <header>
    <identifier>oai:arXiv.org:1300.0001</identifier>
    <datestamp>2026-01-03</datestamp>
    <setSpec>math:math:NA</setSpec>
  </header>
  <metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>1300.0001</id>
      <created>2013-01-01</created>
      <updated>2013-01-01</updated>
      <authors><author><keyname>Doe</keyname><forenames>Jane</forenames></author></authors>
      <title>A Pure Math Paper</title>
      <categories>math.NA</categories>
      <abstract>Pure math, no cs relevance.</abstract>
    </arXiv>
  </metadata>
</record>
"""

RECORD_DELETED = """
<record>
  <header status="deleted">
    <identifier>oai:arXiv.org:1400.0002</identifier>
    <datestamp>2026-01-04</datestamp>
    <setSpec>cs:cs:AI</setSpec>
  </header>
</record>
"""

RECORD_CS_2026 = """
<record>
  <header>
    <identifier>oai:arXiv.org:2601.00099</identifier>
    <datestamp>2026-01-06</datestamp>
    <setSpec>cs:cs:AI</setSpec>
  </header>
  <metadata>
    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>2601.00099</id>
      <created>2026-01-06</created>
      <updated>2026-01-06</updated>
      <authors><author><keyname>Lovelace</keyname><forenames>Ada</forenames></author></authors>
      <title>Cross-listed Agents Paper</title>
      <categories>cs.AI stat.ML</categories>
      <abstract>An agents paper cross-listed in cs and stat.</abstract>
      <doi>10.1000/xyz</doi>
    </arXiv>
  </metadata>
</record>
"""


def _oai_response(records_xml, resumption_token=None):
    body = OAI_ENVELOPE_OPEN + records_xml
    if resumption_token is not None:
        body += f"    <resumptionToken>{resumption_token}</resumptionToken>\n"
    body += OAI_ENVELOPE_CLOSE
    return body


def _no_records_match_response():
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">\n'
        "  <responseDate>2026-08-31T00:00:00Z</responseDate>\n"
        '  <error code="noRecordsMatch">no records match</error>\n'
        "</OAI-PMH>\n"
    )


def _mock_response(status_code=200, text="", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    if status_code == 200:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# published_at from <created>, never <datestamp>
# ---------------------------------------------------------------------------


def test_published_at_from_created_not_datestamp():
    xml = _oai_response(RECORD_1211_3613)
    with patch.object(ab, "_fetch_page", return_value=xml):
        records = list(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))
    assert len(records) == 1
    rec = records[0]
    assert rec["created"] == "2012-11-15"
    assert rec["datestamp"] == "2026-01-05"

    item = ab.record_to_item(rec)
    assert item["published_at"] == datetime(2012, 11, 15, tzinfo=timezone.utc)
    assert item["published_at"] != datetime(2026, 1, 5, tzinfo=timezone.utc)
    assert item["title"] == "A Numerical Method Example"
    assert item["summary"] == "An example abstract with odd whitespace."


def test_canonical_url_matches_inoreader_form():
    xml = _oai_response(RECORD_1211_3613)
    with patch.object(ab, "_fetch_page", return_value=xml):
        rec = next(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))
    item = ab.record_to_item(rec)
    assert item["canonical_url"] == normalize_url("https://arxiv.org/abs/1211.3613")
    assert item["canonical_url"] == "https://arxiv.org/abs/1211.3613"


# ---------------------------------------------------------------------------
# Deleted records
# ---------------------------------------------------------------------------


def test_deleted_record_is_skipped():
    xml = _oai_response(RECORD_1211_3613 + RECORD_DELETED)
    with patch.object(ab, "_fetch_page", return_value=xml):
        records = list(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))
    assert len(records) == 2
    kept = [r for r in records if not r["deleted"]]
    deleted = [r for r in records if r["deleted"]]
    assert len(kept) == 1
    assert len(deleted) == 1
    assert deleted[0]["identifier"] == "oai:arXiv.org:1400.0002"


def test_no_records_match_yields_nothing():
    with patch.object(ab, "_fetch_page", return_value=_no_records_match_response()):
        records = list(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))
    assert records == []


def test_other_oai_error_raises():
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">\n'
        '  <error code="badArgument">bad from date</error>\n'
        "</OAI-PMH>\n"
    )
    with patch.object(ab, "_fetch_page", return_value=body):
        with pytest.raises(ab.ArxivOAIError):
            list(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))


# ---------------------------------------------------------------------------
# Category filter
# ---------------------------------------------------------------------------


def test_category_filter_bare_cs_matches_subcategory():
    assert ab.category_matches(["cs.NA", "math.NA"]) is True


def test_category_filter_rejects_pure_math():
    assert ab.category_matches(["math.NA"]) is False


def test_category_filter_exact_stat_ml():
    assert ab.category_matches(["stat.ML"]) is True
    assert ab.category_matches(["stat.AP"]) is False


# ---------------------------------------------------------------------------
# Cross-listed dedupe across cs + stat
# ---------------------------------------------------------------------------


def test_cross_listed_paper_deduped_across_sets():
    """Same arXiv id appears in both the cs and stat OAI responses; must
    collapse to one new row via upsert_item's canonical_url uniqueness."""
    seen_urls = {}

    def fake_upsert_item(conn, item):
        url = item["canonical_url"]
        if url in seen_urls:
            return seen_urls[url], False
        content_id = len(seen_urls) + 1
        seen_urls[url] = content_id
        return content_id, True

    cs_xml = _oai_response(RECORD_CS_2026)
    stat_xml = _oai_response(RECORD_CS_2026)  # arXiv serves the same record under both sets

    def fake_fetch_page(params):
        return cs_xml if params.get("set") == "cs" else stat_xml

    conn = MagicMock()
    with patch.object(ab, "_fetch_page", side_effect=fake_fetch_page):
        with patch("research_radar.arxiv_backfill.upsert_item", side_effect=fake_upsert_item):
            with patch.object(ab, "upsert_backfill_paper_metadata"):
                cs_stats = ab._run_one_window(conn, "cs", date(2026, 1, 1), date(2026, 1, 7))
                stat_stats = ab._run_one_window(conn, "stat", date(2026, 1, 1), date(2026, 1, 7))

    assert cs_stats.records_kept == 1
    assert cs_stats.records_new == 1
    assert stat_stats.records_kept == 1
    assert stat_stats.records_dupe == 1
    assert stat_stats.records_new == 0
    assert len(seen_urls) == 1


# ---------------------------------------------------------------------------
# Idempotency: never reset an already-scored paper
# ---------------------------------------------------------------------------


def test_scored_paper_not_reset_by_backfill_upsert():
    """The module must go through upsert_item and never call anything that
    touches content_items.status."""
    import inspect

    src = inspect.getsource(ab)
    assert "set_status" not in src

    xml = _oai_response(RECORD_1211_3613)
    conn = MagicMock()
    with patch.object(ab, "_fetch_page", return_value=xml):
        with patch("research_radar.arxiv_backfill.upsert_item", return_value=(42, False)) as mock_upsert:
            with patch.object(ab, "upsert_backfill_paper_metadata") as mock_meta:
                stats = ab._run_one_window(conn, "cs", date(2026, 1, 1), date(2026, 1, 7))

    assert stats.records_new == 0
    assert stats.records_dupe == 1
    mock_upsert.assert_called_once()
    mock_meta.assert_called_once()
    for c in conn.execute.call_args_list + conn.method_calls:
        assert "SET status" not in str(c)


# ---------------------------------------------------------------------------
# 503 flow control
# ---------------------------------------------------------------------------


def test_503_with_retry_after_sleeps_and_retries_not_an_error():
    responses = [
        _mock_response(503, headers={"Retry-After": "3"}),
        _mock_response(200, text=_oai_response(RECORD_1211_3613)),
    ]
    with patch.object(ab.SESSION, "get", side_effect=responses) as mock_get:
        with patch.object(ab.time, "sleep") as mock_sleep:
            text = ab._fetch_page({"verb": "ListRecords"})
    assert mock_get.call_count == 2
    assert any(c == call(3.0) for c in mock_sleep.call_args_list)
    assert "1211.3613" in text


def test_503_default_retry_after_when_header_missing():
    responses = [
        _mock_response(503, headers={}),
        _mock_response(200, text=_oai_response(RECORD_1211_3613)),
    ]
    with patch.object(ab.SESSION, "get", side_effect=responses):
        with patch.object(ab.time, "sleep") as mock_sleep:
            ab._fetch_page({"verb": "ListRecords"})
    assert any(c == call(ab.OAI_DEFAULT_RETRY_AFTER) for c in mock_sleep.call_args_list)


# ---------------------------------------------------------------------------
# Resumption tokens
# ---------------------------------------------------------------------------


def test_resumption_token_followed_until_absent():
    page1 = _oai_response(RECORD_1211_3613, resumption_token="TOK123")
    page2 = _oai_response(RECORD_MATH_ONLY, resumption_token="")

    with patch.object(ab, "_fetch_page", side_effect=[page1, page2]) as mock_fetch:
        with patch.object(ab, "_throttle"):
            records = list(ab.fetch_window_records("cs", date(2026, 1, 1), date(2026, 1, 7)))

    assert len(records) == 2
    assert mock_fetch.call_count == 2
    first_params, second_params = (c.args[0] for c in mock_fetch.call_args_list)
    assert first_params["from"] == "2026-01-01"
    assert second_params == {"verb": "ListRecords", "resumptionToken": "TOK123"}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_complete_checkpoint_skipped_without_force():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"status": "COMPLETE"}
    with patch.object(ab, "_run_one_window") as mock_run:
        totals = ab.run_backfill(conn, date(2026, 1, 1), date(2026, 1, 1), force=False)
    mock_run.assert_not_called()
    assert totals.windows_skipped == len(ab.OAI_SETS)
    assert totals.windows_run == 0


def test_complete_checkpoint_reharvested_with_force():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"status": "COMPLETE"}
    with patch.object(ab, "_run_one_window", return_value=ab.WindowStats(records_seen=1, records_kept=1, records_new=1)) as mock_run:
        totals = ab.run_backfill(conn, date(2026, 1, 1), date(2026, 1, 1), force=True)
    assert mock_run.call_count == len(ab.OAI_SETS)
    assert totals.windows_run == len(ab.OAI_SETS)
    assert totals.windows_skipped == 0


# ---------------------------------------------------------------------------
# Not in PAID_STAGES / not in `all`
# ---------------------------------------------------------------------------


def test_arxiv_backfill_is_free_and_not_in_all():
    import inspect

    from research_radar import pipeline

    assert "arxiv-backfill" not in pipeline.PAID_STAGES

    src = inspect.getsource(pipeline.run_stage)
    all_block = src.split('elif stage == "all":')[1].split("else:")[0]
    assert "stage_arxiv_backfill" not in all_block
