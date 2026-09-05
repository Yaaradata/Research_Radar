"""Unit tests for the topics stage (topic enrichment + claim extraction).

Mocked DB only — no real API calls, no RDS. Pure functions (normalisation,
grounding, value parsing, vocabulary validation) are tested directly; DB
interactions are tested against a MagicMock connection.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from research_radar import topics as tp


def _vocab():
    return {
        "domains_by_name_lower": {
            "natural language processing": {"topic_id": 1, "canonical_name": "Natural Language Processing"},
            "other": {"topic_id": 2, "canonical_name": "Other"},
        },
        "subdomains_by_name_lower": {
            "text classification": {"topic_id": 10, "canonical_name": "Text Classification", "parent_topic_id": 1},
            "ai-generated content detection": {
                "topic_id": 11,
                "canonical_name": "AI-Generated Content Detection",
                "parent_topic_id": 1,
            },
        },
        "applications_by_name_lower": {
            "education": {"topic_id": 20, "canonical_name": "education"},
            "academic-integrity": {"topic_id": 21, "canonical_name": "academic-integrity"},
        },
        "domains_list": ["Natural Language Processing", "Other"],
        "subdomains_by_domain": {
            "Natural Language Processing": ["Text Classification", "AI-Generated Content Detection"],
            "Other": [],
        },
        "applications_list": ["education", "academic-integrity"],
    }


def _fetchone(row):
    m = MagicMock()
    m.fetchone.return_value = row
    return m


# ---------------------------------------------------------------------------
# Closed vocabulary: domain/subdomain/application
# ---------------------------------------------------------------------------


def test_domain_outside_closed_vocabulary_is_dropped_and_counted():
    vocab = _vocab()
    parsed = {"domain": "Underwater Basket Weaving", "subdomains": [], "topics": [], "applications": [], "claims": []}
    out = tp.validate_against_vocabulary(parsed, vocab)
    assert out.domain_id is None
    assert out.unknown_domain is True


def test_known_domain_resolves_to_seeded_topic_id():
    vocab = _vocab()
    parsed = {"domain": "natural language processing", "subdomains": [], "topics": [], "applications": [], "claims": []}
    out = tp.validate_against_vocabulary(parsed, vocab)
    assert out.domain_id == 1
    assert out.unknown_domain is False


def test_applications_empty_when_abstract_never_mentions_one():
    vocab = _vocab()
    parsed = {"domain": "Other", "subdomains": [], "topics": [], "applications": [], "claims": []}
    out = tp.validate_against_vocabulary(parsed, vocab)
    assert out.application_ids == []
    assert out.unknown_applications == 0


def test_unknown_application_is_dropped_and_counted_not_inserted():
    vocab = _vocab()
    parsed = {"domain": "Other", "subdomains": [], "topics": [], "applications": ["astrology"], "claims": []}
    out = tp.validate_against_vocabulary(parsed, vocab)
    assert out.application_ids == []
    assert out.unknown_applications == 1


# ---------------------------------------------------------------------------
# Level-3 topic normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["AI Text Detection", "ai-text detection", "AI text detection"])
def test_level3_normalization_collapses_variants_to_one_form(raw):
    assert tp.normalize_topic_name(raw) == "ai-text-detection"


def test_normalization_strips_punctuation_and_collapses_whitespace():
    assert tp.normalize_topic_name("  Chain-of-Thought,  Prompting!! ") == "chain-of-thought-prompting"


# ---------------------------------------------------------------------------
# Level-3 topic resolution: alias hit vs new topic
# ---------------------------------------------------------------------------


def test_existing_alias_match_attaches_to_existing_topic_and_increments_usage():
    conn = MagicMock()
    conn.execute.side_effect = [
        _fetchone({"topic_id": 42}),  # alias SELECT hit
        MagicMock(),  # UPDATE usage_count
    ]
    topic_id, created = tp.resolve_or_create_level3_topic(conn, "ai-text-detection", parent_topic_id=11)
    assert topic_id == 42
    assert created is False
    update_call = conn.execute.call_args_list[1]
    assert "usage_count = usage_count + 1" in update_call.args[0]
    assert update_call.args[1] == (42,)


def test_new_level3_topic_gets_llm_origin_and_parent_topic_id():
    conn = MagicMock()
    conn.execute.side_effect = [
        _fetchone(None),  # no alias hit
        _fetchone({"topic_id": 99, "inserted": True}),  # INSERT ... RETURNING
    ]
    topic_id, created = tp.resolve_or_create_level3_topic(conn, "reward-hacking", parent_topic_id=11)
    assert topic_id == 99
    assert created is True
    insert_call = conn.execute.call_args_list[1]
    sql, params = insert_call.args
    assert "'topic'" in sql
    assert "'llm'" in sql
    assert params == ("reward-hacking", 11)


# ---------------------------------------------------------------------------
# Claim grounding + value parsing
# ---------------------------------------------------------------------------


def test_claim_with_evidence_not_in_abstract_is_dropped():
    abstract = "We propose a new method for detecting synthetic text."
    claims, ungrounded = tp.build_grounded_claims(
        [
            {
                "metric": "true negative rate",
                "value_text": "99%",
                "unit": "%",
                "task": "detection",
                "dataset": None,
                "evidence": "Our classifier achieves a 99% true negative rate.",  # not literally present
            }
        ],
        abstract,
        "self_reported",
    )
    assert claims == []
    assert ungrounded == 1


def test_claim_with_evidence_literally_in_abstract_is_kept():
    abstract = "Our classifier achieves a 99% true negative rate on held-out data."
    claims, ungrounded = tp.build_grounded_claims(
        [
            {
                "metric": "true negative rate",
                "value_text": "99%",
                "unit": "%",
                "task": "detection",
                "dataset": None,
                "evidence": "Our classifier achieves a 99% true negative rate on held-out data.",
            }
        ],
        abstract,
        "self_reported",
    )
    assert len(claims) == 1
    assert ungrounded == 0
    assert claims[0]["value_num"] == 99.0


def test_abstract_with_no_numbers_yields_zero_claims_not_fabricated():
    claims, ungrounded = tp.build_grounded_claims([], "We discuss qualitative trends in the field.", "self_reported")
    assert claims == []
    assert ungrounded == 0


def test_value_num_null_when_not_a_plain_number_but_value_text_preserved():
    abstract = "Our method runs approximately 3x faster than the baseline."
    claims, _ = tp.build_grounded_claims(
        [
            {
                "metric": "speedup",
                "value_text": "approximately 3x faster",
                "unit": None,
                "task": "inference",
                "dataset": None,
                "evidence": "Our method runs approximately 3x faster than the baseline.",
            }
        ],
        abstract,
        "self_reported",
    )
    assert len(claims) == 1
    assert claims[0]["value_num"] is None
    assert claims[0]["value_text"] == "approximately 3x faster"


def test_parse_value_num_plain_percentage():
    assert tp.parse_value_num("99%") == 99.0


def test_parse_value_num_none_for_prose_value():
    assert tp.parse_value_num("approximately 3x faster") is None


# ---------------------------------------------------------------------------
# Idempotency: re-running skips COMPLETED papers unless --force
# ---------------------------------------------------------------------------


def test_topics_assessment_exists_true_only_for_completed():
    conn = MagicMock()
    conn.execute.return_value = _fetchone({"ok": 1})
    assert tp.topics_assessment_exists(conn, 123) is True
    sql = conn.execute.call_args.args[0]
    assert "status = 'COMPLETED'" in sql


def test_topics_assessment_exists_false_when_no_row():
    conn = MagicMock()
    conn.execute.return_value = _fetchone(None)
    assert tp.topics_assessment_exists(conn, 123) is False


# ---------------------------------------------------------------------------
# The stage never mutates content_items.status — it is an annotation stage.
# ---------------------------------------------------------------------------


def test_stage_never_touches_content_items_status():
    src = inspect.getsource(tp)
    assert "UPDATE research_radar.content_items" not in src
    assert "set_status" not in src


# ---------------------------------------------------------------------------
# corpus-search --tag matches canonical names and aliases
# ---------------------------------------------------------------------------


def test_corpus_search_tag_filter_matches_canonical_name_and_alias():
    from research_radar import corpus_query as cq

    conn = MagicMock()
    conn.execute.return_value = MagicMock(fetchall=MagicMock(return_value=[]))
    cq.search_corpus(conn, tag="ai text detection", top=5)
    sql, params = conn.execute.call_args.args
    assert "t.canonical_name" in sql
    assert "t.aliases" in sql
    assert params["tag"] == "ai text detection"
