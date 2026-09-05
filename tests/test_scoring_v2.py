"""Unit tests for scoring v2 (quality scorer, independence classifier, final score v2).

Mocked OpenRouter only — no real API calls, no RDS.
"""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest

from research_radar import final_score as fs
from research_radar import independence as ind
from research_radar import semantic_scoring as ss
from research_radar.llm_batch import random_batches


def _paper(cid, title="Paper", abstract="We study X on a public benchmark.", categories=None, affiliation_text=None):
    return {
        "content_id": cid,
        "title": title,
        "abstract": abstract,
        "categories": categories or ["cs.LG"],
        "affiliation_text": affiliation_text or [],
    }


def _quality_item(pid, **overrides):
    base = {
        "paper_id": pid,
        "ai_relevance": 8.0,
        "technical_significance": 7.0,
        "apparent_novelty": 6.0,
        "practical_applicability": 6.5,
        "professional_value": 7.0,
        "learning_value": 6.0,
        "evidence_strength": 7.0,
        "newsletter_fit": 5.0,
        "confidence": 8.0,
        "so_what": "Useful for a working professional.",
        "reason_not_higher": "Evaluation is limited to one benchmark.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The quality scorer must never see affiliation/author/organisation data.
# ---------------------------------------------------------------------------


def test_quality_prompt_excludes_affiliation_even_when_present_on_paper_dict():
    paper = _paper(
        1,
        title="Benchmarking retrieval systems",
        abstract="We evaluate retrieval quality on public benchmarks.",
        affiliation_text=["Stanford University, Department of Computer Science", "author@openai.com"],
    )
    block = ss.build_quality_paper_block(paper)
    assert "Stanford" not in block
    assert "openai.com" not in block
    assert "AFFILIATIONS" not in block
    ss.assert_quality_prompt_is_paper_only(block)


def test_prompt_validator_rejects_affiliations_section():
    bad_prompt = "PAPER 1\nTITLE: T\nCATEGORIES: cs.AI\nABSTRACT: A\nAFFILIATIONS: MIT\n"
    with pytest.raises(ss.QualityParseError):
        ss.assert_quality_prompt_is_paper_only(bad_prompt)


# ---------------------------------------------------------------------------
# Eligibility: ENTITY_RESOLVED, SCORED and CANDIDATE are all selectable.
#
# The brief originally said ENTITY_RESOLVED *instead of* SCORED/CANDIDATE.
# That was corrected: papers ingested before the deterministic `score` stage
# was removed from `all` already sit at SCORED/CANDIDATE and never transition
# back, so excluding them would make both paid stages select nothing for the
# entire pre-existing corpus. Widening the filter — not resetting status — is
# the fix. This test pins the widened filter so it cannot regress.
# ---------------------------------------------------------------------------


def test_quality_and_independence_select_entity_resolved_scored_and_candidate():
    """load_quality_candidates is the pass-1 (screen) eligibility pool — still
    widened to all three statuses. Independence now runs only on pass-2
    ("full") papers (tiering brief §1), so it is checked separately below."""
    q_src = inspect.getsource(ss.load_quality_candidates)
    assert "IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE')" in q_src


def test_independence_selects_only_pass2_full_tier_papers():
    i_src = inspect.getsource(ind.load_independence_candidates)
    assert "scoring_tier = 'full'" in i_src


def test_final_score_skips_screen_tier_papers():
    """Two models, two scales — screen-tier assessments must never reach
    load_scored_papers, so stage_final_score never writes a final_score row
    for a screen-only paper (tiering brief §2)."""
    src = inspect.getsource(fs.load_scored_papers) + fs._SCORED_PAPERS_SELECT
    assert "scoring_tier = 'full'" in src
    assert "scoring_tier IS NULL" in src


def test_report_filters_to_full_tier():
    """report reads through load_top, which must exclude screen-tier rows
    even though content_final_scores should never contain one in the first
    place — defense in depth (tiering brief §2)."""
    src = inspect.getsource(fs.load_top)
    assert "scoring_tier = 'full'" in src


# ---------------------------------------------------------------------------
# 0.5-increment scores; non-conforming values round and warn.
# ---------------------------------------------------------------------------


def _quality_response(*items) -> str:
    """The object-with-papers shape the strict json_schema response_format enforces."""
    return json.dumps({"papers": list(items)})


def test_scores_parse_as_multiples_of_half():
    text = _quality_response(_quality_item(1))
    parsed, warnings = ss.parse_quality_batch(text, {1})
    assert parsed[1]["technical_significance"] == 7.0
    assert warnings == 0


def test_non_half_increment_score_rounds_and_warns():
    text = _quality_response(_quality_item(1, technical_significance=7.3))
    parsed, warnings = ss.parse_quality_batch(text, {1})
    assert parsed[1]["technical_significance"] == 7.5
    assert warnings == 1


def test_bare_array_response_is_rejected_now_that_schema_is_object_rooted():
    """Guards against silently accepting the old bare-array shape again."""
    with pytest.raises(ss.QualityParseError):
        ss.parse_quality_batch(json.dumps([_quality_item(1)]), {1})


def test_markdown_fenced_object_response_still_parses():
    """Fence-stripping stays as a fallback even with the object-rooted schema."""
    fenced = "```json\n" + _quality_response(_quality_item(1)) + "\n```"
    parsed, _ = ss.parse_quality_batch(fenced, {1})
    assert parsed[1]["technical_significance"] == 7.0


def test_normalize_half_point():
    assert ss.normalize_half_point(7.24) == (7.0, True)
    assert ss.normalize_half_point(7.26) == (7.5, True)
    assert ss.normalize_half_point(7.5) == (7.5, False)
    assert ss.normalize_half_point(9.8) == (10.0, True)


# ---------------------------------------------------------------------------
# Missing paper_id is never silently dropped.
# ---------------------------------------------------------------------------


def test_missing_paper_id_not_silently_padded_by_parser():
    """Parser only returns ids the model actually echoed — no invented rows."""
    text = _quality_response(_quality_item(1))
    parsed, _ = ss.parse_quality_batch(text, {1, 2})
    assert set(parsed.keys()) == {1}
    assert 2 not in parsed


def test_missing_paper_triggers_individual_retry_not_silent_loss():
    """stage_semantic_score_v2 must retry a paper dropped from the batch reply
    individually, and must write a COMPLETED assessment for it — never discard."""
    papers = [_paper(1), _paper(2)]
    call_log = []

    def fake_call(batch, *, client=None, **_kwargs):
        ids = sorted(p["content_id"] for p in batch)
        call_log.append(ids)
        if len(batch) == 2:
            # Model only echoes paper 1 — paper 2 goes missing from the reply.
            return {
                "results": {1: _quality_item(1)},
                "rounding_warnings": 0,
                "input_tokens": 10,
                "output_tokens": 10,
                "response_id": "batch",
                "estimated_cost_usd": 0.0,
            }
        pid = ids[0]
        return {
            "results": {pid: _quality_item(pid)},
            "rounding_warnings": 0,
            "input_tokens": 5,
            "output_tokens": 5,
            "response_id": "single",
            "estimated_cost_usd": 0.0,
        }

    fake_conn_cm = MagicMock()
    fake_conn_cm.__enter__.return_value = MagicMock()
    fake_conn_cm.__exit__.return_value = False

    with patch("research_radar.semantic_scoring.load_gated_quality_candidates", return_value=papers), \
         patch("research_radar.semantic_scoring.quality_assessment_exists", return_value=False), \
         patch("research_radar.semantic_scoring.require_scoring_enabled"), \
         patch("research_radar.semantic_scoring.require_api_key"), \
         patch("research_radar.semantic_scoring.call_quality_batch_with_retry", side_effect=fake_call), \
         patch("research_radar.pipeline.connect", return_value=fake_conn_cm), \
         patch("research_radar.pipeline.event"), \
         patch("research_radar.pipeline.bump"):
        conn = MagicMock()
        stats = ss.stage_semantic_score_v2(conn, "run-id", sample=100, client=MagicMock())

    assert stats.completed == 2
    assert stats.failed == 0
    assert sorted(call_log) == [[1, 2], [2]]


# ---------------------------------------------------------------------------
# Scoring v3: prompts must not gate-collapse other dimensions when ai_relevance
# is low — the code gate handles drops; stored scores stay honest measurements.
# ---------------------------------------------------------------------------


def test_prompt_versions_default_to_v3():
    assert ss.SCREEN_PROMPT_VERSION == "research-screen-v3"
    assert ss.QUALITY_PROMPT_VERSION == "research-semantic-v3"


def test_gate_prompt_does_not_collapse_other_dimensions():
    gate_collapse = "score every other dimension at 3 or below"
    assert gate_collapse not in ss.QUALITY_SYSTEM_PROMPT
    assert gate_collapse not in ss.SCREEN_SYSTEM_PROMPT
    for prompt in (ss.QUALITY_SYSTEM_PROMPT, ss.SCREEN_SYSTEM_PROMPT):
        assert "Do NOT lower other dimensions because ai_relevance is low" in prompt
        assert "must remain a real measurement" in prompt


# ---------------------------------------------------------------------------
# Tiering: pass 1 (screen) is cheap, prose-free and reasoning-disabled.
# ---------------------------------------------------------------------------


def test_screen_prompt_requests_no_prose_fields():
    # The strict schema is the real guarantee: it has no string properties at
    # all, so the model has nowhere to put prose even if it tried.
    assert set(ss.SCREEN_ITEM_SCHEMA["properties"].keys()) == {"paper_id", *ss.SCREEN_NUMERIC_FIELDS}
    assert all(p["type"] in ("integer", "number") for p in ss.SCREEN_ITEM_SCHEMA["properties"].values())
    # The JSON example in the prompt itself must not ask for a prose field.
    json_example = ss.SCREEN_SYSTEM_PROMPT.rsplit("\n", 2)[-2]
    for banned in ("so_what", "reason_not_higher", "reason"):
        assert banned not in json_example


def test_screen_uses_screen_model_with_reasoning_disabled():
    captured = {}

    def fake_call_chat_completion(client, **kwargs):
        captured.update(kwargs)
        return {"text": json.dumps({"papers": []}), "input_tokens": 1, "output_tokens": 1, "response_id": "r"}

    papers = [_paper(1)]
    with patch("research_radar.semantic_scoring.call_chat_completion", side_effect=fake_call_chat_completion):
        ss.call_screen_batch(papers, client=MagicMock())

    assert captured["model"] == ss.resolve_screen_model()
    assert captured["reasoning_effort"] is None


# ---------------------------------------------------------------------------
# Tiering: the gate — top GATE_PERCENTILE of screen scores, ai_relevance > 3.
# ---------------------------------------------------------------------------


def _screen_row(cid, ai_relevance, mean_other=5.0):
    return {
        "content_id": cid,
        "ai_relevance": ai_relevance,
        "technical_significance": mean_other,
        "apparent_novelty": mean_other,
        "evidence_strength": mean_other,
    }


class _FakeCursorResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def test_pass2_selects_only_top_gate_percentile_of_screen_scores():
    # 20 papers, screen mean = content_id (1..20) so ranking is unambiguous.
    rows = [_screen_row(i, ai_relevance=8.0, mean_other=float(i)) for i in range(1, 21)]
    conn = MagicMock()
    conn.execute.return_value = _FakeCursorResult(rows)

    gated = ss.select_gated_content_ids(conn, gate_percentile=15)
    # top 15% of 20 = ceil(3.0) = 3 highest-mean papers: 20, 19, 18
    assert gated == [20, 19, 18]

    gated_30 = ss.select_gated_content_ids(conn, gate_percentile=30)
    assert gated_30 == [20, 19, 18, 17, 16, 15]


def test_screen_ai_relevance_le_3_never_reaches_pass2():
    rows = [
        _screen_row(1, ai_relevance=9.0, mean_other=9.0),  # would rank #1 on the other three dims
        _screen_row(2, ai_relevance=3.0, mean_other=9.0),  # excluded: ai_relevance <= 3
        _screen_row(3, ai_relevance=8.0, mean_other=1.0),
    ]
    conn = MagicMock()
    conn.execute.return_value = _FakeCursorResult(rows)

    gated = ss.select_gated_content_ids(conn, gate_percentile=100)
    assert 2 not in gated
    assert set(gated) == {1, 3}


# ---------------------------------------------------------------------------
# GATE_PERCENTILE is env-configurable, never hardcoded.
# ---------------------------------------------------------------------------


def test_gate_percentile_reads_from_env(monkeypatch):
    import importlib

    monkeypatch.setenv("GATE_PERCENTILE", "42")
    try:
        importlib.reload(ss)
        assert ss.GATE_PERCENTILE == 42.0
    finally:
        monkeypatch.delenv("GATE_PERCENTILE", raising=False)
        importlib.reload(ss)


# ---------------------------------------------------------------------------
# Concurrency: each worker thread gets its own DB connection.
# ---------------------------------------------------------------------------


def test_worker_pool_never_shares_a_connection_across_batches():
    """Each batch's writes must go through a freshly-obtained connection, not
    one connection object reused across threads (tiering brief §3)."""
    papers = [_paper(i) for i in range(1, 31)]  # 2 screen batches of 15
    connections_used = []

    def fake_connect():
        cm = MagicMock()
        conn_obj = MagicMock()
        cm.__enter__.return_value = conn_obj
        cm.__exit__.return_value = False
        connections_used.append(conn_obj)
        return cm

    def fake_screen_call(batch, *, client=None, **_kwargs):
        return {
            "results": {p["content_id"]: {"ai_relevance": 8.0, "technical_significance": 7.0, "apparent_novelty": 7.0, "evidence_strength": 7.0} for p in batch},
            "rounding_warnings": 0,
            "input_tokens": 10,
            "output_tokens": 10,
            "response_id": "r",
            "estimated_cost_usd": 0.0,
        }

    with patch("research_radar.semantic_scoring.load_quality_candidates", return_value=papers), \
         patch("research_radar.semantic_scoring.screen_assessment_exists", return_value=False), \
         patch("research_radar.semantic_scoring.require_scoring_enabled"), \
         patch("research_radar.semantic_scoring.require_api_key"), \
         patch("research_radar.semantic_scoring.call_screen_batch_with_retry", side_effect=fake_screen_call), \
         patch("research_radar.pipeline.connect", side_effect=fake_connect), \
         patch("research_radar.pipeline.event"), \
         patch("research_radar.pipeline.bump"):
        conn = MagicMock()
        stats = ss.stage_screen(conn, "run-id", client=MagicMock())

    assert stats.completed == 30
    assert len(connections_used) >= 2
    # No single connection object was reused for more than one batch's writes.
    assert len(connections_used) == len(set(id(c) for c in connections_used))


# ---------------------------------------------------------------------------
# Independence: categorical, four statuses, unclear default, narrow self_evaluation.
# ---------------------------------------------------------------------------


def test_independence_factors_cover_all_four_statuses():
    assert set(ind.INDEPENDENCE_FACTORS) == set(ind.INDEPENDENCE_STATUSES)
    assert len(ind.INDEPENDENCE_STATUSES) == 4


def test_missing_independence_classification_defaults_to_unclear():
    assert fs.independence_factor_for(None) == ind.INDEPENDENCE_FACTORS["unclear"]
    assert fs.independence_factor_for("") == ind.INDEPENDENCE_FACTORS["unclear"]
    assert fs.independence_factor_for("not_a_real_status") == ind.INDEPENDENCE_FACTORS["unclear"]


def test_self_evaluation_narrowness_documented_in_system_prompt():
    """Model is mocked; the narrowness guarantee lives in the prompt wording."""
    prompt = ind.SYSTEM_PROMPT
    assert "not_applicable, not self_evaluation" in prompt
    assert "Almost all research does this" in prompt
    assert "Only use\nself_evaluation for claims about a pre-existing organisational product." in prompt


# ---------------------------------------------------------------------------
# research_score / newsletter_score capped at 10.0.
# ---------------------------------------------------------------------------


def test_final_scores_never_exceed_ten_with_maximum_boosts():
    dims = {
        "technical_significance": 10.0,
        "apparent_novelty": 10.0,
        "practical_applicability": 10.0,
        "professional_value": 10.0,
        "learning_value": 10.0,
        "evidence_strength": 10.0,
        "newsletter_fit": 10.0,
        "independence_status": "independent",
    }
    org_rows = [{"priority": 10, "confidence": 1.0}]
    person_rows = [{"priority": 10, "confidence": 1.0}]
    r = fs.compute_final_v2(dims, org_rows, person_rows, fs.WEIGHT_PROFILES["radar-v2"])
    assert r["final_score"] == 10.0
    assert r["newsletter_score"] == 10.0
    assert r["org_boost"] > 0  # cap is doing real work, not a vacuous pass


# ---------------------------------------------------------------------------
# Weight profile radar-v2.
# ---------------------------------------------------------------------------


def test_radar_v2_weights_sum_to_one_and_exclude_ai_relevance():
    w = fs.WEIGHT_PROFILES["radar-v2"]
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert "ai_relevance" not in w


# ---------------------------------------------------------------------------
# Batch composition is randomised.
# ---------------------------------------------------------------------------


def test_batch_composition_is_randomised():
    papers = [_paper(i) for i in range(1, 41)]
    a = random_batches(papers, 5)
    b = random_batches(papers, 5)
    order_a = [p["content_id"] for batch in a for p in batch]
    order_b = [p["content_id"] for batch in b for p in batch]
    assert order_a != order_b


# ---------------------------------------------------------------------------
# Golden test analogue: a strong unwatchlisted paper still beats a weak one
# with a maximum-priority watchlist org.
# ---------------------------------------------------------------------------


def test_strong_paper_no_org_beats_weak_watchlist_paper_v2():
    w = fs.WEIGHT_PROFILES["radar-v2"]
    strong_dims = {
        "technical_significance": 9.0,
        "apparent_novelty": 9.0,
        "practical_applicability": 9.0,
        "professional_value": 9.0,
        "learning_value": 9.0,
        "evidence_strength": 8.0,
        "newsletter_fit": 7.0,
        "independence_status": "not_applicable",
    }
    weak_dims = {
        "technical_significance": 5.0,
        "apparent_novelty": 5.0,
        "practical_applicability": 5.0,
        "professional_value": 5.0,
        "learning_value": 5.0,
        "evidence_strength": 5.0,
        "newsletter_fit": 5.0,
        "independence_status": "not_applicable",
    }
    strong = fs.compute_final_v2(strong_dims, [], [], w)
    weak = fs.compute_final_v2(weak_dims, [{"priority": 10, "confidence": 1.0}], [], w)
    assert strong["final_score"] > weak["final_score"]
    assert strong["org_boost"] == 0.0
