"""Unit tests for GPT semantic scoring (mocked OpenAI; no real API; no RDS)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from research_radar import semantic_scoring as ss


def _paper(cid, score, status="SCORED", title=None):
    return {
        "content_id": cid,
        "id": cid,
        "intrinsic_candidate_score": score,
        "status": status,
        "title": title or f"Paper {cid}",
        "abstract": f"Abstract for paper {cid} about machine learning systems.",
        "categories": ["cs.LG", "cs.AI"],
    }


def _valid_payload(**overrides):
    base = {
        "ai_relevance": {"score": 8.0, "reason": "Central AI/ML contribution."},
        "technical_significance": {"score": 7.0, "reason": "Solid method contribution."},
        "practical_applicability": {"score": 6.5, "reason": "Useful for deployment."},
        "professional_value": {"score": 7.0, "reason": "Relevant for practitioners."},
        "student_learning_value": {"score": 6.0, "reason": "Clear learning value."},
        "apparent_novelty": {"score": 5.5, "reason": "Apparent novelty from abstract only."},
        "explainability": {"score": 8.0, "reason": "Clear problem and method."},
        "industry_relevance": {"score": 6.0, "reason": "Potential industry use."},
        "industry_labels": ["enterprise_ai", "ai_evaluation"],
    }
    base.update(overrides)
    return base


def test_sample_exactly_25_per_group():
    rows = [_paper(i, score=10.0 - (i * 0.01), status="CANDIDATE" if i < 300 else "SCORED") for i in range(1, 401)]
    selected = ss.select_semantic_sample(rows, threshold=5.5, n_per_group=25, seed=20260824)
    assert len(selected) == 100
    groups = {g: [r for r in selected if r["sample_group"] == g] for g in ("top", "threshold", "low", "random")}
    for g, items in groups.items():
        assert len(items) == 25, g
    ids = [r["content_id"] for r in selected]
    assert len(ids) == len(set(ids))


def test_sample_excludes_rejected():
    rows = [_paper(i, 7.0, status="SCORED") for i in range(1, 120)]
    rows += [_paper(9000 + i, 9.9, status="REJECTED") for i in range(50)]
    selected = ss.select_semantic_sample(rows, threshold=5.5, n_per_group=25, seed=1)
    assert all(r["status"] != "REJECTED" for r in selected)
    assert all(r["content_id"] < 9000 for r in selected)


def test_sample_reproducible_with_seed():
    rows = [_paper(i, score=(i % 100) / 10.0) for i in range(1, 301)]
    a = ss.select_semantic_sample(rows, threshold=5.5, n_per_group=25, seed=42)
    b = ss.select_semantic_sample(rows, threshold=5.5, n_per_group=25, seed=42)
    assert [r["content_id"] for r in a] == [r["content_id"] for r in b]
    assert [r["sample_group"] for r in a] == [r["sample_group"] for r in b]


def test_parse_structured_assessment_valid():
    scores, reasons, labels = ss.parse_structured_assessment(_valid_payload())
    assert scores["ai_relevance"] == 8.0
    assert "Central" in reasons["ai_relevance"]
    assert labels == ["enterprise_ai", "ai_evaluation"]


def test_scores_restricted_0_to_10():
    with pytest.raises(ss.SemanticParseError):
        ss.parse_structured_assessment(
            _valid_payload(ai_relevance={"score": 11.0, "reason": "too high"})
        )
    with pytest.raises(ss.SemanticParseError):
        ss.parse_structured_assessment(
            _valid_payload(ai_relevance={"score": -1.0, "reason": "too low"})
        )


def test_invalid_gpt_response_rejected():
    with pytest.raises(ss.SemanticParseError):
        ss.parse_structured_assessment({"ai_relevance": {"score": 5}})
    with pytest.raises(ss.SemanticParseError):
        ss.parse_structured_assessment(
            _valid_payload(industry_labels=["not_a_real_label"])
        )


def test_semantic_score_formula():
    scores = {k: 10.0 for k in ss.DIMENSION_KEYS}
    assert ss.compute_semantic_score(scores) == 10.0
    scores = {
        "ai_relevance": 10.0,
        "technical_significance": 0.0,
        "practical_applicability": 0.0,
        "professional_value": 0.0,
        "student_learning_value": 0.0,
        "apparent_novelty": 0.0,
        "explainability": 0.0,
        "industry_relevance": 0.0,
    }
    assert ss.compute_semantic_score(scores) == 2.0  # 10 * 0.20


def test_prompt_excludes_organisation_and_scores():
    prompt = ss.build_user_prompt(
        title="Agent Systems",
        abstract="We evaluate agent workflows.",
        categories=["cs.AI"],
    )
    assert "TITLE:" in prompt
    assert "ABSTRACT:" in prompt
    assert "ARXIV CATEGORIES:" in prompt
    ss.assert_prompt_is_paper_only(prompt)


def test_prompt_allows_legitimate_words_in_paper_text():
    prompt = ss.build_user_prompt(
        title="OpenAlex candidate organization employer analysis in AI systems",
        abstract=(
            "We introduce a candidate model for organization-level retrieval. "
            "Baselines mention OpenAlex and employer affiliation graphs."
        ),
        categories=["cs.AI"],
    )
    ss.assert_prompt_is_paper_only(prompt)
    for word in ("candidate", "organization", "OpenAlex", "employer"):
        assert word in prompt


def test_semantic_scoring_allows_candidate_organization_openalex_words():
    """Full call path must not reject paper text containing formerly banned words."""
    ok_json = (
        '{"ai_relevance":{"score":8,"reason":"r"},'
        '"technical_significance":{"score":7,"reason":"r"},'
        '"practical_applicability":{"score":6,"reason":"r"},'
        '"professional_value":{"score":6,"reason":"r"},'
        '"student_learning_value":{"score":5,"reason":"r"},'
        '"apparent_novelty":{"score":5,"reason":"r"},'
        '"explainability":{"score":8,"reason":"r"},'
        '"industry_relevance":{"score":6,"reason":"r"},'
        '"industry_labels":[]}'
    )
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        id="resp_legit_words",
        choices=[MagicMock(message=MagicMock(content=ok_json))],
        usage=MagicMock(prompt_tokens=12, completion_tokens=8),
    )
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        result = ss.call_semantic_assessment(
            title="Evaluating a candidate model against OpenAlex organization graphs",
            abstract="Our candidate model compares organization and employer signals from OpenAlex.",
            categories=["cs.AI", "cs.LG"],
            client=client,
        )
    assert result["status"] == "COMPLETED"
    kwargs = client.chat.completions.create.call_args.kwargs
    user_msg = kwargs["messages"][1]["content"]
    assert "candidate model" in user_msg
    assert "organization" in user_msg
    assert "OpenAlex" in user_msg
    assert "notable_org_signal" not in user_msg
    assert "intrinsic_candidate_score" not in user_msg


def test_api_request_uses_only_title_abstract_categories():
    builder_kwargs = {}
    ok_json = (
        '{"ai_relevance":{"score":8,"reason":"r"},'
        '"technical_significance":{"score":7,"reason":"r"},'
        '"practical_applicability":{"score":6,"reason":"r"},'
        '"professional_value":{"score":6,"reason":"r"},'
        '"student_learning_value":{"score":5,"reason":"r"},'
        '"apparent_novelty":{"score":5,"reason":"r"},'
        '"explainability":{"score":8,"reason":"r"},'
        '"industry_relevance":{"score":6,"reason":"r"},'
        '"industry_labels":[]}'
    )

    real_builder = ss.build_user_prompt

    def _tracking_builder(*, title, abstract, categories):
        builder_kwargs.update(
            {
                "title": title,
                "abstract": abstract,
                "categories": categories,
            }
        )
        assert set(builder_kwargs) == {"title", "abstract", "categories"}
        return real_builder(title=title, abstract=abstract, categories=categories)

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        id="resp_only_fields",
        choices=[MagicMock(message=MagicMock(content=ok_json))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=10),
    )
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0), patch.object(
        ss, "build_user_prompt", side_effect=_tracking_builder
    ):
        ss.call_semantic_assessment(
            title="Paper title",
            abstract="Paper abstract with candidate model",
            categories=["cs.AI"],
            client=client,
        )

    assert builder_kwargs == {
        "title": "Paper title",
        "abstract": "Paper abstract with candidate model",
        "categories": ["cs.AI"],
    }
    api_kwargs = client.chat.completions.create.call_args.kwargs
    user_content = api_kwargs["messages"][1]["content"]
    assert user_content.startswith("TITLE:")
    assert "ARXIV CATEGORIES:" in user_content
    assert "ABSTRACT:" in user_content
    assert "Paper title" in user_content
    assert "Paper abstract with candidate model" in user_content
    assert "cs.AI" in user_content
    for forbidden_field in (
        "notable_org_signal",
        "notable_person_signal",
        "intrinsic_candidate_score",
        "content_status",
        "openalex_id",
        "watchlist_id",
    ):
        assert forbidden_field not in user_content


def test_duplicate_assessment_skipped_without_force():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"ok": 1}
    assert ss.assessment_exists(conn, 42) is True
    sql = conn.execute.call_args[0][0]
    assert "status = 'COMPLETED'" in sql


def test_assessment_exists_false_when_no_completed_row():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    assert ss.assessment_exists(conn, 42) is False


def test_force_upsert_uses_on_conflict_update():
    conn = MagicMock()
    result = {
        "scores": {k: 5.0 for k in ss.DIMENSION_KEYS},
        "reasons": {k: "ok" for k in ss.DIMENSION_KEYS},
        "industry_labels": [],
        "semantic_score": 5.0,
        "input_tokens": 10,
        "output_tokens": 20,
        "estimated_cost_usd": 0.001,
        "response_id": "resp_1",
        "status": "COMPLETED",
        "error_message": None,
    }
    ss.upsert_assessment(conn, content_id=1, sample_group="top", result=result, force=True)
    sql = conn.execute.call_args[0][0]
    assert "DO UPDATE SET" in sql


def test_non_force_upsert_overwrites_non_completed_only():
    conn = MagicMock()
    result = {
        "scores": {},
        "reasons": {},
        "industry_labels": [],
        "semantic_score": None,
        "status": "ERROR",
        "error_message": "boom",
    }
    ss.upsert_assessment(conn, content_id=1, sample_group="low", result=result, force=False)
    sql = conn.execute.call_args[0][0]
    assert "DO UPDATE SET" in sql
    assert "status <> 'COMPLETED'" in sql
    assert "DO NOTHING" not in sql


@patch("research_radar.semantic_scoring.time.sleep")
def test_429_retry_then_rate_limited(mock_sleep):
    class RateLimitError(Exception):
        status_code = 429

    client = MagicMock()
    client.chat.completions.create.side_effect = RateLimitError("429")
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_MAX_RETRIES", 3), patch.object(
        ss, "SEMANTIC_REQUEST_SLEEP", 0
    ):
        with pytest.raises(ss.SemanticAPIError) as ei:
            ss.call_semantic_assessment(
                title="T", abstract="A", categories=["cs.AI"], client=client
            )
    assert ei.value.status == "RATE_LIMITED"
    assert client.chat.completions.create.call_count == 3
    assert mock_sleep.called


@patch("research_radar.semantic_scoring.time.sleep")
def test_5xx_retry(mock_sleep):
    class ServerError(Exception):
        status_code = 503

    class OkResponse:
        id = "resp_ok"
        choices = []
        usage = MagicMock(prompt_tokens=11, completion_tokens=22)

        def __init__(self):
            import json

            self.choices = [MagicMock(message=MagicMock(content=json.dumps(_valid_payload())))]

    client = MagicMock()
    client.chat.completions.create.side_effect = [ServerError("503"), OkResponse()]
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_MAX_RETRIES", 3), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        out = ss.call_semantic_assessment(
            title="T", abstract="A", categories=["cs.AI"], client=client
        )
    assert out["status"] == "COMPLETED"
    assert client.chat.completions.create.call_count == 2


def test_openrouter_model_resolution():
    with patch.object(ss, "OPENROUTER_MODEL", "gpt-5.6-sol"):
        assert ss.resolve_model_name() == "openai/gpt-5.6-sol"
    with patch.object(ss, "OPENROUTER_MODEL", "openai/gpt-5.6-sol"):
        assert ss.resolve_model_name() == "openai/gpt-5.6-sol"


@patch("research_radar.semantic_scoring.create_llm_client")
@patch("research_radar.semantic_scoring.time.sleep")
def test_openrouter_uses_chat_completions(mock_sleep, mock_client_factory):
    import json

    class OkResponse:
        id = "resp_or"
        choices = [MagicMock(message=MagicMock(content=json.dumps(_valid_payload())))]
        usage = MagicMock(prompt_tokens=120, completion_tokens=60)

    client = MagicMock()
    client.chat.completions.create.return_value = OkResponse()
    mock_client_factory.return_value = client

    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        out = ss.call_semantic_assessment(
            title="Evaluating Tool-Using AI Agents",
            abstract="We propose a benchmark for production agent systems.",
            categories=["cs.AI"],
        )
    assert out["status"] == "COMPLETED"
    assert out["provider"] == "openrouter"
    client.chat.completions.create.assert_called_once()


def test_call_semantic_assessment_reaches_openrouter_for_normal_paper():
    """SYSTEM_PROMPT is not paper-template-validated; a normal paper must reach the API."""
    import json

    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        id="resp_normal_paper",
        choices=[MagicMock(message=MagicMock(content=json.dumps(_valid_payload())))],
        usage=MagicMock(prompt_tokens=50, completion_tokens=40),
    )
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        out = ss.call_semantic_assessment(
            title="A Survey of Retrieval-Augmented Generation",
            abstract="We review retrieval-augmented generation methods for large language models.",
            categories=["cs.CL", "cs.AI"],
            client=client,
        )
    assert out["status"] == "COMPLETED"
    client.chat.completions.create.assert_called_once()
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("TITLE:")


def test_permanent_api_failure_status_error():
    class BadRequest(Exception):
        status_code = 400

    client = MagicMock()
    client.chat.completions.create.side_effect = BadRequest("bad request")
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_MAX_RETRIES", 2), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0), patch(
        "research_radar.semantic_scoring.time.sleep"
    ):
        with pytest.raises(ss.SemanticAPIError) as ei:
            ss.call_semantic_assessment(
                title="T", abstract="A", categories=[], client=client
            )
    assert ei.value.status == "ERROR"


def test_refusal_status():
    class RefusalError(Exception):
        pass

    client = MagicMock()
    client.chat.completions.create.side_effect = RefusalError("ContentFilter refusal")
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        with pytest.raises(ss.SemanticAPIError) as ei:
            ss.call_semantic_assessment(
                title="T", abstract="A", categories=[], client=client
            )
    assert ei.value.status == "REFUSED"


def test_dry_run_zero_api_calls():
    papers = [_paper(i, 6.0 + (i % 40) * 0.05) for i in range(1, 201)]
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = papers

    with patch.object(ss, "load_relevant_scored_papers", return_value=papers):
        with patch.object(ss, "call_semantic_assessment") as mock_call:
            stats = ss.stage_semantic_score(
                conn, "run-dry", sample=100, dry_run=True, force=False
            )
    mock_call.assert_not_called()
    assert stats.requested == 100
    assert stats.sample_groups["top"] == 25
    sqls = " ".join(str(c[0][0]) for c in conn.execute.call_args_list)
    assert "content_score_assessments" not in sqls


def test_experiment_does_not_modify_status_or_scores_in_stage_path():
    """Guard: stage_semantic_score must not call set_status / store_scores."""
    papers = [_paper(i, 7.0) for i in range(1, 101)]
    conn = MagicMock()
    with patch.object(ss, "load_relevant_scored_papers", return_value=papers):
        with patch("research_radar.pipeline.set_status") as mock_status:
            with patch("research_radar.pipeline.store_scores") as mock_scores:
                ss.stage_semantic_score(conn, "run-dry", sample=100, dry_run=True)
    mock_status.assert_not_called()
    mock_scores.assert_not_called()


def test_cost_calculation():
    with patch.object(ss, "OPENROUTER_INPUT_COST_PER_MILLION", 2.0), patch.object(
        ss, "OPENROUTER_OUTPUT_COST_PER_MILLION", 10.0
    ):
        assert ss.estimate_cost_usd(1_000_000, 1_000_000) == 12.0
        assert ss.estimate_cost_usd(0, 0) == 0.0
        assert ss.estimate_cost_usd(2000, 1000) == round(2000 / 1e6 * 2.0 + 1000 / 1e6 * 10.0, 6)


def test_missing_api_key_fails_safely():
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", ""
    ):
        with pytest.raises(ss.SemanticScoringConfigError):
            ss.require_api_key()
        with pytest.raises(ss.SemanticScoringConfigError):
            ss.call_semantic_assessment(title="T", abstract="A", categories=[])


def test_scoring_disabled_fails_safely():
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "false"}, clear=False):
        with pytest.raises(ss.SemanticScoringDisabled):
            ss.require_scoring_enabled()


@patch("research_radar.semantic_scoring.time.sleep")
def test_successful_openrouter_call_parses_and_scores(mock_sleep):
    import json

    class OkResponse:
        id = "resp_123"
        choices = [MagicMock(message=MagicMock(content=json.dumps(_valid_payload())))]
        usage = MagicMock(prompt_tokens=100, completion_tokens=50)

    client = MagicMock()
    client.chat.completions.create.return_value = OkResponse()
    with patch.dict(os.environ, {"SEMANTIC_SCORING_ENABLED": "true"}, clear=False), patch.object(
        ss, "OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ss, "SEMANTIC_REQUEST_SLEEP", 0):
        out = ss.call_semantic_assessment(
            title="Evaluating Tool-Using AI Agents",
            abstract="We propose a benchmark for production agent systems.",
            categories=["cs.AI"],
            client=client,
        )
    assert out["status"] == "COMPLETED"
    assert out["response_id"] == "resp_123"
    assert 0 <= out["semantic_score"] <= 10
    assert out["input_tokens"] == 100
    call_kwargs = client.chat.completions.create.call_args.kwargs
    user_content = call_kwargs["messages"][1]["content"]
    assert call_kwargs["model"] == ss.resolve_model_name()
    assert call_kwargs["extra_body"]["reasoning"]["effort"] == ss.SEMANTIC_REASONING_EFFORT
    assert call_kwargs["response_format"]["type"] == "json_schema"
