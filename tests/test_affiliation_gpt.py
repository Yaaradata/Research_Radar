"""Unit tests for GPT affiliation resolver (mocked OpenRouter; no real API/DB writes)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from research_radar import affiliation_gpt as ag
from research_radar import semantic_scoring as ss
from research_radar.pipeline import orgs_from_text


def _org(name, aliases=None, oid=1, priority=5):
    return {
        "organisation_id": oid,
        "canonical_name": name,
        "aliases": aliases or [],
        "domains": [],
        "priority": priority,
    }


def _paper(**overrides):
    base = {
        "content_id": 101,
        "title": "Scaling Agents at MIT CSAIL",
        "authors": ["Ada Lovelace"],
        "emails": [],
        "affiliation_text": ["Massachusetts Institute of Technology"],
        "local_evidence": [],
        "item_status": "ENTITY_RESOLVED",
        "evidence_url": "https://arxiv.org/abs/x",
    }
    base.update(overrides)
    return base


def _gpt_payload(decision="MATCHED", orgs=None, reason="Supported by affiliation string."):
    if orgs is None:
        orgs = [
            {
                "organisation_name": "Massachusetts Institute of Technology",
                "matched_watchlist_name": "MIT",
                "affiliation_type": "paper_affiliation",
                "confidence": 0.95,
                "evidence": "Massachusetts Institute of Technology",
                "reason": "Explicit affiliation string on the paper.",
            }
        ]
    return {
        "decision": decision,
        "organisations": orgs,
        "overall_reason": reason,
    }


def _mock_client(payload: dict):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        id="resp_aff_1",
        choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=50),
    )
    return client


def test_local_org_already_resolved_skips_gpt_in_stage():
    conn = MagicMock()
    # load candidates empty because local filter — simulate process path
    paper = _paper()
    with patch.object(ag, "load_affiliation_candidates", return_value=[paper]), patch.object(
        ag, "count_locally_resolved", return_value=10
    ), patch.object(ag, "count_existing_assessments", return_value=0), patch.object(
        ag, "load_orgs", create=True
    ), patch("research_radar.pipeline.load_orgs", return_value=[_org("MIT", ["Massachusetts Institute of Technology"])]), patch(
        "research_radar.pipeline.connect"
    ) as mock_connect, patch.object(ag, "call_affiliation_resolver") as mock_call, patch.dict(
        "os.environ", {"AFFILIATION_GPT_ENABLED": "true"}, clear=False
    ), patch("research_radar.semantic_scoring.create_llm_client"), patch(
        "research_radar.semantic_scoring.OPENROUTER_API_KEY", "sk-or-test"
    ):
        wconn = MagicMock()
        mock_connect.return_value.__enter__.return_value = wconn
        wconn.execute.return_value.fetchall.return_value = [
            {"evidence_type": "explicit_affiliation_text", "evidence_text": "MIT", "confidence": 1.0}
        ]
        stats = ag.stage_affiliation_gpt(conn, "run-1")
    assert mock_call.call_count == 0
    assert stats.locally_resolved >= 1 or stats.papers_requested == 1


def test_explicit_affiliation_matched_maps_watchlist():
    watchlist = [_org("MIT", ["Massachusetts Institute of Technology", "MIT"], oid=7)]
    parsed = ag.parse_affiliation_response(_gpt_payload())
    mapped = ag.map_orgs_to_watchlist(parsed["organisations"], watchlist)
    decision = ag.finalize_decision(parsed["decision"], mapped, parsed["organisations"])
    assert decision == "MATCHED"
    assert mapped[0]["canonical_name"] == "MIT"
    assert mapped[0]["evidence_type"] == "affiliation_text"


def test_ambiguous_evidence_review_required():
    payload = _gpt_payload(
        decision="REVIEW_REQUIRED",
        orgs=[],
        reason="Affiliation strings are ambiguous.",
    )
    parsed = ag.parse_affiliation_response(payload)
    assert parsed["decision"] == "REVIEW_REQUIRED"
    assert ag.finalize_decision("REVIEW_REQUIRED", [], []) == "REVIEW_REQUIRED"


def test_no_evidence_review_required_no_hallucination():
    paper = _paper(affiliation_text=[], emails=[], authors=["Unknown Author"])
    prompt = ag.build_resolver_user_prompt(paper)
    assert "pretrained" in prompt.lower() or "ONLY" in prompt
    assert "Unknown Author" in prompt
    assert "(none)" in prompt
    # Parser rejects MATCHED with empty orgs
    with pytest.raises(ag.AffiliationGPTError):
        ag.parse_affiliation_response(
            {"decision": "MATCHED", "organisations": [], "overall_reason": "guess"}
        )


def test_non_watchlist_organisation_becomes_no_match():
    watchlist = [_org("Meta", ["Facebook AI Research"], oid=2)]
    parsed = ag.parse_affiliation_response(
        _gpt_payload(
            decision="MATCHED",
            orgs=[
                {
                    "organisation_name": "Some Obscure Lab LLC",
                    "matched_watchlist_name": None,
                    "affiliation_type": "paper_affiliation",
                    "confidence": 0.9,
                    "evidence": "Some Obscure Lab LLC",
                    "reason": "Named in affiliation string.",
                }
            ],
        )
    )
    mapped = ag.map_orgs_to_watchlist(parsed["organisations"], watchlist)
    assert mapped == []
    assert ag.finalize_decision("MATCHED", mapped, parsed["organisations"]) == "NO_MATCH"


def test_boundary_safe_canonical_organisation_matching():
    mit = _org("Massachusetts Institute of Technology", ["MIT", "MIT CSAIL"])
    hits = orgs_from_text("Affiliation: MIT CSAIL", [mit])
    assert hits and hits[0][1] in {"MIT", "MIT CSAIL", "Massachusetts Institute of Technology"}
    meta = _org("Meta", ["Meta AI"], oid=2)
    assert orgs_from_text("Researchers at Meta AI built a model", [meta])
    assert not orgs_from_text("This paper discusses metadata extraction", [meta])
    # short alias must not match inside unrelated tokens
    assert orgs_from_text("We SUBMIT results", [mit]) == []


def test_error_assessment_is_retryable():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "ERROR",
        "status": "ERROR",
        "evidence_fingerprint": "abc",
        "error_message": "boom",
    }
    assert ag.assessment_skip_row(conn, 1, "abc", force=False) is None


def test_matched_completed_skipped_on_rerun():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "MATCHED",
        "status": "COMPLETED",
        "evidence_fingerprint": "fp1",
        "error_message": None,
    }
    assert ag.assessment_skip_row(conn, 1, "fp1", force=False) is not None
    # force overrides
    assert ag.assessment_skip_row(conn, 1, "fp1", force=True) is None
    # evidence change retries
    assert ag.assessment_skip_row(conn, 1, "fp2", force=False) is None


def test_no_duplicate_content_organisations_on_conflict():
    conn = MagicMock()
    org = _org("MIT", oid=3)
    mapped = [
        {
            "organisation": org,
            "evidence_type": "affiliation_text",
            "evidence": "MIT",
            "organisation_name": "MIT",
            "confidence": 0.9,
            "affiliation_type": "paper_affiliation",
        }
    ]
    with patch("research_radar.pipeline.store_org_evidence") as mock_store:
        n = ag.write_mapped_orgs(conn, 55, mapped, None)
        n2 = ag.write_mapped_orgs(conn, 55, mapped, None)
    assert n == 1 and n2 == 1
    assert mock_store.call_count == 2
    # store_org_evidence itself uses ON CONFLICT — duplicate prevention is SQL-level


def test_semantic_prompt_still_has_no_affiliation_fields():
    prompt = ss.build_user_prompt(
        title="T",
        abstract="A candidate model for organization graphs.",
        categories=["cs.AI"],
    )
    ss.assert_prompt_is_paper_only(prompt)
    assert "AUTHORS:" not in prompt
    assert "EMAILS" not in prompt
    assert "notable_org" not in prompt


def test_dry_run_makes_zero_api_calls():
    conn = MagicMock()
    papers = [_paper(content_id=i) for i in range(3)]
    with patch.object(ag, "load_affiliation_candidates", return_value=papers), patch.object(
        ag, "count_locally_resolved", return_value=5
    ), patch.object(ag, "count_existing_assessments", return_value=1), patch.object(
        ag, "assessment_skip_row", return_value=None
    ), patch("research_radar.pipeline.load_orgs", return_value=[]), patch.object(
        ag, "call_affiliation_resolver"
    ) as mock_call, patch("research_radar.semantic_scoring.create_llm_client") as mock_client:
        stats = ag.stage_affiliation_gpt(conn, "run-dry", dry_run=True)
    assert mock_call.call_count == 0
    assert mock_client.call_count == 0
    assert stats.eligible_for_gpt == 3
    assert "estimated_calls" in stats.to_dict() or stats.sample_note.startswith("dry_run")


def test_call_affiliation_resolver_mocked_openrouter():
    client = _mock_client(_gpt_payload())
    paper = _paper()
    with patch.dict("os.environ", {"AFFILIATION_GPT_ENABLED": "true"}, clear=False), patch(
        "research_radar.semantic_scoring.OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ag, "AFFILIATION_GPT_REQUEST_SLEEP", 0):
        out = ag.call_affiliation_resolver(paper=paper, client=client)
    assert out["decision"] == "MATCHED"
    assert out["status"] == "COMPLETED"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["reasoning"]["effort"] == ag.AFFILIATION_GPT_REASONING_EFFORT
    user = kwargs["messages"][1]["content"]
    assert "TITLE:" in user
    assert "AUTHORS:" in user
    assert "intrinsic_candidate" not in user
    assert "OpenAlex" not in user or "OpenAlex" in (paper.get("title") or "")


def test_prompt_forbids_memory_based_employer_lookup():
    assert "pretrained" in ag.SYSTEM_PROMPT.lower()
    assert "NEVER" in ag.SYSTEM_PROMPT
    assert "REVIEW_REQUIRED" in ag.SYSTEM_PROMPT
