"""Unit tests for GPT affiliation resolver (mocked OpenRouter; no real API/DB writes)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"

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
    paper = _paper()
    with patch.object(ag, "load_affiliation_candidates", return_value=[paper]), patch.object(
        ag, "count_locally_resolved", return_value=10
    ), patch.object(ag, "count_existing_assessments", return_value=0), patch(
        "research_radar.pipeline.load_orgs",
        return_value=[_org("MIT", ["Massachusetts Institute of Technology"])],
    ), patch("research_radar.pipeline.connect") as mock_connect, patch.object(
        ag, "call_affiliation_resolver"
    ) as mock_call, patch.dict(
        "os.environ", {"AFFILIATION_GPT_ENABLED": "true"}, clear=False
    ), patch("research_radar.semantic_scoring.create_llm_client"), patch(
        "research_radar.semantic_scoring.OPENROUTER_API_KEY", "sk-or-test"
    ):
        wconn = MagicMock()
        mock_connect.return_value.__enter__.return_value = wconn
        wconn.execute.return_value.fetchall.return_value = [
            {
                "evidence_type": "explicit_affiliation_text",
                "evidence_text": "MIT",
                "confidence": 1.0,
            }
        ]
        stats = ag.stage_affiliation_gpt(conn, "run-1")
    assert mock_call.call_count == 0
    assert stats.locally_resolved >= 1 or stats.papers_requested == 1


def test_valid_affiliation_evidence_maps_to_canonical_watchlist():
    watchlist = [
        _org(
            "Massachusetts Institute of Technology",
            ["MIT", "MIT CSAIL"],
            oid=7,
        )
    ]
    paper = _paper(affiliation_text=["Affiliation: MIT CSAIL"])
    parsed = [
        {
            "organisation_name": "MIT CSAIL",
            "affiliation_type": "paper_affiliation",
            "confidence": 0.95,
            "evidence": "Affiliation: MIT CSAIL",
            "reason": "Explicit affiliation string on the paper.",
        }
    ]
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=parsed,
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "MATCHED"
    assert mapped[0]["canonical_name"] == "Massachusetts Institute of Technology"
    assert mapped[0]["evidence_type"] == "affiliation_text"
    assert "MIT CSAIL" in mapped[0]["source_evidence_text"]


def test_gpt_matched_watchlist_name_cannot_create_match():
    """Only organisation_name is passed to orgs_from_text — GPT watchlist hints ignored."""
    watchlist = [_org("MIT", ["MIT"], oid=1)]
    # organisation_name does not match watchlist; a GPT hint of "MIT" must not help.
    parsed = [
        {
            "organisation_name": "Some Obscure Lab LLC",
            "matched_watchlist_name": "MIT",  # legacy / ignored
            "affiliation_type": "paper_affiliation",
            "confidence": 0.99,
            "evidence": "Some Obscure Lab LLC",
            "reason": "hinted MIT",
        }
    ]
    paper = _paper(affiliation_text=["Some Obscure Lab LLC"])
    grounded = ag.filter_grounded_organisations(parsed, paper)
    mapped = ag.map_orgs_to_watchlist(grounded, watchlist)
    assert mapped == []
    assert "matched_watchlist_name" not in ag.RESPONSE_SCHEMA["properties"]["organisations"]["items"]["properties"]


def test_ungrounded_organisation_name_becomes_review_required():
    """Org name not present in affiliation_text → REVIEW_REQUIRED (GPT evidence prose ignored)."""
    watchlist = [_org("MIT", ["MIT", "Massachusetts Institute of Technology"], oid=1)]
    paper = _paper(affiliation_text=["We thank anonymous reviewers."], emails=[])
    parsed = [
        {
            "organisation_name": "Massachusetts Institute of Technology",
            "affiliation_type": "paper_affiliation",
            "confidence": 0.99,
            "evidence": "Author is known to work at MIT",  # GPT prose — must not self-ground
            "reason": "pretrained guess",
        }
    ]
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=parsed,
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "REVIEW_REQUIRED"
    assert mapped == []


def test_gpt_evidence_prose_mismatch_still_grounds_org_name():
    """GPT evidence sentence need not match source; organisation_name containment is enough."""
    watchlist = []
    paper = _paper(affiliation_text=["Affiliation: Rutgers University"])
    parsed = [
        {
            "organisation_name": "Rutgers University",
            "affiliation_type": "paper_affiliation",
            "confidence": 0.9,
            "evidence": 'Explicit paper affiliation: “Rutgers University”; emails listed separately',
            "reason": "named in affiliation",
        }
    ]
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=parsed,
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "NO_MATCH"
    assert mapped == []
    assert ag.find_affiliation_blob_for_org("Rutgers University", paper)


def test_rutgers_university_explicit_affiliation_grounded():
    paper = {"affiliation_text": ["Affiliation: Rutgers University"], "emails": [], "local_evidence": []}
    assert ag.find_affiliation_blob_for_org("Rutgers University", paper)


def test_university_of_utah_with_city_state_suffix_grounded():
    paper = {
        "affiliation_text": ["Affiliation: University of Utah , Salt Lake City , UT , USA"],
        "emails": [],
        "local_evidence": [],
    }
    assert ag.find_affiliation_blob_for_org("University of Utah", paper)


def test_university_of_luxembourg_embedded_in_institute_address_grounded():
    paper = {
        "affiliation_text": [
            "Interdisciplinary Centre for Security, Reliability and Trust (SnT) "
            "University of Luxembourg 29 Avenue J.F. Kennedy L-1855, Luxembourg"
        ],
        "emails": [],
        "local_evidence": [],
    }
    assert ag.find_affiliation_blob_for_org("University of Luxembourg", paper)


def test_shanghai_jiao_tong_inside_school_of_computer_science_grounded():
    paper = {
        "affiliation_text": [
            "Affiliation: School of Computer Science, Shanghai Jiao Tong University, Shanghai, 200240, China"
        ],
        "emails": [],
        "local_evidence": [],
    }
    assert ag.find_affiliation_blob_for_org("Shanghai Jiao Tong University", paper)
    assert ag.find_affiliation_blob_for_org("Zhongguancun Academy", {
        "affiliation_text": ["Affiliation: Zhongguancun Academy, Beijing, 100097, China"],
        "emails": [],
        "local_evidence": [],
    })


def test_grounded_organisation_not_in_watchlist_is_no_match():
    watchlist = [_org("Meta", ["Meta AI"], oid=1)]
    paper = _paper(affiliation_text=["Affiliation: Rutgers University"])
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=[
            {
                "organisation_name": "Rutgers University",
                "affiliation_type": "paper_affiliation",
                "confidence": 0.95,
                "evidence": "whatever GPT wrote",
                "reason": "explicit",
            }
        ],
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "NO_MATCH"
    assert mapped == []


def test_grounded_organisation_in_watchlist_is_matched():
    watchlist = [_org("Rutgers University", ["Rutgers University", "Rutgers"], oid=9)]
    paper = _paper(affiliation_text=["Affiliation: Rutgers University"])
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=[
            {
                "organisation_name": "Rutgers University",
                "affiliation_type": "paper_affiliation",
                "confidence": 0.95,
                "evidence": "mismatched GPT prose that must be ignored",
                "reason": "explicit",
            }
        ],
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "MATCHED"
    assert mapped[0]["canonical_name"] == "Rutgers University"


def test_no_affiliation_gmail_only_is_review_required():
    watchlist = [_org("Google", ["Google"], oid=1)]
    paper = _paper(affiliation_text=[], emails=["alice@gmail.com"])
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=[
            {
                "organisation_name": "Google",
                "affiliation_type": "email_domain",
                "confidence": 0.8,
                "evidence": "alice@gmail.com",
                "reason": "gmail guess",
            }
        ],
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "REVIEW_REQUIRED"
    assert mapped == []


def test_unknown_institutional_email_domain_without_mapping_is_review_required():
    watchlist = [_org("MIT", ["MIT"], oid=1)]  # no ustc.edu.cn domain
    paper = _paper(affiliation_text=[], emails=["wyf666@mail.ustc.edu.cn"])
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision="MATCHED",
        parsed_orgs=[
            {
                "organisation_name": "University of Science and Technology of China",
                "affiliation_type": "email_domain",
                "confidence": 0.9,
                "evidence": "wyf666@mail.ustc.edu.cn",
                "reason": "inferred from domain",
            }
        ],
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "REVIEW_REQUIRED"
    assert mapped == []


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
    assert "ONLY" in prompt
    assert "Unknown Author" in prompt
    with pytest.raises(ag.AffiliationGPTError):
        ag.parse_affiliation_response(
            {"decision": "MATCHED", "organisations": [], "overall_reason": "guess"}
        )


def test_non_watchlist_organisation_becomes_no_match():
    watchlist = [_org("Meta", ["Facebook AI Research"], oid=2)]
    paper = _paper(affiliation_text=["Some Obscure Lab LLC"])
    parsed = ag.parse_affiliation_response(
        _gpt_payload(
            decision="MATCHED",
            orgs=[
                {
                    "organisation_name": "Some Obscure Lab LLC",
                    "affiliation_type": "paper_affiliation",
                    "confidence": 0.9,
                    "evidence": "Some Obscure Lab LLC",
                    "reason": "Named in affiliation string.",
                }
            ],
        )
    )
    decision, mapped = ag.resolve_orgs_from_gpt_result(
        decision=parsed["decision"],
        parsed_orgs=parsed["organisations"],
        paper=paper,
        watchlist_orgs=watchlist,
    )
    assert decision == "NO_MATCH"
    assert mapped == []



def test_boundary_safe_canonical_organisation_matching():
    mit = _org("Massachusetts Institute of Technology", ["MIT", "MIT CSAIL"])
    hits = orgs_from_text("Affiliation: MIT CSAIL", [mit])
    assert hits and hits[0][1] in {
        "MIT",
        "MIT CSAIL",
        "Massachusetts Institute of Technology",
    }
    meta = _org("Meta", ["Meta AI"], oid=2)
    assert orgs_from_text("Researchers at Meta AI built a model", [meta])
    assert not orgs_from_text("This paper discusses metadata extraction", [meta])
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


def test_review_required_unchanged_evidence_skipped():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "REVIEW_REQUIRED",
        "status": "COMPLETED",
        "evidence_fingerprint": "fp1",
        "error_message": None,
    }
    assert ag.assessment_skip_row(conn, 1, "fp1", force=False) is not None


def test_review_required_changed_evidence_retries():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "REVIEW_REQUIRED",
        "status": "COMPLETED",
        "evidence_fingerprint": "fp1",
        "error_message": None,
    }
    assert ag.assessment_skip_row(conn, 1, "fp2", force=False) is None


def test_no_match_changed_evidence_retries():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "NO_MATCH",
        "status": "COMPLETED",
        "evidence_fingerprint": "fp-old",
        "error_message": None,
    }
    assert ag.assessment_skip_row(conn, 1, "fp-new", force=False) is None
    assert ag.assessment_skip_row(conn, 1, "fp-old", force=False) is not None


def test_load_candidates_includes_review_and_no_match_statuses():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    ag.load_affiliation_candidates(conn, limit=10)
    sql = conn.execute.call_args[0][0]
    assert "REVIEW_REQUIRED" in sql
    assert "NO_MATCH" in sql
    assert "PENDING" in sql
    assert "ERROR" in sql


def test_historical_openalex_status_fix_sql_marks_matched():
    sql = (SQL_DIR / "008_affiliation_historical_status_fix.sql").read_text()
    assert "paper_specific_openalex" in sql
    assert "historical_external_affiliation_preserved" in sql
    assert "MATCHED" in sql
    # Must not rewrite evidence as GPT
    assert "gpt_affiliation" not in sql.lower() or "resolver" not in sql


def test_matched_completed_skipped_on_rerun():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {
        "decision": "MATCHED",
        "status": "COMPLETED",
        "evidence_fingerprint": "fp1",
        "error_message": None,
    }
    assert ag.assessment_skip_row(conn, 1, "fp1", force=False) is not None
    assert ag.assessment_skip_row(conn, 1, "fp1", force=True) is None
    assert ag.assessment_skip_row(conn, 1, "fp2", force=False) is None


def test_no_duplicate_content_organisations_on_conflict():
    conn = MagicMock()
    org = _org("MIT", oid=3)
    mapped = [
        {
            "organisation": org,
            "evidence_type": "affiliation_text",
            "source_evidence_text": "MIT",
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
    # evidence text must be source metadata, not invented GPT prose
    assert mock_store.call_args_list[0][0][4] == "MIT"


def test_semantic_prompt_still_only_title_abstract_categories():
    prompt = ss.build_user_prompt(
        title="T",
        abstract="A candidate model for organization graphs.",
        categories=["cs.AI"],
    )
    ss.assert_prompt_is_paper_only(prompt)
    assert prompt.startswith("TITLE:")
    assert "ARXIV CATEGORIES:" in prompt
    assert "ABSTRACT:" in prompt
    assert "AUTHORS:" not in prompt
    assert "EMAILS" not in prompt
    assert "notable_org" not in prompt
    assert "affiliation" not in prompt.lower()


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
    assert stats.skipped_no_evidence == 0
    assert stats.to_dict()["estimated_calls"] == 3


def test_empty_affiliations_and_emails_not_gpt_eligible():
    paper = _paper(affiliation_text=[], emails=[], local_evidence=[])
    assert not ag.is_gpt_eligible_paper(paper)
    assert not ag.has_usable_affiliation_evidence(paper)


def test_gmail_only_not_gpt_eligible():
    paper = _paper(affiliation_text=[], emails=["alice@gmail.com"], local_evidence=[])
    assert not ag.is_gpt_eligible_paper(paper)
    assert ag.is_institutional_email("alice@gmail.com") is False


def test_explicit_affiliation_is_gpt_eligible():
    paper = _paper(affiliation_text=["Affiliation: Rutgers University"], emails=[])
    assert ag.is_gpt_eligible_paper(paper)


def test_institutional_email_is_gpt_eligible():
    paper = _paper(affiliation_text=[], emails=["wyf666@mail.ustc.edu.cn"])
    assert ag.is_institutional_email("wyf666@mail.ustc.edu.cn")
    assert ag.is_gpt_eligible_paper(paper)


def test_local_evidence_skips_gpt_without_openrouter():
    conn = MagicMock()
    paper = _paper(affiliation_text=[], emails=[])
    with patch.object(ag, "load_affiliation_candidates", return_value=[paper]), patch.object(
        ag, "count_locally_resolved", return_value=1
    ), patch.object(ag, "count_existing_assessments", return_value=0), patch(
        "research_radar.pipeline.load_orgs", return_value=[]
    ), patch("research_radar.pipeline.connect") as mock_connect, patch.object(
        ag, "call_affiliation_resolver"
    ) as mock_call, patch.dict(
        "os.environ", {"AFFILIATION_GPT_ENABLED": "true"}, clear=False
    ), patch("research_radar.semantic_scoring.create_llm_client"), patch(
        "research_radar.semantic_scoring.OPENROUTER_API_KEY", "sk-or-test"
    ):
        wconn = MagicMock()
        mock_connect.return_value.__enter__.return_value = wconn
        wconn.execute.return_value.fetchall.return_value = [
            {
                "evidence_type": "explicit_affiliation_text",
                "evidence_text": "MIT",
                "confidence": 1.0,
            }
        ]
        stats = ag.stage_affiliation_gpt(conn, "run-local")
    assert mock_call.call_count == 0
    assert stats.locally_resolved >= 1


def test_dry_run_reports_skipped_no_evidence():
    conn = MagicMock()
    papers = [
        _paper(content_id=1, affiliation_text=["Affiliation: MIT"], emails=[]),
        _paper(content_id=2, affiliation_text=[], emails=["a@gmail.com"]),
        _paper(content_id=3, affiliation_text=[], emails=[]),
    ]
    with patch.object(ag, "load_affiliation_candidates", return_value=papers), patch.object(
        ag, "count_locally_resolved", return_value=0
    ), patch.object(ag, "count_existing_assessments", return_value=0), patch.object(
        ag, "assessment_skip_row", return_value=None
    ), patch("research_radar.pipeline.load_orgs", return_value=[]), patch.object(
        ag, "call_affiliation_resolver"
    ) as mock_call, patch("research_radar.semantic_scoring.create_llm_client") as mock_client:
        stats = ag.stage_affiliation_gpt(conn, "run-dry", dry_run=True)
    assert mock_call.call_count == 0
    assert mock_client.call_count == 0
    d = stats.to_dict()
    assert d["total_unresolved"] == 3
    assert d["eligible_for_gpt"] == 1
    assert d["skipped_no_evidence"] == 2
    assert d["estimated_calls"] == 1
    assert "estimated_tokens" in d
    assert "estimated_cost" in d


def test_no_evidence_runtime_skips_openrouter_and_sets_review_required():
    conn = MagicMock()
    paper = _paper(
        content_id=99,
        affiliation_text=[],
        emails=["x@yahoo.com"],
        affiliation_status="PENDING",
        affiliation_last_error=None,
    )
    with patch.object(ag, "load_affiliation_candidates", return_value=[paper]), patch.object(
        ag, "count_locally_resolved", return_value=0
    ), patch.object(ag, "count_existing_assessments", return_value=0), patch(
        "research_radar.pipeline.load_orgs", return_value=[]
    ), patch("research_radar.pipeline.connect") as mock_connect, patch.object(
        ag, "call_affiliation_resolver"
    ) as mock_call, patch.dict(
        "os.environ", {"AFFILIATION_GPT_ENABLED": "true"}, clear=False
    ), patch("research_radar.semantic_scoring.create_llm_client"), patch(
        "research_radar.semantic_scoring.OPENROUTER_API_KEY", "sk-or-test"
    ), patch.object(ag, "set_affiliation_status") as mock_status, patch(
        "research_radar.pipeline.event"
    ), patch("research_radar.pipeline.bump"):
        wconn = MagicMock()
        mock_connect.return_value.__enter__.return_value = wconn
        wconn.execute.return_value.fetchall.return_value = []
        stats = ag.stage_affiliation_gpt(conn, "run-noev")
    assert mock_call.call_count == 0
    assert stats.skipped_no_evidence == 1
    assert mock_status.call_args.kwargs.get("error") == "no_affiliation_evidence" or (
        mock_status.call_args[0][2] == "REVIEW_REQUIRED"
    )


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
    assert "matched_watchlist_name" not in kwargs["response_format"]["json_schema"]["schema"][
        "properties"
    ]["organisations"]["items"]["properties"]
    user = kwargs["messages"][1]["content"]
    assert "TITLE:" in user
    assert "AUTHORS:" in user
    assert "intrinsic_candidate" not in user


def test_prompt_forbids_memory_based_employer_lookup():
    assert "pretrained" in ag.SYSTEM_PROMPT.lower()
    assert "NEVER" in ag.SYSTEM_PROMPT
    assert "REVIEW_REQUIRED" in ag.SYSTEM_PROMPT


def test_pytest_has_no_real_openrouter_or_db_side_effects_in_unit_helpers():
    # Smoke: grounding/helpers are pure and do not touch network.
    paper = _paper()
    assert ag.is_grounded_in_supplied_evidence(
        "Massachusetts Institute of Technology", paper
    )
    assert not ag.is_grounded_in_supplied_evidence("OpenAI secretly", paper)