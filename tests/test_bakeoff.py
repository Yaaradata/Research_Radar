"""Tests for classification bake-off harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from openpyxl import load_workbook

from research_radar import bakeoff as bo
from research_radar.bakeoff import (
    analyze_raw_classify_response,
    compute_force_fit_rate,
    cost_from_tokens,
    cost_per_thousand_from_measured,
    count_exclusivity_violations,
    estimate_bakeoff_run_cost,
    import_haiku_baseline,
    is_force_fit,
    is_general_method,
    load_bakeoff_config,
    reconstruct_raw_response,
    select_stratified_sample,
    self_consistency_rate,
)
from research_radar.classify import CLASSIFY_PROMPT_VERSION


def _classified_paper(cid: int, domains: list[str], paper_kind: str = "method") -> dict:
    return {
        "content_id": cid,
        "title": f"Paper {cid}",
        "abstract": "Abstract text.",
        "categories": ["cs.AI"],
        "application_domain": domains,
        "audience_relevance": ["practitioner"],
        "paper_kind": paper_kind,
        "geography_focus": "none",
        "domain_confidence": 7.0,
        "baseline_model": "anthropic/claude-haiku-4.5",
    }


def _make_pool(n: int = 560) -> list[dict]:
    pool = []
    for i in range(200):
        pool.append(_classified_paper(1000 + i, ["general_method"], paper_kind="method"))
    for i in range(200, 400):
        pool.append(_classified_paper(1000 + i, ["healthcare_life_sciences"], paper_kind="method"))
    for i in range(400, 460):
        pool.append(_classified_paper(1000 + i, ["financial_services"], paper_kind="survey_review"))
    for i in range(460, 560):
        pool.append(_classified_paper(1000 + i, ["education"], paper_kind="empirical_study"))
    return pool


def test_raw_response_stored_before_validation():
    raw = '{"papers": [{"paper_id": 1, "application_domain": ["not_a_domain"], "audience_relevance": ["practitioner"], "paper_kind": "method", "geography_focus": "none", "domain_confidence": 5}]}'
    out = analyze_raw_classify_response(raw, {1})[1]
    assert out.raw_response == raw
    assert out.json_valid is True
    assert out.schema_valid is False
    assert any("invalid" in d for d in out.dropped_values)


def test_exclusivity_violations_counted_not_silently_dropped():
    raw = json.dumps(
        {
            "papers": [
                {
                    "paper_id": 2,
                    "application_domain": ["general_method", "healthcare_life_sciences"],
                    "audience_relevance": ["practitioner"],
                    "paper_kind": "method",
                    "geography_focus": "none",
                    "domain_confidence": 5.0,
                }
            ]
        }
    )
    v = analyze_raw_classify_response(raw, {2})[2]
    assert v.exclusivity_violation is True
    assert count_exclusivity_violations(["general_method", "healthcare_life_sciences"]) == 1
    assert v.schema_valid is False


def test_force_fit_only_model_specific_vs_human_general():
    assert is_force_fit(["healthcare_life_sciences"], True) is True
    assert is_force_fit(["general_method"], True) is False
    assert is_force_fit(["healthcare_life_sciences"], False) is False
    human = {1: {"application_domain": ["general_method"]}, 2: {"application_domain": ["healthcare_life_sciences"]}}
    model = {
        1: {"application_domain": ["financial_services"]},
        2: {"application_domain": ["general_method"]},
    }
    assert compute_force_fit_rate(model, human) == 0.5


def test_sample_stratification_fixed_seed():
    gm_method = [_classified_paper(i, ["general_method"], "method") for i in range(200)]
    gm_survey = [_classified_paper(200 + i, ["general_method"], "survey_review") for i in range(120)]
    ss = [_classified_paper(300 + i, ["healthcare_life_sciences"], "method") for i in range(200)]
    rem = [_classified_paper(500 + i, ["scientific_research"], "empirical_study") for i in range(100)]
    pool = gm_method + gm_survey + ss + rem
    sample, counts = select_stratified_sample(pool, seed=20260907)
    assert counts["general_method"] == 150
    assert counts["specific_sector"] == 150
    assert counts["non_standard_kind"] == 50
    assert counts["remainder"] == 50
    assert len(sample) == 400
    _, counts2 = select_stratified_sample(pool, seed=20260907)
    assert counts == counts2


def test_haiku_baseline_imports_without_api():
    conn = MagicMock()
    run_id = uuid4()
    paper = _classified_paper(42, ["general_method"])
    n = import_haiku_baseline(conn, run_id, [paper])
    assert n == 1
    assert conn.execute.called
    sql = conn.execute.call_args[0][0]
    assert "bakeoff_results" in sql
    assert "content_classifications" not in sql


def test_self_consistency_compares_by_content_id_not_row_order():
    p1 = {
        1: {"application_domain": ["general_method"]},
        2: {"application_domain": ["healthcare_life_sciences"]},
    }
    p2 = {
        2: {"application_domain": ["healthcare_life_sciences"]},
        1: {"application_domain": ["general_method"]},
    }
    assert self_consistency_rate(p1, p2) == 1.0
    p2[1]["application_domain"] = ["financial_services"]
    assert self_consistency_rate(p1, p2) == 0.5


def test_export_labelling_sheets_have_no_model_outputs(tmp_path):
    from scripts.bakeoff_export_labels import export_workbook

    papers = [_classified_paper(1, ["general_method"]), _classified_paper(2, ["education"])]
    results = [
        {
            "content_id": 1,
            "candidate_id": "haiku",
            "pass_index": 1,
            "application_domain": ["general_method"],
            "paper_kind": "method",
            "schema_valid": True,
            "dropped_values": [],
        }
    ]
    out = tmp_path / "labels.xlsx"
    export_workbook(
        out,
        disagreement_papers=papers,
        control_papers=[papers[0]],
        results=results,
        all_sample=papers,
    )
    wb = load_workbook(out, read_only=True)
    for name in ("Disagreements", "Agreement control"):
        ws = wb[name]
        headers = [c.value for c in ws[1]]
        assert "candidate_id" not in headers
        assert not any(h and "haiku" in str(h) for h in headers)
    assert "Model outputs" in wb.sheetnames


def test_import_handles_blank_rows_without_corrupting():
    from scripts.bakeoff_import_labels import _import_sheet
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["content_id", "title", "abstract", "subhashini_application_domain", "subhashini_audience_relevance", "subhashini_paper_kind", "subhashini_reasoning"])
    ws.append([1, "t", "a", "general_method", "practitioner", "method", "clear general method"])
    ws.append([None, "", "", "", "", "", ""])
    conn = MagicMock()
    ins, sk = _import_sheet(conn, uuid4(), ws, ("subhashini",))
    assert ins == 1
    assert sk == 1


def test_cost_per_thousand_from_measured_tokens_not_static_table():
    rows = [{"tokens_in": 1000, "tokens_out": 200}, {"tokens_in": 1000, "tokens_out": 200}]
    c1k = cost_per_thousand_from_measured(rows, input_cost_per_million=1.0, output_cost_per_million=5.0)
    direct = cost_from_tokens(2000, 400, input_cost_per_million=1.0, output_cost_per_million=5.0) / 2 * 1000
    assert c1k == round(direct, 4)


def test_nothing_writes_to_content_classifications():
    conn = MagicMock()
    import_haiku_baseline(conn, uuid4(), [_classified_paper(1, ["general_method"])])
    for call in conn.execute.call_args_list:
        sql = call[0][0]
        assert "content_classifications" not in sql.lower()


def test_reconstruct_raw_response_roundtrip():
    paper = _classified_paper(9, ["general_method"])
    raw = reconstruct_raw_response(paper)
    parsed = analyze_raw_classify_response(raw, {9})[9]
    assert parsed.schema_valid is True
    assert parsed.application_domain == ["general_method"]


def test_estimate_bakeoff_cost_non_baseline_candidates_two_passes():
    config = load_bakeoff_config()
    est = estimate_bakeoff_run_cost(config, 400, n_passes=2)
    assert est["non_baseline_candidates"] == 5
    assert est["total_estimated_cost_usd"] > 0
    assert len(est["by_candidate"]) == 5


def test_load_config_matches_spec_candidates():
    config = load_bakeoff_config()
    ids = {c.id for c in config.candidates}
    assert ids == {"haiku", "gpt-4o-mini", "gemini-flash", "glm-flash", "gpt-luna", "qwen-max"}
    assert config.batch_size == 15
