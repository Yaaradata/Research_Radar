#!/usr/bin/env python3
"""Export human labelling workbook — no model outputs on labelling sheets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

from research_radar.bakeoff import (  # noqa: E402
    LABELLERS,
    agreement_control_ids,
    candidate_disagreement_rows,
    compute_candidate_metrics,
    load_bakeoff_config,
    load_human_labels,
    load_results_for_run,
    load_sample_papers,
    self_consistency_rate,
)
from research_radar.classification_vocab import APPLICATION_DOMAINS, AUDIENCE_RELEVANCE, PAPER_KINDS
from research_radar.pipeline import connect

REPORTS_DIR = ROOT / "reports"


def _add_enum_validation(ws, col_letter: str, start_row: int, end_row: int, values: tuple[str, ...]) -> None:
    formula = '"' + ",".join(values) + '"'
    dv = DataValidation(type="list", formula1=formula, allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{col_letter}{start_row}:{col_letter}{end_row}")


def _labelling_headers(labeller: str) -> list[str]:
    return [
        f"{labeller}_application_domain",
        f"{labeller}_audience_relevance",
        f"{labeller}_paper_kind",
        f"{labeller}_reasoning",
    ]


def _write_labelling_sheet(ws, papers: list[dict], labellers: tuple[str, ...]) -> None:
    headers = ["content_id", "title", "abstract"]
    for lab in labellers:
        headers.extend(_labelling_headers(lab))
    ws.append(headers)
    for p in papers:
        row = [p["content_id"], p.get("title", ""), p.get("abstract", "")]
        row.extend([""] * (4 * len(labellers)))
        ws.append(row)
    end_row = max(2, len(papers) + 1)
    col = 4
    for _lab in labellers:
        _add_enum_validation(ws, _col_letter(col), 2, end_row, APPLICATION_DOMAINS)
        _add_enum_validation(ws, _col_letter(col + 1), 2, end_row, AUDIENCE_RELEVANCE)
        _add_enum_validation(ws, _col_letter(col + 2), 2, end_row, PAPER_KINDS)
        col += 4


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _write_reference_sheet(ws) -> None:
    ws.append(["APPLICATION_DOMAINS"])
    for v in APPLICATION_DOMAINS:
        ws.append([v])
    ws.append([])
    ws.append(["AUDIENCE_RELEVANCE"])
    for v in AUDIENCE_RELEVANCE:
        ws.append([v])
    ws.append([])
    ws.append(["PAPER_KINDS"])
    for v in PAPER_KINDS:
        ws.append([v])


def _write_model_outputs_sheet(ws, results: list[dict], papers: list[dict]) -> None:
    ws.append(
        [
            "content_id",
            "candidate_id",
            "pass_index",
            "application_domain",
            "paper_kind",
            "schema_valid",
            "dropped_values",
        ]
    )
    for r in results:
        ws.append(
            [
                r["content_id"],
                r["candidate_id"],
                r["pass_index"],
                ",".join(r.get("application_domain") or []),
                r.get("paper_kind"),
                r.get("schema_valid"),
                ",".join(r.get("dropped_values") or []),
            ]
        )


def _candidate_order(results: list[dict]) -> list[str]:
    """Stable column order: haiku first, then remaining candidates alphabetically."""
    ids = sorted({str(r["candidate_id"]) for r in results})
    preferred = ["haiku", "gpt-4o-mini", "gemini-flash", "qwen-max"]
    ordered = [c for c in preferred if c in ids]
    ordered.extend(c for c in ids if c not in ordered)
    return ordered


def _write_model_comparison_sheet(ws, results: list[dict], papers: list[dict]) -> None:
    """One row per paper; pass-1 labels side-by-side per candidate (not for labelling)."""
    candidates = _candidate_order(results)
    headers = ["content_id", "title"]
    for cid in candidates:
        headers.extend([f"{cid}_domain", f"{cid}_paper_kind"])
    ws.append(headers)

    pass1: dict[tuple[int, str], dict] = {}
    for r in results:
        if int(r.get("pass_index") or 1) != 1:
            continue
        pass1[(int(r["content_id"]), str(r["candidate_id"]))] = r

    for p in papers:
        content_id = int(p["content_id"])
        row: list = [content_id, p.get("title", "")]
        for cid in candidates:
            r = pass1.get((content_id, cid))
            if r is None:
                row.extend(["", ""])
            else:
                row.append(",".join(r.get("application_domain") or []))
                row.append(r.get("paper_kind") or "")
        ws.append(row)


def compute_rates_rows(results: list[dict], labels: list[dict], config) -> list[dict]:
    """One metrics dict per candidate (pass-1 rates + self-consistency)."""
    cand_cfg = {c.id: c for c in config.candidates}
    rows_out: list[dict] = []
    for cid in _candidate_order(results):
        rows_p1 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 1]
        rows_p2 = [r for r in results if r["candidate_id"] == cid and int(r["pass_index"]) == 2]
        cfg = cand_cfg.get(cid)
        metrics = compute_candidate_metrics(
            rows_p1,
            labels,
            input_cost_per_million=cfg.input_cost_per_million if cfg else 1.0,
            output_cost_per_million=cfg.output_cost_per_million if cfg else 5.0,
        )
        p1 = {int(r["content_id"]): r for r in rows_p1}
        p2 = {int(r["content_id"]): r for r in rows_p2}
        sc = self_consistency_rate(p1, p2) if p2 else None
        metrics["candidate_id"] = cid
        metrics["model"] = cfg.model if cfg else ""
        metrics["self_consistency"] = round(sc, 4) if sc is not None else None
        rows_out.append(metrics)
    return rows_out


def _write_rates_sheet(ws, rate_rows: list[dict], *, labels_imported: bool) -> None:
    headers = [
        "candidate_id",
        "model",
        "n",
        "accuracy",
        "general_method_rate",
        "force_fit_rate",
        "invalid_rate",
        "valid_json_rate",
        "schema_valid_rate",
        "exclusivity_violations",
        "self_consistency",
        "cost_per_1000",
        "mean_latency_ms",
    ]
    ws.append(headers)
    for m in rate_rows:
        ws.append([m.get(h) for h in headers])
    ws.append([])
    if labels_imported:
        ws.append(["note", "accuracy and force_fit_rate use imported human labels"])
    else:
        ws.append(
            [
                "note",
                "accuracy and force_fit_rate blank until labels are imported; re-export or re-run rates update",
            ]
        )


def export_workbook(
    path: Path,
    *,
    disagreement_papers: list[dict],
    control_papers: list[dict],
    results: list[dict],
    all_sample: list[dict],
    rate_rows: list[dict] | None = None,
    labels_imported: bool = False,
) -> None:
    wb = Workbook()
    ws_dis = wb.active
    ws_dis.title = "Disagreements"
    _write_labelling_sheet(ws_dis, disagreement_papers, LABELLERS)

    ws_ctrl = wb.create_sheet("Agreement control")
    _write_labelling_sheet(ws_ctrl, control_papers, LABELLERS)

    ws_ref = wb.create_sheet("Reference")
    _write_reference_sheet(ws_ref)

    ws_models = wb.create_sheet("Model outputs")
    _write_model_outputs_sheet(ws_models, results, all_sample)

    ws_cmp = wb.create_sheet("Model comparison")
    _write_model_comparison_sheet(ws_cmp, results, all_sample)

    if rate_rows is not None:
        ws_rates = wb.create_sheet("Rates", 0)
        _write_rates_sheet(ws_rates, rate_rows, labels_imported=labels_imported)

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bake-off labelling workbook")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--control-n", type=int, default=30)
    args = parser.parse_args()

    run_id = UUID(args.run_id)
    config = load_bakeoff_config()
    out = Path(args.out) if args.out else REPORTS_DIR / f"bakeoff-labels-{run_id}.xlsx"

    with connect() as conn:
        all_sample = load_sample_papers(conn, run_id)
        results = load_results_for_run(conn, run_id)
        labels = load_human_labels(conn, run_id)
        by_id = {int(p["content_id"]): p for p in all_sample}
        disagree_ids = candidate_disagreement_rows(results)
        control_ids = agreement_control_ids(results, seed=config.sample_seed, n=args.control_n)
        disagreement_papers = [by_id[cid] for cid in sorted(disagree_ids) if cid in by_id]
        control_papers = [by_id[cid] for cid in control_ids if cid in by_id]

    rate_rows = compute_rates_rows(results, labels, config)
    export_workbook(
        out,
        disagreement_papers=disagreement_papers,
        control_papers=control_papers,
        results=results,
        all_sample=all_sample,
        rate_rows=rate_rows,
        labels_imported=bool(labels),
    )
    print(f"Wrote {out}")
    print(f"Disagreements: {len(disagreement_papers)} | Agreement control: {len(control_papers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
