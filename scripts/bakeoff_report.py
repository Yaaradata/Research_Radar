#!/usr/bin/env python3
"""Generate markdown bake-off metrics report from stored results + human labels."""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from research_radar.bakeoff import (  # noqa: E402
    batch_stability_passes,
    compute_accuracy,
    compute_force_fit_rate,
    compute_general_method_rate,
    compute_valid_json_rate,
    cost_per_thousand_from_measured,
    inter_labeller_agreement,
    is_general_method,
    load_bakeoff_config,
    self_consistency_passes,
)
from research_radar.pipeline import connect

REPORTS_DIR = ROOT / "reports"


def _rows_by_content(results: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in results:
        if r.get("domain") is None and r.get("application_domains") is None:
            continue
        apps = r.get("application_domains")
        if isinstance(apps, str):
            apps = json.loads(apps)
        out[int(r["content_id"])] = {
            "domain": r.get("domain"),
            "subdomains": r.get("subdomains"),
            "application_domains": apps or [],
            "primary_audience": r.get("primary_audience"),
            "ai_relevance": float(r["ai_relevance"]) if r.get("ai_relevance") is not None else None,
        }
    return out


def _human_labels(conn, run_id: str) -> dict[int, dict]:
    rows = conn.execute(
        """
        SELECT content_id, labeller, domain, subdomains, application_domains,
               is_general_method, reasoning
        FROM research_radar.bakeoff_labels
        WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    # Use majority vote per content_id when multiple labellers
    by_cid: dict[int, list[dict]] = {}
    for r in rows:
        by_cid.setdefault(int(r["content_id"]), []).append(dict(r))
    out: dict[int, dict] = {}
    for cid, labs in by_cid.items():
        domains = [l["domain"] for l in labs if l.get("domain")]
        gm_votes = [bool(l.get("is_general_method")) for l in labs if l.get("is_general_method") is not None]
        out[cid] = {
            "domain": max(set(domains), key=domains.count) if domains else None,
            "is_general_method": sum(gm_votes) > len(gm_votes) / 2 if gm_votes else None,
            "labellers": labs,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Bake-off metrics report")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_id = args.run_id
    config = load_bakeoff_config()

    with connect() as conn:
        human = _human_labels(conn, run_id)
        candidates = [c.id for c in config.candidates]
        lines = [
            f"# Classification bake-off report",
            "",
            f"**run_id:** `{run_id}`  ",
            f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
            f"**Human-labelled papers:** {len(human)}  ",
            "",
            "| Candidate | Accuracy (domain) | GM rate | Force-fit rate | Other/invalid | Valid JSON | Cost/1k | Self-consist | Batch stab | Mean latency ms |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for cid in candidates:
            pass1 = conn.execute(
                """
                SELECT * FROM research_radar.bakeoff_results
                WHERE run_id = %s AND candidate_id = %s AND pass_index = 1 AND batch_arrangement = 'A'
                """,
                (run_id, cid),
            ).fetchall()
            pass2 = conn.execute(
                """
                SELECT * FROM research_radar.bakeoff_results
                WHERE run_id = %s AND candidate_id = %s AND pass_index = 2 AND batch_arrangement = 'A'
                """,
                (run_id, cid),
            ).fetchall()
            pass_b = conn.execute(
                """
                SELECT * FROM research_radar.bakeoff_results
                WHERE run_id = %s AND candidate_id = %s AND pass_index = 1 AND batch_arrangement = 'B'
                """,
                (run_id, cid),
            ).fetchall()

            rows1 = _rows_by_content([dict(r) for r in pass1])
            rows2 = _rows_by_content([dict(r) for r in pass2])
            rowsb = _rows_by_content([dict(r) for r in pass_b])

            acc = compute_accuracy(rows1, human, field="domain") if human else float("nan")
            gm_rate = compute_general_method_rate(rows1)
            human_gm = {k: v for k, v in human.items() if v.get("is_general_method") is not None}
            ff_rate = compute_force_fit_rate(rows1, human_gm) if human_gm else float("nan")
            other_rate = sum(1 for r in rows1.values() if (r.get("domain") or "").strip() == "Other") / max(
                1, len(rows1)
            )
            valid_json = compute_valid_json_rate([dict(r) for r in pass1])
            total_cost = sum(float(r.get("cost_usd") or 0) for r in pass1)
            cost_1k = cost_per_thousand_from_measured(total_cost, len(pass1))
            sc_ok = self_consistency_passes(rows1, rows2)
            bs_ok = batch_stability_passes(rows1, rowsb)
            latencies = [int(r["latency_ms"]) for r in pass1 if r.get("latency_ms")]
            mean_lat = int(st.mean(latencies)) if latencies else 0

            lines.append(
                f"| {cid} | {acc:.1%} | {gm_rate:.1%} | {ff_rate:.1%} | {other_rate:.1%} | "
                f"{valid_json:.1%} | ${cost_1k:.4f} | {'PASS' if sc_ok else 'FAIL'} | "
                f"{'PASS' if bs_ok else 'FAIL'} | {mean_lat} |"
            )

        lines.extend(["", "## Per-candidate detail", ""])
        for cid in candidates:
            lines.append(f"### {cid}")
            pass1 = conn.execute(
                """
                SELECT domain, application_domains, primary_audience, ai_relevance
                FROM research_radar.bakeoff_results
                WHERE run_id = %s AND candidate_id = %s AND pass_index = 1 AND batch_arrangement = 'A'
                """,
                (run_id, cid),
            ).fetchall()
            domains: dict[str, int] = {}
            audiences: dict[str, int] = {}
            for r in pass1:
                d = r.get("domain") or "(null)"
                domains[d] = domains.get(d, 0) + 1
                a = r.get("primary_audience") or "(null)"
                audiences[a] = audiences.get(a, 0) + 1
            lines.append("Domain distribution:")
            for k, v in sorted(domains.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"- {k}: {v}")
            lines.append("")

        if human:
            label_rows = conn.execute(
                "SELECT * FROM research_radar.bakeoff_labels WHERE run_id = %s",
                (run_id,),
            ).fetchall()
            agreement = inter_labeller_agreement([dict(r) for r in label_rows], field="domain")
            lines.append(f"Inter-labeller domain agreement: {agreement:.1%}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"bakeoff-{run_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
