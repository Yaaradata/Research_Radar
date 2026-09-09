#!/usr/bin/env python3
"""Generate date-bounded Tech and Product/Business newsletter shortlists."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from research_radar.pipeline import connect


TECH_AUDIENCES = {"practitioner", "technical_leadership"}
GENERAL_METHOD = "general_method"
NOTABLE_SHARE = 0.50
NOTABLE_RANK_WEIGHT = 1.50


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    return [str(item) for item in value]


def research_base(row: dict[str, Any]) -> float:
    return (
        float(row["semantic_core"])
        * float(row["evidence_factor"])
        * float(row["independence_factor"])
    )


def newsletter_base(row: dict[str, Any]) -> float:
    return float(row["newsletter_fit"] or 0) * float(row["evidence_factor"])


def breadth_bonus(row: dict[str, Any], *, product: bool) -> float:
    applications = as_list(row["application_domain"])
    if len(applications) > 1:
        return min(0.40, 0.30 + 0.10 * (len(applications) - 2))
    if not product and applications == [GENERAL_METHOD]:
        return 0.15
    return 0.0


def notable_priority(row: dict[str, Any]) -> float:
    """Return the verified organisation/person signal."""
    return float(row["org_boost"] or 0) + float(row["person_boost"] or 0)


def editorial_priority(row: dict[str, Any], priority_fn) -> float:
    return priority_fn(row) + NOTABLE_RANK_WEIGHT * notable_priority(row)


def select_rows(
    pool: list[dict[str, Any]],
    *,
    top: int,
    priority_fn,
) -> list[dict[str, Any]]:
    """Balance impact leaders with the strongest verified notable-affiliation papers."""
    notable = sorted(
        (row for row in pool if notable_priority(row) > 0),
        key=lambda row: (priority_fn(row), notable_priority(row)),
        reverse=True,
    )
    notable_target = min(len(notable), max(1, round(top * NOTABLE_SHARE)))
    selected = notable[:notable_target]
    selected_ids = {row["content_id"] for row in selected}

    impact_ranked = sorted(
        pool,
        key=lambda row: (
            priority_fn(row),
            float(row["newsletter_score"] or 0),
            float(row["final_score"]),
        ),
        reverse=True,
    )
    selected.extend(
        row for row in impact_ranked if row["content_id"] not in selected_ids
    )
    selected = selected[:top]
    return sorted(
        selected,
        key=lambda row: (
            editorial_priority(row, priority_fn),
            priority_fn(row),
        ),
        reverse=True,
    )


def tech_priority(row: dict[str, Any]) -> float:
    impact = 0.60 * research_base(row) + 0.40 * newsletter_base(row)
    return impact + breadth_bonus(row, product=False)


def product_priority(row: dict[str, Any]) -> float:
    evidence = float(row["evidence_factor"])
    independence = float(row["independence_factor"])
    practical = float(row["practical_applicability"] or 0) * evidence * independence
    professional = float(row["professional_value"] or 0) * evidence * independence
    impact = (
        0.20 * research_base(row)
        + 0.25 * newsletter_base(row)
        + 0.275 * practical
        + 0.275 * professional
    )
    return impact + breadth_bonus(row, product=True)


def is_tech_candidate(row: dict[str, Any]) -> bool:
    return bool(set(as_list(row["audience_relevance"])) & TECH_AUDIENCES)


def is_product_candidate(row: dict[str, Any]) -> bool:
    audiences = set(as_list(row["audience_relevance"]))
    applications = set(as_list(row["application_domain"]))
    has_named_sector = bool(applications - {GENERAL_METHOD})
    high_applied_value = (
        float(row["practical_applicability"] or 0) >= 8.0
        and float(row["professional_value"] or 0) >= 8.0
    )
    return "enterprise_adoption" in audiences or (
        has_named_sector and high_applied_value
    )


def load_candidates(date_from: date, date_until: date) -> list[dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute(
            """
            SELECT
                v.*,
                cc.application_domain,
                cc.audience_relevance,
                cc.paper_kind,
                cc.geography_focus
            FROM research_radar.v_research_radar_top v
            JOIN research_radar.content_classifications cc
              ON cc.content_id = v.content_id
            WHERE v.score_version = 'radar-v2'
              AND v.scoring_tier = 'full'
              AND v.published_at >= %s::date
              AND v.published_at < (%s::date + INTERVAL '1 day')
            """,
            (str(date_from), str(date_until)),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def label(value: str) -> str:
    return value.replace("_", " ").title()


def breadth_label(row: dict[str, Any]) -> str:
    applications = as_list(row["application_domain"])
    if len(applications) > 1:
        return f"Wide: {len(applications)} named sectors"
    if applications == [GENERAL_METHOD]:
        return "General method; no single sector asserted"
    if applications:
        return f"Focused: {label(applications[0])}"
    return "No application label"


def notable_label(row: dict[str, Any]) -> str:
    parts = []
    if float(row["org_boost"] or 0) > 0:
        parts.append(f"organisation +{float(row['org_boost']):.2f}")
    if float(row["person_boost"] or 0) > 0:
        parts.append(f"person +{float(row['person_boost']):.2f}")
    return ", ".join(parts) if parts else "none"


def render_report(
    rows: list[dict[str, Any]],
    *,
    audience_name: str,
    date_from: date,
    date_until: date,
    pool_size: int,
    notable_pool_size: int,
    priority_fn,
) -> str:
    if audience_name == "Tech":
        audience_rule = (
            "Practitioner or Technical Leadership classification."
        )
        score_rule = (
            "60% evidence-adjusted research impact + 40% evidence-adjusted "
            "newsletter fit, plus a modest application-breadth bonus."
        )
    else:
        audience_rule = (
            "Enterprise Adoption classification, or a named application sector with "
            "both Practical Applicability and Professional Value at least 8.0."
        )
        score_rule = (
            "20% evidence-adjusted research impact + 25% newsletter fit + 27.5% "
            "practical applicability + 27.5% professional value, plus an "
            "application-breadth bonus."
        )
    lines = [
        f"# TheNeural Newsletter — {audience_name} Top 20",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Hard publication window:** {date_from} through {date_until} (inclusive)  ",
        f"**Eligible audience-specific pool:** {pool_size} fully scored papers  ",
        f"**Verified notable-organisation candidates:** {notable_pool_size}  ",
        f"**Selected with a notable signal:** "
        f"{sum(notable_priority(row) > 0 for row in rows)} of {len(rows)}  ",
        f"**Audience rule:** {audience_rule}  ",
        f"**Primary-impact rule:** {score_rule}  ",
        "",
        "## Selection order",
        "",
        "1. Enforce the publication window and audience eligibility.",
        "2. Rank by broad applicability and high reader impact using stored research, "
        "newsletter-fit, practical-applicability and professional-value scores.",
        "3. Reserve up to half the list for the highest-impact papers with verified "
        "notable organisations/people; if fewer exist, include all available.",
        "4. Fill remaining places with the strongest breadth-and-impact papers, then "
        "order the selected set using the notable signal as a substantial secondary factor.",
        "",
        "Notable papers are chosen by impact within the verified-notable subset—not by "
        "affiliation alone. All listed organisations are evidence-backed paper-author "
        "affiliations. The people watchlist is currently empty, so the current notable "
        "signals come from organisations.",
        "",
        "---",
        "",
    ]

    for index, row in enumerate(rows, 1):
        audiences = ", ".join(label(x) for x in as_list(row["audience_relevance"]))
        applications = ", ".join(label(x) for x in as_list(row["application_domain"]))
        lines += [
            f"## {index}. {row['title']}",
            "",
            f"**Editorial priority:** {editorial_priority(row, priority_fn):.2f}  ",
            f"**Primary impact score:** {priority_fn(row):.2f}  ",
            f"**Application breadth:** {breadth_label(row)}  ",
            f"**Notable tie-break:** {notable_label(row)}  ",
            f"**Research score:** {float(row['final_score']):.2f}  ",
            f"**Newsletter score:** {float(row['newsletter_score']):.2f}  ",
            f"**Audience:** {audiences or 'Unclassified'}  ",
            f"**Application:** {applications or 'No application label'}  ",
            f"**Published:** {row['published_at']:%Y-%m-%d}  ",
        ]
        if row.get("arxiv_id"):
            lines.append(f"**arXiv:** `{row['arxiv_id']}`  ")
        if row.get("canonical_url"):
            lines.append(f"[{row['canonical_url']}]({row['canonical_url']})")
        lines += [
            "",
            f"**Why it matters:** {row.get('so_what') or 'No editorial summary stored.'}  ",
            f"**Limitation:** {row.get('reason_not_higher') or 'No limitation stored.'}  ",
            "",
        ]

        organisations = row.get("organisations") or []
        if isinstance(organisations, str):
            try:
                organisations = json.loads(organisations)
            except json.JSONDecodeError:
                organisations = []
        if organisations:
            names = []
            for organisation in organisations:
                name = organisation.get("organisation")
                if name and name not in names:
                    names.append(name)
            lines.append(f"**Notable organisations:** {', '.join(names)}")
        else:
            lines.append("**Notable organisations:** none verified")
        lines += ["", "---", ""]

    return "\n".join(lines)


def render_combined(
    tech_report: str,
    product_report: str,
    *,
    date_from: date,
    date_until: date,
    overlap: int,
    tech_notable: int,
    product_notable: int,
) -> str:
    tech_body = tech_report.replace(
        "# TheNeural Newsletter — Tech Top 20", "## Part I — Tech Top 20", 1
    )
    product_body = product_report.replace(
        "# TheNeural Newsletter — Product & Business Top 20",
        "## Part II — Product & Business Top 20",
        1,
    )
    header = [
        "# TheNeural Newsletter — Revised 20 + 20 Selection",
        "",
        f"**Hard publication window:** {date_from} through {date_until} (inclusive)  ",
        "**Deliverable:** 20 papers for Tech readers + 20 papers for Product/Business readers  ",
        f"**Selected with notable organisations:** Tech {tech_notable}/20; "
        f"Product/Business {product_notable}/20  ",
        f"**Cross-audience overlap:** {overlap} papers (retained where genuinely useful to both)  ",
        "",
        "## Editorial brief applied",
        "",
        "> First filter by start and end date. Then prioritize wide application areas "
        "and high impact from reading the paper. Next prioritize notable organisations "
        "and people. Select separately for Tech readers and Product/Business readers.",
        "",
        "The two lists are independently ranked, so overlap is allowed rather than "
        "replacing a strong cross-audience paper with a weaker one.",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + tech_body + "\n\n" + product_body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", type=date.fromisoformat, required=True)
    parser.add_argument("--until", dest="date_until", type=date.fromisoformat, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    candidates = load_candidates(args.date_from, args.date_until)
    tech_pool = [row for row in candidates if is_tech_candidate(row)]
    product_pool = [row for row in candidates if is_product_candidate(row)]

    tech = select_rows(tech_pool, top=args.top, priority_fn=tech_priority)
    product = select_rows(product_pool, top=args.top, priority_fn=product_priority)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.date_from}-to-{args.date_until}"
    tech_path = args.out_dir / f"newsletter-top20-tech-{suffix}.md"
    product_path = args.out_dir / f"newsletter-top20-product-business-{suffix}.md"
    combined_path = args.out_dir / f"newsletter-revised-20-plus-20-{suffix}.md"
    tech_report = render_report(
        tech,
        audience_name="Tech",
        date_from=args.date_from,
        date_until=args.date_until,
        pool_size=len(tech_pool),
        notable_pool_size=sum(notable_priority(row) > 0 for row in tech_pool),
        priority_fn=tech_priority,
    )
    product_report = render_report(
        product,
        audience_name="Product & Business",
        date_from=args.date_from,
        date_until=args.date_until,
        pool_size=len(product_pool),
        notable_pool_size=sum(notable_priority(row) > 0 for row in product_pool),
        priority_fn=product_priority,
    )
    overlap = len({row["content_id"] for row in tech} & {row["content_id"] for row in product})
    tech_path.write_text(tech_report + "\n")
    product_path.write_text(product_report + "\n")
    combined_path.write_text(
        render_combined(
            tech_report,
            product_report,
            date_from=args.date_from,
            date_until=args.date_until,
            overlap=overlap,
            tech_notable=sum(notable_priority(row) > 0 for row in tech),
            product_notable=sum(notable_priority(row) > 0 for row in product),
        )
    )
    print(f"Wrote {tech_path} ({len(tech)}/{len(tech_pool)} selected)")
    print(f"Wrote {product_path} ({len(product)}/{len(product_pool)} selected)")
    print(f"Wrote {combined_path}")
    print(f"Top-list overlap: {overlap}")


if __name__ == "__main__":
    main()
