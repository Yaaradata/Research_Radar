"""
Final Research Radar ranking.

Combines the stored LLM semantic assessment with verified deterministic
organisation/person signals. Makes ZERO API calls — reads only what is
already persisted, so it is free and safe to re-run.

Does NOT modify content_items.status. Candidate labelling remains a separate,
deliberately calibrated decision.
"""

from __future__ import annotations

import json
import logging
import os
import statistics as st
from datetime import datetime, timezone

from research_radar.independence import DEFAULT_INDEPENDENCE_FACTOR, INDEPENDENCE_FACTORS

log = logging.getLogger("research_radar.final_score")

# ---------------------------------------------------------------- weights

# ai_relevance is deliberately excluded from every ranking profile: the corpus
# is pre-filtered for relevance, so the dimension is near-constant and only
# compresses the scale. It remains stored and usable as a gate.
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    # Recommended. Novelty is up-weighted because it is what a research radar
    # is for, and it carries the most variance across the corpus.
    "radar-v1": {
        "technical_significance": 0.25,
        "apparent_novelty": 0.20,
        "practical_applicability": 0.20,
        "professional_value": 0.15,
        "student_learning_value": 0.10,
        "explainability": 0.05,
        "industry_relevance": 0.05,
    },
    # Mechanical renormalisation of the original weights minus ai_relevance.
    # Kept for A/B comparison against radar-v1.
    "renormalised-v0": {
        "technical_significance": 0.20 / 0.80,
        "practical_applicability": 0.15 / 0.80,
        "professional_value": 0.15 / 0.80,
        "student_learning_value": 0.10 / 0.80,
        "apparent_novelty": 0.10 / 0.80,
        "explainability": 0.05 / 0.80,
        "industry_relevance": 0.05 / 0.80,
    },
    # Scoring v2 (research-semantic-v2 quality assessments). Quality is a
    # weighted mean same as v1, but the final formula also multiplies by
    # evidence_factor and independence_factor — see compute_final_v2.
    "radar-v2": {
        "technical_significance": 0.28,
        "apparent_novelty": 0.24,
        "practical_applicability": 0.20,
        "professional_value": 0.16,
        "learning_value": 0.12,
    },
}

# Profiles that use the v2 formula (quality * evidence_factor * independence_factor
# + boosts, plus a separately-computed newsletter_score). Everything else uses the
# v1 additive formula (compute_final). Keep this in sync with WEIGHT_PROFILES.
V2_PROFILES = frozenset({"radar-v2"})

DEFAULT_PROFILE = os.getenv("FINAL_SCORE_PROFILE", "radar-v1").strip() or "radar-v1"

# Boost caps. Organisation must never dominate research quality (brief §10,
# golden test #2). 0.5 reorders within a narrow band and nothing more.
ORG_BOOST_MAX = float(os.getenv("FINAL_SCORE_ORG_BOOST_MAX", "0.5"))
PERSON_BOOST_MAX = float(os.getenv("FINAL_SCORE_PERSON_BOOST_MAX", "0.3"))
MIN_EVIDENCE_CONFIDENCE = float(os.getenv("FINAL_SCORE_MIN_EVIDENCE_CONFIDENCE", "0.90"))

DIMENSIONS = sorted({d for p in WEIGHT_PROFILES.values() for d in p})

DIMENSION_LABEL = {
    "technical_significance": "Technical significance",
    "apparent_novelty": "Apparent novelty",
    "practical_applicability": "Practical applicability",
    "professional_value": "Professional value",
    "student_learning_value": "Student learning value",
    "explainability": "Explainability",
    "industry_relevance": "Industry relevance",
    "learning_value": "Learning value",
    "ai_relevance": "AI relevance (gate only)",
    "evidence_strength": "Evidence strength (evidence_factor input, gate only)",
    "newsletter_fit": "Newsletter fit (newsletter_score only)",
}


class UnknownProfileError(ValueError):
    pass


def resolve_profile(name: str | None) -> tuple[str, dict[str, float]]:
    name = (name or DEFAULT_PROFILE).strip()
    if name not in WEIGHT_PROFILES:
        raise UnknownProfileError(
            f"Unknown weight profile '{name}'. Available: {sorted(WEIGHT_PROFILES)}"
        )
    return name, WEIGHT_PROFILES[name]


def clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, round(float(x), 2)))


def compute_semantic_core(dims: dict, weights: dict[str, float]) -> float:
    """Weighted mean over available dimensions, renormalised if any are NULL."""
    total_w = 0.0
    acc = 0.0
    for key, w in weights.items():
        val = dims.get(key)
        if val is None:
            continue
        acc += w * float(val)
        total_w += w
    if total_w <= 0:
        raise ValueError("no scored dimensions available")
    return clamp(acc / total_w)


def org_boost_for(rows: list[dict]) -> tuple[float, list[dict]]:
    """
    Boost from verified paper-affiliation watchlist organisations only.

    Requires current_affiliation = FALSE (a current employer is not a paper
    affiliation, brief §9) and confidence >= MIN_EVIDENCE_CONFIDENCE.
    """
    qualifying = [
        r for r in rows if float(r.get("confidence") or 0) >= MIN_EVIDENCE_CONFIDENCE
    ]
    if not qualifying:
        return 0.0, []
    top_priority = max(int(r.get("priority") or 0) for r in qualifying)
    boost = round(ORG_BOOST_MAX * (top_priority / 10.0), 2)
    return boost, qualifying


def person_boost_for(rows: list[dict]) -> tuple[float, list[dict]]:
    """Structurally present; returns 0 while the people watchlist is empty."""
    if not rows:
        return 0.0, []
    top_priority = max(int(r.get("priority") or 0) for r in rows)
    return round(PERSON_BOOST_MAX * (top_priority / 10.0), 2), rows


def compute_final(
    dims: dict,
    org_rows: list[dict],
    person_rows: list[dict],
    weights: dict[str, float],
) -> dict:
    core = compute_semantic_core(dims, weights)
    ob, org_detail = org_boost_for(org_rows)
    pb, person_detail = person_boost_for(person_rows)
    return {
        "semantic_core": core,
        "org_boost": ob,
        "person_boost": pb,
        "final_score": clamp(core + ob + pb),
        "org_detail": org_detail,
        "person_detail": person_detail,
    }


def evidence_factor_for(evidence_strength) -> float:
    """0.70 (no evidence / NULL) to 1.00 (evidence_strength == 10)."""
    es = float(evidence_strength) if evidence_strength is not None else 0.0
    es = max(0.0, min(10.0, es))
    return round(0.70 + 0.03 * es, 3)


def independence_factor_for(status: str | None) -> float:
    """Missing/unrecognised classification defaults to 'unclear' — unknown is
    not the same as independent (brief design decision 3)."""
    return INDEPENDENCE_FACTORS.get(status or "", DEFAULT_INDEPENDENCE_FACTOR)


def compute_final_v2(
    dims: dict,
    org_rows: list[dict],
    person_rows: list[dict],
    weights: dict[str, float],
) -> dict:
    """
    research_score = min(10, quality * evidence_factor * independence_factor + boosts)
    newsletter_score = min(10, newsletter_fit * evidence_factor + org_boost)

    Both are capped at 10.0 via clamp() — quality/newsletter_fit are each
    already <= 10 and both factors are <= 1.0, so only the additive boosts can
    push a value over the cap.
    """
    quality = compute_semantic_core(dims, weights)
    evidence_factor = evidence_factor_for(dims.get("evidence_strength"))
    independence_status = dims.get("independence_status")
    independence_factor = independence_factor_for(independence_status)
    ob, org_detail = org_boost_for(org_rows)
    pb, person_detail = person_boost_for(person_rows)

    research_score = clamp(quality * evidence_factor * independence_factor + ob + pb)
    newsletter_fit = dims.get("newsletter_fit")
    newsletter_fit = float(newsletter_fit) if newsletter_fit is not None else 0.0
    newsletter_score = clamp(newsletter_fit * evidence_factor + ob)

    return {
        "semantic_core": quality,
        "org_boost": ob,
        "person_boost": pb,
        "final_score": research_score,
        "newsletter_score": newsletter_score,
        "evidence_factor": evidence_factor,
        "independence_factor": independence_factor,
        "independence_status": independence_status,
        "org_detail": org_detail,
        "person_detail": person_detail,
    }


# ------------------------------------------------------------------ load


_SCORED_PAPERS_SELECT = """
    SELECT
        a.assessment_id,
        a.content_id,
        a.technical_significance,
        a.practical_applicability,
        a.professional_value,
        a.student_learning_value,
        a.apparent_novelty,
        a.explainability,
        a.industry_relevance,
        a.learning_value,
        a.evidence_strength,
        a.newsletter_fit,
        a.so_what,
        a.reason_not_higher,
        a.confidence,
        a.ai_relevance,
        a.semantic_score,
        a.model_name,
        a.prompt_version,
        ind.independence_status,
        ind.independence_reason,
        ci.title
    FROM research_radar.content_score_assessments a
    JOIN research_radar.content_items ci ON ci.id = a.content_id
    LEFT JOIN LATERAL (
        SELECT cia.status AS independence_status, cia.reason AS independence_reason
        FROM research_radar.content_independence_assessments cia
        WHERE cia.content_id = a.content_id AND cia.status <> 'ERROR'
        ORDER BY cia.created_at DESC
        LIMIT 1
    ) ind ON TRUE
    WHERE a.status = 'COMPLETED'
      AND (a.scoring_tier IS NULL OR a.scoring_tier = 'full')
"""
# scoring_tier IS NULL keeps v1 rows (v1 never sets scoring_tier) and any
# other untiered rows; scoring_tier = 'full' keeps v2 pass-2 rows. Screen-tier
# rows are permanently excluded here — two models, two scales, never mixed
# into one ranking (tiering brief §2). This is the ONLY place that matters:
# final_score computes only for scoring_tier='full' papers, so screen-only
# papers get no final_score row at all.


def load_scored_papers(conn, *, prompt_version: str | None = None) -> list[dict]:
    """Papers with a COMPLETED semantic assessment. ERROR/missing are skipped."""
    if prompt_version:
        rows = conn.execute(
            _SCORED_PAPERS_SELECT + "  AND a.prompt_version = %s\n  ORDER BY a.content_id",
            (prompt_version,),
        ).fetchall()
    else:
        rows = conn.execute(_SCORED_PAPERS_SELECT + "  ORDER BY a.content_id").fetchall()
    return [dict(r) for r in rows]


def load_org_rows(conn, content_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT o.canonical_name, o.priority, co.evidence_type,
                   co.evidence_text, co.confidence
            FROM research_radar.content_organisations co
            JOIN research_radar.organisations o
              ON o.organisation_id = co.organisation_id
            WHERE co.content_id = %s
              AND co.relationship_type = 'paper_author_affiliation'
              AND co.current_affiliation = FALSE
            """,
            (content_id,),
        ).fetchall()
    ]


def load_person_rows(conn, content_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT p.canonical_name, p.priority, cp.match_confidence
            FROM research_radar.content_people cp
            JOIN research_radar.people p ON p.person_id = cp.person_id
            WHERE cp.content_id = %s AND cp.is_notable = TRUE
            """,
            (content_id,),
        ).fetchall()
    ]


def upsert_final_score(conn, *, content_id, score_version, assessment_id, result, weights):
    conn.execute(
        """
        INSERT INTO research_radar.content_final_scores(
            content_id, score_version, semantic_core, org_boost, person_boost,
            final_score, newsletter_score, evidence_factor, independence_factor,
            independence_status, assessment_id, components, provenance, computed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,NOW())
        ON CONFLICT (content_id, score_version) DO UPDATE SET
            semantic_core        = EXCLUDED.semantic_core,
            org_boost            = EXCLUDED.org_boost,
            person_boost         = EXCLUDED.person_boost,
            final_score          = EXCLUDED.final_score,
            newsletter_score     = EXCLUDED.newsletter_score,
            evidence_factor      = EXCLUDED.evidence_factor,
            independence_factor  = EXCLUDED.independence_factor,
            independence_status  = EXCLUDED.independence_status,
            assessment_id        = EXCLUDED.assessment_id,
            components           = EXCLUDED.components,
            provenance           = EXCLUDED.provenance,
            computed_at          = NOW()
        """,
        (
            content_id,
            score_version,
            result["semantic_core"],
            result["org_boost"],
            result["person_boost"],
            result["final_score"],
            result.get("newsletter_score"),
            result.get("evidence_factor"),
            result.get("independence_factor"),
            result.get("independence_status"),
            assessment_id,
            json.dumps(
                {
                    "org_evidence": result["org_detail"],
                    "person_evidence": result["person_detail"],
                },
                default=str,
            ),
            json.dumps(
                {
                    "method": (
                        "quality_times_evidence_times_independence_plus_verified_signals"
                        if score_version in V2_PROFILES
                        else "semantic_core_plus_verified_signals"
                    ),
                    "score_version": score_version,
                    "weights": weights,
                    "excluded_dimensions": ["ai_relevance"],
                    "excluded_reason": "near-constant post relevance-filter; compresses scale",
                    "org_boost_max": ORG_BOOST_MAX,
                    "person_boost_max": PERSON_BOOST_MAX,
                    "min_evidence_confidence": MIN_EVIDENCE_CONFIDENCE,
                    "freshness_included": False,
                },
                default=str,
            ),
        ),
    )


def stage_final_score(conn, run_id, *, profile=None, limit=None, dry_run=False):
    score_version, weights = resolve_profile(profile)
    papers = load_scored_papers(conn)
    if limit:
        papers = papers[:limit]
    log.info(
        "Final score: %d papers profile=%s dry_run=%s",
        len(papers),
        score_version,
        dry_run,
    )

    is_v2 = score_version in V2_PROFILES
    written = skipped = 0
    for p in papers:
        dims = {k: p.get(k) for k in DIMENSIONS}
        if all(dims.get(k) is None for k in weights):
            skipped += 1
            continue
        if is_v2:
            dims["evidence_strength"] = p.get("evidence_strength")
            dims["newsletter_fit"] = p.get("newsletter_fit")
            dims["independence_status"] = p.get("independence_status")
            result = compute_final_v2(
                dims,
                load_org_rows(conn, p["content_id"]),
                load_person_rows(conn, p["content_id"]),
                weights,
            )
        else:
            result = compute_final(
                dims,
                load_org_rows(conn, p["content_id"]),
                load_person_rows(conn, p["content_id"]),
                weights,
            )
        if not dry_run:
            upsert_final_score(
                conn,
                content_id=p["content_id"],
                score_version=score_version,
                assessment_id=p["assessment_id"],
                result=result,
                weights=weights,
            )
        written += 1
    if not dry_run:
        conn.commit()
    log.info("Final score: written=%d skipped=%d", written, skipped)
    return {"written": written, "skipped": skipped, "score_version": score_version}


def print_distribution(conn, *, profile=None):
    score_version, weights = resolve_profile(profile)
    papers = load_scored_papers(conn)

    print(f"\n=== Dimension spread (n={len(papers)}) ===")
    print(f"{'dimension':<26}{'mean':>8}{'stdev':>8}{'min':>7}{'max':>7}")
    for dim in DIMENSIONS:
        vals = [float(p[dim]) for p in papers if p.get(dim) is not None]
        if len(vals) < 2:
            continue
        flag = "  <-- in profile" if dim in weights else ""
        print(
            f"{dim:<26}{st.mean(vals):>8.2f}{st.stdev(vals):>8.2f}"
            f"{min(vals):>7.1f}{max(vals):>7.1f}{flag}"
        )

    print("\n=== Composite comparison ===")
    for name, w in WEIGHT_PROFILES.items():
        vals = []
        for p in papers:
            try:
                vals.append(compute_semantic_core({k: p.get(k) for k in DIMENSIONS}, w))
            except ValueError:
                continue
        if len(vals) < 2:
            continue
        vals.sort()

        def pct(q, _vals=vals):
            return _vals[min(len(_vals) - 1, int(len(_vals) * q))]

        print(f"\n{name}: mean={st.mean(vals):.2f} stdev={st.stdev(vals):.2f}")
        print(
            f"  p50={pct(0.50):.2f}  p75={pct(0.75):.2f}  "
            f"p90={pct(0.90):.2f}  p95={pct(0.95):.2f}  p99={pct(0.99):.2f}"
        )
        top20_cut = vals[-20] if len(vals) >= 20 else vals[0]
        print(f"  top20 cutoff = {top20_cut:.2f}")
        if name == score_version:
            print("  <-- selected profile")

    legacy = [
        float(p["semantic_score"])
        for p in papers
        if p.get("semantic_score") is not None
    ]
    if len(legacy) >= 2:
        legacy.sort()
        print(
            f"\nlegacy semantic_score (includes ai_relevance): "
            f"mean={st.mean(legacy):.2f} stdev={st.stdev(legacy):.2f}"
        )
        print(
            f"  p50={legacy[int(len(legacy) * 0.50)]:.2f}  "
            f"p90={legacy[int(len(legacy) * 0.90)]:.2f}  "
            f"p95={legacy[int(len(legacy) * 0.95)]:.2f}"
        )


RANK_BY_ORDER = {
    "research": "final_score DESC, published_at DESC NULLS LAST",
    "newsletter": "newsletter_score DESC NULLS LAST, final_score DESC, published_at DESC NULLS LAST",
}


def load_top(conn, *, profile=None, top=20, since_days=None, rank_by="research"):
    score_version, _ = resolve_profile(profile)
    if rank_by not in RANK_BY_ORDER:
        raise ValueError(f"Unknown --rank-by '{rank_by}'. Available: {sorted(RANK_BY_ORDER)}")
    order_sql = RANK_BY_ORDER[rank_by]
    return [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT * FROM research_radar.v_research_radar_top
            WHERE score_version = %s
              AND (scoring_tier IS NULL OR scoring_tier = 'full')
              AND (%s::int IS NULL
                   OR published_at >= NOW() - (%s::int || ' days')::interval)
            ORDER BY {order_sql}
            LIMIT %s
            """,
            (score_version, since_days, since_days, top),
        ).fetchall()
    ]


def count_ranked(conn, *, profile=None) -> int:
    score_version, _ = resolve_profile(profile)
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM research_radar.content_final_scores
        WHERE score_version = %s
        """,
        (score_version,),
    ).fetchone()
    return int(row["n"] if row else 0)


def _dimension_score(row: dict, dim: str):
    val = row.get(dim)
    if val is None:
        return "—"
    return f"{float(val):.1f}"


def _dimension_reason(reasons, dim: str) -> str:
    block = (reasons or {}).get(dim) if isinstance(reasons, dict) else None
    if block is None:
        return "—"
    if isinstance(block, dict):
        why = block.get("reason") or "—"
    else:
        why = str(block)
    return why.replace("|", "\\|")


def render_markdown(rows, *, profile, since_days=None, corpus_n=None, rank_by="research", pool=None) -> str:
    score_version, weights = resolve_profile(profile)
    window = f"last {since_days} days" if since_days else "full corpus"
    out = [
        "# TheNeural Research Radar — Top Papers",
        "",
        f"**Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"**Window:** {window}  ",
        f"**Ranking:** `{score_version}` (rank by {rank_by})  ",
    ]
    if corpus_n:
        out.append(f"**Ranked from:** {corpus_n} semantically assessed papers  ")
    if pool:
        out.append(
            f"**Scoring pool:** {pool.get('screened', 0)} screened, "
            f"{pool.get('full', 0)} fully scored — this Top N is drawn from the "
            f"fully-scored pool only, never the screened-only pool  "
        )
    out += [
        "",
        "Scores are LLM assessments of the paper's own title and abstract. The model "
        "never sees author, organisation or watchlist data, so it judges the work "
        "rather than the institution. Organisation is an evidence-backed boost "
        "capped at "
        f"{ORG_BOOST_MAX}, never a filter.",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(rows, 1):
        reasons = r.get("reasons") or {}
        out.append(f"## {i}. {r['title']}")
        out.append("")
        line = (
            f"**{float(r['final_score']):.2f}** — quality {float(r['semantic_core']):.2f}"
        )
        if r.get("evidence_factor") is not None:
            line += f" × evidence {float(r['evidence_factor']):.2f}"
        if r.get("independence_factor") is not None:
            line += f" × independence {float(r['independence_factor']):.2f}"
        if float(r["org_boost"]) > 0:
            line += f" + org {float(r['org_boost']):.2f}"
        if float(r["person_boost"]) > 0:
            line += f" + person {float(r['person_boost']):.2f}"
        out.append(line + "  ")
        if r.get("newsletter_score") is not None:
            out.append(f"Newsletter score: **{float(r['newsletter_score']):.2f}**  ")
        if r.get("independence_status"):
            out.append(f"Independence: `{r['independence_status']}`  ")
        if r.get("published_at"):
            out.append(f"Published: {r['published_at']:%Y-%m-%d}  ")
        if r.get("arxiv_id"):
            out.append(f"arXiv: `{r['arxiv_id']}`  ")
        if r.get("canonical_url"):
            out.append(f"[{r['canonical_url']}]({r['canonical_url']})")
        out.append("")

        out.append("| Dimension | Score | Assessment |")
        out.append("|---|---|---|")
        for dim in weights:
            score_cell = _dimension_score(r, dim)
            why = _dimension_reason(reasons, dim)
            out.append(
                f"| {DIMENSION_LABEL.get(dim, dim)} | {score_cell} | {why} |"
            )
        out.append("")

        if r.get("so_what"):
            out.append(f"**So what:** {r['so_what']}  ")
        if r.get("reason_not_higher"):
            out.append(f"**Why not higher:** {r['reason_not_higher']}  ")
        if r.get("so_what") or r.get("reason_not_higher"):
            out.append("")

        orgs = r.get("organisations") or []
        if isinstance(orgs, str):
            try:
                orgs = json.loads(orgs)
            except json.JSONDecodeError:
                orgs = []
        if orgs:
            out.append("**Organisations (evidence-backed):**")
            for o in orgs:
                out.append(
                    f"- {o['organisation']} — {o['evidence_type']}: "
                    f"`{o.get('evidence_text') or 'n/a'}` (confidence {o['confidence']})"
                )
            out.append("")
        else:
            out.append("**Organisations:** none resolved from paper evidence")
            out.append("")

        labels = r.get("industry_labels") or []
        if isinstance(labels, str):
            try:
                labels = json.loads(labels)
            except json.JSONDecodeError:
                labels = []
        if labels:
            out.append(f"**Industry:** {', '.join(str(x) for x in labels)}")
            out.append("")
        out.append("---")
        out.append("")

    return "\n".join(out)
