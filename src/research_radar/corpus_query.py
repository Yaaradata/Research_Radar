"""Free, pure-SQL query interface over the topics/claims corpus.

This is the point of the topics stage: once every RELEVANT-or-later paper
carries a domain/subdomains/topics/applications and grounded claims, the
corpus should answer "show me papers on X", "how is this applied in Y",
"what results have we seen in Z" without another model call. Zero API calls,
ever — filters combine with AND.
"""

from __future__ import annotations

import json


def _tag_condition(alias_ci: str, param_name: str) -> str:
    """content_id filter clause matching a topic by canonical name OR alias,
    case-insensitively. `param_name` is bound twice."""
    return (
        f"EXISTS (SELECT 1 FROM research_radar.content_topics ct "
        f"JOIN research_radar.topics t ON t.topic_id = ct.topic_id "
        f"WHERE ct.content_id = {alias_ci}.id "
        f"AND (lower(t.canonical_name) = lower(%({param_name})s) "
        f"OR lower(%({param_name})s) = ANY(SELECT lower(a) FROM unnest(t.aliases) a)))"
    )


def _level_condition(alias_ci: str, param_name: str, level: str) -> str:
    return (
        f"EXISTS (SELECT 1 FROM research_radar.content_topics ct "
        f"JOIN research_radar.topics t ON t.topic_id = ct.topic_id "
        f"WHERE ct.content_id = {alias_ci}.id AND t.level = '{level}' "
        f"AND lower(t.canonical_name) = lower(%({param_name})s))"
    )


def search_corpus(
    conn,
    *,
    tag: str | None = None,
    subdomain: str | None = None,
    application: str | None = None,
    domain: str | None = None,
    with_claims: bool = False,
    since: str | None = None,
    top: int = 20,
) -> list[dict]:
    filters = ["ci.status <> 'REJECTED'"]
    params: dict = {}

    if tag:
        filters.append(_tag_condition("ci", "tag"))
        params["tag"] = tag
    if subdomain:
        filters.append(_level_condition("ci", "subdomain", "subdomain"))
        params["subdomain"] = subdomain
    if application:
        filters.append(_level_condition("ci", "application", "application"))
        params["application"] = application
    if domain:
        filters.append(_level_condition("ci", "domain", "domain"))
        params["domain"] = domain
    if with_claims:
        filters.append("EXISTS (SELECT 1 FROM research_radar.content_claims cc WHERE cc.content_id = ci.id)")
    if since:
        filters.append("ci.published_at >= %(since)s::timestamptz")
        params["since"] = since

    params["top"] = top
    sql = f"""
        SELECT
            ci.id AS content_id,
            ci.title,
            ci.canonical_url,
            ci.published_at,
            fs.final_score,
            (SELECT t.canonical_name FROM research_radar.content_topics ct
             JOIN research_radar.topics t ON t.topic_id = ct.topic_id
             WHERE ct.content_id = ci.id AND t.level = 'domain' LIMIT 1) AS domain,
            (SELECT array_agg(t.canonical_name) FROM research_radar.content_topics ct
             JOIN research_radar.topics t ON t.topic_id = ct.topic_id
             WHERE ct.content_id = ci.id AND t.level = 'subdomain') AS subdomains,
            (SELECT array_agg(t.canonical_name) FROM research_radar.content_topics ct
             JOIN research_radar.topics t ON t.topic_id = ct.topic_id
             WHERE ct.content_id = ci.id AND t.level = 'topic') AS tags,
            (SELECT array_agg(t.canonical_name) FROM research_radar.content_topics ct
             JOIN research_radar.topics t ON t.topic_id = ct.topic_id
             WHERE ct.content_id = ci.id AND t.level = 'application') AS applications
        FROM research_radar.content_items ci
        LEFT JOIN LATERAL (
            SELECT final_score FROM research_radar.content_final_scores
            WHERE content_id = ci.id ORDER BY computed_at DESC LIMIT 1
        ) fs ON TRUE
        WHERE {' AND '.join(filters)}
        ORDER BY fs.final_score DESC NULLS LAST, ci.published_at DESC NULLS LAST
        LIMIT %(top)s
    """
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    if with_claims:
        for row in rows:
            row["claims"] = load_claims_for_content(conn, row["content_id"])
    return rows


def load_claims_for_content(conn, content_id: int) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            """
            SELECT metric, value_text, value_num, unit, task, dataset, qualifier, evidence
            FROM research_radar.content_claims
            WHERE content_id = %s
            ORDER BY claim_id
            """,
            (content_id,),
        ).fetchall()
    ]


def claims_for_metric(conn, metric: str, *, top: int = 50) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            cc.content_id, ci.title, ci.canonical_url, ci.published_at,
            cc.metric, cc.value_text, cc.value_num, cc.unit, cc.task, cc.dataset,
            cc.qualifier, cc.evidence
        FROM research_radar.content_claims cc
        JOIN research_radar.content_items ci ON ci.id = cc.content_id
        WHERE lower(cc.metric) = lower(%s)
        ORDER BY cc.value_num DESC NULLS LAST, ci.published_at DESC NULLS LAST
        LIMIT %s
        """,
        (metric, top),
    ).fetchall()
    return [dict(r) for r in rows]


def list_topics(conn, *, min_usage: int | None = None, level: str | None = None) -> list[dict]:
    filters = ["1=1"]
    params: list = []
    if min_usage is not None:
        filters.append("usage_count >= %s")
        params.append(min_usage)
    if level:
        filters.append("level = %s")
        params.append(level)
    sql = f"""
        SELECT topic_id, canonical_name, level, origin, usage_count, aliases
        FROM research_radar.topics
        WHERE {' AND '.join(filters)}
        ORDER BY level, usage_count DESC, canonical_name
    """
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------------
# CLI-facing rendering
# ---------------------------------------------------------------------------


def render_results_table(rows: list[dict]) -> str:
    if not rows:
        return "No results."
    lines = []
    for r in rows:
        score = f"{r['final_score']:.2f}" if r.get("final_score") is not None else "—"
        published = r["published_at"].date().isoformat() if r.get("published_at") else "—"
        lines.append(f"[{score}] {r['title']}  ({published})")
        lines.append(f"    {r['canonical_url']}")
        lines.append(f"    domain: {r.get('domain') or '—'}")
        if r.get("subdomains"):
            lines.append(f"    subdomains: {', '.join(r['subdomains'])}")
        if r.get("tags"):
            lines.append(f"    tags: {', '.join(r['tags'])}")
        if r.get("applications"):
            lines.append(f"    applications: {', '.join(r['applications'])}")
        for c in r.get("claims") or []:
            lines.append(f"    claim: {c['metric']} = {c['value_text']} ({c.get('task') or 'n/a'}, {c['qualifier']})")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_topics_table(rows: list[dict]) -> str:
    if not rows:
        return "No topics."
    lines = [f"{'level':<11} {'usage':>6}  canonical_name"]
    for r in rows:
        lines.append(f"{r['level']:<11} {r['usage_count']:>6}  {r['canonical_name']}")
    return "\n".join(lines)


def render_claims_table(rows: list[dict]) -> str:
    if not rows:
        return "No claims."
    lines = []
    for r in rows:
        lines.append(f"{r['value_text']:>15}  {r['title']}  ({r.get('task') or 'n/a'}, {r['qualifier']})")
        lines.append(f"    {r['canonical_url']}")
    return "\n".join(lines)


def _json_default(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def run_corpus_search(
    *,
    tag=None,
    subdomain=None,
    application=None,
    domain=None,
    with_claims=False,
    since=None,
    list_topics_flag=False,
    min_usage=None,
    claims_for=None,
    top=20,
    as_json=False,
    out=None,
):
    from research_radar.pipeline import connect

    with connect() as conn:
        if list_topics_flag:
            rows = list_topics(conn, min_usage=min_usage)
            text = json.dumps(rows, default=_json_default) if as_json else render_topics_table(rows)
        elif claims_for:
            rows = claims_for_metric(conn, claims_for, top=top)
            text = json.dumps(rows, default=_json_default) if as_json else render_claims_table(rows)
        else:
            rows = search_corpus(
                conn,
                tag=tag,
                subdomain=subdomain,
                application=application,
                domain=domain,
                with_claims=with_claims,
                since=since,
                top=top,
            )
            text = json.dumps(rows, default=_json_default) if as_json else render_results_table(rows)

    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text if as_json else f"# Corpus search results\n\n```\n{text}\n```\n")
        print(f"Wrote {out}")
    else:
        print(text)
