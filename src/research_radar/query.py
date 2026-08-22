import argparse
import json
from research_radar.pipeline import connect


def main():
    p = argparse.ArgumentParser(description="Query Research Radar candidates")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-score", type=float, default=0)
    p.add_argument("--org")
    p.add_argument("--topic")
    p.add_argument("--days", type=int, default=None, help="Only items published in the last N days")
    p.add_argument("--since", help="ISO date/datetime lower bound for published_at, e.g. 2026-08-13")
    a = p.parse_args()

    filters = ["cs.intrinsic_candidate_score >= %s"]
    params = [a.min_score]
    if a.days is not None:
        filters.append("ci.published_at >= NOW() - (%s || ' days')::interval")
        params.append(str(a.days))
    if a.since:
        filters.append("ci.published_at >= %s::timestamptz")
        params.append(a.since)
    if a.org:
        filters.append(
            "EXISTS (SELECT 1 FROM research_radar.content_organisations co "
            "JOIN research_radar.organisations o ON o.organisation_id=co.organisation_id "
            "WHERE co.content_id=ci.id AND (lower(o.canonical_name)=lower(%s) OR lower(COALESCE(o.org_key,''))=lower(%s)))"
        )
        params.extend([a.org, a.org])
    if a.topic:
        filters.append(
            "EXISTS (SELECT 1 FROM research_radar.content_topics ct "
            "JOIN research_radar.topics t ON t.topic_id=ct.topic_id "
            "WHERE ct.content_id=ci.id AND lower(t.canonical_name) LIKE lower(%s))"
        )
        params.append(f"%{a.topic}%")
    params.append(a.limit)

    sql = f"""
    SELECT
      ci.id,
      ci.title,
      ci.canonical_url,
      ci.published_at,
      ci.published_at::date AS published_date,
      ci.status,
      cs.intrinsic_candidate_score,
      cs.ai_relevance,
      cs.professional_value,
      cs.student_learning_value,
      cs.notable_org_signal
    FROM research_radar.content_items ci
    JOIN research_radar.content_scores cs ON cs.content_id=ci.id
    WHERE ci.status='CANDIDATE'
      AND {' AND '.join(filters)}
    ORDER BY cs.intrinsic_candidate_score DESC, ci.published_at DESC NULLS LAST
    LIMIT %s
    """
    with connect() as conn:
        for row in conn.execute(sql, params).fetchall():
            print(json.dumps(row, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
