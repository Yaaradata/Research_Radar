"""Schema preflight — one-line diagnosis when a required migration is missing.

Stages call ``require_stage_schema(conn, stage)`` before doing work (including
dry-run). Missing objects raise ``SchemaPreflightError`` naming the migration
file to apply. No DDL is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


class SchemaPreflightError(RuntimeError):
    """Clear, single-line schema miss — apply the named migration."""


@dataclass(frozen=True)
class SchemaCheck:
    migration: str
    description: str
    probe: Callable  # (conn) -> bool


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM information_schema.tables
        WHERE table_schema = 'research_radar' AND table_name = %s
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return bool(row)


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM information_schema.columns
        WHERE table_schema = 'research_radar'
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table, column),
    ).fetchone()
    return bool(row)


def _view_exists(conn, view: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM information_schema.views
        WHERE table_schema = 'research_radar' AND table_name = %s
        LIMIT 1
        """,
        (view,),
    ).fetchone()
    return bool(row)


def _seeded_domain_exists(conn) -> bool:
    if not _column_exists(conn, "topics", "level"):
        return False
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.topics
        WHERE level = 'domain'
          AND canonical_name = 'Natural Language Processing'
        LIMIT 1
        """
    ).fetchone()
    return bool(row)


def _index_exists(conn, index_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM pg_indexes
        WHERE schemaname = 'research_radar' AND indexname = %s
        LIMIT 1
        """,
        (index_name,),
    ).fetchone()
    return bool(row)


# Key object probes for each numbered migration under sql/.
# Order matches scripts/setup_db.sh application order.
MIGRATION_CHECKS: list[SchemaCheck] = [
    SchemaCheck("sql/001_schema.sql", "research_radar.content_items", lambda c: _table_exists(c, "content_items")),
    SchemaCheck(
        "sql/002_seed_watchlists.sql",
        "research_radar.organisations (seeded)",
        lambda c: bool(
            c.execute("SELECT 1 AS ok FROM research_radar.organisations LIMIT 1").fetchone()
        )
        if _table_exists(c, "organisations")
        else False,
    ),
    SchemaCheck(
        "sql/003_org_watchlist_extend.sql",
        "research_radar.organisations.org_key",
        lambda c: _column_exists(c, "organisations", "org_key"),
    ),
    SchemaCheck(
        "sql/004_openalex_and_source_seen.sql",
        "research_radar.paper_metadata.openalex_status",
        lambda c: _column_exists(c, "paper_metadata", "openalex_status"),
    ),
    SchemaCheck(
        "sql/005_content_analysis_view.sql",
        "research_radar.v_content_analysis",
        lambda c: _view_exists(c, "v_content_analysis"),
    ),
    SchemaCheck(
        "sql/006_content_score_assessments.sql",
        "research_radar.content_score_assessments",
        lambda c: _table_exists(c, "content_score_assessments"),
    ),
    SchemaCheck(
        "sql/007_affiliation_gpt.sql",
        "research_radar.affiliation_assessments",
        lambda c: _table_exists(c, "affiliation_assessments"),
    ),
    SchemaCheck(
        "sql/008_affiliation_historical_status_fix.sql",
        "research_radar.affiliation_assessments (008 is DML-only; gated on 007)",
        lambda c: _table_exists(c, "affiliation_assessments"),
    ),
    SchemaCheck(
        "sql/009_content_final_scores.sql",
        "research_radar.content_final_scores",
        lambda c: _table_exists(c, "content_final_scores"),
    ),
    SchemaCheck(
        "sql/010_scoring_v2.sql",
        "research_radar.content_score_assessments.scoring_tier",
        lambda c: _column_exists(c, "content_score_assessments", "scoring_tier"),
    ),
    SchemaCheck(
        "sql/011_backfill_checkpoints.sql",
        "research_radar.backfill_checkpoints",
        lambda c: _table_exists(c, "backfill_checkpoints"),
    ),
    SchemaCheck(
        "sql/012_topic_hierarchy.sql",
        "research_radar.topics.level",
        lambda c: _column_exists(c, "topics", "level"),
    ),
    SchemaCheck(
        "sql/013_seed_topic_hierarchy.sql",
        "seeded domain 'Natural Language Processing'",
        _seeded_domain_exists,
    ),
    SchemaCheck(
        "sql/014_scoring_v3.sql",
        "research_radar.content_classifications",
        lambda c: _table_exists(c, "content_classifications"),
    ),
    SchemaCheck(
        "sql/015_general_method_application.sql",
        "research_radar.topics general-method application seed",
        lambda c: bool(
            c.execute(
                """
                SELECT 1 AS ok FROM research_radar.topics
                WHERE canonical_name = 'general-method' AND level = 'application'
                LIMIT 1
                """
            ).fetchone()
        )
        if _table_exists(c, "topics")
        else False,
    ),
    SchemaCheck(
        "sql/015_relevance_version.sql",
        "research_radar.content_items.relevance_version",
        lambda c: _column_exists(c, "content_items", "relevance_version"),
    ),
    SchemaCheck(
        "sql/017_s3_manifest.sql",
        "research_radar.s3_archives",
        lambda c: _table_exists(c, "s3_archives"),
    ),
    SchemaCheck(
        "sql/018_relevance_version.sql",
        "research_radar.ix_content_relevance_version (REJECTED backfill index)",
        lambda c: _index_exists(c, "ix_content_relevance_version"),
    ),
    SchemaCheck(
        "sql/019_bakeoff.sql",
        "research_radar.bakeoff_runs",
        lambda c: _table_exists(c, "bakeoff_runs"),
    ),
]


# Stage → migration files that must be present before the stage may run.
STAGE_REQUIRED_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "relevance": ("sql/015_relevance_version.sql",),
    "classify": ("sql/014_scoring_v3.sql",),
    "topics": ("sql/012_topic_hierarchy.sql", "sql/013_seed_topic_hierarchy.sql"),
}


def _checks_for_migration(path: str) -> list[SchemaCheck]:
    return [c for c in MIGRATION_CHECKS if c.migration == path]


def probe_migration(conn, migration: str) -> tuple[bool, str]:
    checks = _checks_for_migration(migration)
    if not checks:
        return False, f"no probe registered for {migration}"
    for check in checks:
        try:
            ok = bool(check.probe(conn))
        except Exception as exc:
            return False, f"{check.description} (probe error: {exc})"
        if not ok:
            return False, check.description
    return True, checks[0].description


def require_stage_schema(conn, stage: str) -> None:
    """Raise SchemaPreflightError with a one-line fix if stage prerequisites are missing."""
    required = STAGE_REQUIRED_MIGRATIONS.get(stage)
    if not required:
        return
    for migration in required:
        ok, description = probe_migration(conn, migration)
        if not ok:
            raise SchemaPreflightError(
                f"{description} missing; apply {migration}"
            )


def report_all_migrations(conn) -> list[dict]:
    """Return applied/missing status for every registered migration probe."""
    out = []
    for check in MIGRATION_CHECKS:
        try:
            applied = bool(check.probe(conn))
            detail = check.description
        except Exception as exc:
            applied = False
            detail = f"{check.description} (probe error: {exc})"
        out.append(
            {
                "migration": check.migration,
                "applied": applied,
                "object": detail,
            }
        )
    return out
