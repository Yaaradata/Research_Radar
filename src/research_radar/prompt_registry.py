"""Prompt / policy version registry — SHA-256 integrity for selective recompute."""

from __future__ import annotations

import hashlib
from typing import Any


class PromptRegistryError(RuntimeError):
    """Raised when a version string would silently change under a new body hash."""


def sha256_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def register_prompt(
    conn,
    kind: str,
    version: str,
    body: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Upsert prompt body. Raises if (kind, version) exists with a different hash."""
    digest = sha256_body(body)
    existing = conn.execute(
        """
        SELECT prompt_id, kind, version, model_name, body_sha256, body, active, created_at
        FROM research_radar.prompt_versions
        WHERE kind = %s AND version = %s
        """,
        (kind, version),
    ).fetchone()
    if existing:
        if existing["body_sha256"] != digest:
            raise PromptRegistryError(
                f"prompt {kind}/{version} already registered with hash "
                f"{existing['body_sha256'][:12]}…; refusing to overwrite with {digest[:12]}…"
            )
        return dict(existing)

    row = conn.execute(
        """
        INSERT INTO research_radar.prompt_versions (kind, version, model_name, body_sha256, body)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING prompt_id, kind, version, model_name, body_sha256, body, active, created_at
        """,
        (kind, version, model_name, digest, body),
    ).fetchone()
    return dict(row)


def get_prompt(conn, kind: str, version: str | None = None) -> dict[str, Any] | None:
    if version is None:
        row = conn.execute(
            """
            SELECT prompt_id, kind, version, model_name, body_sha256, body, active, created_at
            FROM research_radar.prompt_versions
            WHERE kind = %s AND active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (kind,),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT prompt_id, kind, version, model_name, body_sha256, body, active, created_at
            FROM research_radar.prompt_versions
            WHERE kind = %s AND version = %s
            """,
            (kind, version),
        ).fetchone()
    return dict(row) if row else None


def assert_registered(conn, kind: str, version: str, body: str, model_name: str | None = None) -> dict[str, Any]:
    """Register on stage startup and assert stored hash matches in-code body."""
    return register_prompt(conn, kind, version, body, model_name=model_name)


def papers_per_prompt_version(conn) -> list[dict[str, Any]]:
    """Counts of assessments keyed by stage table + prompt_version (read-only)."""
    queries = [
        (
            "screen",
            """
            SELECT 'screen'::text AS stage, prompt_version, COUNT(*)::int AS n
            FROM research_radar.content_score_assessments
            WHERE scoring_tier = 'screen'
            GROUP BY prompt_version
            """,
        ),
        (
            "paper_scoring",
            """
            SELECT 'paper_scoring'::text AS stage, prompt_version, COUNT(*)::int AS n
            FROM research_radar.content_score_assessments
            WHERE scoring_tier = 'full'
            GROUP BY prompt_version
            """,
        ),
        (
            "classify",
            """
            SELECT 'classify'::text AS stage, prompt_version, COUNT(*)::int AS n
            FROM research_radar.content_classifications
            GROUP BY prompt_version
            """,
        ),
        (
            "independence",
            """
            SELECT 'independence'::text AS stage, prompt_version, COUNT(*)::int AS n
            FROM research_radar.content_independence_assessments
            GROUP BY prompt_version
            """,
        ),
    ]
    out: list[dict[str, Any]] = []
    for _label, sql in queries:
        try:
            rows = conn.execute(sql).fetchall()
        except Exception:
            continue
        out.extend(dict(r) for r in rows)
    out.sort(key=lambda r: (r["stage"], r["prompt_version"]))
    return out
