"""Pull-state tracking — last run + last successfully stored published_at per source."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

PULL_OVERLAP_HOURS = float(os.getenv("PULL_OVERLAP_HOURS", "24"))

SOURCE_INOREADER = "inoreader"
SOURCE_ARXIV_OAI = "arxiv_oai"


def mark_run_started(conn, source: str, run_id: UUID) -> None:
    conn.execute(
        """
        INSERT INTO research_radar.pull_state (
            source, last_run_started_at, last_run_id, updated_at
        ) VALUES (%s, NOW(), %s, NOW())
        ON CONFLICT (source) DO UPDATE SET
            last_run_started_at = EXCLUDED.last_run_started_at,
            last_run_id = EXCLUDED.last_run_id,
            updated_at = NOW()
        """,
        (source, str(run_id)),
    )


def mark_run_completed(
    conn,
    source: str,
    *,
    run_id: UUID,
    last_published_at: datetime | None,
    last_external_id: str | None,
    items_last_run: int,
) -> None:
    """Advance pull_state only after a successful run with committed rows.

    last_published_at uses GREATEST so a partial newer window never rewinds,
    and NULL input leaves the prior watermark untouched.
    """
    conn.execute(
        """
        INSERT INTO research_radar.pull_state (
            source, last_run_started_at, last_run_completed_at, last_run_id,
            last_published_at, last_external_id, items_last_run, items_total, updated_at
        ) VALUES (
            %s, NOW(), NOW(), %s, %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (source) DO UPDATE SET
            last_run_completed_at = NOW(),
            last_run_id = EXCLUDED.last_run_id,
            last_published_at = CASE
                WHEN EXCLUDED.last_published_at IS NULL THEN research_radar.pull_state.last_published_at
                WHEN research_radar.pull_state.last_published_at IS NULL THEN EXCLUDED.last_published_at
                ELSE GREATEST(
                    research_radar.pull_state.last_published_at,
                    EXCLUDED.last_published_at
                )
            END,
            last_external_id = COALESCE(
                EXCLUDED.last_external_id,
                research_radar.pull_state.last_external_id
            ),
            items_last_run = EXCLUDED.items_last_run,
            items_total = research_radar.pull_state.items_total + EXCLUDED.items_last_run,
            updated_at = NOW()
        """,
        (
            source,
            str(run_id),
            last_published_at,
            last_external_id,
            int(items_last_run),
            int(items_last_run),
        ),
    )


def get_pull_state(conn, source: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM research_radar.pull_state WHERE source = %s",
        (source,),
    ).fetchone()
    return dict(row) if row else None


def list_pull_states(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM research_radar.pull_state ORDER BY source"
    ).fetchall()
    return [dict(r) for r in rows]


def resume_cutoff(
    conn,
    source: str,
    *,
    overlap_hours: float | None = None,
) -> datetime | None:
    """Return last_published_at - overlap, or None if no watermark yet."""
    hours = PULL_OVERLAP_HOURS if overlap_hours is None else overlap_hours
    state = get_pull_state(conn, source)
    if not state or state.get("last_published_at") is None:
        return None
    pub = state["last_published_at"]
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return pub - timedelta(hours=hours)


def format_resume_banner(source: str, state: dict[str, Any] | None, cutoff: datetime) -> str:
    last_run = (state or {}).get("last_run_completed_at") or (state or {}).get("last_run_started_at")
    items = (state or {}).get("items_last_run")
    last_run_s = last_run.isoformat().replace("+00:00", "Z") if last_run else "never"
    items_s = str(items) if items is not None else "?"
    cutoff_s = cutoff.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    return f"{source}: resume from {cutoff_s} (last run {last_run_s}, {items_s} items)"


def track_newest_published(
    current: tuple[datetime | None, str | None],
    published_at: datetime | None,
    external_id: str | None,
) -> tuple[datetime | None, str | None]:
    """Keep (max published_at, external_id of that row) across successfully stored items."""
    best_pub, best_ext = current
    if published_at is None:
        return current
    if best_pub is None or published_at > best_pub:
        return published_at, external_id
    return current
