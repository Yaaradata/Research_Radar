"""Shared published_at window filters for paid scoring candidate loaders."""

from __future__ import annotations

from datetime import date
from typing import Any


def published_at_sql_filters(
    since: date | str | None,
    until: date | str | None,
) -> tuple[str, list[Any]]:
    """Return SQL fragment and params for ci.published_at window.

    ``until`` is inclusive of that calendar day (exclusive upper bound is +1 day).
    When both are None, returns ("", []) so callers behave as before.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        clauses.append("ci.published_at >= %s::date")
        params.append(str(since))
    if until is not None:
        clauses.append("ci.published_at < (%s::date + INTERVAL '1 day')")
        params.append(str(until))
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def format_published_window(since: date | str | None, until: date | str | None) -> dict[str, Any]:
    if since is None and until is None:
        label = "all dates (no published_at filter)"
    else:
        parts = []
        if since is not None:
            parts.append(f"since={since}")
        if until is not None:
            parts.append(f"until={until} (inclusive)")
        label = ", ".join(parts)
    return {
        "published_since": str(since) if since is not None else None,
        "published_until": str(until) if until is not None else None,
        "published_window": label,
    }


def merge_window_summary(summary: dict[str, Any], since: date | str | None, until: date | str | None) -> dict[str, Any]:
    out = dict(summary)
    out.update(format_published_window(since, until))
    return out


def print_empty_candidate_pool(stage_label: str, since: date | str | None, until: date | str | None) -> None:
    window = format_published_window(since, until)
    print(f"\n{stage_label}")
    print(f"  published_window: {window['published_window']}")
    print("  candidates: 0")
    print("  nothing to do")
