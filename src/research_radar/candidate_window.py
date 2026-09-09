"""Shared published_at window filters for stage candidate loaders."""

from __future__ import annotations

from datetime import date
from typing import Any


def published_at_sql_filters(
    date_from: date | str | None,
    date_until: date | str | None,
) -> tuple[str, list[Any]]:
    """Return SQL fragment and params for ci.published_at window.

    ``date_until`` is inclusive of that calendar day (exclusive upper bound is +1 day).
    When both are None, returns ("", []) so callers behave as before.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if date_from is not None:
        clauses.append("ci.published_at >= %s::date")
        params.append(str(date_from))
    if date_until is not None:
        clauses.append("ci.published_at < (%s::date + INTERVAL '1 day')")
        params.append(str(date_until))
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def format_window_label(date_from: date | str | None, date_until: date | str | None) -> str:
    if date_from is None and date_until is None:
        return "all"
    start = str(date_from) if date_from is not None else ""
    end = str(date_until) if date_until is not None else ""
    return f"{start}..{end}"
