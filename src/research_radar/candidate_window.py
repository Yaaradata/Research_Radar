"""Shared published_at window filters for paid scoring candidate loaders."""

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


def format_run_line(
    stage: str,
    date_from: date | str | None,
    date_until: date | str | None,
    *,
    candidates: int,
    batches: int | None = None,
    est_cost: float | None = None,
) -> str:
    parts = [
        f"{stage}: window={format_window_label(date_from, date_until)}",
        f"candidates={candidates}",
    ]
    if batches is not None:
        parts.append(f"batches={batches}")
    if est_cost is not None:
        parts.append(f"est_cost=${est_cost:.2f}")
    return " ".join(parts)


def merge_window_summary(
    summary: dict[str, Any],
    date_from: date | str | None,
    date_until: date | str | None,
) -> dict[str, Any]:
    out = dict(summary)
    out["window"] = format_window_label(date_from, date_until)
    out["date_from"] = str(date_from) if date_from is not None else None
    out["date_until"] = str(date_until) if date_until is not None else None
    return out


def print_stage_run_line(
    stage: str,
    date_from: date | str | None,
    date_until: date | str | None,
    *,
    candidates: int,
    batches: int | None = None,
    est_cost: float | None = None,
) -> None:
    print(
        format_run_line(
            stage,
            date_from,
            date_until,
            candidates=candidates,
            batches=batches,
            est_cost=est_cost,
        )
    )
