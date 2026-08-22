"""Crossref + OpenAlex affiliation resolution (DOI-first, budget-aware)."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import quote

log = logging.getLogger("research-radar")

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
OPENALEX_REQUEST_SLEEP = float(os.getenv("OPENALEX_REQUEST_SLEEP", "1.2"))
OPENALEX_MAX_RETRIES = int(os.getenv("OPENALEX_MAX_RETRIES", "5"))
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip() or os.getenv("OPENALEX_MAILTO", "").strip()

OPENALEX_COST_DOI_USD = 0.0001
OPENALEX_COST_TITLE_SEARCH_USD = 0.001

_OPENALEX_LOCK = threading.Lock()
_OPENALEX_NEXT_ALLOWED = 0.0
_OPENALEX_BUDGET_EXHAUSTED = False


class OpenAlexBudgetExhausted(Exception):
    """Daily OpenAlex credit budget exhausted."""


class OpenAlexRateLimited(Exception):
    """Transient OpenAlex 429 while credits may still remain."""


@dataclass
class AffiliationStageStats:
    total_unresolved: int = 0
    crossref_attempted: int = 0
    crossref_matched: int = 0
    openalex_doi_attempted: int = 0
    openalex_doi_matched: int = 0
    openalex_title_searched: int = 0
    openalex_title_matched: int = 0
    pending_remaining: int = 0
    rate_limited: int = 0
    no_match: int = 0
    errors: int = 0
    http_calls_start: int = 0
    http_calls_end: int = 0
    openalex_estimated_cost_usd: float = 0.0
    budget_exhausted: bool = False

    def record_http(self):
        from research_radar.pipeline import HTTP_CALLS

        self.http_calls_end = HTTP_CALLS

    def to_dict(self) -> dict:
        return {
            "total_unresolved": self.total_unresolved,
            "crossref_attempted": self.crossref_attempted,
            "crossref_matched": self.crossref_matched,
            "openalex_doi_attempted": self.openalex_doi_attempted,
            "openalex_doi_matched": self.openalex_doi_matched,
            "openalex_title_searched": self.openalex_title_searched,
            "openalex_title_matched": self.openalex_title_matched,
            "pending_remaining": self.pending_remaining,
            "rate_limited": self.rate_limited,
            "no_match": self.no_match,
            "errors": self.errors,
            "http_calls": max(0, self.http_calls_end - self.http_calls_start),
            "openalex_estimated_cost_usd": round(self.openalex_estimated_cost_usd, 4),
            "budget_exhausted": self.budget_exhausted,
        }


def reset_openalex_budget_flag():
    global _OPENALEX_BUDGET_EXHAUSTED
    _OPENALEX_BUDGET_EXHAUSTED = False


def is_openalex_budget_exhausted() -> bool:
    return _OPENALEX_BUDGET_EXHAUSTED


def normalize_doi(doi):
    if not doi:
        return None
    d = str(doi).strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:\s*", "", d, flags=re.I).strip()
    return d or None


def _http_get(url, **kwargs):
    from research_radar.pipeline import http_get

    return http_get(url, **kwargs)


def _openalex_auth():
    params = {}
    headers = {}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
        headers["Authorization"] = f"Bearer {OPENALEX_API_KEY}"
    return params, headers


def _openalex_throttle():
    global _OPENALEX_NEXT_ALLOWED
    with _OPENALEX_LOCK:
        now = time.time()
        wait = _OPENALEX_NEXT_ALLOWED - now
        if wait > 0:
            time.sleep(wait)
        _OPENALEX_NEXT_ALLOWED = time.time() + max(OPENALEX_REQUEST_SLEEP, 0.5)


def _openalex_mark_wait(seconds: float):
    global _OPENALEX_NEXT_ALLOWED
    with _OPENALEX_LOCK:
        _OPENALEX_NEXT_ALLOWED = max(_OPENALEX_NEXT_ALLOWED, time.time() + seconds)


def openalex_http_get(url, *, params=None, allow_not_found: bool = False):
    global _OPENALEX_BUDGET_EXHAUSTED
    if _OPENALEX_BUDGET_EXHAUSTED:
        raise OpenAlexBudgetExhausted("OpenAlex daily budget already exhausted this process")
    if not OPENALEX_API_KEY:
        log.warning("OPENALEX_API_KEY is empty; OpenAlex calls will likely be heavily limited")

    base_params, headers = _openalex_auth()
    merged = dict(base_params)
    if params:
        merged.update(params)

    last = None
    for attempt in range(OPENALEX_MAX_RETRIES):
        _openalex_throttle()
        last = _http_get(url, params=merged, headers=headers)
        remaining = last.headers.get("X-RateLimit-Remaining")
        reset = last.headers.get("X-RateLimit-Reset")
        credits_used = last.headers.get("X-RateLimit-Credits-Used")

        if last.status_code == 200:
            return last

        if last.status_code == 404 and allow_not_found:
            return last

        if last.status_code == 429:
            rem_usd = last.headers.get("X-RateLimit-Remaining-USD")
            log.warning(
                "OpenAlex 429 remaining=%s remaining_usd=%s reset=%s credits_used=%s attempt=%s url=%s",
                remaining,
                rem_usd,
                reset,
                credits_used,
                attempt + 1,
                url,
            )
            budget_gone = (
                (remaining is not None and str(remaining).strip() == "0")
                or (rem_usd is not None and str(rem_usd).strip() in {"0", "0.0"})
                or ("insufficient budget" in (last.text or "").lower())
            )
            if budget_gone:
                _OPENALEX_BUDGET_EXHAUSTED = True
                _openalex_mark_wait(300)
                raise OpenAlexBudgetExhausted(
                    f"OpenAlex daily budget exhausted (remaining={remaining} remaining_usd={rem_usd} reset={reset})"
                )
            wait = min(60.0, 2 ** attempt)
            _openalex_mark_wait(wait)
            time.sleep(wait)
            continue

        if last.status_code >= 500:
            wait = min(60.0, 2 ** attempt)
            log.warning("OpenAlex %s; backing off %.1fs", last.status_code, wait)
            time.sleep(wait)
            continue

        last.raise_for_status()

    if last is not None and last.status_code == 429:
        raise OpenAlexRateLimited("OpenAlex still rate-limited after retries")
    if last is not None:
        last.raise_for_status()
    raise RuntimeError("OpenAlex request failed with no response")


def _institutions_from_openalex_work(work: dict) -> dict:
    inst = []
    for a in (work or {}).get("authorships", []) or []:
        for i in a.get("institutions", []) or []:
            if i.get("display_name"):
                inst.append(
                    {
                        "author": (a.get("author") or {}).get("display_name"),
                        "institution": i["display_name"],
                    }
                )
    return {
        "source_url": (work or {}).get("id"),
        "work_id": (work or {}).get("id"),
        "title": (work or {}).get("title") or (work or {}).get("display_name"),
        "institutions": inst,
    }


def resolve_openalex_by_doi(doi: str, stats: AffiliationStageStats | None = None):
    doi = normalize_doi(doi)
    if not doi:
        return None
    url = f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}"
    if stats is not None:
        stats.openalex_doi_attempted += 1
        stats.openalex_estimated_cost_usd += OPENALEX_COST_DOI_USD
    r = openalex_http_get(url, allow_not_found=True)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise RuntimeError(f"Unexpected OpenAlex status {r.status_code} for DOI lookup")
    return _institutions_from_openalex_work(r.json())


def resolve_openalex_by_title(title: str, stats: AffiliationStageStats | None = None):
    if not title:
        return None
    clean = re.sub(r'[?&=:"<>{}]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip()[:300]
    if not clean:
        return None
    if stats is not None:
        stats.openalex_title_searched += 1
        stats.openalex_estimated_cost_usd += OPENALEX_COST_TITLE_SEARCH_USD
    params = {"filter": f"title.search:{clean}", "per-page": 3}
    r = openalex_http_get("https://api.openalex.org/works", params=params)
    results = r.json().get("results", []) or []
    if not results:
        return None
    norm = lambda s: " ".join((s or "").lower().split())
    best = results[0]
    if norm(best.get("title") or best.get("display_name")) not in {norm(title), norm(clean)}:
        return None
    return _institutions_from_openalex_work(best)


def _institutions_from_crossref_message(message: dict, doi: str) -> dict:
    inst = []
    for author in message.get("author", []) or []:
        author_name = " ".join(x for x in [author.get("given"), author.get("family")] if x).strip()
        affiliations = author.get("affiliation") or []
        if not affiliations:
            continue
        for aff in affiliations:
            name = aff.get("name") if isinstance(aff, dict) else None
            if not name:
                continue
            inst.append({"author": author_name or None, "institution": name})
    source_url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    return {
        "source_url": source_url,
        "doi": doi,
        "title": message.get("title", [None])[0] if isinstance(message.get("title"), list) else message.get("title"),
        "institutions": inst,
    }


def resolve_crossref_by_doi(doi: str, mailto: str | None = None, stats: AffiliationStageStats | None = None):
    doi = normalize_doi(doi)
    if not doi:
        return None
    if stats is not None:
        stats.crossref_attempted += 1
    mailto = (mailto or CROSSREF_MAILTO or "").strip()
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    params = {"mailto": mailto} if mailto else {}
    headers = {
        "User-Agent": (
            f"TheNeural-Research-Radar/0.1 (mailto:{mailto})" if mailto else "TheNeural-Research-Radar/0.1"
        )
    }
    r = _http_get(url, params=params, headers=headers)
    if r.status_code == 404:
        return None
    if not r.ok:
        log.warning("Crossref lookup failed status=%s doi=%s", r.status_code, doi)
        return None
    message = r.json().get("message") or {}
    payload = _institutions_from_crossref_message(message, doi)
    if not payload.get("institutions"):
        return None
    return payload
