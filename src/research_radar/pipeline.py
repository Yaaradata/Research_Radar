from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote

import feedparser
import psycopg
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or os.getenv("PG_DSN", "").strip()
INOREADER_ACCESS_TOKEN = (
    os.getenv("INOREADER_ACCESS_TOKEN", "").strip()
    or os.getenv("INOREADER_OAUTH_TOKEN", "").strip()
)
INOREADER_BASE_URL = os.getenv("INOREADER_BASE_URL", "https://www.inoreader.com")
INOREADER_STREAM = os.getenv("INOREADER_STREAM", "user/-/state/com.google/reading-list")
INOREADER_BATCH_SIZE = int(os.getenv("INOREADER_BATCH_SIZE", "100"))
INOREADER_LOOKBACK_DAYS = int(os.getenv("INOREADER_LOOKBACK_DAYS", "7"))
INOREADER_MAX_PAGES = int(os.getenv("INOREADER_MAX_PAGES", "50"))
INOREADER_FIXTURE = os.getenv("INOREADER_FIXTURE", "")
# OpenAlex/Crossref are deprecated for active affiliation enrichment (use affiliation-gpt).
# Defaults off so the normal pipeline makes no OpenAlex/Crossref HTTP requests.
OPENALEX_ENABLED = os.getenv("OPENALEX_ENABLED", "false").lower() in {"1","true","yes","on"}
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
# Kept for logging only; OpenAlex polite-pool mailto is deprecated in favour of API keys.
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()
OPENALEX_REQUEST_SLEEP = float(os.getenv("OPENALEX_REQUEST_SLEEP", "1.2"))
OPENALEX_MAX_RETRIES = int(os.getenv("OPENALEX_MAX_RETRIES", "5"))
OPENALEX_WORKERS = int(os.getenv("OPENALEX_WORKERS", "1"))
CROSSREF_ENABLED = os.getenv("CROSSREF_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "").strip() or OPENALEX_MAILTO
OPENALEX_TITLE_SEARCH_ENABLED = os.getenv("OPENALEX_TITLE_SEARCH_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
AFFILIATION_GPT_ENABLED = os.getenv("AFFILIATION_GPT_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "20"))
HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", "TheNeural-Research-Radar/0.1")
ARXIV_REQUEST_SLEEP = float(os.getenv("ARXIV_REQUEST_SLEEP", "1.0"))
ARXIV_MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "6"))
ARXIV_WORKERS = int(os.getenv("ARXIV_WORKERS", "4"))
ARXIV_COMMIT_EVERY = int(os.getenv("ARXIV_COMMIT_EVERY", "10"))
PIPELINE_WORKERS = int(os.getenv("PIPELINE_WORKERS", str(ARXIV_WORKERS)))
MIN_AI_RELEVANCE = float(os.getenv("MIN_AI_RELEVANCE_FOR_ENRICHMENT", "5.0"))
MIN_CANDIDATE_SCORE = float(os.getenv("MIN_INTRINSIC_CANDIDATE_SCORE", "5.5"))
SCORE_INCLUDE_PERSON_SIGNAL = os.getenv("SCORE_INCLUDE_PERSON_SIGNAL", "auto").strip().lower()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("research-radar")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": HTTP_USER_AGENT})
HTTP_CALLS = 0
_ARXIV_LOCK = threading.Lock()
_ARXIV_NEXT_ALLOWED = 0.0

ARXIV_RE = re.compile(r'(?:arxiv\.org/(?:abs|html|pdf)/|arxiv:)?(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?P<version>v\d+)?', re.I)
EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I)
TRACKING_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","mc_cid","mc_eid"}
AI_ARXIV = {"cs.AI","cs.CL","cs.CV","cs.LG","cs.MA","cs.RO","stat.ML"}

TOPIC_RULES = {
    "AI agents": [r"\bagent(ic|s)?\b", r"\bmulti[- ]agent\b", r"\btool use\b", r"\bcomputer use\b"],
    "LLMs / foundation models": [r"\bllm(s)?\b", r"\blarge language model", r"\bfoundation model", r"\btransformer(s)?\b"],
    "Machine learning": [r"\bmachine learning\b", r"\bdeep learning\b", r"\bneural network"],
    "Data science": [r"\bdata science\b", r"\bstatistical learning\b"],
    "Multimodal AI": [r"\bmultimodal\b", r"\bvision[- ]language\b", r"\bvlm(s)?\b"],
    "AI engineering / ML systems": [r"\bmlops\b", r"\bllmops\b", r"\binference serving\b", r"\bmodel serving\b", r"\bai engineering\b"],
    "AI evaluation": [r"\beval(s|uation)?\b", r"\bbenchmark(s|ing)?\b", r"\bgrader(s)?\b"],
    "AI safety and security": [r"\balignment\b", r"\bai safety\b", r"\bmodel safety\b", r"\bjailbreak", r"\bprompt injection\b", r"\bai security\b"],
    "Retrieval / RAG": [r"\brag\b", r"\bretrieval[- ]augmented\b", r"\bvector search\b", r"\bretrieval\b"],
    "Model training, inference and efficiency": [r"\btraining\b", r"\binference\b", r"\bquantization\b", r"\bdistillation\b", r"\bmixture of experts\b", r"\bmoe\b"],
    "Applied AI": [r"\bapplied ai\b", r"\bai application", r"\bgenerative ai\b"],
    "Human-AI interaction": [r"\bhuman[- ]ai\b", r"\bhuman computer interaction\b", r"\bhci\b"],
    "AI product and experimentation": [r"\ba/b test", r"\bexperimentation\b", r"\bai product\b", r"\bproduct management\b"],
}


def http_get(url, **kwargs):
    global HTTP_CALLS
    HTTP_CALLS += 1
    kwargs.setdefault("timeout", HTTP_TIMEOUT)
    return SESSION.get(url, **kwargs)


def http_get_retry(url, *, retries=None, sleep_base=None, **kwargs):
    """GET with backoff for arXiv rate limits (429/5xx)."""
    retries = ARXIV_MAX_RETRIES if retries is None else retries
    sleep_base = ARXIV_REQUEST_SLEEP if sleep_base is None else sleep_base
    last = None
    for attempt in range(1, retries + 1):
        _arxiv_throttle()
        last = http_get(url, **kwargs)
        if last.status_code == 429 or last.status_code >= 500:
            wait = min(max(sleep_base, 1.0) * (2 ** (attempt - 1)), 90.0)
            log.warning(
                "HTTP %s on %s (attempt %d/%d); sleeping %.1fs",
                last.status_code,
                url.split("?", 1)[0],
                attempt,
                retries,
                wait,
            )
            time.sleep(wait)
            continue
        return last
    return last


def _arxiv_throttle():
    """Shared polite spacing across parallel workers."""
    global _ARXIV_NEXT_ALLOWED
    with _ARXIV_LOCK:
        now = time.time()
        wait = _ARXIV_NEXT_ALLOWED - now
        if wait > 0:
            time.sleep(wait)
        _ARXIV_NEXT_ALLOWED = time.time() + max(ARXIV_REQUEST_SLEEP, 0.2)


def _entry_to_base(e, fallback_id=None, fallback_version=None):
    arxiv_id, version = extract_arxiv_id(getattr(e, "id", None) or "")
    if not arxiv_id:
        arxiv_id, version = fallback_id, fallback_version
    categories = [t["term"] for t in getattr(e, "tags", [])]
    authors = [a.get("name") for a in getattr(e, "authors", []) if a.get("name")]
    return {
        "arxiv_id": arxiv_id,
        "arxiv_version": version,
        "paper_title": " ".join(e.title.split()),
        "abstract": " ".join(e.summary.split()),
        "authors": authors,
        "submission_date": getattr(e, "published", None),
        "latest_revision_date": getattr(e, "updated", None),
        "arxiv_categories": categories,
        "doi": getattr(e, "arxiv_doi", None),
        "journal_reference": getattr(e, "arxiv_journal_ref", None),
        "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
        "html_url": f"https://arxiv.org/html/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def extract_arxiv_affiliations(arxiv_id):
    emails, affs, evidence_url = [], [], None
    for url in [f"https://arxiv.org/html/{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}"]:
        page = http_get_retry(url)
        if page is not None and page.ok:
            evidence_url = url
            emails = sorted(set(EMAIL_RE.findall(page.text)))
            soup = BeautifulSoup(page.text, "html.parser")
            for selector in [".ltx_contact", ".ltx_role_affiliation", ".ltx_affiliation", ".authors", ".author", ".dateline"]:
                for node in soup.select(selector):
                    txt = " ".join(node.get_text(" ", strip=True).split())
                    if txt and txt not in affs:
                        affs.append(txt[:1500])
            break
    return emails, affs[:20], evidence_url


def enrich_arxiv_atom_batch(arxiv_ids):
    """One Atom API call for many IDs (much fewer 429s than 1 call/paper)."""
    ids = [a for a in arxiv_ids if a]
    if not ids:
        return {}
    r = http_get_retry(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids)},
    )
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    out = {}
    for e in feed.entries:
        base = _entry_to_base(e)
        if base.get("arxiv_id"):
            out[base["arxiv_id"]] = base
    return out


def enrich_arxiv(url_or_id):
    arxiv_id, version = extract_arxiv_id(url_or_id)
    if not arxiv_id:
        raise ValueError(f"Could not extract arXiv ID from {url_or_id}")
    batch = enrich_arxiv_atom_batch([arxiv_id])
    base = batch.get(arxiv_id)
    if not base:
        raise ValueError(f"arXiv metadata not found for {arxiv_id}")
    if version is not None:
        base["arxiv_version"] = version
    emails, affs, evidence_url = extract_arxiv_affiliations(arxiv_id)
    base["emails"] = emails
    base["affiliation_text"] = affs
    base["evidence_url"] = evidence_url
    return base


def refresh_inoreader_access_token():
    global INOREADER_ACCESS_TOKEN
    refresh_token = os.getenv("INOREADER_REFRESH_TOKEN", "").strip()
    client_id = os.getenv("INOREADER_CLIENT_ID", "").strip()
    client_secret = os.getenv("INOREADER_CLIENT_SECRET", "").strip()
    if not refresh_token or not client_id or not client_secret:
        log.warning("Cannot refresh Inoreader token: missing REFRESH_TOKEN / CLIENT_ID / CLIENT_SECRET")
        return None
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    r = requests.post(
        f"{INOREADER_BASE_URL.rstrip('/')}/oauth2/token",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=HTTP_TIMEOUT,
    )
    if not r.ok:
        log.error("Inoreader token refresh failed status=%s body=%s", r.status_code, r.text[:500])
        return None
    data = r.json()
    token = (data.get("access_token") or "").strip()
    new_refresh = (data.get("refresh_token") or "").strip()
    if token:
        os.environ["INOREADER_ACCESS_TOKEN"] = token
        INOREADER_ACCESS_TOKEN = token
    if new_refresh:
        os.environ["INOREADER_REFRESH_TOKEN"] = new_refresh
    return token or None


def inoreader_headers(access_token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": HTTP_USER_AGENT,
    }
    client_id = os.getenv("INOREADER_CLIENT_ID", "").strip()
    client_secret = os.getenv("INOREADER_CLIENT_SECRET", "").strip()
    if client_id:
        headers["AppId"] = client_id
    if client_secret:
        headers["AppKey"] = client_secret
    return headers


def connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def extract_arxiv_id(value):
    if not value:
        return None, None
    m = ARXIV_RE.search(value)
    if not m:
        return None, None
    version = m.group("version")
    return m.group("id"), int(version[1:]) if version else None


def normalize_url(url):
    arxiv_id, _ = extract_arxiv_id(url)
    if arxiv_id and "arxiv.org" in url.lower():
        return f"https://arxiv.org/abs/{arxiv_id}"
    p = urlsplit(url.strip())
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    query = [(k,v) for k,v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
    return urlunsplit((scheme, netloc, p.path.rstrip("/") or "/", urlencode(query), ""))


def detect_source_type(url, feed="", categories=None):
    text = " ".join([url or "", feed or "", " ".join(categories or [])]).lower()
    if "arxiv.org" in text: return "arxiv"
    if any(x in text for x in ["research.","/research/","acm.org","ieee.org"]): return "research_paper"
    if any(x in text for x in ["openai.com","anthropic.com","deepmind","microsoft.com","meta.com","amazon.science"]): return "company_research"
    if any(x in text for x in ["blog","engineering"]): return "technical_blog"
    if any(x in text for x in ["news","reuters","techcrunch","theverge"]): return "news"
    return "other"


def parse_iso_datetime(value):
    """Parse ISO-8601 datetime strings to timezone-aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def parse_unix_seconds(value):
    """Inoreader `published` / `updated` — Unix seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        return parse_iso_datetime(text)
    return None


def parse_unix_milliseconds(value):
    """Inoreader `crawlTimeMsec` — Unix milliseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return datetime.fromtimestamp(float(text) / 1000.0, tz=timezone.utc)
    return None


def parse_unix_microseconds(value):
    """Inoreader `timestampUsec` — Unix microseconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1_000_000.0, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return datetime.fromtimestamp(float(text) / 1_000_000.0, tz=timezone.utc)
    return None


def effective_item_date(published_at, source_seen_at):
    """Lookback helper only — never persisted as published_at."""
    return published_at or source_seen_at


LOCAL_ORG_EVIDENCE_TYPES = ("email_domain", "explicit_affiliation_text")


def timestamps_from_inoreader_raw(raw_metadata):
    """Re-parse Inoreader timestamps from stored raw_metadata JSON (no API call)."""
    if not raw_metadata:
        return None, None, None
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            return None, None, None
    if not isinstance(raw_metadata, dict):
        return None, None, None
    published_at = parse_unix_seconds(raw_metadata.get("published"))
    source_seen_at = parse_unix_microseconds(raw_metadata.get("timestampUsec"))
    if source_seen_at is None:
        source_seen_at = parse_unix_milliseconds(raw_metadata.get("crawlTimeMsec"))
    updated_at = parse_unix_seconds(raw_metadata.get("updated"))
    return published_at, source_seen_at, updated_at


def lookback_cutoff():
    days = max(1, INOREADER_LOOKBACK_DAYS)
    return datetime.now(timezone.utc) - timedelta(days=days)


def inoreader_item_to_canonical(item):
    url = None
    for alt in item.get("alternate",[]) or []:
        if alt.get("href"):
            url = alt["href"]; break
    if not url:
        for can in item.get("canonical",[]) or []:
            if can.get("href"):
                url = can["href"]; break
    if not url: return None
    origin = item.get("origin") or {}
    feed = origin.get("title") or origin.get("streamId") or ""
    categories = item.get("categories") or []
    summary_obj = item.get("summary") or item.get("content") or {}
    summary = summary_obj.get("content","") if isinstance(summary_obj,dict) else str(summary_obj)
    authors = [item["author"]] if item.get("author") else []
    published_at = parse_unix_seconds(item.get("published"))
    source_seen_at = parse_unix_microseconds(item.get("timestampUsec"))
    if source_seen_at is None:
        source_seen_at = parse_unix_milliseconds(item.get("crawlTimeMsec"))
    return {
        "source":"inoreader", "source_external_id":item.get("id"), "source_feed":feed,
        "source_type":detect_source_type(url,feed,categories), "canonical_url":normalize_url(url),
        "title":item.get("title") or "(untitled)", "summary":summary,
        "published_at": published_at,
        "source_seen_at": source_seen_at,
        "updated_at": parse_unix_seconds(item.get("updated")), "authors_raw":authors,
        "categories_raw":categories, "inoreader_tags":categories, "raw_metadata":item
    }


def fetch_inoreader_items():
    cutoff = lookback_cutoff()
    cutoff_usec = int(cutoff.timestamp() * 1_000_000)
    log.info(
        "Inoreader lookback_days=%s cutoff_utc=%s cutoff_usec=%s",
        INOREADER_LOOKBACK_DAYS,
        cutoff.isoformat(),
        cutoff_usec,
    )

    if INOREADER_FIXTURE:
        data = json.loads(Path(INOREADER_FIXTURE).read_text(encoding="utf-8"))
        raw = data.get("items", data if isinstance(data, list) else [])
    else:
        access_token = INOREADER_ACCESS_TOKEN
        if not access_token:
            access_token = refresh_inoreader_access_token() or ""
        if not access_token:
            raise RuntimeError("Set INOREADER_ACCESS_TOKEN or INOREADER_FIXTURE")
        stream = INOREADER_STREAM.strip("/")
        # Encode folder names with spaces (e.g. "10 RESEARCH - T3 Firehose")
        encoded_stream = "/".join(quote(part, safe="") for part in stream.split("/") if part != "")
        url = f"{INOREADER_BASE_URL.rstrip('/')}/reader/api/0/stream/contents/{encoded_stream}"
        log.info("Inoreader stream=%s", stream)

        raw = []
        continuation = None
        for page in range(1, INOREADER_MAX_PAGES + 1):
            params = {
                "n": INOREADER_BATCH_SIZE,
                "ot": cutoff_usec,
                "output": "json",
            }
            if continuation:
                params["c"] = continuation
            r = http_get(url, headers=inoreader_headers(access_token), params=params)
            if r.status_code == 401:
                log.warning("Inoreader returned 401; attempting token refresh")
                refreshed = refresh_inoreader_access_token()
                if refreshed:
                    time.sleep(0.5)
                    access_token = refreshed
                    r = http_get(url, headers=inoreader_headers(access_token), params=params)
                else:
                    raise RuntimeError(
                        "Inoreader access token expired and refresh failed. "
                        "Re-auth and update INOREADER_ACCESS_TOKEN / INOREADER_REFRESH_TOKEN in .env"
                    )
            if not r.ok:
                raise RuntimeError(f"Inoreader stream fetch failed status={r.status_code} body={r.text[:800]}")
            payload = r.json()
            page_items = payload.get("items", []) or []
            raw.extend(page_items)
            log.info("Inoreader page=%d fetched=%d cumulative=%d", page, len(page_items), len(raw))
            continuation = payload.get("continuation")
            if not continuation or not page_items:
                break
            # Soft stop if this page is entirely older than the lookback window.
            page_pubs = []
            for i in page_items:
                pub = parse_unix_seconds(i.get("published"))
                seen = parse_unix_microseconds(i.get("timestampUsec"))
                if seen is None:
                    seen = parse_unix_milliseconds(i.get("crawlTimeMsec"))
                eff = effective_item_date(pub, seen)
                if eff is not None:
                    page_pubs.append(eff)
            page_pubs = [p for p in page_pubs if p is not None]
            if page_pubs and max(page_pubs) < cutoff:
                log.info("Inoreader page older than lookback cutoff; stopping pagination")
                break
            time.sleep(0.4)

    items = []
    skipped_old = 0
    skipped_nodate = 0
    for item in raw:
        canonical = inoreader_item_to_canonical(item)
        if not canonical:
            continue
        effective = effective_item_date(
            canonical.get("published_at"),
            canonical.get("source_seen_at"),
        )
        if effective is None:
            skipped_nodate += 1
            # Keep undated items only if we cannot determine age; still store date as NULL.
            items.append(canonical)
            continue
        if effective < cutoff:
            skipped_old += 1
            continue
        items.append(canonical)
    log.info(
        "Inoreader kept=%d skipped_older_than_lookback=%d undated=%d",
        len(items),
        skipped_old,
        skipped_nodate,
    )
    return items


def score_relevance(title, summary, categories, source_type):
    text = f"{title}\n{summary}".lower()
    hits = {}
    for topic, patterns in TOPIC_RULES.items():
        n = sum(1 for p in patterns if re.search(p,text,re.I))
        if n: hits[topic]=n
    score = min(6.5, 3.0 + sum(hits.values())*0.7) if hits else 0
    if any(c in AI_ARXIV for c in categories or []): score += 2.0
    if source_type in {"arxiv","research_paper","company_research"}: score += 0.8
    score = min(10.0, round(score,2))
    ordered = sorted(hits.items(), key=lambda x:(-x[1],x[0]))
    primary = ordered[0][0] if ordered else None
    secondary = [x[0] for x in ordered[1:4]]
    reasons = []
    if primary: reasons.append(f"matched topic '{primary}'")
    if any(c in AI_ARXIV for c in categories or []): reasons.append("AI/ML arXiv category")
    if source_type in {"arxiv","research_paper","company_research"}: reasons.append(f"source prior={source_type}")
    return score, primary, secondary, "; ".join(reasons) or "no strong deterministic AI signal"


def upsert_item(conn,item):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM research_radar.content_items WHERE canonical_url=%s LIMIT 1",(item["canonical_url"],))
        is_new = cur.fetchone() is None
        cur.execute("""
        INSERT INTO research_radar.content_items(source_type,source,source_external_id,source_feed,canonical_url,title,summary,authors_raw,categories_raw,inoreader_tags,published_at,source_seen_at,updated_at,raw_metadata,status,ingested_at,modified_at)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb,'INGESTED',NOW(),NOW())
        ON CONFLICT(canonical_url) DO UPDATE SET
          source_external_id=COALESCE(EXCLUDED.source_external_id,research_radar.content_items.source_external_id),
          source_feed=COALESCE(EXCLUDED.source_feed,research_radar.content_items.source_feed), title=EXCLUDED.title,
          summary=COALESCE(NULLIF(EXCLUDED.summary,''),research_radar.content_items.summary),
          authors_raw=CASE WHEN EXCLUDED.authors_raw<>'[]'::jsonb THEN EXCLUDED.authors_raw ELSE research_radar.content_items.authors_raw END,
          categories_raw=CASE WHEN EXCLUDED.categories_raw<>'[]'::jsonb THEN EXCLUDED.categories_raw ELSE research_radar.content_items.categories_raw END,
          inoreader_tags=EXCLUDED.inoreader_tags,
          published_at=COALESCE(EXCLUDED.published_at,research_radar.content_items.published_at),
          source_seen_at=COALESCE(research_radar.content_items.source_seen_at, EXCLUDED.source_seen_at),
          updated_at=COALESCE(EXCLUDED.updated_at,research_radar.content_items.updated_at),raw_metadata=EXCLUDED.raw_metadata,modified_at=NOW()
        RETURNING id
        """,(item["source_type"],item["source"],item.get("source_external_id"),item.get("source_feed"),item["canonical_url"],item["title"],item.get("summary"),json.dumps(item.get("authors_raw",[])),json.dumps(item.get("categories_raw",[])),json.dumps(item.get("inoreader_tags",[])),item.get("published_at"),item.get("source_seen_at"),item.get("updated_at"),json.dumps(item.get("raw_metadata",{}),default=str)))
        return cur.fetchone()["id"], is_new


def bump(conn,run_id,field,amount=1):
    allowed={"items_received","items_new","items_duplicate","items_relevant","items_enriched","orgs_resolved","people_resolved","items_scored","candidates_created","errors","http_calls","llm_calls"}
    if field not in allowed: raise ValueError(field)
    with conn.cursor() as cur: cur.execute(f"UPDATE research_radar.pipeline_runs SET {field}={field}+%s WHERE run_id=%s",(amount,run_id))


def set_status(conn,content_id,status):
    with conn.cursor() as cur: cur.execute("UPDATE research_radar.content_items SET status=%s,modified_at=NOW() WHERE id=%s",(status,content_id))


def event(conn,run_id,content_id,stage,event_type,success,details=None,error=None):
    with conn.cursor() as cur: cur.execute("INSERT INTO research_radar.processing_events(run_id,content_id,stage,event_type,success,error_message,details) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)",(run_id,content_id,stage,event_type,success,error,json.dumps(details or {},default=str)))


def store_relevance(conn,content_id,score,primary,secondary,reason):
    with conn.cursor() as cur:
        for idx,name in enumerate([x for x in [primary,*secondary] if x]):
            cur.execute("""
            INSERT INTO research_radar.content_topics(content_id,topic_id,is_primary,confidence,reason)
            SELECT %s,topic_id,%s,%s,%s FROM research_radar.topics WHERE canonical_name=%s
            ON CONFLICT(content_id,topic_id) DO UPDATE SET is_primary=EXCLUDED.is_primary,confidence=EXCLUDED.confidence,reason=EXCLUDED.reason
            """,(content_id,idx==0,min(1.0,score/10),reason,name))
        cur.execute("""
        INSERT INTO research_radar.content_scores(content_id,ai_relevance,scoring_reason)
        VALUES(%s,%s,%s::jsonb)
        ON CONFLICT(content_id) DO UPDATE SET ai_relevance=EXCLUDED.ai_relevance,scoring_reason=research_radar.content_scores.scoring_reason||EXCLUDED.scoring_reason,scored_at=NOW()
        """,(content_id,score,json.dumps({"relevance_reason":reason})))


def store_paper(conn,content_id,p):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO research_radar.paper_metadata(content_id,arxiv_id,arxiv_version,doi,abstract,categories,authors_raw,submission_date,latest_revision_date,journal_reference,paper_url,html_url,pdf_url,affiliation_text,extracted_emails,enrichment_metadata,modified_at)
        VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,NOW())
        ON CONFLICT(content_id) DO UPDATE SET arxiv_id=EXCLUDED.arxiv_id,arxiv_version=COALESCE(EXCLUDED.arxiv_version,research_radar.paper_metadata.arxiv_version),doi=COALESCE(EXCLUDED.doi,research_radar.paper_metadata.doi),abstract=EXCLUDED.abstract,categories=EXCLUDED.categories,authors_raw=EXCLUDED.authors_raw,submission_date=EXCLUDED.submission_date,latest_revision_date=EXCLUDED.latest_revision_date,journal_reference=EXCLUDED.journal_reference,paper_url=EXCLUDED.paper_url,html_url=EXCLUDED.html_url,pdf_url=EXCLUDED.pdf_url,affiliation_text=EXCLUDED.affiliation_text,extracted_emails=EXCLUDED.extracted_emails,enrichment_metadata=EXCLUDED.enrichment_metadata,modified_at=NOW()
        """,(content_id,p["arxiv_id"],p.get("arxiv_version"),p.get("doi"),p.get("abstract"),json.dumps(p.get("arxiv_categories",[])),json.dumps(p.get("authors",[])),p.get("submission_date"),p.get("latest_revision_date"),p.get("journal_reference"),p.get("paper_url"),p.get("html_url"),p.get("pdf_url"),json.dumps(p.get("affiliation_text",[])),json.dumps(p.get("emails",[])),json.dumps({"evidence_url":p.get("evidence_url")})))
        cur.execute("UPDATE research_radar.content_items SET title=%s,summary=%s,authors_raw=%s::jsonb,categories_raw=%s::jsonb,modified_at=NOW() WHERE id=%s",(p.get("paper_title"),p.get("abstract"),json.dumps(p.get("authors",[])),json.dumps(p.get("arxiv_categories",[])),content_id))


def ensure_paper_metadata_row(conn, content_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_radar.paper_metadata(content_id)
            VALUES (%s)
            ON CONFLICT (content_id) DO NOTHING
            """,
            (content_id,),
        )


def set_openalex_status(
    conn,
    content_id,
    status,
    *,
    work_id=None,
    error=None,
    bump_attempts=False,
):
    ensure_paper_metadata_row(conn, content_id)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_radar.paper_metadata
            SET openalex_status = %s,
                openalex_work_id = COALESCE(%s, openalex_work_id),
                openalex_checked_at = NOW(),
                openalex_attempts = CASE WHEN %s THEN openalex_attempts + 1 ELSE openalex_attempts END,
                openalex_last_error = %s,
                modified_at = NOW()
            WHERE content_id = %s
            """,
            (status, work_id, bump_attempts, error, content_id),
        )


def load_orgs(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT organisation_id,canonical_name,aliases,domains,priority FROM research_radar.organisations WHERE active=TRUE")
        return cur.fetchall()


def load_people(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT person_id,canonical_name,aliases,openalex_id,orcid,priority FROM research_radar.people WHERE active=TRUE")
        return cur.fetchall()


def org_from_email(email,orgs):
    if "@" not in email: return None
    domain=email.rsplit("@",1)[1].lower().strip(" >).,;")
    matches=[]
    for o in orgs:
        if any(domain==d.lower() or domain.endswith("."+d.lower()) for d in o["domains"] or []): matches.append(o)
    return sorted(matches,key=lambda o:(-o["priority"],o["canonical_name"]))[0] if matches else None


def _alias_boundary_pattern(alias: str) -> re.Pattern | None:
    """Token/boundary-safe alias matcher — supports short aliases (MIT, IBM, xAI)."""
    name = (alias or "").strip()
    if not name:
        return None
    escaped = re.escape(name)
    escaped = re.sub(r"\s+", r"\\s+", escaped)
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)


def orgs_from_text(text, orgs):
    if not text:
        return []
    out = []
    for o in orgs:
        for name in [o["canonical_name"], *(o["aliases"] or [])]:
            pat = _alias_boundary_pattern(name)
            if pat and pat.search(text):
                out.append((o, name))
                break
    return out


def store_org_evidence(conn,content_id,org,etype,etext,eurl,confidence):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO research_radar.content_organisations(content_id,organisation_id,relationship_type,evidence_type,evidence_text,evidence_url,confidence,current_affiliation)
        VALUES(%s,%s,'paper_author_affiliation',%s,%s,%s,%s,FALSE)
        ON CONFLICT(content_id,organisation_id,relationship_type,evidence_type,COALESCE(evidence_text,'')) DO UPDATE SET confidence=GREATEST(research_radar.content_organisations.confidence,EXCLUDED.confidence),evidence_url=COALESCE(EXCLUDED.evidence_url,research_radar.content_organisations.evidence_url),observed_at=NOW()
        """,(content_id,org["organisation_id"],etype,etext,eurl,confidence))


def resolve_orgs_local(conn, content_id, p, orgs):
    """Email domain + explicit affiliation only (no OpenAlex)."""
    seen = set()
    count = 0
    for email in p.get("emails", []):
        org = org_from_email(email, orgs)
        if org and (org["organisation_id"], "email_domain", email) not in seen:
            store_org_evidence(conn, content_id, org, "email_domain", email, p.get("evidence_url"), 0.98)
            seen.add((org["organisation_id"], "email_domain", email))
            count += 1
    for txt in p.get("affiliation_text", []):
        for org, name in orgs_from_text(txt, orgs):
            key = (org["organisation_id"], "explicit_affiliation_text", name)
            if key not in seen:
                store_org_evidence(conn, content_id, org, "explicit_affiliation_text", name, p.get("evidence_url"), 1.0)
                seen.add(key)
                count += 1
    return count


def delete_local_org_evidence(conn, content_id):
    """Remove only deterministic local affiliation evidence (preserve Crossref/OpenAlex)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM research_radar.content_organisations
            WHERE content_id = %s
              AND relationship_type = 'paper_author_affiliation'
              AND current_affiliation = FALSE
              AND evidence_type = ANY(%s)
            """,
            (content_id, list(LOCAL_ORG_EVIDENCE_TYPES)),
        )
        return cur.rowcount


def reprocess_orgs_local(conn, content_id, p, orgs):
    """Re-run local org matching from stored enrichment; preserve external evidence."""
    deleted = delete_local_org_evidence(conn, content_id)
    count = resolve_orgs_local(conn, content_id, p, orgs)
    if count > 0:
        set_affiliation_status(conn, content_id, "NOT_NEEDED", error=None)
        set_openalex_status(conn, content_id, "NOT_NEEDED", error=None)
    elif count_content_org_matches(conn, content_id) == 0:
        set_affiliation_status(
            conn, content_id, "PENDING", error="awaiting_affiliation_gpt"
        )
    return count, deleted


def count_content_org_matches(conn, content_id):
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM research_radar.content_organisations
        WHERE content_id = %s
          AND relationship_type = 'paper_author_affiliation'
          AND current_affiliation = FALSE
        """,
        (content_id,),
    ).fetchone()
    return int(row["n"] or 0)


def apply_external_orgs(conn, content_id, payload, orgs, evidence_type, confidence):
    """Map external API institution strings onto watchlist orgs."""
    if not payload:
        return 0
    seen = set()
    count = 0
    for inst in payload.get("institutions") or []:
        for org, name in orgs_from_text(inst["institution"], orgs):
            etext = f"{inst.get('author') or ''} — {inst['institution']}".strip(" —")
            key = (org["organisation_id"], evidence_type, etext)
            if key in seen:
                continue
            store_org_evidence(
                conn,
                content_id,
                org,
                evidence_type,
                etext,
                payload.get("source_url"),
                confidence,
            )
            seen.add(key)
            count += 1
    return count


def apply_crossref_orgs(conn, content_id, cr, orgs):
    return apply_external_orgs(conn, content_id, cr, orgs, "paper_specific_crossref", 0.88)


def apply_openalex_orgs(conn, content_id, oa, orgs, evidence_type="paper_specific_openalex_doi"):
    return apply_external_orgs(conn, content_id, oa, orgs, evidence_type, 0.90)


def set_affiliation_status(
    conn,
    content_id,
    status,
    *,
    error=None,
    bump_attempts=False,
):
    """Generic affiliation-resolution status (GPT stage queue)."""
    ensure_paper_metadata_row(conn, content_id)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_radar.paper_metadata
            SET affiliation_status = %s,
                affiliation_checked_at = NOW(),
                affiliation_attempts = CASE WHEN %s THEN affiliation_attempts + 1 ELSE affiliation_attempts END,
                affiliation_last_error = %s,
                modified_at = NOW()
            WHERE content_id = %s
            """,
            (status, bump_attempts, error, content_id),
        )


def resolve_orgs(conn, content_id, p, orgs):
    """
    Deterministic org resolution only. Unresolved papers are queued for
    affiliation-gpt (not OpenAlex/Crossref).
    """
    count = resolve_orgs_local(conn, content_id, p, orgs)
    ensure_paper_metadata_row(conn, content_id)
    if count > 0:
        set_affiliation_status(conn, content_id, "NOT_NEEDED", error=None)
        # Keep historical openalex_* columns consistent for legacy tooling.
        set_openalex_status(conn, content_id, "NOT_NEEDED", error=None)
    else:
        set_affiliation_status(
            conn,
            content_id,
            "PENDING",
            error="awaiting_affiliation_gpt",
        )
        # Do not queue OpenAlex in the active pipeline.
        set_openalex_status(
            conn,
            content_id,
            "NOT_NEEDED",
            error="replaced_by_affiliation_gpt",
        )
    return count


def norm_name(s): return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()


def resolve_people(conn,content_id,authors,people):
    count=0
    with conn.cursor() as cur:
        for pos,author in enumerate(authors,start=1):
            na=norm_name(author)
            for p in people:
                if any(norm_name(x)==na for x in [p["canonical_name"],*(p["aliases"] or [])]):
                    conf=0.98 if p.get("openalex_id") or p.get("orcid") else 0.85
                    cur.execute("""
                    INSERT INTO research_radar.content_people(content_id,person_id,author_position,is_notable,match_confidence,evidence_type,evidence_text)
                    VALUES(%s,%s,%s,TRUE,%s,'author_name_watchlist_match',%s)
                    ON CONFLICT(content_id,person_id) DO UPDATE SET author_position=EXCLUDED.author_position,is_notable=TRUE,match_confidence=GREATEST(research_radar.content_people.match_confidence,EXCLUDED.match_confidence),evidence_type=EXCLUDED.evidence_type,evidence_text=EXCLUDED.evidence_text,observed_at=NOW()
                    """,(content_id,p["person_id"],pos,conf,author))
                    count+=1; break
    return count


def clamp(x): return max(0.0,min(10.0,round(float(x),2)))


def person_signal_enabled(active_people_count=0):
    if SCORE_INCLUDE_PERSON_SIGNAL in {"1", "true", "yes", "on"}:
        return True
    if SCORE_INCLUDE_PERSON_SIGNAL in {"0", "false", "no", "off"}:
        return False
    # auto: only weight person signal when a people watchlist exists
    return active_people_count > 0


def intrinsic_scores(ai_relevance, title, abstract, org_prios, person_prios, *, include_person_signal=True):
    text = f"{title} {abstract}".lower()

    technical_rules = []
    technical = 4.0
    if any(k in text for k in ["benchmark", "evaluation", "architecture", "algorithm", "framework", "system"]):
        technical += 1.5
        technical_rules.append("technical_method_signal")
    if any(k in text for k in ["experiment", "empirical", "dataset", "ablation"]):
        technical += 1.0
        technical_rules.append("technical_empirical_signal")

    practical_rules = []
    practical = 3.5
    if any(k in text for k in ["deployment", "production", "engineering", "workflow", "agent", "inference", "retrieval"]):
        practical += 2.0
        practical_rules.append("deployment_or_production_signal")
    if any(k in text for k in ["enterprise", "developer", "product", "application"]):
        practical += 1.0
        practical_rules.append("enterprise_or_product_signal")

    professional_rules = []
    professional = 4.0
    if practical >= 5.5:
        professional += 1.5
        professional_rules.append("high_practical_applicability_threshold")
    if technical >= 5.5:
        professional += 1.0
        professional_rules.append("high_technical_significance_threshold")

    learner_rules = []
    learner = 4.0
    if any(k in text for k in ["tutorial", "survey", "benchmark", "evaluation"]):
        learner += 1.0
        learner_rules.append("tutorial_survey_or_benchmark_signal")

    explainability_rules = []
    explainability = 5.0
    if len(abstract or "") < 2500:
        explainability += 1.0
        explainability_rules.append("short_abstract_heuristic")

    weights_with_person = {
        "ai_relevance": 0.30,
        "technical_significance": 0.15,
        "practical_applicability": 0.15,
        "professional_value": 0.10,
        "student_learning_value": 0.10,
        "notable_org_signal": 0.10,
        "notable_person_signal": 0.10,
    }
    weights_without_person = {
        "ai_relevance": 0.30 / 0.90,
        "technical_significance": 0.15 / 0.90,
        "practical_applicability": 0.15 / 0.90,
        "professional_value": 0.10 / 0.90,
        "student_learning_value": 0.10 / 0.90,
        "notable_org_signal": 0.10 / 0.90,
    }

    s = {
        "ai_relevance": clamp(ai_relevance),
        "technical_significance": clamp(technical),
        "novelty": 5.0,
        "notable_person_signal": clamp(max(person_prios or [0])),
        "notable_org_signal": clamp(max(org_prios or [0])),
        "professional_value": clamp(professional),
        "student_learning_value": clamp(learner),
        "practical_applicability": clamp(practical),
        "explainability": clamp(explainability),
    }
    if include_person_signal:
        weights = weights_with_person
        s["intrinsic_candidate_score"] = clamp(
            weights["ai_relevance"] * s["ai_relevance"]
            + weights["technical_significance"] * s["technical_significance"]
            + weights["practical_applicability"] * s["practical_applicability"]
            + weights["professional_value"] * s["professional_value"]
            + weights["student_learning_value"] * s["student_learning_value"]
            + weights["notable_org_signal"] * s["notable_org_signal"]
            + weights["notable_person_signal"] * s["notable_person_signal"]
        )
    else:
        weights = weights_without_person
        s["intrinsic_candidate_score"] = clamp(
            weights["ai_relevance"] * s["ai_relevance"]
            + weights["technical_significance"] * s["technical_significance"]
            + weights["practical_applicability"] * s["practical_applicability"]
            + weights["professional_value"] * s["professional_value"]
            + weights["student_learning_value"] * s["student_learning_value"]
            + weights["notable_org_signal"] * s["notable_org_signal"]
        )

    provenance = {
        "method": "deterministic_heuristic",
        "version": "deterministic-v0.2",
        "dimensions": {
            "technical_significance": {
                "score": s["technical_significance"],
                "base": 4.0,
                "matched_rules": technical_rules,
            },
            "practical_applicability": {
                "score": s["practical_applicability"],
                "base": 3.5,
                "matched_rules": practical_rules,
            },
            "professional_value": {
                "score": s["professional_value"],
                "base": 4.0,
                "matched_rules": professional_rules,
            },
            "student_learning_value": {
                "score": s["student_learning_value"],
                "base": 4.0,
                "matched_rules": learner_rules,
            },
            "explainability": {
                "score": s["explainability"],
                "base": 5.0,
                "matched_rules": explainability_rules,
            },
            "novelty": {
                "score": s["novelty"],
                "matched_rules": [],
                "novelty_method": "fixed_proxy",
            },
            "notable_org_signal": {
                "score": s["notable_org_signal"],
                "matched_rules": ["watchlist_org_priority_max"] if s["notable_org_signal"] > 0 else [],
            },
            "notable_person_signal": {
                "score": s["notable_person_signal"],
                "matched_rules": ["watchlist_person_priority_max"] if s["notable_person_signal"] > 0 else [],
            },
            "ai_relevance": {
                "score": s["ai_relevance"],
                "matched_rules": ["deterministic_relevance_stage"],
            },
        },
        "weights": weights,
        "person_weight_enabled": include_person_signal,
        "intrinsic_candidate_score": s["intrinsic_candidate_score"],
        "industry_relevance": {
            "status": "not_yet_semantically_scored",
            "note": "Industry relevance is not populated by deterministic-v0.2; planned for LLM scoring stage.",
        },
    }
    return s, provenance


def store_scores(conn, content_id, s, scoring_reason, relevance_reason=None):
    reason = dict(scoring_reason)
    if relevance_reason:
        reason["relevance_reason"] = relevance_reason
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO research_radar.content_scores(content_id,ai_relevance,technical_significance,novelty,notable_person_signal,notable_org_signal,professional_value,student_learning_value,practical_applicability,explainability,intrinsic_candidate_score,score_version,scoring_reason,scored_at)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'deterministic-v0.2',%s::jsonb,NOW())
        ON CONFLICT(content_id) DO UPDATE SET ai_relevance=EXCLUDED.ai_relevance,technical_significance=EXCLUDED.technical_significance,novelty=EXCLUDED.novelty,notable_person_signal=EXCLUDED.notable_person_signal,notable_org_signal=EXCLUDED.notable_org_signal,professional_value=EXCLUDED.professional_value,student_learning_value=EXCLUDED.student_learning_value,practical_applicability=EXCLUDED.practical_applicability,explainability=EXCLUDED.explainability,intrinsic_candidate_score=EXCLUDED.intrinsic_candidate_score,score_version=EXCLUDED.score_version,scoring_reason=EXCLUDED.scoring_reason,scored_at=NOW()
        """,(content_id,s["ai_relevance"],s["technical_significance"],s["novelty"],s["notable_person_signal"],s["notable_org_signal"],s["professional_value"],s["student_learning_value"],s["practical_applicability"],s["explainability"],s["intrinsic_candidate_score"],json.dumps(reason)))


def store_opportunities(conn,content_id,s,primary):
    opp=[("raw_paper","all",1.0,"Canonical research record retained.")]
    if s["notable_org_signal"]>=8 or s["notable_person_signal"]>=8: opp.append(("notable_research","tech_product_professional",0.85,"Strong watchlist organisation/person signal."))
    if s["professional_value"]>=6: opp.append(("professional_learning","tech_product_professional",0.80,"Deterministic professional-value score is high."))
    if s["student_learning_value"]>=6 and s["explainability"]>=5: opp.append(("student_learning","student_early_career",0.72,"Learning value and explainability are sufficient."))
    if s["technical_significance"]>=6: opp.append(("technical_deep_dive","tech_product_professional",0.75,"Technical-significance proxy is high."))
    if s["practical_applicability"]>=6: opp.append(("explainer_candidate","tech_product_professional",0.76,f"Practical applicability is high for topic: {primary or 'AI'}."))
    if s["professional_value"]>=6 and s["intrinsic_candidate_score"]>=MIN_CANDIDATE_SCORE: opp.append(("weekend_read","tech_product_professional",0.70,"Strong intrinsic and professional-value signals."))
    with conn.cursor() as cur:
        for kind,aud,conf,why in opp:
            cur.execute("""
            INSERT INTO research_radar.content_opportunities(content_id,opportunity_type,audience,confidence,reason)
            VALUES(%s,%s,%s,%s,%s)
            ON CONFLICT(content_id,opportunity_type,COALESCE(audience,'')) DO UPDATE SET confidence=EXCLUDED.confidence,reason=EXCLUDED.reason
            """,(content_id,kind,aud,conf,why))


def signal_priorities(conn,content_id):
    with conn.cursor() as cur:
        cur.execute("SELECT o.priority FROM research_radar.content_organisations co JOIN research_radar.organisations o ON o.organisation_id=co.organisation_id WHERE co.content_id=%s AND co.relationship_type='paper_author_affiliation' AND co.current_affiliation=FALSE",(content_id,))
        org=[r["priority"] for r in cur.fetchall()]
        cur.execute("SELECT p.priority FROM research_radar.content_people cp JOIN research_radar.people p ON p.person_id=cp.person_id WHERE cp.content_id=%s AND cp.is_notable=TRUE",(content_id,))
        people=[r["priority"] for r in cur.fetchall()]
    return org,people


def process_item(conn,run_id,item,orgs,people):
    content_id,is_new=upsert_item(conn,item); bump(conn,run_id,"items_new" if is_new else "items_duplicate")
    try:
        score,primary,secondary,reason=score_relevance(item["title"],item.get("summary") or "",item.get("categories_raw") or [],item["source_type"])
        store_relevance(conn,content_id,score,primary,secondary,reason); set_status(conn,content_id,"RELEVANCE_CHECKED")
        event(conn,run_id,content_id,"relevance","deterministic_relevance",True,{"score":score,"primary_topic":primary,"reason":reason})
        if score<MIN_AI_RELEVANCE:
            set_status(conn,content_id,"REJECTED"); return
        bump(conn,run_id,"items_relevant"); set_status(conn,content_id,"RELEVANT")
        enriched={"paper_title":item["title"],"abstract":item.get("summary") or "","authors":item.get("authors_raw") or [],"arxiv_categories":item.get("categories_raw") or [],"emails":[],"affiliation_text":[],"evidence_url":None}
        if item["source_type"]=="arxiv":
            enriched=enrich_arxiv(item["canonical_url"]); store_paper(conn,content_id,enriched); bump(conn,run_id,"items_enriched"); set_status(conn,content_id,"ENRICHED")
            event(conn,run_id,content_id,"paper_enrichment","arxiv_enrichment",True,{"arxiv_id":enriched.get("arxiv_id"),"emails":enriched.get("emails",[])})
        oc=resolve_orgs(conn,content_id,enriched,orgs); pc=resolve_people(conn,content_id,enriched.get("authors",[]),people)
        bump(conn,run_id,"orgs_resolved",oc); bump(conn,run_id,"people_resolved",pc); set_status(conn,content_id,"ENTITY_RESOLVED")
        op,pp=signal_priorities(conn,content_id)
        include_person = person_signal_enabled(len(people))
        s, provenance = intrinsic_scores(score,enriched.get("paper_title") or item["title"],enriched.get("abstract") or item.get("summary") or "",op,pp,include_person_signal=include_person)
        store_scores(conn,content_id,s,provenance,reason); bump(conn,run_id,"items_scored"); set_status(conn,content_id,"SCORED"); store_opportunities(conn,content_id,s,primary)
        if s["intrinsic_candidate_score"]>=MIN_CANDIDATE_SCORE:
            set_status(conn,content_id,"CANDIDATE"); bump(conn,run_id,"candidates_created")
    except Exception as exc:
        log.exception("Item failed id=%s title=%s",content_id,item.get("title")); set_status(conn,content_id,"ERROR"); bump(conn,run_id,"errors"); event(conn,run_id,content_id,"item","processing_error",False,{"title":item.get("title")},str(exc))


def start_run(conn, stage):
    run_id = uuid.uuid4()
    conn.execute(
        "INSERT INTO research_radar.pipeline_runs(run_id, notes) VALUES(%s, %s::jsonb)",
        (run_id, json.dumps({"stage": stage})),
    )
    # Commit immediately so parallel worker connections can reference this run_id.
    conn.commit()
    log.info("Started pipeline_run=%s stage=%s", run_id, stage)
    return run_id


def finish_run(conn, run_id, starting_calls, ok=True, error=None):
    if ok:
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET ended_at=NOW(), http_calls=http_calls+%s, status='COMPLETED' WHERE run_id=%s",
            (HTTP_CALLS - starting_calls, run_id),
        )
    else:
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET ended_at=NOW(), http_calls=http_calls+%s, errors=errors+1, status='FAILED', notes=notes||%s::jsonb WHERE run_id=%s",
            (HTTP_CALLS - starting_calls, json.dumps({"fatal_error": str(error)}), run_id),
        )


def print_status_counts(conn):
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM research_radar.content_items GROUP BY status ORDER BY status"
    ).fetchall()
    if not rows:
        log.info("DB status counts: (no content_items yet)")
        return
    summary = ", ".join(f"{r['status']}={r['n']}" for r in rows)
    log.info("DB status counts: %s", summary)


def stage_ingest(conn, run_id, limit=None):
    items = fetch_inoreader_items()
    if limit is not None:
        items = items[:limit]
    bump(conn, run_id, "items_received", len(items))
    log.info("Ingest: received %d items from Inoreader/fixture", len(items))
    for i, item in enumerate(items, 1):
        content_id, is_new = upsert_item(conn, item)
        bump(conn, run_id, "items_new" if is_new else "items_duplicate")
        if is_new:
            set_status(conn, content_id, "INGESTED")
            event(conn, run_id, content_id, "ingest", "upsert", True, {"title": item.get("title"), "is_new": True})
        else:
            existing = conn.execute(
                "SELECT status FROM research_radar.content_items WHERE id=%s",
                (content_id,),
            ).fetchone()
            event(
                conn,
                run_id,
                content_id,
                "ingest",
                "ingest_duplicate_preserved",
                True,
                {
                    "title": item.get("title"),
                    "is_new": False,
                    "existing_status": existing["status"] if existing else None,
                    "canonical_url": item.get("canonical_url"),
                },
            )
        if i % 25 == 0 or i == len(items):
            log.info("Ingest progress %d/%d last_id=%s title=%s", i, len(items), content_id, (item.get("title") or "")[:80])
    return len(items)


def stage_repair_timestamps(conn, run_id, limit=None):
    """One-time repair: re-parse published_at / source_seen_at from stored Inoreader raw_metadata."""
    rows = conn.execute(
        """
        SELECT id, title, raw_metadata, published_at, source_seen_at, updated_at
        FROM research_radar.content_items
        WHERE source = 'inoreader'
          AND raw_metadata IS NOT NULL
          AND raw_metadata::text <> '{}'
        ORDER BY id
        LIMIT %s
        """,
        (limit or 10_000,),
    ).fetchall()
    log.info("Repair timestamps: scanning %d inoreader rows with raw_metadata", len(rows))
    updated = unchanged = 0
    for i, row in enumerate(rows, 1):
        pub, seen, upd = timestamps_from_inoreader_raw(row["raw_metadata"])
        new_pub = pub if pub is not None else row["published_at"]
        new_seen = seen if seen is not None else row["source_seen_at"]
        new_upd = upd if upd is not None else row.get("updated_at")
        if (
            new_pub == row["published_at"]
            and new_seen == row["source_seen_at"]
            and new_upd == row.get("updated_at")
        ):
            unchanged += 1
            continue
        conn.execute(
            """
            UPDATE research_radar.content_items
            SET published_at = %s,
                source_seen_at = %s,
                updated_at = %s,
                modified_at = NOW()
            WHERE id = %s
            """,
            (new_pub, new_seen, new_upd, row["id"]),
        )
        event(
            conn,
            run_id,
            row["id"],
            "repair_timestamps",
            "updated_from_raw_metadata",
            True,
            {
                "published_at": str(new_pub) if new_pub else None,
                "source_seen_at": str(new_seen) if new_seen else None,
                "updated_at": str(new_upd) if new_upd else None,
            },
        )
        updated += 1
        if i % 100 == 0 or i == len(rows):
            log.info("Repair timestamps progress %d/%d updated=%d unchanged=%d", i, len(rows), updated, unchanged)
    log.info("Repair timestamps done updated=%d unchanged=%d", updated, unchanged)
    bump(conn, run_id, "items_received", updated)
    return updated


def stage_relevance(conn, run_id, limit=None):
    rows = conn.execute(
        """
        SELECT id, title, summary, categories_raw, source_type
        FROM research_radar.content_items
        WHERE status IN ('INGESTED', 'RELEVANCE_CHECKED', 'ERROR')
        ORDER BY id
        LIMIT %s
        """,
        (limit or 10_000,),
    ).fetchall()
    log.info("Relevance: processing %d items", len(rows))
    for i, row in enumerate(rows, 1):
        try:
            score, primary, secondary, reason = score_relevance(
                row["title"], row.get("summary") or "", row.get("categories_raw") or [], row["source_type"]
            )
            store_relevance(conn, row["id"], score, primary, secondary, reason)
            set_status(conn, row["id"], "RELEVANCE_CHECKED")
            event(conn, run_id, row["id"], "relevance", "deterministic_relevance", True, {"score": score, "primary_topic": primary, "reason": reason})
            if score < MIN_AI_RELEVANCE:
                set_status(conn, row["id"], "REJECTED")
                log.info("Relevance REJECTED id=%s score=%.2f title=%s", row["id"], score, (row["title"] or "")[:80])
            else:
                bump(conn, run_id, "items_relevant")
                set_status(conn, row["id"], "RELEVANT")
                log.info("Relevance KEEP id=%s score=%.2f topic=%s title=%s", row["id"], score, primary, (row["title"] or "")[:80])
        except Exception as exc:
            log.exception("Relevance failed id=%s", row["id"])
            set_status(conn, row["id"], "ERROR")
            bump(conn, run_id, "errors")
            event(conn, run_id, row["id"], "relevance", "processing_error", False, {"title": row.get("title")}, str(exc))
        if i % 25 == 0 or i == len(rows):
            log.info("Relevance progress %d/%d", i, len(rows))
    return len(rows)


def stage_enrich(conn, run_id, limit=None):
    rows = conn.execute(
        """
        SELECT id, title, summary, authors_raw, categories_raw, source_type, canonical_url
        FROM research_radar.content_items
        WHERE status IN ('RELEVANT', 'ERROR')
          AND source_type = 'arxiv'
        ORDER BY id
        LIMIT %s
        """,
        (limit or 10_000,),
    ).fetchall()
    workers = max(1, ARXIV_WORKERS)
    commit_every = max(1, ARXIV_COMMIT_EVERY)
    log.info(
        "Enrich: processing %d arXiv items workers=%d commit_every=%d",
        len(rows),
        workers,
        commit_every,
    )

    done = 0
    for start in range(0, len(rows), commit_every):
        chunk = rows[start : start + commit_every]
        id_list = []
        row_by_arxiv = {}
        for row in chunk:
            aid, ver = extract_arxiv_id(row["canonical_url"])
            if not aid:
                set_status(conn, row["id"], "ERROR")
                bump(conn, run_id, "errors")
                event(conn, run_id, row["id"], "paper_enrichment", "processing_error", False, {"title": row.get("title")}, "missing arxiv id")
                continue
            id_list.append(aid)
            row_by_arxiv[aid] = (row, ver)

        try:
            meta_by_id = enrich_arxiv_atom_batch(id_list)
        except Exception as exc:
            log.exception("Batch Atom fetch failed for %d ids; falling back per-item", len(id_list))
            meta_by_id = {}
            for aid in id_list:
                try:
                    meta_by_id.update(enrich_arxiv_atom_batch([aid]))
                except Exception:
                    log.warning("Atom fetch failed for %s: %s", aid, exc)

        def _work(aid: str):
            row, ver = row_by_arxiv[aid]
            base = meta_by_id.get(aid)
            if not base:
                raise ValueError(f"arXiv metadata not found for {aid}")
            if ver is not None:
                base = dict(base)
                base["arxiv_version"] = ver
            emails, affs, evidence_url = extract_arxiv_affiliations(aid)
            enriched = dict(base)
            enriched["emails"] = emails
            enriched["affiliation_text"] = affs
            enriched["evidence_url"] = evidence_url
            return row, enriched

        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for aid in list(row_by_arxiv.keys()):
                futures[pool.submit(_work, aid)] = aid
            for fut in as_completed(futures):
                aid = futures[fut]
                row, _ver = row_by_arxiv[aid]
                try:
                    row, enriched = fut.result()
                    store_paper(conn, row["id"], enriched)
                    bump(conn, run_id, "items_enriched")
                    set_status(conn, row["id"], "ENRICHED")
                    event(
                        conn,
                        run_id,
                        row["id"],
                        "paper_enrichment",
                        "arxiv_enrichment",
                        True,
                        {"arxiv_id": enriched.get("arxiv_id"), "emails": enriched.get("emails", [])},
                    )
                    log.info(
                        "Enrich OK id=%s arxiv=%s emails=%d title=%s",
                        row["id"],
                        enriched.get("arxiv_id"),
                        len(enriched.get("emails") or []),
                        (enriched.get("paper_title") or row["title"] or "")[:80],
                    )
                except Exception as exc:
                    log.exception("Enrich failed id=%s arxiv=%s", row["id"], aid)
                    set_status(conn, row["id"], "ERROR")
                    bump(conn, run_id, "errors")
                    event(
                        conn,
                        run_id,
                        row["id"],
                        "paper_enrichment",
                        "processing_error",
                        False,
                        {"title": row.get("title"), "arxiv_id": aid},
                        str(exc),
                    )

        conn.commit()
        done += len(chunk)
        log.info("Enrich progress %d/%d (committed batch of up to %d)", done, len(rows), commit_every)

    non_arxiv = conn.execute(
        "SELECT COUNT(*) AS n FROM research_radar.content_items WHERE status='RELEVANT' AND source_type <> 'arxiv'"
    ).fetchone()["n"]
    if non_arxiv:
        log.info("Enrich: %d non-arxiv RELEVANT items left for entities/score without paper_metadata", non_arxiv)
    return len(rows)


def _load_enriched_payload(conn, row):
    paper = conn.execute(
        "SELECT * FROM research_radar.paper_metadata WHERE content_id=%s",
        (row["id"],),
    ).fetchone()
    if paper:
        return {
            "paper_title": row["title"],
            "abstract": paper.get("abstract") or row.get("summary") or "",
            "authors": paper.get("authors_raw") or row.get("authors_raw") or [],
            "arxiv_categories": paper.get("categories") or row.get("categories_raw") or [],
            "emails": paper.get("extracted_emails") or [],
            "affiliation_text": paper.get("affiliation_text") or [],
            "evidence_url": (paper.get("enrichment_metadata") or {}).get("evidence_url"),
        }
    return {
        "paper_title": row["title"],
        "abstract": row.get("summary") or "",
        "authors": row.get("authors_raw") or [],
        "arxiv_categories": row.get("categories_raw") or [],
        "emails": [],
        "affiliation_text": [],
        "evidence_url": None,
    }


def stage_entities(conn, run_id, limit=None):
    orgs = [dict(o) for o in load_orgs(conn)]
    people = [dict(p) for p in load_people(conn)]
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT id, title, summary, authors_raw, categories_raw, source_type, status
            FROM research_radar.content_items
            WHERE status IN ('RELEVANT', 'ENRICHED', 'ERROR')
            ORDER BY id
            LIMIT %s
            """,
            (limit or 10_000,),
        ).fetchall()
    ]
    workers = max(1, PIPELINE_WORKERS)
    commit_every = max(1, ARXIV_COMMIT_EVERY)
    log.info(
        "Entities: processing %d items workers=%d commit_every=%d (orgs=%d people=%d)",
        len(rows),
        workers,
        commit_every,
        len(orgs),
        len(people),
    )

    def _work(row):
        with connect() as wconn:
            try:
                enriched = _load_enriched_payload(wconn, row)
                oc = resolve_orgs(wconn, row["id"], enriched, orgs)
                pc = resolve_people(wconn, row["id"], enriched.get("authors", []), people)
                bump(wconn, run_id, "orgs_resolved", oc)
                bump(wconn, run_id, "people_resolved", pc)
                set_status(wconn, row["id"], "ENTITY_RESOLVED")
                event(wconn, run_id, row["id"], "entities", "resolve", True, {"orgs": oc, "people": pc})
                wconn.commit()
                return ("ok", row["id"], oc, pc, row.get("title") or "")
            except Exception as exc:
                wconn.rollback()
                try:
                    set_status(wconn, row["id"], "ERROR")
                    bump(wconn, run_id, "errors")
                    event(
                        wconn,
                        run_id,
                        row["id"],
                        "entities",
                        "processing_error",
                        False,
                        {"title": row.get("title")},
                        str(exc),
                    )
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("err", row["id"], str(exc), row.get("title") or "")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, row) for row in rows]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result[0] == "ok":
                _, cid, oc, pc, title = result
                log.info("Entities id=%s orgs=%d people=%d title=%s", cid, oc, pc, title[:80])
            else:
                _, cid, err, title = result
                log.error("Entities failed id=%s title=%s err=%s", cid, title[:80], err)
            if done % commit_every == 0 or done == len(rows):
                log.info("Entities progress %d/%d", done, len(rows))
    return len(rows)


def stage_entities_reprocess(conn, run_id, limit=None):
    """
    Re-run local org matching from stored paper_metadata only.
    Deletes/recalculates email_domain + explicit_affiliation_text evidence.
    Preserves Crossref/OpenAlex evidence and content workflow status.
    """
    orgs = [dict(o) for o in load_orgs(conn)]
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT ci.id, ci.title, ci.summary, ci.authors_raw, ci.categories_raw,
                   ci.source_type, ci.status
            FROM research_radar.content_items ci
            INNER JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
            WHERE ci.status IN ('ENRICHED', 'ENTITY_RESOLVED', 'SCORED', 'CANDIDATE', 'RELEVANT', 'ERROR')
            ORDER BY ci.id
            LIMIT %s
            """,
            (limit or 10_000,),
        ).fetchall()
    ]
    log.info(
        "Entities-reprocess: %d enriched papers (orgs=%d) — local evidence only, no status change",
        len(rows),
        len(orgs),
    )
    total_added = total_deleted = errors = 0
    for i, row in enumerate(rows, 1):
        try:
            enriched = _load_enriched_payload(conn, row)
            added, deleted = reprocess_orgs_local(conn, row["id"], enriched, orgs)
            total_added += added
            total_deleted += deleted
            if added or deleted:
                bump(conn, run_id, "orgs_resolved", added)
                event(
                    conn,
                    run_id,
                    row["id"],
                    "entities_reprocess",
                    "local_orgs_recalculated",
                    True,
                    {
                        "orgs_added": added,
                        "local_evidence_deleted": deleted,
                        "status_preserved": row["status"],
                    },
                )
                log.info(
                    "Entities-reprocess id=%s +%d -%d status=%s title=%s",
                    row["id"],
                    added,
                    deleted,
                    row["status"],
                    (row.get("title") or "")[:80],
                )
        except Exception as exc:
            errors += 1
            bump(conn, run_id, "errors")
            event(
                conn,
                run_id,
                row["id"],
                "entities_reprocess",
                "processing_error",
                False,
                {"title": row.get("title"), "status": row["status"]},
                str(exc),
            )
            log.exception("Entities-reprocess failed id=%s", row["id"])
        if i % 50 == 0 or i == len(rows):
            log.info(
                "Entities-reprocess progress %d/%d added=%d deleted=%d errors=%d",
                i,
                len(rows),
                total_added,
                total_deleted,
                errors,
            )
    return len(rows)


def stage_openalex(conn, run_id, limit=None):
    """
    DEPRECATED for active pipeline — use affiliation-gpt instead.

    Legacy Crossref/OpenAlex DOI enrichment. Remains callable only when
    OPENALEX_ENABLED or CROSSREF_ENABLED is explicitly true. Historical
    OpenAlex evidence in the DB is preserved.
    """
    from research_radar.affiliation_external import (
        AffiliationStageStats,
        OpenAlexBudgetExhausted,
        OpenAlexRateLimited,
        is_openalex_budget_exhausted,
        normalize_doi,
        reset_openalex_budget_flag,
        resolve_crossref_by_doi,
        resolve_openalex_by_doi,
        resolve_openalex_by_title,
    )

    reset_openalex_budget_flag()

    if not CROSSREF_ENABLED and not OPENALEX_ENABLED:
        log.warning(
            "stage_openalex is deprecated/inactive "
            "(OPENALEX_ENABLED=false, CROSSREF_ENABLED=false). "
            "Use affiliation-gpt instead."
        )
        return 0

    log.warning(
        "Running deprecated stage_openalex (legacy). Prefer affiliation-gpt for new runs."
    )
    if OPENALEX_ENABLED and not OPENALEX_API_KEY:
        log.warning("OpenAlex enabled but OPENALEX_API_KEY missing — lookups will likely fail")

    orgs = [dict(o) for o in load_orgs(conn)]
    doi_clause = ""
    if not OPENALEX_TITLE_SEARCH_ENABLED:
        doi_clause = " AND pm.doi IS NOT NULL AND trim(pm.doi) <> ''"
    rows = [
        dict(r)
        for r in conn.execute(
            f"""
            SELECT pm.content_id AS id,
                   ci.title,
                   ci.status AS item_status,
                   pm.doi,
                   pm.abstract,
                   pm.openalex_status,
                   pm.openalex_attempts
            FROM research_radar.paper_metadata pm
            JOIN research_radar.content_items ci ON ci.id = pm.content_id
            WHERE pm.openalex_status IN ('PENDING', 'RATE_LIMITED', 'ERROR')
              AND ci.status IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE', 'ENRICHED', 'RELEVANT', 'ERROR')
              {doi_clause}
            ORDER BY
              CASE pm.openalex_status
                WHEN 'PENDING' THEN 0
                WHEN 'RATE_LIMITED' THEN 1
                ELSE 2
              END,
              pm.openalex_attempts ASC,
              pm.content_id ASC
            LIMIT %s
            """,
            (limit or 10_000,),
        ).fetchall()
    ]

    stats = AffiliationStageStats(
        total_unresolved=len(rows),
        http_calls_start=HTTP_CALLS,
    )
    workers = max(1, min(OPENALEX_WORKERS, 1))
    log.info(
        "External affiliation: %d unresolved crossref=%s openalex=%s workers=%d title_search=%s",
        len(rows),
        CROSSREF_ENABLED,
        OPENALEX_ENABLED,
        workers,
        OPENALEX_TITLE_SEARCH_ENABLED,
    )
    if not rows:
        return 0

    def _maybe_rescore(wconn, row, matched_n):
        if matched_n > 0 and row.get("item_status") in {"SCORED", "CANDIDATE", "ENTITY_RESOLVED"}:
            set_status(wconn, row["id"], "ENTITY_RESOLVED")

    def _process_row(row):
        if is_openalex_budget_exhausted():
            return ("budget_skip", row["id"], row.get("title") or "")

        with connect() as wconn:
            content_id = row["id"]
            doi = normalize_doi(row.get("doi"))
            title = row.get("title") or ""
            orgs_before = count_content_org_matches(wconn, content_id)
            work_id = None
            openalex_doi_tried = False
            openalex_doi_found = False

            try:
                # 1) Crossref DOI (explicit affiliations only)
                if CROSSREF_ENABLED and doi:
                    cr = resolve_crossref_by_doi(doi, CROSSREF_MAILTO, stats)
                    if cr:
                        matched = apply_crossref_orgs(wconn, content_id, cr, orgs)
                        if matched:
                            stats.crossref_matched += 1
                            bump(wconn, run_id, "orgs_resolved", matched)
                            event(
                                wconn,
                                run_id,
                                content_id,
                                "openalex",
                                "crossref_matched",
                                True,
                                {"orgs": matched, "doi": doi},
                            )

                # 2) OpenAlex singleton DOI (never title-search on 429/5xx)
                if OPENALEX_ENABLED and doi and count_content_org_matches(wconn, content_id) == orgs_before:
                    openalex_doi_tried = True
                    oa = resolve_openalex_by_doi(doi, stats)
                    if oa:
                        openalex_doi_found = True
                        matched = apply_openalex_orgs(
                            wconn,
                            content_id,
                            oa,
                            orgs,
                            evidence_type="paper_specific_openalex_doi",
                        )
                        if matched:
                            stats.openalex_doi_matched += 1
                            bump(wconn, run_id, "orgs_resolved", matched)
                        work_id = oa.get("work_id")

                # 3) Optional explicit title search (disabled by default)
                if (
                    OPENALEX_ENABLED
                    and OPENALEX_TITLE_SEARCH_ENABLED
                    and not doi
                    and count_content_org_matches(wconn, content_id) == orgs_before
                ):
                    oa_title = resolve_openalex_by_title(title, stats)
                    if oa_title:
                        matched = apply_openalex_orgs(
                            wconn,
                            content_id,
                            oa_title,
                            orgs,
                            evidence_type="paper_specific_openalex",
                        )
                        if matched:
                            stats.openalex_title_matched += 1
                            bump(wconn, run_id, "orgs_resolved", matched)
                        work_id = oa_title.get("work_id")

                orgs_after = count_content_org_matches(wconn, content_id)
                gained = orgs_after - orgs_before

                if gained > 0:
                    set_openalex_status(
                        wconn,
                        content_id,
                        "MATCHED",
                        work_id=work_id,
                        error=None,
                        bump_attempts=True,
                    )
                    _maybe_rescore(wconn, row, gained)
                    event(
                        wconn,
                        run_id,
                        content_id,
                        "openalex",
                        "matched",
                        True,
                        {"orgs": gained, "work_id": work_id, "doi": doi},
                    )
                    wconn.commit()
                    return ("matched", content_id, gained, title)

                if not doi:
                    if OPENALEX_TITLE_SEARCH_ENABLED:
                        set_openalex_status(
                            wconn,
                            content_id,
                            "PENDING",
                            error="no_doi_awaiting_title_or_manual",
                            bump_attempts=True,
                        )
                        stats.pending_remaining += 1
                        event(wconn, run_id, content_id, "openalex", "pending_no_doi", True, {})
                        wconn.commit()
                        return ("pending", content_id, title)
                    # Title search disabled — should not be selected; leave status unchanged.
                    wconn.commit()
                    return ("deferred_no_doi", content_id, title)

                if openalex_doi_tried and not openalex_doi_found:
                    set_openalex_status(
                        wconn,
                        content_id,
                        "PENDING",
                        error="openalex_doi_not_found_allow_title_fallback",
                        bump_attempts=True,
                    )
                    stats.pending_remaining += 1
                    event(wconn, run_id, content_id, "openalex", "doi_not_found", True, {"doi": doi})
                    wconn.commit()
                    return ("pending", content_id, title)

                set_openalex_status(
                    wconn,
                    content_id,
                    "NO_MATCH",
                    error="no_external_affiliation_evidence",
                    bump_attempts=True,
                )
                stats.no_match += 1
                event(wconn, run_id, content_id, "openalex", "no_match", True, {"doi": doi})
                wconn.commit()
                return ("no_match", content_id, title)

            except OpenAlexBudgetExhausted as exc:
                stats.budget_exhausted = True
                wconn.rollback()
                try:
                    set_openalex_status(
                        wconn,
                        content_id,
                        "RATE_LIMITED",
                        error=str(exc)[:500],
                        bump_attempts=True,
                    )
                    stats.rate_limited += 1
                    event(wconn, run_id, content_id, "openalex", "budget_exhausted", False, {}, str(exc))
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("budget", content_id, title)
            except OpenAlexRateLimited as exc:
                wconn.rollback()
                try:
                    set_openalex_status(
                        wconn,
                        content_id,
                        "RATE_LIMITED",
                        error=str(exc)[:500],
                        bump_attempts=True,
                    )
                    stats.rate_limited += 1
                    event(wconn, run_id, content_id, "openalex", "rate_limited", False, {}, str(exc))
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("rate", content_id, str(exc), title)
            except Exception as exc:
                wconn.rollback()
                try:
                    set_openalex_status(
                        wconn,
                        content_id,
                        "ERROR",
                        error=str(exc)[:500],
                        bump_attempts=True,
                    )
                    stats.errors += 1
                    bump(wconn, run_id, "errors")
                    event(wconn, run_id, content_id, "openalex", "error", False, {}, str(exc))
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("err", content_id, str(exc), title)

    done = matched_n = pending_n = no_match_n = rate_n = err_n = 0
    for row in rows:
        if is_openalex_budget_exhausted():
            stats.pending_remaining += 1
            stats.budget_exhausted = True
            done += 1
            continue

        result = _process_row(row)
        done += 1
        kind = result[0]
        if kind == "matched":
            matched_n += 1
            log.info("Affiliation MATCHED id=%s orgs=%s title=%s", result[1], result[2], result[3][:80])
        elif kind == "pending":
            pending_n += 1
            log.info("Affiliation PENDING id=%s title=%s", result[1], result[2][:80])
        elif kind == "no_match":
            no_match_n += 1
            log.info("Affiliation NO_MATCH id=%s title=%s", result[1], result[2][:80])
        elif kind == "budget":
            rate_n += 1
            log.warning("OpenAlex BUDGET exhausted at id=%s — zero further API calls this run", result[1])
        elif kind == "budget_skip":
            stats.pending_remaining += 1
        elif kind == "rate":
            rate_n += 1
            log.warning("Affiliation RATE_LIMITED id=%s err=%s", result[1], result[2])
        else:
            err_n += 1
            log.error("Affiliation ERROR id=%s err=%s", result[1], result[2])

        if done % max(1, ARXIV_COMMIT_EVERY) == 0 or done == len(rows):
            log.info(
                "Affiliation progress %d/%d matched=%d pending=%d no_match=%d rate=%d err=%d",
                done,
                len(rows),
                matched_n,
                pending_n,
                no_match_n,
                rate_n,
                err_n,
            )

    stats.record_http()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"affiliation_stage_stats": stats.to_dict()}), run_id),
    )
    log.info("Affiliation stage stats: %s", json.dumps(stats.to_dict()))
    return len(rows)


def stage_score(conn, run_id, limit=None):
    people_n = conn.execute(
        "SELECT COUNT(*) AS n FROM research_radar.people WHERE active=TRUE"
    ).fetchone()["n"]
    include_person = person_signal_enabled(people_n)
    log.info(
        "Score: person_signal_enabled=%s (active_people=%s threshold=%.2f)",
        include_person,
        people_n,
        MIN_CANDIDATE_SCORE,
    )
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT ci.id, ci.title, ci.summary, ci.authors_raw, ci.categories_raw, ci.source_type,
                   cs.ai_relevance, cs.scoring_reason
            FROM research_radar.content_items ci
            LEFT JOIN research_radar.content_scores cs ON cs.content_id = ci.id
            WHERE ci.status IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE', 'ERROR')
            ORDER BY ci.id
            LIMIT %s
            """,
            (limit or 10_000,),
        ).fetchall()
    ]
    workers = max(1, PIPELINE_WORKERS)
    commit_every = max(1, ARXIV_COMMIT_EVERY)
    log.info("Score: processing %d items workers=%d commit_every=%d", len(rows), workers, commit_every)

    def _work(row):
        with connect() as wconn:
            try:
                topic = wconn.execute(
                    """
                    SELECT t.canonical_name
                    FROM research_radar.content_topics ct
                    JOIN research_radar.topics t ON t.topic_id = ct.topic_id
                    WHERE ct.content_id=%s AND ct.is_primary=TRUE
                    LIMIT 1
                    """,
                    (row["id"],),
                ).fetchone()
                primary = topic["canonical_name"] if topic else None
                reason = ((row.get("scoring_reason") or {}) or {}).get("relevance_reason") or ""
                ai_rel = float(row.get("ai_relevance") or 0)
                if ai_rel <= 0:
                    score, primary, _, reason = score_relevance(
                        row["title"],
                        row.get("summary") or "",
                        row.get("categories_raw") or [],
                        row["source_type"],
                    )
                    ai_rel = score
                    store_relevance(wconn, row["id"], score, primary, [], reason)
                enriched = _load_enriched_payload(wconn, row)
                op, pp = signal_priorities(wconn, row["id"])
                s, provenance = intrinsic_scores(
                    ai_rel,
                    enriched.get("paper_title") or row["title"],
                    enriched.get("abstract") or row.get("summary") or "",
                    op,
                    pp,
                    include_person_signal=include_person,
                )
                store_scores(wconn, row["id"], s, provenance, reason)
                bump(wconn, run_id, "items_scored")
                set_status(wconn, row["id"], "SCORED")
                store_opportunities(wconn, row["id"], s, primary)
                is_candidate = s["intrinsic_candidate_score"] >= MIN_CANDIDATE_SCORE
                if is_candidate:
                    set_status(wconn, row["id"], "CANDIDATE")
                    bump(wconn, run_id, "candidates_created")
                wconn.commit()
                return (
                    "ok",
                    row["id"],
                    float(s["intrinsic_candidate_score"]),
                    float(s["notable_org_signal"]),
                    is_candidate,
                    row.get("title") or "",
                )
            except Exception as exc:
                wconn.rollback()
                try:
                    set_status(wconn, row["id"], "ERROR")
                    bump(wconn, run_id, "errors")
                    event(
                        wconn,
                        run_id,
                        row["id"],
                        "score",
                        "processing_error",
                        False,
                        {"title": row.get("title")},
                        str(exc),
                    )
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("err", row["id"], str(exc), row.get("title") or "")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, row) for row in rows]
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result[0] == "ok":
                _, cid, intrinsic, org_signal, is_candidate, title = result
                if is_candidate:
                    log.info(
                        "Score CANDIDATE id=%s intrinsic=%.2f org_signal=%.2f title=%s",
                        cid,
                        intrinsic,
                        org_signal,
                        title[:80],
                    )
                else:
                    log.info("Score SCORED id=%s intrinsic=%.2f title=%s", cid, intrinsic, title[:80])
            else:
                _, cid, err, title = result
                log.error("Score failed id=%s title=%s err=%s", cid, title[:80], err)
            if done % commit_every == 0 or done == len(rows):
                log.info("Score progress %d/%d", done, len(rows))
    return len(rows)


def print_top_candidates(top=10):
    print(f"\nTOP CANDIDATES (limit={top})")
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT content_id, title, canonical_url, arxiv_id, published_at,
                   intrinsic_candidate_score, organisations, opportunities
            FROM research_radar.v_candidates
            ORDER BY intrinsic_candidate_score DESC NULLS LAST, published_at DESC NULLS LAST
            LIMIT %s
            """,
            (top,),
        ).fetchall()
        if not rows:
            print("(none yet)")
            return
        for i, row in enumerate(rows, 1):
            print(
                f"\n{i}. {row['title']}\n"
                f"   Published: {row.get('published_at')}\n"
                f"   URL: {row['canonical_url']}\n"
                f"   arXiv: {row.get('arxiv_id')}\n"
                f"   Intrinsic: {row.get('intrinsic_candidate_score')}\n"
                f"   Organisations: {json.dumps(row.get('organisations'), default=str)}\n"
                f"   Opportunities: {json.dumps(row.get('opportunities'), default=str)}"
            )


PAID_STAGES = frozenset({"affiliation-gpt", "screen", "independence", "semantic-score"})


class PaidStageNotAuthorised(RuntimeError):
    """Raised when a money-spending stage is invoked without explicit authorisation."""


def run_stage(
    stage,
    limit=None,
    *,
    sample=None,
    full=False,
    dry_run=False,
    force=False,
    allow_paid=False,
    profile=None,
    since_days=None,
    diagnose=False,
    out=None,
    top=None,
    rank_by=None,
    gate_percentile=None,
):
    if stage in PAID_STAGES and not dry_run and not allow_paid:
        raise PaidStageNotAuthorised(
            f"Stage '{stage}' makes paid OpenRouter calls. "
            f"Re-run with --allow-paid to authorise spend, or --dry-run to estimate only."
        )
    global HTTP_CALLS
    starting_calls = HTTP_CALLS
    with connect() as conn:
        run_id = start_run(conn, stage)
        try:
            print_status_counts(conn)
            if stage == "ingest":
                stage_ingest(conn, run_id, limit=limit)
            elif stage == "relevance":
                stage_relevance(conn, run_id, limit=limit)
            elif stage == "enrich":
                stage_enrich(conn, run_id, limit=limit)
            elif stage == "entities":
                stage_entities(conn, run_id, limit=limit)
            elif stage == "entities-reprocess":
                stage_entities_reprocess(conn, run_id, limit=limit)
            elif stage == "repair-timestamps":
                stage_repair_timestamps(conn, run_id, limit=limit)
            elif stage in {"openalex", "openalex-retry"}:
                stage_openalex(conn, run_id, limit=limit)
            elif stage == "affiliation-gpt":
                from research_radar.affiliation_gpt import stage_affiliation_gpt

                stage_affiliation_gpt(
                    conn,
                    run_id,
                    limit=limit,
                    dry_run=dry_run,
                    force=force,
                )
            elif stage == "score":
                # Deterministic scoring — kept unwired from `all` (brief §7). The
                # function stays available for manual/legacy use via --stage score.
                stage_score(conn, run_id, limit=limit)
            elif stage == "screen":
                from research_radar.semantic_scoring import stage_screen

                stage_screen(
                    conn,
                    run_id,
                    limit=limit,
                    dry_run=dry_run,
                    force=force,
                )
            elif stage == "semantic-score":
                from research_radar.semantic_scoring import stage_semantic_score_v2

                stage_semantic_score_v2(
                    conn,
                    run_id,
                    sample=sample if sample is not None else (100 if not full else None),
                    full=full,
                    dry_run=dry_run,
                    force=force,
                    gate_percentile=gate_percentile,
                )
            elif stage == "independence":
                from research_radar.independence import stage_independence

                stage_independence(
                    conn,
                    run_id,
                    limit=limit,
                    dry_run=dry_run,
                    force=force,
                )
            elif stage == "semantic-compare":
                from research_radar.semantic_scoring import print_semantic_compare

                print_semantic_compare(conn, limit=limit or 20)
            elif stage == "scoring-cost":
                # Free — zero API calls, ever. Cost projection for a full run
                # (pass 1 over everything not yet screened, pass 2 +
                # independence over the gated top gate_percentile%) over
                # whatever's currently eligible. Deliverable §3 of the tiering
                # brief: check a full-corpus cost before committing to it.
                from research_radar.llm_batch import print_scoring_cost_summary
                from research_radar.semantic_scoring import project_full_run_costs

                projection = project_full_run_costs(conn, gate_percentile=gate_percentile)
                print("\nSCORING COST PROJECTION (dry, zero API calls)")
                print_scoring_cost_summary(**projection)
            elif stage == "final-score":
                from research_radar.final_score import print_distribution, stage_final_score

                if diagnose:
                    print_distribution(conn, profile=profile)
                else:
                    stage_final_score(
                        conn,
                        run_id,
                        profile=profile,
                        limit=limit,
                        dry_run=dry_run,
                    )
            elif stage == "report":
                from research_radar.final_score import count_ranked, load_top, render_markdown
                from research_radar.semantic_scoring import GATE_PERCENTILE, count_scoring_pool

                rows = load_top(
                    conn,
                    profile=profile,
                    top=top or 20,
                    since_days=since_days,
                    rank_by=rank_by or "research",
                )
                pool = count_scoring_pool(conn)
                print(
                    f"\nSCORING POOL: {pool['screened']} papers screened, "
                    f"{pool['full']} fully scored (top {GATE_PERCENTILE:g}% gate) "
                    f"— Top {top or 20} is drawn from the {pool['full']} fully-scored pool, not the {pool['screened']} screened."
                )
                md = render_markdown(
                    rows,
                    profile=profile,
                    since_days=since_days,
                    corpus_n=count_ranked(conn, profile=profile),
                    rank_by=rank_by or "research",
                    pool=pool,
                )
                if out:
                    with open(out, "w", encoding="utf-8") as fh:
                        fh.write(md)
                    print(f"Wrote {out} ({len(rows)} papers)")
                else:
                    print(md)
            elif stage == "all":
                # `all` is the free/unattended path. Paid stages (affiliation-gpt,
                # screen, semantic-score, independence — run in that order) are
                # invoked explicitly with --allow-paid, never on cron. The
                # deterministic `score` stage is no longer part of this flow
                # (brief §7) — it stays available via --stage score for
                # manual/legacy use, but final ranking now comes from
                # screen -> semantic-score -> independence -> final-score.
                stage_ingest(conn, run_id, limit=limit)
                stage_relevance(conn, run_id, limit=limit)
                stage_enrich(conn, run_id, limit=limit)
                stage_entities(conn, run_id, limit=limit)
            else:
                raise ValueError(f"Unknown stage: {stage}")
            finish_run(conn, run_id, starting_calls, ok=True)
            print_status_counts(conn)
            conn.commit()
            log.info("Stage %s completed run_id=%s", stage, run_id)
            return run_id
        except Exception as exc:
            finish_run(conn, run_id, starting_calls, ok=False, error=exc)
            conn.commit()
            raise


def run_pipeline(limit=None):
    return run_stage("all", limit=limit)


def main():
    ap = argparse.ArgumentParser(description="Research Radar pipeline (run all or one stage)")
    ap.add_argument(
        "--stage",
        choices=[
            "ingest",
            "repair-timestamps",
            "relevance",
            "enrich",
            "entities",
            "entities-reprocess",
            "openalex",
            "openalex-retry",
            "affiliation-gpt",
            "score",
            "screen",
            "semantic-score",
            "independence",
            "semantic-compare",
            "scoring-cost",
            "final-score",
            "report",
            "show",
            "all",
        ],
        default="all",
        help="Run one stage at a time so you can inspect DB/logs between steps",
    )
    ap.add_argument("--top", type=int, default=10, help="How many candidates to print after score/all/show/report")
    ap.add_argument("--limit", type=int, default=None, help="Optional max items for this stage (useful for smoke tests)")
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="semantic-score: sample size, randomly composed (default 100)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="semantic-score / independence: score all eligible ENTITY_RESOLVED papers (explicit; not default)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="semantic-score / independence / affiliation-gpt / final-score: estimate only; zero OpenRouter calls / writes",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="semantic-score / independence / affiliation-gpt: overwrite existing assessment for same model/prompt version",
    )
    ap.add_argument(
        "--allow-paid",
        action="store_true",
        help="Required to run a stage that makes paid OpenRouter calls (affiliation-gpt, screen, semantic-score, independence)",
    )
    ap.add_argument(
        "--gate-percentile",
        type=float,
        default=None,
        help="semantic-score / scoring-cost: override GATE_PERCENTILE env for this run (top X%% of screen scores that reach pass 2)",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="final-score / report: weight profile (default radar-v1; use radar-v2 for scoring-v2 assessments)",
    )
    ap.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="report: only include papers published in the last N days",
    )
    ap.add_argument(
        "--diagnose",
        action="store_true",
        help="final-score: print dimension spread and composite comparison",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="report: write markdown to this path instead of stdout",
    )
    ap.add_argument(
        "--rank-by",
        choices=["research", "newsletter"],
        default="research",
        help="report: rank by research_score or newsletter_score (default research; radar-v2 only)",
    )
    args = ap.parse_args()

    if args.stage == "show":
        print_top_candidates(args.top)
        return

    if args.stage == "semantic-score" and args.full is False and args.sample is None:
        args.sample = 100

    run_id = run_stage(
        args.stage,
        limit=args.limit,
        sample=args.sample,
        full=args.full,
        dry_run=args.dry_run,
        force=args.force,
        allow_paid=args.allow_paid,
        profile=args.profile,
        since_days=args.since_days,
        diagnose=args.diagnose,
        out=args.out,
        top=args.top,
        rank_by=args.rank_by,
        gate_percentile=args.gate_percentile,
    )
    print(f"\nSTAGE COMPLETED: {args.stage} run_id={run_id}")
    if args.stage in {"score", "all"}:
        print_top_candidates(args.top)


if __name__ == "__main__":
    main()
