"""arXiv OAI-PMH backfill harvester — metadata only.

Pulls `content_items` (+ `paper_metadata` where the OAI record already
supplies it) directly from arXiv's OAI-PMH endpoint so the corpus can be
backfilled to dates Inoreader never captured. This module does not enrich,
score or resolve affiliations — those stay in `enrich` / `affiliation-gpt`.

`from`/`until` on the OAI endpoint filter on `<datestamp>` (created OR
updated), not `<published_at>`. A 2012 paper revised in January 2026 comes
back inside a January 2026 window. `published_at` must always come from
`<created>` — never `<datestamp>` — or every time-windowed report downstream
gets corrupted by revision noise. See `record_to_item`.
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from os import getenv

import requests

from research_radar.pipeline import (
    ensure_paper_metadata_row,
    normalize_url,
    parse_iso_datetime,
    upsert_item,
)

log = logging.getLogger("research-radar")

OAI_BASE = getenv("ARXIV_OAI_BASE", "https://oaipmh.arxiv.org/oai")
OAI_DELAY_SECONDS = float(getenv("ARXIV_OAI_DELAY", "5"))
OAI_SETS = ("cs", "stat")
OAI_TIMEOUT_SECONDS = int(getenv("ARXIV_OAI_TIMEOUT_SECONDS", "60"))
OAI_DEFAULT_RETRY_AFTER = float(getenv("ARXIV_OAI_DEFAULT_RETRY_AFTER", "20"))
WINDOW_DAYS = 7

BACKFILL_CATEGORIES = getenv(
    "ARXIV_BACKFILL_CATEGORIES",
    "cs,cs.AI,cs.CL,cs.CV,cs.LG,cs.NE,stat.ML",
).split(",")

OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": getenv("HTTP_USER_AGENT", "TheNeural-Research-Radar/0.1")})

_last_request_at = 0.0
_request_count = 0


class ArxivOAIError(RuntimeError):
    """Non-recoverable OAI-PMH error response (not a 503 flow-control pause)."""


class ArxivBackfillWindowFailed(RuntimeError):
    """One or more (set, window) harvests failed; see .failures."""

    def __init__(self, failures):
        self.failures = failures
        super().__init__(f"{len(failures)} window(s) failed: {failures}")


def _throttle():
    """Enforce OAI_DELAY_SECONDS between every request, including 503 retries."""
    global _last_request_at
    now = time.monotonic()
    wait = _last_request_at + OAI_DELAY_SECONDS - now
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _fetch_page(params):
    """One OAI-PMH HTTP GET. 503+Retry-After is normal flow control, not an error."""
    global _request_count
    while True:
        _throttle()
        _request_count += 1
        resp = SESSION.get(OAI_BASE, params=params, timeout=OAI_TIMEOUT_SECONDS)
        if resp.status_code == 503:
            retry_after = resp.headers.get("Retry-After", "")
            try:
                wait = float(retry_after.strip())
            except (TypeError, ValueError):
                wait = OAI_DEFAULT_RETRY_AFTER
            log.info("arxiv-backfill 503 (flow control) Retry-After=%ss; sleeping", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.text


def _text(elem, path):
    if elem is None:
        return None
    node = elem.find(path, OAI_NS)
    return node.text.strip() if node is not None and node.text else None


def category_matches(categories, allowed=BACKFILL_CATEGORIES):
    """A bare entry (e.g. 'cs') matches any 'cs.*' category, not just an exact 'cs' token."""
    allowed_exact = set(allowed)
    allowed_bare = {a for a in allowed if "." not in a}
    for c in categories or []:
        if c in allowed_exact:
            return True
        if c.split(".", 1)[0] in allowed_bare:
            return True
    return False


def parse_record(record_elem):
    """One <record> -> dict. Deleted records return {"deleted": True} with no metadata."""
    header = record_elem.find("oai:header", OAI_NS)
    identifier = _text(header, "oai:identifier")
    datestamp = _text(header, "oai:datestamp")
    if header is not None and header.get("status") == "deleted":
        return {"identifier": identifier, "datestamp": datestamp, "deleted": True}

    arxiv_elem = record_elem.find("oai:metadata/arxiv:arXiv", OAI_NS)
    if arxiv_elem is None:
        # Defensive: no metadata block and not flagged deleted. Skip like deleted.
        return {"identifier": identifier, "datestamp": datestamp, "deleted": True}

    authors = []
    authors_elem = arxiv_elem.find("arxiv:authors", OAI_NS)
    if authors_elem is not None:
        for a in authors_elem.findall("arxiv:author", OAI_NS):
            keyname = _text(a, "arxiv:keyname") or ""
            forenames = _text(a, "arxiv:forenames") or ""
            name = " ".join(x for x in (forenames, keyname) if x)
            if name:
                authors.append(name)

    categories_text = _text(arxiv_elem, "arxiv:categories") or ""

    return {
        "identifier": identifier,
        "datestamp": datestamp,
        "deleted": False,
        "arxiv_id": _text(arxiv_elem, "arxiv:id"),
        "created": _text(arxiv_elem, "arxiv:created"),
        "updated": _text(arxiv_elem, "arxiv:updated"),
        "title": " ".join((_text(arxiv_elem, "arxiv:title") or "").split()),
        "abstract": " ".join((_text(arxiv_elem, "arxiv:abstract") or "").split()),
        "categories": categories_text.split(),
        "doi": _text(arxiv_elem, "arxiv:doi"),
        "journal_ref": _text(arxiv_elem, "arxiv:journal-ref"),
        "authors": authors,
    }


def fetch_window_records(set_spec, window_from: date, window_until: date):
    """Yield parsed record dicts for one (set, window), following resumption tokens.

    Never persists a token as a resume point — on failure the caller re-enters
    this generator from window_from again on the next run.
    """
    params = {
        "verb": "ListRecords",
        "set": set_spec,
        "metadataPrefix": "arXiv",
        "from": window_from.isoformat(),
        "until": window_until.isoformat(),
    }
    while True:
        xml_text = _fetch_page(params)
        root = ET.fromstring(xml_text)
        error = root.find("oai:error", OAI_NS)
        if error is not None:
            code = error.get("code")
            if code == "noRecordsMatch":
                return
            raise ArxivOAIError(f"OAI error code={code} message={error.text}")

        list_records = root.find("oai:ListRecords", OAI_NS)
        if list_records is None:
            return
        for record_elem in list_records.findall("oai:record", OAI_NS):
            yield parse_record(record_elem)

        token_elem = list_records.find("oai:resumptionToken", OAI_NS)
        token = token_elem.text.strip() if token_elem is not None and token_elem.text else None
        if not token:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}


def record_to_item(rec):
    """Map a parsed OAI record onto the `content_items` upsert_item() shape."""
    arxiv_id = rec["arxiv_id"]
    return {
        "source": "arxiv_oai",
        "source_external_id": rec["identifier"],
        "source_type": "arxiv",
        "canonical_url": normalize_url(f"https://arxiv.org/abs/{arxiv_id}"),
        "title": rec["title"] or "(untitled)",
        "summary": rec["abstract"],
        "published_at": parse_iso_datetime(rec["created"]),
        "source_seen_at": datetime.now(timezone.utc),
        "updated_at": parse_iso_datetime(rec["updated"]) if rec.get("updated") else None,
        "authors_raw": rec["authors"],
        "categories_raw": rec["categories"],
        "inoreader_tags": [],
        "raw_metadata": {
            "oai_identifier": rec["identifier"],
            "datestamp": rec["datestamp"],
            "created": rec["created"],
            "updated": rec["updated"],
            "categories": rec["categories"],
            "authors": rec["authors"],
            "doi": rec["doi"],
            "journal_ref": rec["journal_ref"],
            "title": rec["title"],
            "abstract": rec["abstract"],
        },
    }


def upsert_backfill_paper_metadata(conn, content_id, rec):
    """Fill paper_metadata gaps from an OAI record. Never overwrites an existing
    non-empty value (in particular: never touches affiliation_text/extracted_emails,
    which only the HTML-scraping `enrich` stage populates) — a scored/enriched
    paper must not be degraded by a later backfill pass over the same arXiv id.
    """
    arxiv_id = rec["arxiv_id"]
    ensure_paper_metadata_row(conn, content_id)
    conn.execute(
        """
        UPDATE research_radar.paper_metadata SET
            arxiv_id = COALESCE(arxiv_id, %s),
            doi = COALESCE(doi, %s),
            abstract = CASE WHEN COALESCE(abstract, '') = '' THEN %s ELSE abstract END,
            categories = CASE WHEN categories = '[]'::jsonb THEN %s::jsonb ELSE categories END,
            authors_raw = CASE WHEN authors_raw = '[]'::jsonb THEN %s::jsonb ELSE authors_raw END,
            submission_date = COALESCE(submission_date, %s),
            latest_revision_date = COALESCE(latest_revision_date, %s),
            journal_reference = COALESCE(journal_reference, %s),
            paper_url = COALESCE(paper_url, %s),
            html_url = COALESCE(html_url, %s),
            pdf_url = COALESCE(pdf_url, %s),
            enrichment_metadata = enrichment_metadata || %s::jsonb,
            modified_at = NOW()
        WHERE content_id = %s
        """,
        (
            arxiv_id,
            rec.get("doi"),
            rec.get("abstract") or "",
            json.dumps(rec.get("categories") or []),
            json.dumps(rec.get("authors") or []),
            parse_iso_datetime(rec["created"]),
            parse_iso_datetime(rec["updated"]) if rec.get("updated") else None,
            rec.get("journal_ref"),
            f"https://arxiv.org/abs/{arxiv_id}",
            f"https://arxiv.org/html/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}",
            json.dumps({"oai_backfill": {"datestamp": rec.get("datestamp"), "set_spec": rec.get("_set_spec")}}),
            content_id,
        ),
    )


def _iter_windows(date_from: date, date_until: date):
    cur = date_from
    while cur <= date_until:
        end = min(cur + timedelta(days=WINDOW_DAYS - 1), date_until)
        yield cur, end
        cur = end + timedelta(days=1)


def _created_year(created_str):
    dt = parse_iso_datetime(created_str) if created_str else None
    return dt.year if dt else None


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def checkpoint_status(conn, source, set_spec, window_from, window_until):
    row = conn.execute(
        """
        SELECT status FROM research_radar.backfill_checkpoints
        WHERE source=%s AND set_spec=%s AND window_from=%s AND window_until=%s
        """,
        (source, set_spec, window_from, window_until),
    ).fetchone()
    return row["status"] if row else None


def start_checkpoint(conn, source, set_spec, window_from, window_until):
    conn.execute(
        """
        INSERT INTO research_radar.backfill_checkpoints
            (source, set_spec, window_from, window_until, status, started_at)
        VALUES (%s, %s, %s, %s, 'RUNNING', NOW())
        ON CONFLICT (source, set_spec, window_from, window_until) DO UPDATE SET
            status = 'RUNNING', started_at = NOW(), ended_at = NULL, error = NULL,
            records_seen = 0, records_kept = 0, records_new = 0, records_dupe = 0
        """,
        (source, set_spec, window_from, window_until),
    )


def finish_checkpoint(conn, source, set_spec, window_from, window_until, *, status, stats, error=None):
    conn.execute(
        """
        UPDATE research_radar.backfill_checkpoints
        SET status=%s, records_seen=%s, records_kept=%s, records_new=%s, records_dupe=%s,
            error=%s, ended_at=NOW()
        WHERE source=%s AND set_spec=%s AND window_from=%s AND window_until=%s
        """,
        (
            status,
            stats.records_seen,
            stats.records_kept,
            stats.records_new,
            stats.records_dupe,
            error,
            source,
            set_spec,
            window_from,
            window_until,
        ),
    )


# ---------------------------------------------------------------------------
# Real harvest
# ---------------------------------------------------------------------------

SOURCE = "arxiv_oai"


@dataclass
class WindowStats:
    records_seen: int = 0
    records_kept: int = 0
    records_new: int = 0
    records_dupe: int = 0
    records_deleted: int = 0
    records_revision: int = 0  # <created> year earlier than the window's year


@dataclass
class BackfillTotals:
    records_seen: int = 0
    records_kept: int = 0
    records_new: int = 0
    records_dupe: int = 0
    records_deleted: int = 0
    records_revision: int = 0
    windows_run: int = 0
    windows_skipped: int = 0
    windows_failed: int = 0
    failures: list = field(default_factory=list)

    def add(self, w: WindowStats):
        self.records_seen += w.records_seen
        self.records_kept += w.records_kept
        self.records_new += w.records_new
        self.records_dupe += w.records_dupe
        self.records_deleted += w.records_deleted
        self.records_revision += w.records_revision


def _run_one_window(conn, set_spec, window_from, window_until, *, raw_writer=None) -> WindowStats:
    stats = WindowStats()
    for rec in fetch_window_records(set_spec, window_from, window_until):
        stats.records_seen += 1
        if rec["deleted"] or not rec.get("arxiv_id"):
            stats.records_deleted += 1
            continue
        if not category_matches(rec["categories"]):
            continue
        stats.records_kept += 1

        created_year = _created_year(rec.get("created"))
        if created_year is not None and created_year < window_from.year:
            stats.records_revision += 1

        if raw_writer is not None:
            raw_writer.write(rec)

        item = record_to_item(rec)
        content_id, is_new = upsert_item(conn, item)
        if is_new:
            stats.records_new += 1
        else:
            stats.records_dupe += 1
        rec["_set_spec"] = set_spec
        upsert_backfill_paper_metadata(conn, content_id, rec)
    return stats


def run_backfill(conn, date_from: date, date_until: date, *, force: bool = False, run_id=None) -> BackfillTotals:
    from uuid import UUID

    from research_radar.archive import RawArchiveWriter

    run_uuid = UUID(str(run_id)) if run_id is not None else None
    totals = BackfillTotals()
    writer = None
    if run_uuid is not None:
        writer = RawArchiveWriter(run_uuid, SOURCE)
        writer.__enter__()
    try:
        for set_spec in OAI_SETS:
            for window_from, window_until in _iter_windows(date_from, date_until):
                if not force and checkpoint_status(conn, SOURCE, set_spec, window_from, window_until) == "COMPLETE":
                    log.info(
                        "arxiv-backfill set=%s window=%s..%s SKIP (checkpoint COMPLETE)",
                        set_spec, window_from, window_until,
                    )
                    totals.windows_skipped += 1
                    continue

                start_checkpoint(conn, SOURCE, set_spec, window_from, window_until)
                conn.commit()
                started = time.monotonic()
                try:
                    stats = _run_one_window(conn, set_spec, window_from, window_until, raw_writer=writer)
                    finish_checkpoint(conn, SOURCE, set_spec, window_from, window_until, status="COMPLETE", stats=stats)
                    conn.commit()
                    totals.add(stats)
                    totals.windows_run += 1
                    elapsed = int(time.monotonic() - started)
                    log.info(
                        "arxiv-backfill set=%s window=%s..%s records=%d kept=%d new=%d dupe=%d elapsed=%ds",
                        set_spec, window_from, window_until,
                        stats.records_seen, stats.records_kept, stats.records_new, stats.records_dupe, elapsed,
                    )
                except Exception as exc:
                    conn.rollback()
                    empty = WindowStats()
                    finish_checkpoint(conn, SOURCE, set_spec, window_from, window_until, status="FAILED", stats=empty, error=str(exc)[:2000])
                    conn.commit()
                    totals.windows_failed += 1
                    totals.failures.append((set_spec, str(window_from), str(window_until), str(exc)))
                    log.exception("arxiv-backfill set=%s window=%s..%s FAILED", set_spec, window_from, window_until)
    finally:
        if writer is not None:
            writer.finish(conn, window_from=date_from, window_until=date_until)
    return totals


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run_projection(date_from: date, date_until: date) -> dict:
    """Fetch the first 7-day window (live, both sets), no writes. Project the
    full range from those real counts."""
    windows = list(_iter_windows(date_from, date_until))
    total_windows = len(windows)
    first_from, first_until = windows[0]

    per_set = {}
    kept_arxiv_ids = set()
    seen_total = deleted_total = kept_total = revision_total = 0
    requests_total = 0

    for set_spec in OAI_SETS:
        seen = deleted = kept = revision = 0
        requests_before = _request_count
        for rec in fetch_window_records(set_spec, first_from, first_until):
            seen += 1
            if rec["deleted"] or not rec.get("arxiv_id"):
                deleted += 1
                continue
            if not category_matches(rec["categories"]):
                continue
            kept += 1
            kept_arxiv_ids.add(rec["arxiv_id"])
            created_year = _created_year(rec.get("created"))
            if created_year is not None and created_year < first_from.year:
                revision += 1
        requests_this_set = _request_count - requests_before
        requests_total += requests_this_set
        per_set[set_spec] = {
            "records_seen": seen,
            "records_deleted": deleted,
            "records_kept": kept,
            "records_revision": revision,
            "oai_requests": requests_this_set,
        }
        seen_total += seen
        deleted_total += deleted
        kept_total += kept
        revision_total += revision

    unique_kept = len(kept_arxiv_ids)
    revision_fraction = (revision_total / kept_total) if kept_total else 0.0

    projected_records_seen = seen_total * total_windows
    projected_kept_raw = kept_total * total_windows  # pre-cross-set-dedupe, pre-Inoreader-dupe
    projected_kept_unique = unique_kept * total_windows
    projected_revisions = round(projected_kept_unique * revision_fraction)
    projected_new = projected_kept_unique - projected_revisions
    projected_requests = requests_total * total_windows
    wall_clock_seconds = projected_requests * OAI_DELAY_SECONDS

    return {
        "first_window": {"from": str(first_from), "until": str(first_until)},
        "total_windows": total_windows,
        "per_set_first_window": per_set,
        "first_window_totals": {
            "records_seen": seen_total,
            "records_deleted": deleted_total,
            "records_kept_raw": kept_total,
            "records_kept_unique_across_sets": unique_kept,
            "records_revision": revision_total,
            "oai_requests": requests_total,
        },
        "projected_full_range": {
            "date_from": str(date_from),
            "date_until": str(date_until),
            "records_seen": projected_records_seen,
            "records_kept_raw": projected_kept_raw,
            "records_kept_unique_across_sets": projected_kept_unique,
            "estimated_genuinely_new": projected_new,
            "estimated_revisions_of_older_papers": projected_revisions,
            "oai_requests": projected_requests,
            "wall_clock_seconds": wall_clock_seconds,
            "wall_clock_hours": round(wall_clock_seconds / 3600, 1),
            "wall_clock_days": round(wall_clock_seconds / 86400, 2),
        },
    }


def print_dry_run(projection: dict):
    print("\nARXIV BACKFILL DRY RUN (first window only, live fetch, zero writes)")
    fw = projection["first_window"]
    print(f"  first window: {fw['from']}..{fw['until']} (1 of {projection['total_windows']} windows in the requested range)")
    for set_spec, s in projection["per_set_first_window"].items():
        print(
            f"  set={set_spec} records={s['records_seen']} deleted={s['records_deleted']} "
            f"kept={s['records_kept']} revisions={s['records_revision']} requests={s['oai_requests']}"
        )
    ft = projection["first_window_totals"]
    print(f"  first-window kept, deduped across cs+stat: {ft['records_kept_unique_across_sets']}")
    pr = projection["projected_full_range"]
    print(f"\n  PROJECTION for {pr['date_from']}..{pr['date_until']} ({projection['total_windows']} windows, linear extrapolation from week 1):")
    print(f"    records seen (both sets): ~{pr['records_seen']:,}")
    print(f"    kept after category filter, deduped across sets: ~{pr['records_kept_unique_across_sets']:,}")
    print(f"      estimated genuinely new papers: ~{pr['estimated_genuinely_new']:,}")
    print(f"      estimated revisions of pre-window papers: ~{pr['estimated_revisions_of_older_papers']:,}")
    print(f"    OAI-PMH requests: ~{pr['oai_requests']:,}")
    print(f"    wall-clock at {OAI_DELAY_SECONDS}s/request: ~{pr['wall_clock_hours']:.1f}h (~{pr['wall_clock_days']:.1f} days)")
    print("  NB: this is a single week's real counts scaled linearly — arXiv submission volume is not uniform")
    print("      (conference deadlines, holidays), so treat this as an order-of-magnitude estimate, not a forecast.")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def stage_arxiv_backfill(conn, run_id, *, date_from: date, date_until: date, dry_run: bool = False, force: bool = False):
    if date_from is None or date_until is None:
        raise ValueError("arxiv-backfill requires --from and --until")
    if date_from > date_until:
        raise ValueError(f"--from ({date_from}) must be <= --until ({date_until})")

    if dry_run:
        projection = dry_run_projection(date_from, date_until)
        print_dry_run(projection)
        return projection

    totals = run_backfill(conn, date_from, date_until, force=force, run_id=run_id)

    print("\nARXIV BACKFILL SUMMARY")
    print(f"  windows run={totals.windows_run} skipped(complete)={totals.windows_skipped} failed={totals.windows_failed}")
    print(f"  records seen={totals.records_seen} kept={totals.records_kept} new={totals.records_new} "
          f"dupe={totals.records_dupe} deleted_skipped={totals.records_deleted}")
    print(f"  revisions (created year earlier than window): {totals.records_revision}")
    log.info(
        "arxiv-backfill DONE windows_run=%d skipped=%d failed=%d seen=%d kept=%d new=%d dupe=%d deleted=%d revisions=%d",
        totals.windows_run, totals.windows_skipped, totals.windows_failed,
        totals.records_seen, totals.records_kept, totals.records_new, totals.records_dupe,
        totals.records_deleted, totals.records_revision,
    )

    if totals.windows_failed:
        log.error("!! arxiv-backfill: %d window(s) failed: %s", totals.windows_failed, totals.failures)
        raise ArxivBackfillWindowFailed(totals.failures)

    return totals
