"""GPT affiliation resolver via OpenRouter (evidence-only; never uses pretrained employer knowledge).

Replaces active Crossref/OpenAlex affiliation enrichment. Deterministic local
email/affiliation matching still runs first in stage_entities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("research-radar")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AFFILIATION_GPT_ENABLED = os.getenv("AFFILIATION_GPT_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AFFILIATION_GPT_MODEL = (
    os.getenv("AFFILIATION_GPT_MODEL", "openai/gpt-5.6-sol").strip() or "openai/gpt-5.6-sol"
)
AFFILIATION_GPT_REASONING_EFFORT = (
    os.getenv("AFFILIATION_GPT_REASONING_EFFORT", "medium").strip() or "medium"
)
AFFILIATION_GPT_PROMPT_VERSION = (
    os.getenv("AFFILIATION_GPT_PROMPT_VERSION", "affiliation-resolver-v1").strip()
    or "affiliation-resolver-v1"
)
AFFILIATION_GPT_WORKERS = int(os.getenv("AFFILIATION_GPT_WORKERS", "2"))
AFFILIATION_GPT_MAX_RETRIES = int(os.getenv("AFFILIATION_GPT_MAX_RETRIES", "5"))
AFFILIATION_GPT_REQUEST_SLEEP = float(os.getenv("AFFILIATION_GPT_REQUEST_SLEEP", "0.2"))

RESOLVER_NAME = "gpt_affiliation"
LLM_PROVIDER = "openrouter"

DECISIONS = ("MATCHED", "NO_MATCH", "REVIEW_REQUIRED")
AFFILIATION_TYPES = (
    "paper_affiliation",
    "email_domain",
    "inferred_from_supplied_evidence",
)

# Map GPT affiliation_type → content_organisations.evidence_type (never "gpt").
EVIDENCE_TYPE_MAP = {
    "paper_affiliation": "affiliation_text",
    "email_domain": "email_domain",
    "inferred_from_supplied_evidence": "paper_metadata",
}

SYSTEM_PROMPT = """You are an affiliation resolver for Research Radar.

CRITICAL RULES:
1. You may ONLY use evidence explicitly supplied in the user message.
2. You must NEVER use pretrained knowledge about authors, labs, companies, or universities.
3. Never answer "where does this author work?" from memory.
4. Never invent an employer, university, lab, company, or current affiliation.
5. If evidence is missing, ambiguous, conflicting, or would require guessing, return REVIEW_REQUIRED.
6. MATCHED only when supplied evidence directly or strongly supports an organisation affiliation.
7. NO_MATCH when evidence exists but does not identify any organisation (you extract names from evidence only; watchlist mapping is done externally).
8. Prefer REVIEW_REQUIRED whenever uncertain.

You extract organisation name strings that appear in (or are clearly entailed by) the supplied evidence.
Do not decide watchlist priority. Do not invent organisations not supported by evidence.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "organisations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "organisation_name": {"type": "string"},
                    "matched_watchlist_name": {"type": ["string", "null"]},
                    "affiliation_type": {"type": "string", "enum": list(AFFILIATION_TYPES)},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "organisation_name",
                    "matched_watchlist_name",
                    "affiliation_type",
                    "confidence",
                    "evidence",
                    "reason",
                ],
            },
        },
        "overall_reason": {"type": "string"},
    },
    "required": ["decision", "organisations", "overall_reason"],
}


class AffiliationGPTError(Exception):
    def __init__(self, message: str, *, status: str = "ERROR", retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class AffiliationRunStats:
    papers_requested: int = 0
    locally_resolved: int = 0
    eligible_for_gpt: int = 0
    existing_assessments: int = 0
    completed: int = 0
    matched: int = 0
    no_match: int = 0
    review_required: int = 0
    failed: int = 0
    skipped_existing: int = 0
    orgs_written: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_total_cost_usd: float = 0.0
    sample_note: str = ""

    def to_dict(self) -> dict:
        avg = (
            round(self.estimated_total_cost_usd / self.completed, 6)
            if self.completed
            else 0.0
        )
        return {
            "papers_requested": self.papers_requested,
            "locally_resolved": self.locally_resolved,
            "eligible_for_gpt": self.eligible_for_gpt,
            "existing_assessments": self.existing_assessments,
            "completed": self.completed,
            "matched": self.matched,
            "no_match": self.no_match,
            "review_required": self.review_required,
            "failed": self.failed,
            "skipped_existing": self.skipped_existing,
            "orgs_written": self.orgs_written,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_total_cost_usd, 6),
            "average_cost_per_paper_usd": avg,
            "model": resolve_affiliation_model(),
            "provider": LLM_PROVIDER,
            "prompt_version": AFFILIATION_GPT_PROMPT_VERSION,
            "reasoning_effort": AFFILIATION_GPT_REASONING_EFFORT,
            "sample_note": self.sample_note,
        }


def resolve_affiliation_model() -> str:
    model = AFFILIATION_GPT_MODEL
    if "/" not in model:
        return f"openai/{model}"
    return model


def require_affiliation_enabled():
    # Re-read env so CLI / sourced .env overrides work after import.
    enabled = os.getenv("AFFILIATION_GPT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        raise RuntimeError(
            "AFFILIATION_GPT_ENABLED is false. Set AFFILIATION_GPT_ENABLED=true to run."
        )


def require_api_key():
    from research_radar.semantic_scoring import OPENROUTER_API_KEY

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing")


def evidence_fingerprint(paper: dict) -> str:
    payload = {
        "title": paper.get("title") or "",
        "authors": paper.get("authors") or [],
        "emails": paper.get("emails") or [],
        "affiliation_text": paper.get("affiliation_text") or [],
        "local_evidence": paper.get("local_evidence") or [],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_resolver_user_prompt(paper: dict) -> str:
    """Paper evidence only — no scores, statuses, watchlist priorities, or OpenAlex guesses."""
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors_text = authors
    else:
        authors_text = ", ".join(str(a) for a in authors if a) or "(none)"

    emails = paper.get("emails") or []
    if isinstance(emails, str):
        emails_text = emails
    else:
        emails_text = ", ".join(str(e) for e in emails if e) or "(none)"

    affs = paper.get("affiliation_text") or []
    if isinstance(affs, str):
        aff_lines = [affs]
    else:
        aff_lines = [str(a) for a in affs if a]
    aff_text = "\n".join(f"- {a}" for a in aff_lines) if aff_lines else "(none)"

    local_ev = paper.get("local_evidence") or []
    if local_ev:
        local_lines = "\n".join(
            f"- type={e.get('evidence_type')} text={e.get('evidence_text')}"
            for e in local_ev
        )
    else:
        local_lines = "(none — no deterministic watchlist match yet)"

    return (
        "Resolve organisation affiliations using ONLY the evidence below.\n"
        "Do not use pretrained knowledge about these authors.\n"
        "If you cannot support an affiliation from this evidence, return REVIEW_REQUIRED.\n\n"
        f"TITLE:\n{(paper.get('title') or '').strip() or '(empty)'}\n\n"
        f"AUTHORS:\n{authors_text}\n\n"
        f"AUTHOR EMAILS / DOMAINS:\n{emails_text}\n\n"
        f"EXPLICIT AFFILIATION STRINGS FROM PAPER:\n{aff_text}\n\n"
        f"EXISTING DETERMINISTIC LOCAL EVIDENCE ROWS:\n{local_lines}\n"
    )


def parse_affiliation_response(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise AffiliationGPTError("response payload must be an object", retryable=False)
    decision = (payload.get("decision") or "").strip().upper()
    if decision not in DECISIONS:
        raise AffiliationGPTError(f"invalid decision: {decision}", retryable=False)
    overall = (payload.get("overall_reason") or "").strip()
    if not overall:
        raise AffiliationGPTError("empty overall_reason", retryable=False)
    orgs_raw = payload.get("organisations") or []
    if not isinstance(orgs_raw, list):
        raise AffiliationGPTError("organisations must be a list", retryable=False)
    orgs = []
    for item in orgs_raw:
        if not isinstance(item, dict):
            raise AffiliationGPTError("organisation item must be object", retryable=False)
        name = (item.get("organisation_name") or "").strip()
        if not name:
            raise AffiliationGPTError("empty organisation_name", retryable=False)
        aff_type = (item.get("affiliation_type") or "").strip()
        if aff_type not in AFFILIATION_TYPES:
            raise AffiliationGPTError(f"invalid affiliation_type: {aff_type}", retryable=False)
        conf = item.get("confidence")
        try:
            conf_f = float(conf)
        except (TypeError, ValueError) as exc:
            raise AffiliationGPTError("invalid confidence", retryable=False) from exc
        conf_f = max(0.0, min(1.0, conf_f))
        evidence = (item.get("evidence") or "").strip()
        reason = (item.get("reason") or "").strip()
        if not evidence or not reason:
            raise AffiliationGPTError("organisation evidence/reason required", retryable=False)
        mw = item.get("matched_watchlist_name")
        if mw is not None:
            mw = str(mw).strip() or None
        orgs.append(
            {
                "organisation_name": name,
                "matched_watchlist_name": mw,
                "affiliation_type": aff_type,
                "confidence": conf_f,
                "evidence": evidence,
                "reason": reason,
            }
        )
    if decision == "MATCHED" and not orgs:
        raise AffiliationGPTError("MATCHED requires organisations", retryable=False)
    return {"decision": decision, "organisations": orgs, "overall_reason": overall}


def map_orgs_to_watchlist(parsed_orgs: list[dict], watchlist_orgs: list[dict]) -> list[dict]:
    """Deterministic boundary-safe mapping; GPT does not decide priority."""
    from research_radar.pipeline import orgs_from_text

    mapped = []
    seen = set()
    for item in parsed_orgs:
        candidates_text = [
            item.get("organisation_name") or "",
            item.get("matched_watchlist_name") or "",
            item.get("evidence") or "",
        ]
        hits = []
        for text in candidates_text:
            hits.extend(orgs_from_text(text, watchlist_orgs))
        for org, matched_name in hits:
            oid = org["organisation_id"]
            if oid in seen:
                continue
            seen.add(oid)
            mapped.append(
                {
                    **item,
                    "organisation": org,
                    "canonical_name": org["canonical_name"],
                    "matched_alias": matched_name,
                    "evidence_type": EVIDENCE_TYPE_MAP.get(
                        item["affiliation_type"], "paper_metadata"
                    ),
                }
            )
    return mapped


def finalize_decision(decision: str, mapped: list[dict], parsed_orgs: list[dict]) -> str:
    """Reconcile GPT decision with watchlist mapping safety."""
    if decision == "MATCHED":
        if mapped:
            return "MATCHED"
        # GPT claimed match but nothing maps to watchlist.
        if parsed_orgs:
            return "NO_MATCH"
        return "REVIEW_REQUIRED"
    if decision == "NO_MATCH":
        # If deterministic mapping still finds a watchlist org from GPT strings, accept it.
        if mapped:
            return "MATCHED"
        return "NO_MATCH"
    return "REVIEW_REQUIRED"


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    return int(inp or 0), int(out or 0)


def call_affiliation_resolver(*, paper: dict, client=None) -> dict:
    """Call OpenRouter with evidence-only prompt. Returns parsed result + usage."""
    from research_radar.semantic_scoring import create_llm_client, estimate_cost_usd

    require_api_key()
    user_prompt = build_resolver_user_prompt(paper)
    model = resolve_affiliation_model()
    if client is None:
        client = create_llm_client()

    last_exc: Exception | None = None
    for attempt in range(1, AFFILIATION_GPT_MAX_RETRIES + 1):
        try:
            if AFFILIATION_GPT_REQUEST_SLEEP > 0:
                time.sleep(AFFILIATION_GPT_REQUEST_SLEEP)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                extra_body={"reasoning": {"effort": AFFILIATION_GPT_REASONING_EFFORT}},
                temperature=0.0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "affiliation_resolution",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
            )
            text = (response.choices[0].message.content or "").strip()
            inp, out = _usage_tokens(response)
            response_id = getattr(response, "id", None)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AffiliationGPTError(f"invalid JSON: {exc}", retryable=False) from exc
            parsed = parse_affiliation_response(payload)
            return {
                **parsed,
                "input_tokens": inp,
                "output_tokens": out,
                "estimated_cost_usd": estimate_cost_usd(inp, out),
                "response_id": response_id,
                "provider": LLM_PROVIDER,
                "model_name": model,
                "prompt_version": AFFILIATION_GPT_PROMPT_VERSION,
                "resolver": RESOLVER_NAME,
                "status": "COMPLETED",
                "error_message": None,
            }
        except AffiliationGPTError:
            raise
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status == 429 or (status is not None and int(status) >= 500):
                time.sleep(min(30.0, 2 ** attempt))
                continue
            raise AffiliationGPTError(str(exc), status="ERROR", retryable=False) from exc
    raise AffiliationGPTError(str(last_exc or "affiliation GPT request failed"), status="ERROR")


def set_affiliation_status(
    conn,
    content_id: int,
    status: str,
    *,
    error: str | None = None,
    bump_attempts: bool = False,
):
    from research_radar.pipeline import set_affiliation_status as _set

    _set(conn, content_id, status, error=error, bump_attempts=bump_attempts)


def assessment_skip_row(conn, content_id: int, fingerprint: str, *, force: bool) -> dict | None:
    """Return existing assessment row if it should be skipped."""
    if force:
        return None
    model = resolve_affiliation_model()
    row = conn.execute(
        """
        SELECT decision, status, evidence_fingerprint, error_message
        FROM research_radar.affiliation_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, model, AFFILIATION_GPT_PROMPT_VERSION),
    ).fetchone()
    if not row:
        return None
    row = dict(row)
    if row.get("status") == "ERROR":
        return None  # retryable
    if row.get("status") == "COMPLETED":
        # Re-run only if underlying evidence changed.
        if row.get("evidence_fingerprint") and row["evidence_fingerprint"] != fingerprint:
            return None
        return row
    return None


def upsert_affiliation_assessment(conn, *, content_id: int, result: dict, fingerprint: str):
    sql = """
        INSERT INTO research_radar.affiliation_assessments(
            content_id, resolver, provider, model_name, prompt_version,
            decision, status, confidence, evidence_text, evidence_source, reason,
            organisations, evidence_fingerprint,
            input_tokens, output_tokens, estimated_cost_usd,
            response_id, error_message
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s::jsonb, %s,
            %s, %s, %s,
            %s, %s
        )
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            resolver = EXCLUDED.resolver,
            decision = EXCLUDED.decision,
            status = EXCLUDED.status,
            confidence = EXCLUDED.confidence,
            evidence_text = EXCLUDED.evidence_text,
            evidence_source = EXCLUDED.evidence_source,
            reason = EXCLUDED.reason,
            organisations = EXCLUDED.organisations,
            evidence_fingerprint = EXCLUDED.evidence_fingerprint,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            response_id = EXCLUDED.response_id,
            error_message = EXCLUDED.error_message,
            created_at = NOW()
    """
    orgs = result.get("organisations") or []
    confidences = [float(o.get("confidence") or 0) for o in orgs if isinstance(o, dict)]
    confidence = max(confidences) if confidences else None
    evidence_bits = [o.get("evidence") for o in orgs if isinstance(o, dict) and o.get("evidence")]
    sources = [
        EVIDENCE_TYPE_MAP.get(o.get("affiliation_type"), "paper_metadata")
        for o in orgs
        if isinstance(o, dict)
    ]
    evidence_source = sources[0] if sources else None
    conn.execute(
        sql,
        (
            content_id,
            result.get("resolver") or RESOLVER_NAME,
            result.get("provider") or LLM_PROVIDER,
            result.get("model_name") or resolve_affiliation_model(),
            result.get("prompt_version") or AFFILIATION_GPT_PROMPT_VERSION,
            result.get("decision") or "ERROR",
            result.get("status") or "ERROR",
            confidence,
            "; ".join(evidence_bits) if evidence_bits else None,
            evidence_source,
            result.get("overall_reason") or result.get("error_message"),
            json.dumps(orgs),
            fingerprint,
            result.get("input_tokens"),
            result.get("output_tokens"),
            result.get("estimated_cost_usd"),
            result.get("response_id"),
            result.get("error_message"),
        ),
    )


def write_mapped_orgs(conn, content_id: int, mapped: list[dict], evidence_url: str | None) -> int:
    from research_radar.pipeline import store_org_evidence

    written = 0
    for item in mapped:
        org = item["organisation"]
        etype = item["evidence_type"]
        etext = item.get("evidence") or item.get("organisation_name")
        conf = float(item.get("confidence") or 0.7)
        conf = max(0.0, min(1.0, conf))
        # Prefer not to mark evidence source as GPT — resolver lives in assessment table.
        store_org_evidence(conn, content_id, org, etype, etext, evidence_url, conf)
        written += 1
    return written


def load_local_evidence_rows(conn, content_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT evidence_type, evidence_text, confidence
        FROM research_radar.content_organisations
        WHERE content_id = %s
          AND relationship_type = 'paper_author_affiliation'
          AND current_affiliation = FALSE
          AND evidence_type IN ('email_domain', 'explicit_affiliation_text')
        """,
        (content_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_affiliation_candidates(conn, limit: int | None = None) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            ci.id AS content_id,
            ci.title,
            ci.status AS item_status,
            pm.authors_raw AS authors,
            pm.extracted_emails AS emails,
            pm.affiliation_text,
            pm.doi,
            pm.enrichment_metadata,
            pm.affiliation_status,
            pm.affiliation_attempts
        FROM research_radar.paper_metadata pm
        JOIN research_radar.content_items ci ON ci.id = pm.content_id
        WHERE pm.affiliation_status IN ('PENDING', 'ERROR')
          AND ci.status IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE', 'ENRICHED', 'RELEVANT', 'ERROR')
          AND NOT EXISTS (
              SELECT 1
              FROM research_radar.content_organisations co
              WHERE co.content_id = pm.content_id
                AND co.relationship_type = 'paper_author_affiliation'
                AND co.current_affiliation = FALSE
                AND co.evidence_type IN ('email_domain', 'explicit_affiliation_text')
          )
        ORDER BY
          CASE pm.affiliation_status WHEN 'PENDING' THEN 0 ELSE 1 END,
          pm.affiliation_attempts ASC,
          pm.content_id ASC
        LIMIT %s
        """,
        (limit or 10_000,),
    ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        meta = item.get("enrichment_metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        item["evidence_url"] = meta.get("evidence_url") if isinstance(meta, dict) else None
        item["authors"] = item.get("authors") or []
        item["emails"] = item.get("emails") or []
        item["affiliation_text"] = item.get("affiliation_text") or []
        out.append(item)
    return out


def count_locally_resolved(conn) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM research_radar.paper_metadata pm
        WHERE pm.affiliation_status = 'NOT_NEEDED'
           OR EXISTS (
              SELECT 1 FROM research_radar.content_organisations co
              WHERE co.content_id = pm.content_id
                AND co.relationship_type = 'paper_author_affiliation'
                AND co.current_affiliation = FALSE
                AND co.evidence_type IN ('email_domain', 'explicit_affiliation_text')
           )
        """
    ).fetchone()
    return int(row["n"] or 0)


def count_existing_assessments(conn) -> int:
    model = resolve_affiliation_model()
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM research_radar.affiliation_assessments
        WHERE provider = %s AND model_name = %s AND prompt_version = %s
        """,
        (LLM_PROVIDER, model, AFFILIATION_GPT_PROMPT_VERSION),
    ).fetchone()
    return int(row["n"] or 0)


def estimate_prompt_tokens(paper: dict) -> int:
    # Rough char/4 heuristic for dry-run only.
    prompt = SYSTEM_PROMPT + build_resolver_user_prompt(paper)
    return max(1, len(prompt) // 4)


def stage_affiliation_gpt(
    conn,
    run_id,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    from research_radar.pipeline import bump, event, load_orgs
    from research_radar.semantic_scoring import create_llm_client

    watchlist = [dict(o) for o in load_orgs(conn)]
    candidates = load_affiliation_candidates(conn, limit=limit)
    stats = AffiliationRunStats(
        papers_requested=len(candidates),
        locally_resolved=count_locally_resolved(conn),
        eligible_for_gpt=len(candidates),
        existing_assessments=count_existing_assessments(conn),
    )

    enabled = os.getenv("AFFILIATION_GPT_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not dry_run and not enabled:
        log.warning(
            "AFFILIATION_GPT_ENABLED=false — skipping affiliation-gpt "
            "(%d eligible papers remain PENDING)",
            len(candidates),
        )
        print(
            "\nAFFILIATION GPT SKIPPED (AFFILIATION_GPT_ENABLED=false)\n"
            f"  eligible_for_gpt: {len(candidates)}\n"
            "  Set AFFILIATION_GPT_ENABLED=true to run."
        )
        return stats

    if not dry_run:
        require_api_key()

    if dry_run:
        est_in = 0
        est_out = 0
        calls = 0
        for paper in candidates:
            fp = evidence_fingerprint(paper)
            skip = assessment_skip_row(conn, int(paper["content_id"]), fp, force=force)
            if skip:
                stats.skipped_existing += 1
                continue
            calls += 1
            est_in += estimate_prompt_tokens(paper)
            est_out += 250  # structured JSON heuristic
        from research_radar.semantic_scoring import estimate_cost_usd

        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_total_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.sample_note = f"dry_run_estimated_calls={calls}"
        summary = stats.to_dict()
        summary["estimated_calls"] = calls
        log.info("Affiliation-GPT DRY RUN: %s", json.dumps(summary))
        print("\nAFFILIATION GPT DRY RUN")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return stats

    client = create_llm_client()
    workers = max(1, AFFILIATION_GPT_WORKERS)

    def _process_one(paper: dict) -> tuple[str, dict]:
        from research_radar.pipeline import connect as _connect

        content_id = int(paper["content_id"])
        with _connect() as wconn:
            local_ev = load_local_evidence_rows(wconn, content_id)
            # Safety: if local orgs appeared, mark NOT_NEEDED and skip GPT.
            if local_ev:
                set_affiliation_status(wconn, content_id, "NOT_NEEDED", error=None)
                wconn.commit()
                return ("local_skip", {"content_id": content_id})

            paper = dict(paper)
            paper["local_evidence"] = local_ev
            fp = evidence_fingerprint(paper)
            skip = assessment_skip_row(wconn, content_id, fp, force=force)
            if skip:
                wconn.commit()
                return ("skipped", {"content_id": content_id, "decision": skip.get("decision")})

            try:
                result = call_affiliation_resolver(paper=paper, client=client)
                mapped = map_orgs_to_watchlist(result["organisations"], watchlist)
                decision = finalize_decision(
                    result["decision"], mapped, result["organisations"]
                )
                result["decision"] = decision
                # Persist watchlist-canonical org names in stored organisations payload.
                store_orgs = []
                for m in mapped:
                    store_orgs.append(
                        {
                            "organisation_name": m["organisation_name"],
                            "matched_watchlist_name": m["canonical_name"],
                            "affiliation_type": m["affiliation_type"],
                            "confidence": m["confidence"],
                            "evidence": m["evidence"],
                            "reason": m["reason"],
                        }
                    )
                if not store_orgs:
                    store_orgs = result["organisations"]
                result["organisations"] = store_orgs

                written = 0
                if decision == "MATCHED" and mapped:
                    written = write_mapped_orgs(
                        wconn, content_id, mapped, paper.get("evidence_url")
                    )

                upsert_affiliation_assessment(
                    wconn, content_id=content_id, result=result, fingerprint=fp
                )
                set_affiliation_status(
                    wconn,
                    content_id,
                    decision,
                    error=None if decision != "ERROR" else result.get("overall_reason"),
                    bump_attempts=True,
                )
                # Optionally re-queue scoring when new orgs found.
                if written > 0 and paper.get("item_status") in {
                    "SCORED",
                    "CANDIDATE",
                    "ENTITY_RESOLVED",
                }:
                    from research_radar.pipeline import set_status

                    set_status(wconn, content_id, "ENTITY_RESOLVED")

                event(
                    wconn,
                    run_id,
                    content_id,
                    "affiliation_gpt",
                    decision.lower(),
                    True,
                    {
                        "decision": decision,
                        "orgs_written": written,
                        "response_id": result.get("response_id"),
                    },
                )
                if written:
                    bump(wconn, run_id, "orgs_resolved", written)
                wconn.commit()
                return (
                    "completed",
                    {
                        "content_id": content_id,
                        "decision": decision,
                        "orgs_written": written,
                        **result,
                    },
                )
            except AffiliationGPTError as exc:
                err_result = {
                    "decision": "ERROR",
                    "organisations": [],
                    "overall_reason": str(exc),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "response_id": None,
                    "provider": LLM_PROVIDER,
                    "model_name": resolve_affiliation_model(),
                    "prompt_version": AFFILIATION_GPT_PROMPT_VERSION,
                    "resolver": RESOLVER_NAME,
                    "status": "ERROR",
                    "error_message": str(exc)[:1000],
                }
                try:
                    upsert_affiliation_assessment(
                        wconn, content_id=content_id, result=err_result, fingerprint=fp
                    )
                    set_affiliation_status(
                        wconn,
                        content_id,
                        "ERROR",
                        error=str(exc)[:500],
                        bump_attempts=True,
                    )
                    event(
                        wconn,
                        run_id,
                        content_id,
                        "affiliation_gpt",
                        "error",
                        False,
                        {},
                        str(exc),
                    )
                    bump(wconn, run_id, "errors")
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("failed", {"content_id": content_id, "error": str(exc)})
            except Exception as exc:
                wconn.rollback()
                try:
                    err_result = {
                        "decision": "ERROR",
                        "organisations": [],
                        "overall_reason": str(exc),
                        "status": "ERROR",
                        "error_message": str(exc)[:1000],
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "provider": LLM_PROVIDER,
                        "model_name": resolve_affiliation_model(),
                        "prompt_version": AFFILIATION_GPT_PROMPT_VERSION,
                        "resolver": RESOLVER_NAME,
                    }
                    upsert_affiliation_assessment(
                        wconn, content_id=content_id, result=err_result, fingerprint=fp
                    )
                    set_affiliation_status(
                        wconn, content_id, "ERROR", error=str(exc)[:500], bump_attempts=True
                    )
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("failed", {"content_id": content_id, "error": str(exc)})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, p) for p in candidates]
        done = 0
        for fut in as_completed(futures):
            kind, payload = fut.result()
            done += 1
            if kind == "skipped":
                stats.skipped_existing += 1
            elif kind == "local_skip":
                stats.locally_resolved += 1
            elif kind == "completed":
                stats.completed += 1
                decision = payload.get("decision")
                if decision == "MATCHED":
                    stats.matched += 1
                elif decision == "NO_MATCH":
                    stats.no_match += 1
                elif decision == "REVIEW_REQUIRED":
                    stats.review_required += 1
                stats.orgs_written += int(payload.get("orgs_written") or 0)
                stats.input_tokens += int(payload.get("input_tokens") or 0)
                stats.output_tokens += int(payload.get("output_tokens") or 0)
                stats.estimated_total_cost_usd += float(
                    payload.get("estimated_cost_usd") or 0
                )
                log.info(
                    "Affiliation GPT %s id=%s decision=%s orgs=%s",
                    "COMPLETED",
                    payload.get("content_id"),
                    payload.get("decision"),
                    payload.get("orgs_written"),
                )
            else:
                stats.failed += 1
                log.warning(
                    "Affiliation GPT FAILED id=%s err=%s",
                    payload.get("content_id"),
                    payload.get("error"),
                )
            if done % 10 == 0 or done == len(candidates):
                log.info(
                    "Affiliation GPT progress %d/%d completed=%d failed=%d skipped=%d",
                    done,
                    len(candidates),
                    stats.completed,
                    stats.failed,
                    stats.skipped_existing,
                )

    summary = stats.to_dict()
    log.info("Affiliation GPT stats: %s", json.dumps(summary))
    print("\nAFFILIATION GPT SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return stats
