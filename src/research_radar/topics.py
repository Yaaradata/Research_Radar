"""Topic enrichment + key-claim extraction (topics stage).

A separate, always-on annotation pass over every RELEVANT-or-later paper —
NOT part of scoring. Scoring (screen/semantic-score) runs on a percentile
gated subset; if tagging were folded into that pass, most of the corpus
would never be tagged and would not be queryable. This stage attaches:

  - exactly one closed-vocabulary domain
  - one to three closed-vocabulary subdomains
  - three to eight open-vocabulary (but normalised + deduplicated) topics
  - zero to four closed-vocabulary applications
  - up to five literally-grounded quantitative claims

to content_topics / content_claims. It never changes content_items.status.

Reuses SCREEN_MODEL and the OpenRouter plumbing from semantic_scoring.py
rather than re-declaring them, so "cheap, no-reasoning model" means the same
model in both places.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date

from research_radar.llm_batch import (
    AdaptiveConcurrencyGate,
    LLMBatchError,
    SCORING_CONCURRENCY,
    call_chat_completion,
    cost_summary_line,
    random_batches,
    strip_json_fences,
)
from research_radar.semantic_scoring import (
    LLM_PROVIDER,
    create_llm_client,
    estimate_cost_usd,
    resolve_screen_model,
)

log = logging.getLogger("research-radar")

TOPICS_PROMPT_VERSION = os.getenv("TOPICS_PROMPT_VERSION", "topics-v1").strip()
TOPICS_BATCH_SIZE = int(os.getenv("TOPICS_BATCH_SIZE", "10"))
TOPICS_MAX_RETRIES = int(os.getenv("TOPICS_MAX_RETRIES", "3"))
TOPICS_REQUEST_SLEEP = float(os.getenv("TOPICS_REQUEST_SLEEP", "0.2"))
ASSESSMENT_TYPE = "topics"

# Papers that have reached RELEVANT and are not REJECTED/ERRORed are eligible,
# regardless of how far they have since advanced. `enrich` moves arxiv-sourced
# RELEVANT rows to ENRICHED (and beyond) almost immediately, so a literal
# `status = 'RELEVANT'` filter — as an earlier draft of this stage's brief
# specified — would only ever catch non-arxiv items and whatever's mid-flight,
# never the bulk of the (arxiv-heavy) corpus. See load_quality_candidates()
# in semantic_scoring.py for the same correction made for an identical bug.
TOPICS_ELIGIBLE_STATUSES = ("RELEVANT", "ENRICHED", "ENTITY_RESOLVED", "SCORED", "CANDIDATE")

# Independence classification -> content_claims.qualifier. Most papers have no
# independence assessment at all (it only runs on the pass-2 gated subset) —
# those default to 'self_reported', same as 'self_evaluation' and
# 'not_applicable' (the claim is about the paper's own new artifact, not a
# third-party evaluation of somebody else's).
INDEPENDENCE_TO_QUALIFIER = {
    "independent": "independent",
    "self_evaluation": "self_reported",
    "not_applicable": "self_reported",
    "unclear": "unclear",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TOPICS_SYSTEM_PROMPT = """You tag AI research papers for a searchable research corpus. Your tags are
how people will later find these papers, so precision and consistency matter
more than creativity.

For each paper you are given its title, arXiv categories and abstract.

RETURN FOUR KINDS OF LABEL

domain      Exactly ONE, chosen from the DOMAINS list below. If none fits,
            return "Other".

subdomains  ONE to THREE, chosen from the SUBDOMAINS list below. Choose only
            subdomains the paper genuinely addresses, not ones it mentions in
            passing.

topics      THREE to EIGHT specific free-text tags describing what the paper
            is actually about. These are more specific than subdomains. Use
            established terminology from the field, not invented phrases. Use
            the singular where natural. Examples of the right level of
            specificity: "synthetic text detection", "chain of thought
            prompting", "speculative decoding", "reward hacking".

applications ZERO to FOUR, chosen from the APPLICATIONS list below. Only
            include an application the paper explicitly addresses or
            evaluates. An abstract that never mentions education must not be
            tagged education.

EXTRACTING KEY CLAIMS

Extract up to FIVE quantitative results that are LITERALLY STATED in the
abstract.

For each: the metric name, the value exactly as written, the unit, the task
it measures, and the dataset if one is named.

CRITICAL RULES

Extract only what is written. Never infer, never estimate, never convert
units, never compute a number that is not present. If the abstract says
"significantly outperforms baselines" with no figure, that is NOT a claim -
return nothing for it.

Copy the value verbatim. If it says "99%", return "99%". If it says
"approximately 3x faster", return "approximately 3x faster".

Include the exact sentence the claim came from as evidence. If you cannot
quote a sentence containing the number, do not return the claim.

Most abstracts contain zero to two extractable claims. Returning an empty
list is correct and expected. Do not pad.

Return ONLY a JSON object, no prose, no markdown fences.
"""


def build_vocabulary_block(vocab: dict) -> str:
    lines = ["DOMAINS", *[f"- {d}" for d in vocab["domains_list"]], ""]
    lines.append("SUBDOMAINS")
    for domain in vocab["domains_list"]:
        subs = vocab["subdomains_by_domain"].get(domain) or []
        if not subs:
            continue
        lines.append(f"{domain}:")
        lines.extend(f"- {s}" for s in subs)
    lines.append("")
    lines.append("APPLICATIONS")
    lines.extend(f"- {a}" for a in vocab["applications_list"])
    return "\n".join(lines)


def build_paper_block(paper: dict) -> str:
    if isinstance(paper.get("categories"), (list, tuple)):
        cat_text = ", ".join(str(c) for c in paper["categories"] if c)
    else:
        cat_text = str(paper.get("categories") or "")
    return (
        f"PAPER {paper['content_id']}\n"
        f"TITLE: {(paper.get('title') or '').strip()}\n"
        f"CATEGORIES: {cat_text.strip() or '(none)'}\n"
        f"ABSTRACT: {(paper.get('abstract') or '').strip()}"
    )


def build_topics_batch_user_prompt(papers: list[dict], vocab_block: str) -> str:
    blocks = "\n\n".join(build_paper_block(p) for p in papers)
    return f"{vocab_block}\n\n{blocks}\n"


# ---------------------------------------------------------------------------
# Response schema (strict json_schema, object root)
# ---------------------------------------------------------------------------

TOPICS_CLAIM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "metric": {"type": "string"},
        "value_text": {"type": "string"},
        "unit": {"type": ["string", "null"]},
        "task": {"type": ["string", "null"]},
        "dataset": {"type": ["string", "null"]},
        "evidence": {"type": "string"},
    },
    "required": ["metric", "value_text", "unit", "task", "dataset", "evidence"],
}

TOPICS_PAPER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "integer"},
        "domain": {"type": "string"},
        "subdomains": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "topics": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "applications": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "claims": {"type": "array", "items": TOPICS_CLAIM_SCHEMA, "maxItems": 5},
    },
    "required": ["paper_id", "domain", "subdomains", "topics", "applications", "claims"],
}

TOPICS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "papers": {"type": "array", "items": TOPICS_PAPER_SCHEMA},
    },
    "required": ["papers"],
}


class TopicsParseError(ValueError):
    pass


def parse_topics_batch(text: str, expected_ids: set[int]) -> dict[int, dict]:
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TopicsParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise TopicsParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise TopicsParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise TopicsParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise TopicsParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc
        if pid not in expected_ids:
            continue
        claims = item.get("claims")
        out[pid] = {
            "domain": str(item.get("domain") or "").strip(),
            "subdomains": [str(s).strip() for s in (item.get("subdomains") or []) if str(s).strip()],
            "topics": [str(t).strip() for t in (item.get("topics") or []) if str(t).strip()],
            "applications": [str(a).strip() for a in (item.get("applications") or []) if str(a).strip()],
            "claims": claims if isinstance(claims, list) else [],
        }
    return out


# ---------------------------------------------------------------------------
# Normalisation — the vocabulary-control layer. Uncontrolled free-text
# tagging produces "AI text detection" / "AI-text detection" / "LLM text
# detection" as three distinct rows and the corpus stops being queryable.
# ---------------------------------------------------------------------------


def normalize_topic_name(raw: str) -> str:
    """lowercase, strip punctuation, collapse whitespace, hyphenate.

    "AI Text Detection", "ai-text detection" and "AI text detection" all
    normalise to "ai-text-detection".
    """
    s = (raw or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)  # strip punctuation, keep word chars/space/hyphen
    s = re.sub(r"[\s_]+", " ", s).strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


_PLAIN_NUMBER_RE = re.compile(r"^\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*%?$")


def parse_value_num(value_text: str) -> float | None:
    """Parse a plain number out of value_text where possible, else None.

    "99%" -> 99.0. "$4.50" -> 4.5. "approximately 3x faster" -> None (words
    present). "3x" -> None (not a plain number/percent). Never infer or
    convert units — this is a literal parse, not an estimate.
    """
    if not value_text:
        return None
    m = _PLAIN_NUMBER_RE.match(value_text.strip())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def claim_evidence_is_grounded(evidence: str, abstract: str) -> bool:
    """Same grounding discipline as the affiliation resolver: a claim whose
    evidence sentence is not literally present in the abstract is dropped,
    not repaired or fuzzy-matched — that is what stops the model inventing
    results."""
    if not evidence or not abstract:
        return False
    return evidence.strip() in abstract


# ---------------------------------------------------------------------------
# Vocabulary loading + validation (DB-backed)
# ---------------------------------------------------------------------------


def load_topic_vocabulary(conn) -> dict:
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT topic_id, canonical_name, level, parent_topic_id
            FROM research_radar.topics
            WHERE level IN ('domain', 'subdomain', 'application')
              AND active = TRUE
            ORDER BY topic_id
            """
        ).fetchall()
    ]
    domains = {r["topic_id"]: r for r in rows if r["level"] == "domain"}
    domains_by_name_lower = {r["canonical_name"].lower(): r for r in domains.values()}
    subdomains_by_name_lower: dict[str, dict] = {}
    subdomains_by_domain: dict[str, list[str]] = {d["canonical_name"]: [] for d in domains.values()}
    applications_by_name_lower: dict[str, dict] = {}
    applications_list: list[str] = []

    for r in rows:
        if r["level"] == "subdomain":
            subdomains_by_name_lower[r["canonical_name"].lower()] = r
            parent = domains.get(r["parent_topic_id"])
            if parent:
                subdomains_by_domain[parent["canonical_name"]].append(r["canonical_name"])
        elif r["level"] == "application":
            applications_by_name_lower[r["canonical_name"].lower()] = r
            applications_list.append(r["canonical_name"])

    return {
        "domains_by_name_lower": domains_by_name_lower,
        "subdomains_by_name_lower": subdomains_by_name_lower,
        "applications_by_name_lower": applications_by_name_lower,
        "domains_list": [d["canonical_name"] for d in domains.values()],
        "subdomains_by_domain": subdomains_by_domain,
        "applications_list": applications_list,
    }


@dataclass
class ValidationOutcome:
    domain_id: int | None = None
    domain_name: str | None = None
    subdomain_ids: list[int] = field(default_factory=list)
    subdomain_names: list[str] = field(default_factory=list)
    application_ids: list[int] = field(default_factory=list)
    unknown_domain: bool = False
    unknown_subdomains: int = 0
    unknown_applications: int = 0


def validate_against_vocabulary(parsed: dict, vocab: dict) -> ValidationOutcome:
    out = ValidationOutcome()

    domain_row = vocab["domains_by_name_lower"].get((parsed.get("domain") or "").strip().lower())
    if domain_row:
        out.domain_id = domain_row["topic_id"]
        out.domain_name = domain_row["canonical_name"]
    else:
        out.unknown_domain = True
        log.warning("Topics: unknown domain %r dropped", parsed.get("domain"))

    seen_subdomain_ids: set[int] = set()
    for raw in parsed.get("subdomains") or []:
        row = vocab["subdomains_by_name_lower"].get(raw.strip().lower())
        if row is None:
            out.unknown_subdomains += 1
            log.warning("Topics: unknown subdomain %r dropped", raw)
            continue
        if row["topic_id"] in seen_subdomain_ids:
            continue
        seen_subdomain_ids.add(row["topic_id"])
        out.subdomain_ids.append(row["topic_id"])
        out.subdomain_names.append(row["canonical_name"])

    seen_app_ids: set[int] = set()
    for raw in parsed.get("applications") or []:
        row = vocab["applications_by_name_lower"].get(raw.strip().lower())
        if row is None:
            out.unknown_applications += 1
            log.warning("Topics: unknown application %r dropped", raw)
            continue
        if row["topic_id"] in seen_app_ids:
            continue
        seen_app_ids.add(row["topic_id"])
        out.application_ids.append(row["topic_id"])

    return out


def resolve_or_create_level3_topic(conn, normalized_name: str, parent_topic_id: int | None) -> tuple[int, bool]:
    """Resolve `normalized_name` against canonical_name + aliases of existing
    level='topic' rows, incrementing usage_count on a hit; otherwise create a
    new one with origin='llm'. Returns (topic_id, created)."""
    row = conn.execute(
        """
        SELECT topic_id
        FROM research_radar.topics
        WHERE level = 'topic' AND %s = ANY(aliases)
        LIMIT 1
        """,
        (normalized_name,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE research_radar.topics SET usage_count = usage_count + 1 WHERE topic_id = %s",
            (row["topic_id"],),
        )
        return row["topic_id"], False

    row = conn.execute(
        """
        INSERT INTO research_radar.topics (canonical_name, level, origin, parent_topic_id, usage_count)
        VALUES (%s, 'topic', 'llm', %s, 1)
        ON CONFLICT (canonical_name) DO UPDATE SET
            usage_count = research_radar.topics.usage_count + 1
        RETURNING topic_id, (xmax = 0) AS inserted
        """,
        (normalized_name, parent_topic_id),
    ).fetchone()
    return row["topic_id"], bool(row["inserted"])


def resolve_claim_qualifier(conn, content_id: int) -> str:
    row = conn.execute(
        """
        SELECT status
        FROM research_radar.content_independence_assessments
        WHERE content_id = %s AND status <> 'ERROR'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (content_id,),
    ).fetchone()
    if not row:
        return "self_reported"
    return INDEPENDENCE_TO_QUALIFIER.get(row["status"], "self_reported")


# ---------------------------------------------------------------------------
# DB writes
# ---------------------------------------------------------------------------


def write_content_topics(conn, content_id: int, links: list[tuple[int, bool, float]]):
    """links: (topic_id, is_primary, confidence)."""
    for topic_id, is_primary, confidence in links:
        conn.execute(
            """
            INSERT INTO research_radar.content_topics(content_id, topic_id, is_primary, confidence, reason)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (content_id, topic_id) DO UPDATE SET
                is_primary = EXCLUDED.is_primary,
                confidence = EXCLUDED.confidence,
                reason = EXCLUDED.reason
            """,
            (content_id, topic_id, is_primary, confidence, TOPICS_PROMPT_VERSION),
        )


def write_claims(conn, content_id: int, claims: list[dict]):
    conn.execute("DELETE FROM research_radar.content_claims WHERE content_id = %s", (content_id,))
    for c in claims:
        conn.execute(
            """
            INSERT INTO research_radar.content_claims(
                content_id, metric, value_text, value_num, unit, task, dataset, qualifier, evidence
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                content_id,
                c["metric"],
                c["value_text"],
                c.get("value_num"),
                c.get("unit"),
                c.get("task"),
                c.get("dataset"),
                c["qualifier"],
                c["evidence"],
            ),
        )


def topics_assessment_exists(conn, content_id: int) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_topic_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
          AND status = 'COMPLETED'
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, resolve_screen_model(), TOPICS_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_topics_assessment(conn, *, content_id: int, status: str, tokens_in=None, tokens_out=None, cost_usd=None):
    sql = """
        INSERT INTO research_radar.content_topic_assessments(
            content_id, provider, model_name, prompt_version, status, tokens_in, tokens_out, cost_usd
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            status = EXCLUDED.status,
            tokens_in = EXCLUDED.tokens_in,
            tokens_out = EXCLUDED.tokens_out,
            cost_usd = EXCLUDED.cost_usd,
            created_at = NOW()
    """
    conn.execute(
        sql,
        (content_id, LLM_PROVIDER, resolve_screen_model(), TOPICS_PROMPT_VERSION, status, tokens_in, tokens_out, cost_usd),
    )


def load_topics_candidates(
    conn,
    limit: int | None = None,
    *,
    since: date | str | None = None,
    until: date | str | None = None,
) -> list[dict]:
    from research_radar.candidate_window import published_at_sql_filters

    window_sql, window_params = published_at_sql_filters(since, until)
    rows = conn.execute(
        f"""
        SELECT
            ci.id AS content_id,
            ci.title,
            COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
            COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.status IN ({', '.join(['%s'] * len(TOPICS_ELIGIBLE_STATUSES))})
        {window_sql}
        ORDER BY ci.id
        LIMIT %s
        """,
        (*TOPICS_ELIGIBLE_STATUSES, *window_params, limit or 200_000),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_topics_batch(papers: list[dict], vocab_block: str, *, client=None, on_rate_limited=None) -> dict:
    if client is None:
        client = create_llm_client()
    model = resolve_screen_model()
    user_prompt = build_topics_batch_user_prompt(papers, vocab_block)
    expected_ids = {int(p["content_id"]) for p in papers}

    result = call_chat_completion(
        client,
        model=model,
        system_prompt=TOPICS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=None,
        temperature=0.2,
        max_retries=1,  # outer stage loop owns batch-level retry/backoff
        request_sleep=TOPICS_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "topics_assessment",
                "strict": True,
                "schema": TOPICS_RESPONSE_SCHEMA,
            },
        },
        on_rate_limited=on_rate_limited,
    )
    parsed = parse_topics_batch(result["text"], expected_ids)
    return {
        "results": parsed,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
        "estimated_cost_usd": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
    }


def call_topics_batch_with_retry(
    papers: list[dict],
    vocab_block: str,
    *,
    client=None,
    max_retries: int | None = None,
    on_rate_limited=None,
    batch_tag: str = "",
) -> dict:
    max_retries = max_retries if max_retries is not None else TOPICS_MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_topics_batch(papers, vocab_block, client=client, on_rate_limited=on_rate_limited)
        except (LLMBatchError, TopicsParseError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(60.0, 2 ** (attempt - 1))
                log.warning(
                    "[batch %s] Topics batch retry attempt=%s/%s size=%s err=%s",
                    batch_tag, attempt, max_retries, len(papers), exc,
                )
                time.sleep(wait)
                continue
    raise last_exc


# ---------------------------------------------------------------------------
# Post-processing: parsed model output -> DB rows (pure w.r.t. the model —
# every domain/subdomain/application is checked against the closed
# vocabulary, every claim's evidence is checked against the abstract).
# ---------------------------------------------------------------------------


def process_paper_result(conn, paper: dict, parsed: dict, vocab: dict, stats: "TopicsRunStats") -> None:
    content_id = int(paper["content_id"])
    abstract = paper.get("abstract") or ""

    validated = validate_against_vocabulary(parsed, vocab)
    stats.unknown_domains += int(validated.unknown_domain)
    stats.unknown_subdomains += validated.unknown_subdomains
    stats.unknown_applications += validated.unknown_applications

    # confidence isn't a model output here (no numeric score is requested of
    # it) — it's a fixed Python-side marker distinguishing closed-vocabulary
    # assignments (validated exactly against the seeded list) from
    # open-vocabulary level-3 topics (fuzzier by construction).
    links: list[tuple[int, bool, float]] = []
    if validated.domain_id is not None:
        links.append((validated.domain_id, True, 1.0))
    for sub_id in validated.subdomain_ids:
        links.append((sub_id, False, 1.0))
    for app_id in validated.application_ids:
        links.append((app_id, False, 1.0))

    # Parent for new level-3 topics: first validated subdomain, falling back
    # to the domain itself when every proposed subdomain was off-vocabulary.
    parent_id = validated.subdomain_ids[0] if validated.subdomain_ids else validated.domain_id

    seen_norm: set[str] = set()
    for raw_topic in parsed.get("topics") or []:
        norm = normalize_topic_name(raw_topic)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        topic_id, _created = resolve_or_create_level3_topic(conn, norm, parent_id)
        links.append((topic_id, False, 0.75))

    write_content_topics(conn, content_id, links)

    qualifier = resolve_claim_qualifier(conn, content_id)
    grounded_claims, ungrounded_count = build_grounded_claims(parsed.get("claims") or [], abstract, qualifier)
    stats.claims_ungrounded += ungrounded_count
    write_claims(conn, content_id, grounded_claims)
    stats.claims_written += len(grounded_claims)


def build_grounded_claims(raw_claims: list[dict], abstract: str, qualifier: str) -> tuple[list[dict], int]:
    """Pure: parsed model claims -> DB-ready claim rows. A claim is dropped
    (counted, not stored) when it has no metric/value_text, or its evidence
    is not a literal substring of the abstract. An abstract with no numbers
    yields zero claims here, never a fabricated one — there is nothing to
    match evidence against."""
    grounded: list[dict] = []
    ungrounded_count = 0
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        evidence = str(raw_claim.get("evidence") or "")
        metric = str(raw_claim.get("metric") or "").strip()
        value_text = str(raw_claim.get("value_text") or "").strip()
        if not metric or not value_text:
            continue
        if not claim_evidence_is_grounded(evidence, abstract):
            ungrounded_count += 1
            log.warning("Topics: ungrounded claim dropped metric=%r", metric)
            continue
        grounded.append(
            {
                "metric": metric,
                "value_text": value_text,
                "value_num": parse_value_num(value_text),
                "unit": raw_claim.get("unit"),
                "task": raw_claim.get("task"),
                "dataset": raw_claim.get("dataset"),
                "qualifier": qualifier,
                "evidence": evidence.strip(),
            }
        )
    return grounded, ungrounded_count


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass
class TopicsRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped_existing: int = 0
    batches: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    unknown_domains: int = 0
    unknown_subdomains: int = 0
    unknown_applications: int = 0
    claims_written: int = 0
    claims_ungrounded: int = 0

    def to_dict(self) -> dict:
        # dry-run never sets `completed` (no batches actually ran) — fall
        # back to `requested` so the dry-run summary shows a real average
        # instead of a spurious 0.0.
        denom = self.completed or self.requested
        avg = (self.estimated_cost_usd / denom) if denom else 0.0
        return {
            "requested": self.requested,
            "completed": self.completed,
            "failed": self.failed,
            "skipped_existing": self.skipped_existing,
            "batches": self.batches,
            "calls": self.calls,
            "batch_size": TOPICS_BATCH_SIZE,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "average_cost_per_paper_usd": round(avg, 6),
            "model": resolve_screen_model(),
            "provider": LLM_PROVIDER,
            "prompt_version": TOPICS_PROMPT_VERSION,
            "unknown_domains": self.unknown_domains,
            "unknown_subdomains": self.unknown_subdomains,
            "unknown_applications": self.unknown_applications,
            "claims_written": self.claims_written,
            "claims_ungrounded": self.claims_ungrounded,
        }


def estimate_topics_prompt_tokens(papers: list[dict], vocab_block: str) -> int:
    prompt = TOPICS_SYSTEM_PROMPT + build_topics_batch_user_prompt(papers, vocab_block)
    return max(1, len(prompt) // 4)


def stage_topics(
    conn,
    run_id,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    since: date | str | None = None,
    until: date | str | None = None,
    client=None,
):
    vocab = load_topic_vocabulary(conn)
    if not vocab["domains_list"]:
        raise RuntimeError(
            "No seeded domains found in research_radar.topics. "
            "Run sql/012_topic_hierarchy.sql and sql/013_seed_topic_hierarchy.sql first."
        )
    vocab_block = build_vocabulary_block(vocab)

    candidates = load_topics_candidates(conn, limit=limit, since=since, until=until)
    if not force:
        eligible = [c for c in candidates if not topics_assessment_exists(conn, c["content_id"])]
        skipped = len(candidates) - len(eligible)
        candidates = eligible
    else:
        skipped = 0

    if not candidates:
        from research_radar.candidate_window import merge_window_summary, print_empty_candidate_pool

        print_empty_candidate_pool("TOPICS", since, until)
        summary = merge_window_summary({"requested": 0, "skipped_existing": skipped}, since, until)
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"topics_empty": summary}), run_id),
        )
        return TopicsRunStats(requested=0, skipped_existing=skipped)

    stats = TopicsRunStats(requested=len(candidates), skipped_existing=skipped)
    batches = random_batches(candidates, TOPICS_BATCH_SIZE)
    stats.batches = len(batches)

    if dry_run:
        est_in = sum(estimate_topics_prompt_tokens(b, vocab_block) for b in batches)
        # Output is larger than screen scoring: a domain, up to 3 subdomains,
        # 3-8 topics, up to 4 applications and up to 5 grounded claims (each
        # with a quoted evidence sentence) per paper.
        est_out = 260 * len(candidates)
        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.calls = len(batches)
        summary = stats.to_dict()
        from research_radar.candidate_window import merge_window_summary

        summary = merge_window_summary(summary, since, until)
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"topics_dry_run": summary}), run_id),
        )
        log.info("Topics DRY RUN: %s", json.dumps(summary))
        print("\nTOPICS DRY RUN")
        print(f"  published_window: {summary['published_window']}")
        print(f"  candidates: {summary['requested']}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(cost_summary_line("Topics:", stats.requested, stats.calls, stats.estimated_cost_usd))
        return stats

    if client is None:
        client = create_llm_client()

    gate = AdaptiveConcurrencyGate(SCORING_CONCURRENCY)

    def _write_one(pid: int, paper: dict, parsed: dict | None, result_meta: dict | None, error: str | None):
        from research_radar.pipeline import bump, connect as _connect, event

        with _connect() as wconn:
            if parsed is not None:
                process_paper_result(wconn, paper, parsed, vocab, stats)
                upsert_topics_assessment(
                    wconn,
                    content_id=pid,
                    status="COMPLETED",
                    tokens_in=(result_meta or {}).get("input_tokens"),
                    tokens_out=(result_meta or {}).get("output_tokens"),
                    cost_usd=(result_meta or {}).get("estimated_cost_usd"),
                )
                event(wconn, run_id, pid, "topics", "completed", True, {"domain": parsed.get("domain")})
            else:
                upsert_topics_assessment(wconn, content_id=pid, status="ERROR")
                event(wconn, run_id, pid, "topics", "error", False, {}, error)
                bump(wconn, run_id, "errors")
            wconn.commit()

    def _process_batch(batch: list[dict]) -> list[str]:
        batch_tag = str(uuid.uuid4())[:8]
        by_id = {int(p["content_id"]): p for p in batch}
        outcomes: list[str] = []

        gate.acquire()
        try:
            try:
                result = call_topics_batch_with_retry(
                    batch, vocab_block, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                )
                stats.calls += 1
            except (LLMBatchError, TopicsParseError) as exc:
                log.warning("[batch %s] Topics batch failed entirely; falling back to individual calls: %s", batch_tag, exc)
                result = None

            parsed_map: dict[int, dict] = {}
            if result is not None:
                parsed_map = result["results"]
                stats.input_tokens += int(result["input_tokens"] or 0)
                stats.output_tokens += int(result["output_tokens"] or 0)
                stats.estimated_cost_usd += float(result["estimated_cost_usd"] or 0)

            missing_ids = set(by_id.keys()) - set(parsed_map.keys())
            for pid in missing_ids:
                try:
                    single = call_topics_batch_with_retry(
                        [by_id[pid]], vocab_block, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                    )
                    stats.calls += 1
                    r = single["results"].get(pid)
                    if r is None:
                        raise TopicsParseError(f"paper {pid} missing from individual retry response too")
                    stats.input_tokens += int(single["input_tokens"] or 0)
                    stats.output_tokens += int(single["output_tokens"] or 0)
                    stats.estimated_cost_usd += float(single["estimated_cost_usd"] or 0)
                    _write_one(pid, by_id[pid], r, single, None)
                    outcomes.append("completed")
                except Exception as exc:
                    _write_one(pid, by_id[pid], None, None, str(exc)[:1000])
                    outcomes.append("failed")

            for pid, r in parsed_map.items():
                _write_one(pid, by_id[pid], r, result, None)
                outcomes.append("completed")

            return outcomes
        finally:
            gate.release()

    workers = max(1, SCORING_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        done_batches = 0
        for fut in as_completed(futures):
            for kind in fut.result():
                if kind == "completed":
                    stats.completed += 1
                else:
                    stats.failed += 1
            done_batches += 1
            log.info("Topics batches progress %d/%d concurrency_ceiling=%d", done_batches, len(batches), gate.ceiling)

    summary = stats.to_dict()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"topics_stats": summary}), run_id),
    )
    log.info("Topics stats: %s", json.dumps(summary))
    print("\nTOPICS SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(cost_summary_line("Topics:", stats.completed, stats.calls, stats.estimated_cost_usd))
    return stats
