"""Independence classifier (Call B) — categorical, never numeric.

This is a SEPARATE OpenRouter call from quality scoring (semantic_scoring.py).
It is the only module allowed to see paper_metadata.affiliation_text. The
quality scorer must never import from here and must never receive this
module's inputs — that separation is the architectural guarantee described in
the scoring-v2 brief. Do not merge the two calls to save cost.

Uses paper_metadata.affiliation_text (free, extracted during `enrich`), never
resolved content_organisations watchlist matches — those come from the paid
affiliation-gpt stage which has only run on a small fraction of papers, so
almost everything would classify as `unclear` if we waited on it.
"""

from __future__ import annotations

import json
import logging
import os
import time
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
from research_radar.classification_vocab import PAPER_KINDS
from research_radar.semantic_scoring import (
    OPENROUTER_API_KEY,
    QUALITY_PROMPT_VERSION,
    create_llm_client,
    estimate_cost_usd,
)

log = logging.getLogger("research-radar")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEPENDENCE_MODEL = (os.getenv("INDEPENDENCE_MODEL", "openai/gpt-5.6-sol").strip() or "openai/gpt-5.6-sol")
# Independence is a short categorical task; reasoning is cheap here (brief §2).
INDEPENDENCE_REASONING_EFFORT = os.getenv("INDEPENDENCE_REASONING_EFFORT", "low").strip() or "low"
INDEPENDENCE_PROMPT_VERSION = (
    os.getenv("INDEPENDENCE_PROMPT_VERSION", "independence-v2").strip() or "independence-v2"
)
INDEPENDENCE_BATCH_SIZE = int(os.getenv("INDEPENDENCE_BATCH_SIZE", "20"))
INDEPENDENCE_MAX_RETRIES = int(os.getenv("INDEPENDENCE_MAX_RETRIES", "3"))
INDEPENDENCE_REQUEST_SLEEP = float(os.getenv("INDEPENDENCE_REQUEST_SLEEP", "0.2"))
LLM_PROVIDER = "openrouter"

INDEPENDENCE_STATUSES = ("independent", "self_evaluation", "unclear", "not_applicable")

# Status -> final-score multiplier. A Python constant, deliberately, so this
# can be recalibrated without re-running inference (brief design decision 3).
INDEPENDENCE_FACTORS = {
    "independent": 1.00,
    "not_applicable": 1.00,
    "unclear": 0.95,
    "self_evaluation": 0.90,
}

DEFAULT_INDEPENDENCE_FACTOR = INDEPENDENCE_FACTORS["unclear"]

SYSTEM_PROMPT = """You classify the independence of AI research papers.

For each paper you are given its title, abstract, and author affiliation text
where available.

Decide ONE status:

independent
    The paper evaluates or makes comparative claims about a system, model or
    product built by an organisation the authors are NOT affiliated with.

self_evaluation
    The paper makes evaluative or comparative claims about an EXISTING
    product, model or system built by an organisation the authors ARE
    affiliated with. This is the narrow case of an organisation reporting
    favourably on its own shipped system.

not_applicable
    The paper introduces a new method, model or result of its own and
    evaluates it experimentally. This is ordinary original research and is
    NOT self-evaluation. Most papers are this. Theory papers, surveys,
    datasets and negative results are also not_applicable.

unclear
    You cannot tell from what you were given. Use this when affiliation is
    missing AND the paper makes comparative claims about a named commercial
    system. Unknown is NOT the same as independent - do not guess.

CRITICAL: authors evaluating the new method their own paper introduces is
not_applicable, not self_evaluation. Almost all research does this. Only use
self_evaluation for claims about a pre-existing organisational product.

Give one short reason per paper citing what in the supplied text led to the
decision.

Also emit paper_kind — one value describing what kind of contribution this is:

method · empirical_study · benchmark_dataset · survey_review · theory · position ·
negative_result · system_infrastructure

Theory papers, surveys, datasets and negative results are often not_applicable for
independence; still emit the paper_kind you infer from the abstract.

Return ONLY a JSON object with a single key "papers", whose value is an array
with one object per paper, in the order supplied. No prose, no markdown
fences.
{"papers": [{"paper_id": <int>, "status": "...", "reason": "...", "paper_kind": "..."}]}
"""

INDEPENDENCE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "integer"},
        "status": {"type": "string", "enum": list(INDEPENDENCE_STATUSES)},
        "reason": {"type": "string"},
        "paper_kind": {"type": "string", "enum": list(PAPER_KINDS)},
    },
    "required": ["paper_id", "status", "reason", "paper_kind"],
}

INDEPENDENCE_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "papers": {"type": "array", "items": INDEPENDENCE_ITEM_SCHEMA},
    },
    "required": ["papers"],
}


def resolve_independence_model() -> str:
    model = INDEPENDENCE_MODEL
    return model if "/" in model else f"openai/{model}"


class IndependenceParseError(ValueError):
    pass


def build_paper_block(paper: dict) -> str:
    """PAPER {id} / TITLE / AFFILIATIONS / ABSTRACT — verbatim brief template."""
    aff = paper.get("affiliation_text")
    if isinstance(aff, (list, tuple)):
        aff_text = "; ".join(str(a).strip() for a in aff if a) or "not available"
    else:
        aff_text = (aff or "").strip() or "not available"
    content_id = paper.get("content_id", paper.get("id"))
    return (
        f"PAPER {content_id}\n"
        f"TITLE: {(paper.get('title') or '').strip()}\n"
        f"AFFILIATIONS: {aff_text}\n"
        f"ABSTRACT: {(paper.get('abstract') or '').strip() or '(empty)'}\n"
    )


def build_batch_user_prompt(papers: list[dict]) -> str:
    return "\n\n".join(build_paper_block(p) for p in papers)


def parse_independence_batch(text: str, expected_ids: set[int]) -> dict[int, dict]:
    """Parse the model's {"papers": [...]} object. Only ids we asked about are kept.

    strip_json_fences handles accidental markdown-fence wrapping; the object
    shape itself is enforced API-side by INDEPENDENCE_RESPONSE_SCHEMA
    (strict json_schema), so this is a defensive re-check, not the primary
    validation.
    """
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IndependenceParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise IndependenceParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise IndependenceParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise IndependenceParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise IndependenceParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc
        status = (item.get("status") or "").strip()
        if status not in INDEPENDENCE_STATUSES:
            raise IndependenceParseError(f"invalid status for paper {pid}: {status!r}")
        reason = (item.get("reason") or "").strip()
        if not reason:
            raise IndependenceParseError(f"empty reason for paper {pid}")
        paper_kind = (item.get("paper_kind") or "").strip()
        if paper_kind not in PAPER_KINDS:
            raise IndependenceParseError(f"invalid paper_kind for paper {pid}: {paper_kind!r}")
        if pid in expected_ids:
            out[pid] = {"status": status, "reason": reason, "paper_kind": paper_kind}
    return out


def call_independence_batch(papers: list[dict], *, client=None, on_rate_limited=None) -> dict:
    """One or many papers. Returns {results: {content_id: {status, reason}}, usage...}."""
    if client is None:
        client = create_llm_client()
    model = resolve_independence_model()
    user_prompt = build_batch_user_prompt(papers)
    expected_ids = {int(p.get("content_id", p.get("id"))) for p in papers}

    result = call_chat_completion(
        client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=INDEPENDENCE_REASONING_EFFORT,
        temperature=0.0,
        max_retries=1,  # outer stage loop owns the batch-level retry/backoff
        request_sleep=INDEPENDENCE_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "independence_assessment",
                "strict": True,
                "schema": INDEPENDENCE_RESPONSE_SCHEMA,
            },
        },
        on_rate_limited=on_rate_limited,
    )
    parsed = parse_independence_batch(result["text"], expected_ids)
    return {
        "results": parsed,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
        "estimated_cost_usd": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
    }


def call_independence_batch_with_retry(
    papers: list[dict],
    *,
    client=None,
    max_retries: int | None = None,
    on_rate_limited=None,
    batch_tag: str = "",
) -> dict:
    """Batch call with exponential-backoff retry at the batch level."""
    max_retries = max_retries if max_retries is not None else INDEPENDENCE_MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_independence_batch(papers, client=client, on_rate_limited=on_rate_limited)
        except (LLMBatchError, IndependenceParseError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(60.0, 2 ** (attempt - 1))
                log.warning(
                    "[batch %s] Independence batch retry attempt=%s/%s size=%s err=%s",
                    batch_tag,
                    attempt,
                    max_retries,
                    len(papers),
                    exc,
                )
                time.sleep(wait)
                continue
    raise last_exc


@dataclass
class IndependenceRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped_existing: int = 0
    by_status: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    batches: int = 0
    calls: int = 0

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "completed": self.completed,
            "failed": self.failed,
            "skipped_existing": self.skipped_existing,
            "by_status": self.by_status,
            "batches": self.batches,
            "calls": self.calls,
            "batch_size": INDEPENDENCE_BATCH_SIZE,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "model": resolve_independence_model(),
            "provider": LLM_PROVIDER,
            "prompt_version": INDEPENDENCE_PROMPT_VERSION,
            "reasoning_effort": INDEPENDENCE_REASONING_EFFORT,
        }


def independence_assessment_exists(conn, content_id: int) -> bool:
    model = resolve_independence_model()
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_independence_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
          AND status <> 'ERROR'
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, model, INDEPENDENCE_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_independence_assessment(
    conn,
    *,
    content_id: int,
    status: str,
    reason: str | None,
    paper_kind: str | None,
    evidence_used: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
    response_id: str | None,
):
    conn.execute(
        """
        INSERT INTO research_radar.content_independence_assessments(
            content_id, provider, model_name, prompt_version,
            status, reason, paper_kind, evidence_used,
            tokens_in, tokens_out, cost_usd, response_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            status = EXCLUDED.status,
            reason = EXCLUDED.reason,
            paper_kind = EXCLUDED.paper_kind,
            evidence_used = EXCLUDED.evidence_used,
            tokens_in = EXCLUDED.tokens_in,
            tokens_out = EXCLUDED.tokens_out,
            cost_usd = EXCLUDED.cost_usd,
            response_id = EXCLUDED.response_id,
            created_at = NOW()
        """,
        (
            content_id,
            LLM_PROVIDER,
            resolve_independence_model(),
            INDEPENDENCE_PROMPT_VERSION,
            status,
            reason,
            paper_kind,
            evidence_used,
            input_tokens,
            output_tokens,
            cost_usd,
            response_id,
        ),
    )


def load_independence_candidates(
    conn,
    limit: int | None = None,
    *,
    since: date | str | None = None,
    until: date | str | None = None,
) -> list[dict]:
    """
    Papers with a COMPLETED pass-2 ("full") quality assessment, affiliation_text
    only — never content_organisations. Independence now runs ONLY on pass-2
    papers (tiering brief §1): classifying a paper that screened out and will
    never rank is wasted spend. This replaced the earlier
    ENTITY_RESOLVED/SCORED/CANDIDATE status filter, which predates tiering.

    Optional ``since`` / ``until`` bound ``ci.published_at`` (until is inclusive).
    """
    from research_radar.candidate_window import published_at_sql_filters

    window_sql, window_params = published_at_sql_filters(since, until)
    rows = conn.execute(
        f"""
        SELECT
            ci.id AS content_id,
            ci.title,
            COALESCE(pm.abstract, ci.summary, '') AS abstract,
            pm.affiliation_text
        FROM research_radar.content_score_assessments a
        JOIN research_radar.content_items ci ON ci.id = a.content_id
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE a.prompt_version = %s
          AND a.scoring_tier = 'full'
          AND a.status = 'COMPLETED'
        {window_sql}
        ORDER BY ci.id
        LIMIT %s
        """,
        (QUALITY_PROMPT_VERSION, *window_params, limit or 10_000),
    ).fetchall()
    return [dict(r) for r in rows]


def estimate_prompt_tokens(papers: list[dict]) -> int:
    prompt = SYSTEM_PROMPT + build_batch_user_prompt(papers)
    return max(1, len(prompt) // 4)


def stage_independence(
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
    from research_radar.pipeline import bump, event

    candidates = load_independence_candidates(conn, limit=limit, since=since, until=until)
    if not force:
        candidates = [c for c in candidates if not independence_assessment_exists(conn, c["content_id"])]

    if not candidates:
        from research_radar.candidate_window import merge_window_summary, print_empty_candidate_pool

        print_empty_candidate_pool("INDEPENDENCE", since, until)
        summary = merge_window_summary({"requested": 0}, since, until)
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"independence_empty": summary}), run_id),
        )
        return IndependenceRunStats(requested=0)

    stats = IndependenceRunStats(requested=len(candidates))
    batches = random_batches(candidates, INDEPENDENCE_BATCH_SIZE)
    stats.batches = len(batches)

    if dry_run:
        est_in = sum(estimate_prompt_tokens(b) for b in batches)
        est_out = 60 * len(candidates)  # ~status+short reason per paper, heuristic
        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.calls = len(batches)
        summary = stats.to_dict()
        from research_radar.candidate_window import merge_window_summary

        summary = merge_window_summary(summary, since, until)
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"independence_dry_run": summary}), run_id),
        )
        log.info("Independence DRY RUN: %s", json.dumps(summary))
        print("\nINDEPENDENCE DRY RUN")
        print(f"  published_window: {summary['published_window']}")
        print(f"  candidates: {summary['requested']}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(cost_summary_line("Independence:", stats.requested, stats.calls, stats.estimated_cost_usd))
        return stats

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Set it in .env before independence classification.")

    if client is None:
        client = create_llm_client()

    gate = AdaptiveConcurrencyGate(SCORING_CONCURRENCY)

    def _process_batch(batch: list[dict]) -> list[tuple[str, dict]]:
        from research_radar.pipeline import connect as _connect

        batch_id_tag = str(id(batch))[-6:]  # cheap unique-enough tag for log interleaving
        expected_ids = {int(p["content_id"]) for p in batch}
        by_id = {int(p["content_id"]): p for p in batch}
        outcomes: list[tuple[str, dict]] = []

        gate.acquire()
        try:
            try:
                result = call_independence_batch_with_retry(
                    batch, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_id_tag
                )
                stats.calls += 1
            except (LLMBatchError, IndependenceParseError) as exc:
                log.warning("[batch %s] Independence batch failed entirely; falling back to individual calls: %s", batch_id_tag, exc)
                result = None

            parsed: dict[int, dict] = {}
            usage_in = usage_out = 0
            cost = 0.0
            response_id = None
            if result is not None:
                parsed = result["results"]
                usage_in = result["input_tokens"]
                usage_out = result["output_tokens"]
                cost = result["estimated_cost_usd"]
                response_id = result["response_id"]

            missing_ids = expected_ids - set(parsed.keys())
            # Individual fallback: whole-batch failure, or specific papers the model dropped.
            for pid in missing_ids:
                paper = by_id[pid]
                try:
                    single = call_independence_batch_with_retry(
                        [paper], client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_id_tag
                    )
                    stats.calls += 1
                    r = single["results"].get(pid)
                    if r is None:
                        raise IndependenceParseError(f"paper {pid} missing from individual retry response too")
                    with _connect() as wconn:
                        upsert_independence_assessment(
                            wconn,
                            content_id=pid,
                            status=r["status"],
                            reason=r["reason"],
                            paper_kind=r.get("paper_kind"),
                            evidence_used=paper.get("affiliation_text") and json.dumps(paper.get("affiliation_text"), default=str),
                            input_tokens=single["input_tokens"],
                            output_tokens=single["output_tokens"],
                            cost_usd=single["estimated_cost_usd"],
                            response_id=single["response_id"],
                        )
                        event(wconn, run_id, pid, "independence", r["status"], True, {"individual_retry": True, "batch": batch_id_tag})
                        wconn.commit()
                    outcomes.append(("completed", {"content_id": pid, **r, "individual_retry": True}))
                except Exception as exc:
                    with _connect() as wconn:
                        upsert_independence_assessment(
                            wconn,
                            content_id=pid,
                            status="ERROR",
                            reason=str(exc)[:1000],
                            paper_kind=None,
                            evidence_used=None,
                            input_tokens=None,
                            output_tokens=None,
                            cost_usd=None,
                            response_id=None,
                        )
                        event(wconn, run_id, pid, "independence", "error", False, {"batch": batch_id_tag}, str(exc))
                        bump(wconn, run_id, "errors")
                        wconn.commit()
                    outcomes.append(("failed", {"content_id": pid, "error": str(exc)}))

            if parsed:
                with _connect() as wconn:
                    for pid, r in parsed.items():
                        paper = by_id[pid]
                        upsert_independence_assessment(
                            wconn,
                            content_id=pid,
                            status=r["status"],
                            reason=r["reason"],
                            paper_kind=r.get("paper_kind"),
                            evidence_used=paper.get("affiliation_text") and json.dumps(paper.get("affiliation_text"), default=str),
                            input_tokens=None,
                            output_tokens=None,
                            cost_usd=None,
                            response_id=response_id,
                        )
                        event(wconn, run_id, pid, "independence", r["status"], True, {"batch_size": len(batch), "batch": batch_id_tag})
                    wconn.commit()
                for pid, r in parsed.items():
                    outcomes.append(("completed", {"content_id": pid, **r}))

            if outcomes:
                outcomes[0] = (
                    outcomes[0][0],
                    {**outcomes[0][1], "_batch_input_tokens": usage_in, "_batch_output_tokens": usage_out, "_batch_cost": cost},
                )
            return outcomes
        finally:
            gate.release()

    workers = max(1, SCORING_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        done_batches = 0
        for fut in as_completed(futures):
            for kind, payload in fut.result():
                stats.input_tokens += int(payload.pop("_batch_input_tokens", 0) or 0)
                stats.output_tokens += int(payload.pop("_batch_output_tokens", 0) or 0)
                stats.estimated_cost_usd += float(payload.pop("_batch_cost", 0) or 0)
                if kind == "completed":
                    stats.completed += 1
                    status = payload.get("status")
                    stats.by_status[status] = stats.by_status.get(status, 0) + 1
                else:
                    stats.failed += 1
            done_batches += 1
            log.info("Independence batches progress %d/%d concurrency_ceiling=%d", done_batches, len(batches), gate.ceiling)

    summary = stats.to_dict()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"independence_stats": summary}), run_id),
    )
    log.info("Independence stats: %s", json.dumps(summary))
    print("\nINDEPENDENCE SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(cost_summary_line("Independence:", stats.completed, stats.calls, stats.estimated_cost_usd))
    return stats
