"""Scoring v3 classify pass — domain, audience, paper_kind on the full eligible pool.

Runs before screen on every ENTITY_RESOLVED / SCORED / CANDIDATE paper with
title + categories + abstract only (same blind input as quality scoring).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from research_radar.classification_vocab import (
    APPLICATION_DOMAINS,
    AUDIENCE_RELEVANCE,
    GEOGRAPHY_FOCUS,
    PAPER_KINDS,
)
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
    build_quality_batch_user_prompt,
    create_llm_client,
    estimate_cost_usd,
    load_quality_candidates,
    require_api_key,
    require_scoring_enabled,
    resolve_screen_model,
)

log = logging.getLogger("research-radar")

CLASSIFY_PROMPT_VERSION = (
    os.getenv("CLASSIFY_PROMPT_VERSION", "research-classify-v1").strip() or "research-classify-v1"
)
CLASSIFY_BATCH_SIZE = int(os.getenv("CLASSIFY_BATCH_SIZE", "15"))
CLASSIFY_MAX_RETRIES = int(os.getenv("CLASSIFY_MAX_RETRIES", "3"))
CLASSIFY_REQUEST_SLEEP = float(os.getenv("CLASSIFY_REQUEST_SLEEP", "0.2"))
CLASSIFY_INPUT_KIND = "title_categories_abstract"

CLASSIFY_SYSTEM_PROMPT = """You classify AI research papers for TheNeural, a research intelligence system whose
corpus also feeds a professional newsletter, enterprise advisory work, and student
teaching material.

You are given each paper's title, arXiv categories and abstract. You are NOT told who
wrote it or where they work. Do not guess, and do not let a familiar research style,
dataset or terminology lead you to infer a laboratory.

You are CLASSIFYING, not scoring. Do not judge quality, novelty or importance. A weak
paper and an excellent paper about the same subject get the same labels.

APPLICATION DOMAIN

Which real-world sector could deploy this, if the paper names or clearly implies one?

MOST AI PAPERS HAVE NO APPLICATION DOMAIN. A new attention mechanism, a training
efficiency result, or a general reasoning benchmark is general_method. This is the
correct and most common answer. Do not force-fit a sector.

Assign a sector ONLY when the paper states or unmistakably implies it - a named clinical
dataset, a named financial task, a named industrial process. "Could be useful in
healthcare" is not enough. If you are reaching for a justification, the answer is
general_method.

Return 1 to 3 values. If any real sector applies, do not also return general_method.

AUDIENCE RELEVANCE

Who in our readership would be materially better off having read this? Return every
audience that applies, minimum one.

practitioner           An engineer or data scientist could act on this: a technique they
                       could implement, a result that changes how they would build.
technical_leadership   Bears on an architecture, platform or build-or-buy decision.
enterprise_adoption    Bears on deploying AI in an organisation - governance, risk,
                       evaluation, cost, safety, operating model, workforce.
student                Clarifies a foundation, or is a good entry point into a subfield
                       for someone still learning. Survey and tutorial-like papers often
                       qualify; highly specialised incremental work rarely does.

Audience is about SUBJECT and TREATMENT, not quality. A rigorous narrow paper may serve
only practitioners. A clear survey may serve students and technical leadership both.

PAPER KIND

One value. What kind of contribution is this?

method · empirical_study · benchmark_dataset · survey_review · theory · position ·
negative_result · system_infrastructure

GEOGRAPHY FOCUS

Only when the paper is ABOUT a specific geography - a national policy, a
country-specific dataset, a regional deployment. This is the subject of the paper, NOT
where the authors are. You have not been told where the authors are and must not infer
it. Almost always: none.

CONFIDENCE

domain_confidence   0-10 in 0.5 increments. How sure you are of the application domain
                    given only what this abstract contains. Low confidence with
                    general_method is fine and expected.

Return ONLY a JSON object with a single key "papers", whose value is an array with one
object per paper, in the order supplied. No reasons, no prose, no markdown fences.

{"papers": [{"paper_id": <int>, "application_domain": ["..."],
"audience_relevance": ["..."], "paper_kind": "...", "geography_focus": "...",
"domain_confidence": <0-10>}]}
"""

CLASSIFY_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "integer"},
        "application_domain": {
            "type": "array",
            "items": {"type": "string", "enum": list(APPLICATION_DOMAINS)},
            "minItems": 0,
            "maxItems": 3,
        },
        "audience_relevance": {
            "type": "array",
            "items": {"type": "string", "enum": list(AUDIENCE_RELEVANCE)},
            "minItems": 1,
            "maxItems": 4,
        },
        "paper_kind": {"type": "string", "enum": list(PAPER_KINDS)},
        "geography_focus": {"type": "string", "enum": list(GEOGRAPHY_FOCUS)},
        "domain_confidence": {"type": "number"},
    },
    "required": [
        "paper_id",
        "application_domain",
        "audience_relevance",
        "paper_kind",
        "geography_focus",
        "domain_confidence",
    ],
}

CLASSIFY_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "papers": {"type": "array", "items": CLASSIFY_ITEM_SCHEMA},
    },
    "required": ["papers"],
}


class ClassifyParseError(ValueError):
    pass


def _normalize_half_point(value: float) -> tuple[float, bool]:
    rounded = round(value * 2) / 2.0
    return rounded, abs(rounded - value) > 1e-9


def parse_classify_batch(text: str, expected_ids: set[int]) -> dict[int, dict]:
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClassifyParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise ClassifyParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise ClassifyParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ClassifyParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise ClassifyParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc
        if pid not in expected_ids:
            continue

        app_dom = item.get("application_domain")
        audience = item.get("audience_relevance")
        paper_kind = (item.get("paper_kind") or "").strip()
        geography = (item.get("geography_focus") or "").strip()
        if not isinstance(app_dom, list) or len(app_dom) > 3:
            raise ClassifyParseError(f"paper {pid} invalid application_domain")
        if not isinstance(audience, list) or not (1 <= len(audience) <= 4):
            raise ClassifyParseError(f"paper {pid} invalid audience_relevance")
        if paper_kind not in PAPER_KINDS:
            raise ClassifyParseError(f"paper {pid} invalid paper_kind: {paper_kind!r}")
        if geography not in GEOGRAPHY_FOCUS:
            raise ClassifyParseError(f"paper {pid} invalid geography_focus: {geography!r}")
        for v in app_dom:
            if v not in APPLICATION_DOMAINS:
                raise ClassifyParseError(f"paper {pid} invalid application_domain value: {v!r}")
        for v in audience:
            if v not in AUDIENCE_RELEVANCE:
                raise ClassifyParseError(f"paper {pid} invalid audience value: {v!r}")
        if "general_method" in app_dom and len(app_dom) > 1:
            raise ClassifyParseError(f"paper {pid} general_method must not combine with other domains")

        try:
            conf_raw = float(item["domain_confidence"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ClassifyParseError(f"paper {pid} missing domain_confidence") from exc
        if conf_raw < 0.0 or conf_raw > 10.0:
            raise ClassifyParseError(f"paper {pid} domain_confidence out of range")
        conf, _ = _normalize_half_point(conf_raw)

        out[pid] = {
            "application_domain": app_dom,
            "audience_relevance": audience,
            "paper_kind": paper_kind,
            "geography_focus": geography,
            "domain_confidence": conf,
        }
    return out


def call_classify_batch(papers: list[dict], *, client=None, on_rate_limited=None) -> dict:
    if client is None:
        client = create_llm_client()
    model = resolve_screen_model()
    user_prompt = build_quality_batch_user_prompt(papers)
    expected_ids = {int(p["content_id"]) for p in papers}

    result = call_chat_completion(
        client,
        model=model,
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=None,
        temperature=0.0,
        max_retries=1,
        request_sleep=CLASSIFY_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "classify_assessment",
                "strict": True,
                "schema": CLASSIFY_RESPONSE_SCHEMA,
            },
        },
        on_rate_limited=on_rate_limited,
    )
    parsed = parse_classify_batch(result["text"], expected_ids)
    return {
        "results": parsed,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
        "estimated_cost_usd": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
    }


def call_classify_batch_with_retry(
    papers: list[dict],
    *,
    client=None,
    max_retries: int | None = None,
    on_rate_limited=None,
    batch_tag: str = "",
) -> dict:
    max_retries = max_retries if max_retries is not None else CLASSIFY_MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_classify_batch(papers, client=client, on_rate_limited=on_rate_limited)
        except (LLMBatchError, ClassifyParseError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(60.0, 2 ** (attempt - 1))
                log.warning(
                    "[batch %s] Classify batch retry attempt=%s/%s size=%s err=%s",
                    batch_tag,
                    attempt,
                    max_retries,
                    len(papers),
                    exc,
                )
                time.sleep(wait)
                continue
    raise last_exc


def classification_exists(conn, content_id: int) -> bool:
    row = conn.execute(
        """
        SELECT 1 AS ok FROM research_radar.content_classifications
        WHERE content_id = %s AND prompt_version = %s
        LIMIT 1
        """,
        (content_id, CLASSIFY_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_classification(conn, *, content_id: int, result: dict):
    conn.execute(
        """
        INSERT INTO research_radar.content_classifications(
            content_id, prompt_version, model, classify_input_kind,
            application_domain, audience_relevance, paper_kind,
            geography_focus, domain_confidence
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (content_id, prompt_version) DO UPDATE SET
            model = EXCLUDED.model,
            classify_input_kind = EXCLUDED.classify_input_kind,
            application_domain = EXCLUDED.application_domain,
            audience_relevance = EXCLUDED.audience_relevance,
            paper_kind = EXCLUDED.paper_kind,
            geography_focus = EXCLUDED.geography_focus,
            domain_confidence = EXCLUDED.domain_confidence,
            created_at = NOW()
        """,
        (
            content_id,
            CLASSIFY_PROMPT_VERSION,
            resolve_screen_model(),
            CLASSIFY_INPUT_KIND,
            result["application_domain"],
            result["audience_relevance"],
            result["paper_kind"],
            result["geography_focus"],
            result["domain_confidence"],
        ),
    )


@dataclass
class ClassifyRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    batches: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "completed": self.completed,
            "failed": self.failed,
            "batches": self.batches,
            "calls": self.calls,
            "batch_size": CLASSIFY_BATCH_SIZE,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "model": resolve_screen_model(),
            "provider": LLM_PROVIDER,
            "prompt_version": CLASSIFY_PROMPT_VERSION,
        }


def estimate_classify_prompt_tokens(papers: list[dict]) -> int:
    prompt = CLASSIFY_SYSTEM_PROMPT + build_quality_batch_user_prompt(papers)
    return max(1, len(prompt) // 4)


def stage_classify(
    conn,
    run_id,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    client=None,
):
    """Classify every eligible paper before the screen gate."""
    if not dry_run:
        require_scoring_enabled()
        require_api_key()

    candidates = load_quality_candidates(conn, limit=limit)
    if not force:
        candidates = [c for c in candidates if not classification_exists(conn, c["content_id"])]

    stats = ClassifyRunStats(requested=len(candidates))
    batches = random_batches(candidates, CLASSIFY_BATCH_SIZE)
    stats.batches = len(batches)

    if dry_run:
        est_in = sum(estimate_classify_prompt_tokens(b) for b in batches)
        est_out = 80 * len(candidates)
        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.calls = len(batches)
        summary = stats.to_dict()
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"classify_dry_run": summary}), run_id),
        )
        log.info("Classify DRY RUN: %s", json.dumps(summary))
        print("\nCLASSIFY DRY RUN")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(cost_summary_line("Classify:", stats.requested, stats.calls, stats.estimated_cost_usd))
        return stats

    if client is None:
        client = create_llm_client()

    gate = AdaptiveConcurrencyGate(SCORING_CONCURRENCY)

    def _process_batch(batch: list[dict]) -> list[tuple[str, dict]]:
        from research_radar.pipeline import bump, connect as _connect, event

        batch_tag = str(uuid.uuid4())[:8]
        expected_ids = {int(p["content_id"]) for p in batch}
        by_id = {int(p["content_id"]): p for p in batch}
        outcomes: list[tuple[str, dict]] = []

        gate.acquire()
        try:
            try:
                result = call_classify_batch_with_retry(
                    batch, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                )
                stats.calls += 1
            except (LLMBatchError, ClassifyParseError) as exc:
                log.warning("[batch %s] Classify batch failed; falling back per paper: %s", batch_tag, exc)
                result = None

            parsed: dict[int, dict] = {}
            if result is not None:
                parsed = result["results"]
                stats.input_tokens += int(result["input_tokens"] or 0)
                stats.output_tokens += int(result["output_tokens"] or 0)
                stats.estimated_cost_usd += float(result["estimated_cost_usd"] or 0)

            missing_ids = expected_ids - set(parsed.keys())
            for pid in missing_ids:
                paper = by_id[pid]
                try:
                    single = call_classify_batch_with_retry(
                        [paper], client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                    )
                    stats.calls += 1
                    stats.input_tokens += int(single["input_tokens"] or 0)
                    stats.output_tokens += int(single["output_tokens"] or 0)
                    stats.estimated_cost_usd += float(single["estimated_cost_usd"] or 0)
                    r = single["results"].get(pid)
                    if r is None:
                        raise ClassifyParseError(f"paper {pid} missing from individual retry")
                    with _connect() as wconn:
                        upsert_classification(wconn, content_id=pid, result=r)
                        event(wconn, run_id, pid, "classify", "completed", True, {"batch": batch_tag})
                        wconn.commit()
                    outcomes.append(("completed", {"content_id": pid}))
                except Exception as exc:
                    with _connect() as wconn:
                        event(wconn, run_id, pid, "classify", "error", False, {"batch": batch_tag}, str(exc))
                        bump(wconn, run_id, "errors")
                        wconn.commit()
                    outcomes.append(("failed", {"content_id": pid, "error": str(exc)}))

            if parsed:
                with _connect() as wconn:
                    for pid, r in parsed.items():
                        upsert_classification(wconn, content_id=pid, result=r)
                        event(wconn, run_id, pid, "classify", "completed", True, {"batch_size": len(batch), "batch": batch_tag})
                    wconn.commit()
                for pid in parsed:
                    outcomes.append(("completed", {"content_id": pid}))
            return outcomes
        finally:
            gate.release()

    workers = max(1, SCORING_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        done_batches = 0
        for fut in as_completed(futures):
            for kind, _payload in fut.result():
                if kind == "completed":
                    stats.completed += 1
                else:
                    stats.failed += 1
            done_batches += 1
            log.info("Classify batches progress %d/%d concurrency_ceiling=%d", done_batches, len(batches), gate.ceiling)

    summary = stats.to_dict()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"classify_stats": summary}), run_id),
    )
    log.info("Classify stats: %s", json.dumps(summary))
    print("\nCLASSIFY SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(cost_summary_line("Classify:", stats.completed, stats.calls, stats.estimated_cost_usd))
    return stats
