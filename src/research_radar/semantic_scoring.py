"""GPT semantic paper scoring via OpenRouter (assessment-only; does not mutate content_scores).

Phase 1: reproducible 100-paper experiment using openai/gpt-5.6-sol through OpenRouter.
Organisation / person / deterministic scores are intentionally NOT sent to the model.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from research_radar.llm_batch import (
    AdaptiveConcurrencyGate,
    LLMBatchError,
    SCORING_CONCURRENCY,
    call_chat_completion,
    cost_summary_line,
    random_batches,
    strip_json_fences,
)

log = logging.getLogger("research-radar")

# ---------------------------------------------------------------------------
# Configuration — OpenRouter only (same pattern as Newsletter5)
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").strip()
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://theneural.ai/").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "TheNeural Research Radar").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol").strip() or "openai/gpt-5.6-sol"
SEMANTIC_REASONING_EFFORT = os.getenv("SEMANTIC_REASONING_EFFORT", "medium").strip() or "medium"
SEMANTIC_MAX_RETRIES = int(os.getenv("SEMANTIC_MAX_RETRIES", "5"))
SEMANTIC_REQUEST_SLEEP = float(os.getenv("SEMANTIC_REQUEST_SLEEP", "0.2"))
SEMANTIC_PROMPT_VERSION = os.getenv("SEMANTIC_PROMPT_VERSION", "research-semantic-v1").strip()
SEMANTIC_SCORING_WORKERS = int(os.getenv("SEMANTIC_SCORING_WORKERS", "2"))
LLM_PROVIDER = "openrouter"
ASSESSMENT_TYPE = "llm_semantic"
SAMPLE_SEED = int(os.getenv("SEMANTIC_SAMPLE_SEED", "20260824"))

# Pricing (USD per 1M tokens). Override via env — provisional placeholders.
OPENROUTER_INPUT_COST_PER_MILLION = float(os.getenv("OPENROUTER_INPUT_COST_PER_MILLION", "2.00"))
OPENROUTER_OUTPUT_COST_PER_MILLION = float(os.getenv("OPENROUTER_OUTPUT_COST_PER_MILLION", "10.00"))


def resolve_model_name() -> str:
    """OpenRouter model id, e.g. openai/gpt-5.6-sol."""
    model = OPENROUTER_MODEL
    if "/" not in model:
        return f"openai/{model}"
    return model


def create_llm_client():
    """OpenRouter client (OpenAI-compatible SDK pointed at openrouter.ai)."""
    from openai import OpenAI

    return OpenAI(
        base_url=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
        },
    )

SEMANTIC_WEIGHTS = {
    "ai_relevance": 0.20,
    "technical_significance": 0.20,
    "practical_applicability": 0.15,
    "professional_value": 0.15,
    "student_learning_value": 0.10,
    "apparent_novelty": 0.10,
    "explainability": 0.05,
    "industry_relevance": 0.05,
}

DIMENSION_KEYS = list(SEMANTIC_WEIGHTS.keys())

INDUSTRY_LABELS_ALLOWED = [
    "software_engineering",
    "enterprise_ai",
    "cybersecurity",
    "financial_services",
    "healthcare",
    "education",
    "robotics",
    "autonomous_systems",
    "ecommerce",
    "media_creative",
    "developer_tools",
    "data_analytics",
    "cloud_infrastructure",
    "ai_safety",
    "ai_evaluation",
    "research_tools",
]

SYSTEM_PROMPT = """You are evaluating AI research papers for a professional research intelligence system.

Your job is to judge the supplied paper itself, not the reputation of its authors or institution.

You receive only:
- title
- abstract
- arXiv categories

Score each requested dimension from 0 to 10.

Use the full scale.
Do not cluster everything between 6 and 8.

A score of:
0-2 = weak
3-4 = below average
5-6 = meaningful/moderate
7-8 = strong
9 = exceptional
10 = extremely rare and should require unusually strong evidence in the supplied abstract.

Do not infer facts not present in the supplied text.
Do not reward famous terminology.
Do not assume claimed results are independently verified.
Treat novelty as apparent novelty based only on the abstract.
Give one concise reason for each score (1 sentence; maximum 2 short sentences).
Do not provide chain-of-thought or hidden reasoning.
"""

DIMENSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["score", "reason"],
}

SEMANTIC_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ai_relevance": DIMENSION_SCHEMA,
        "technical_significance": DIMENSION_SCHEMA,
        "practical_applicability": DIMENSION_SCHEMA,
        "professional_value": DIMENSION_SCHEMA,
        "student_learning_value": DIMENSION_SCHEMA,
        "apparent_novelty": DIMENSION_SCHEMA,
        "explainability": DIMENSION_SCHEMA,
        "industry_relevance": DIMENSION_SCHEMA,
        "industry_labels": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string", "enum": INDUSTRY_LABELS_ALLOWED},
        },
    },
    "required": [
        "ai_relevance",
        "technical_significance",
        "practical_applicability",
        "professional_value",
        "student_learning_value",
        "apparent_novelty",
        "explainability",
        "industry_relevance",
        "industry_labels",
    ],
}


@dataclass
class SemanticRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    rate_limited: int = 0
    refused: int = 0
    skipped_existing: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    sample_groups: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        avg = (self.estimated_cost_usd / self.completed) if self.completed else 0.0
        return {
            "papers_requested": self.requested,
            "completed": self.completed,
            "failed": self.failed,
            "rate_limited": self.rate_limited,
            "refused": self.refused,
            "skipped_existing": self.skipped_existing,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "average_cost_per_paper_usd": round(avg, 6),
            "model": resolve_model_name(),
            "provider": LLM_PROVIDER,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "reasoning_effort": SEMANTIC_REASONING_EFFORT,
            "sample_groups": self.sample_groups,
        }


class SemanticScoringDisabled(RuntimeError):
    pass


class SemanticScoringConfigError(RuntimeError):
    pass


class SemanticParseError(ValueError):
    pass


class SemanticAPIError(RuntimeError):
    def __init__(self, message: str, *, status: str = "ERROR", retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def require_scoring_enabled():
    enabled = os.getenv("SEMANTIC_SCORING_ENABLED", "false").lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        raise SemanticScoringDisabled(
            "SEMANTIC_SCORING_ENABLED is false. Set SEMANTIC_SCORING_ENABLED=true in .env."
        )


def require_api_key():
    if not OPENROUTER_API_KEY:
        raise SemanticScoringConfigError(
            "OPENROUTER_API_KEY is missing. Set it in .env before semantic scoring."
        )


def clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticParseError(f"score is not numeric: {value!r}") from exc
    if x < 0.0 or x > 10.0:
        raise SemanticParseError(f"score out of range 0-10: {x}")
    return round(x, 2)


def compute_semantic_score(scores: dict[str, float]) -> float:
    """Weighted semantic score from component dimensions (Python-side; not model-invented)."""
    total = 0.0
    for key, weight in SEMANTIC_WEIGHTS.items():
        if key not in scores:
            raise SemanticParseError(f"missing dimension for semantic_score: {key}")
        total += float(scores[key]) * weight
    return round(total, 2)


def estimate_cost_usd(input_tokens: int | None, output_tokens: int | None) -> float:
    """Separate cost calculation from scoring logic."""
    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    cost = (inp / 1_000_000.0) * OPENROUTER_INPUT_COST_PER_MILLION + (
        out / 1_000_000.0
    ) * OPENROUTER_OUTPUT_COST_PER_MILLION
    return round(cost, 6)


def build_user_prompt(*, title: str, abstract: str, categories) -> str:
    """API input: title + abstract + categories only. No org/person/scores."""
    if isinstance(categories, str):
        cat_text = categories
    elif isinstance(categories, (list, tuple)):
        cat_text = ", ".join(str(c) for c in categories if c)
    else:
        cat_text = ""
    return (
        f"TITLE:\n{(title or '').strip()}\n\n"
        f"ARXIV CATEGORIES:\n{cat_text.strip() or '(none)'}\n\n"
        f"ABSTRACT:\n{(abstract or '').strip() or '(empty)'}\n"
    )


def assert_prompt_is_paper_only(prompt: str):
    """
    Structural boundary only: user prompt must be the paper-only template.

    Do NOT scan for banned words. Paper titles/abstracts may legitimately contain
    terms such as candidate, organization, employer, OpenAlex, or watchlist.
    Non-paper fields (orgs, people, scores, statuses, OpenAlex metadata) must never
    be passed into build_user_prompt — that is enforced by call signature + tests.
    """
    if not isinstance(prompt, str):
        raise SemanticParseError("prompt must be a string")
    if not prompt.startswith("TITLE:"):
        raise SemanticParseError("prompt must start with TITLE section")
    if "\nARXIV CATEGORIES:\n" not in prompt:
        raise SemanticParseError("prompt missing ARXIV CATEGORIES section")
    if "\nABSTRACT:\n" not in prompt:
        raise SemanticParseError("prompt missing ABSTRACT section")


def parse_structured_assessment(payload: dict) -> tuple[dict[str, float], dict, list[str]]:
    if not isinstance(payload, dict):
        raise SemanticParseError("response payload must be an object")
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for key in DIMENSION_KEYS:
        block = payload.get(key)
        if not isinstance(block, dict):
            raise SemanticParseError(f"missing dimension object: {key}")
        scores[key] = clamp_score(block.get("score"))
        reason = (block.get("reason") or "").strip()
        if not reason:
            raise SemanticParseError(f"empty reason for {key}")
        reasons[key] = reason
    labels_raw = payload.get("industry_labels") or []
    if not isinstance(labels_raw, list):
        raise SemanticParseError("industry_labels must be a list")
    labels = []
    for lab in labels_raw[:5]:
        if lab not in INDUSTRY_LABELS_ALLOWED:
            raise SemanticParseError(f"invalid industry label: {lab}")
        if lab not in labels:
            labels.append(lab)
    return scores, reasons, labels


def select_semantic_sample(
    rows: list[dict],
    *,
    threshold: float,
    n_per_group: int = 25,
    seed: int = SAMPLE_SEED,
) -> list[dict]:
    """
    Reproducible 100-paper sample from SCORED/CANDIDATE rows.

    Groups (non-overlapping):
      top        — highest intrinsic_candidate_score
      low        — lowest intrinsic among remaining
      threshold  — closest to candidate threshold among remaining
      random     — seeded random among remaining
    """
    eligible = []
    for r in rows:
        status = (r.get("status") or "").upper()
        if status in {"REJECTED"}:
            continue
        if status not in {"SCORED", "CANDIDATE"}:
            continue
        eligible.append(dict(r))

    if not eligible:
        return []

    def score_of(r):
        return float(r.get("intrinsic_candidate_score") or 0.0)

    def cid(r):
        return int(r["content_id"] if "content_id" in r else r["id"])

    used: set[int] = set()
    selected: list[dict] = []

    def take(candidates: list[dict], group: str, n: int):
        taken = 0
        for r in candidates:
            i = cid(r)
            if i in used:
                continue
            item = dict(r)
            item["content_id"] = i
            item["sample_group"] = group
            selected.append(item)
            used.add(i)
            taken += 1
            if taken >= n:
                break
        return taken

    by_high = sorted(eligible, key=lambda r: (-score_of(r), cid(r)))
    take(by_high, "top", n_per_group)

    by_low = sorted(eligible, key=lambda r: (score_of(r), cid(r)))
    take(by_low, "low", n_per_group)

    remaining = [r for r in eligible if cid(r) not in used]
    by_thresh = sorted(
        remaining,
        key=lambda r: (abs(score_of(r) - float(threshold)), cid(r)),
    )
    take(by_thresh, "threshold", n_per_group)

    remaining = [r for r in eligible if cid(r) not in used]
    rng = random.Random(int(seed))
    if remaining:
        k = min(n_per_group, len(remaining))
        picks = rng.sample(remaining, k)
        # stable order by content_id after sample for reproducibility of output listing
        for r in sorted(picks, key=cid):
            item = dict(r)
            item["content_id"] = cid(r)
            item["sample_group"] = "random"
            selected.append(item)
            used.add(cid(r))

    return selected


def estimate_input_chars(paper: dict) -> int:
    prompt = build_user_prompt(
        title=paper.get("title") or "",
        abstract=paper.get("abstract") or paper.get("summary") or "",
        categories=paper.get("categories") or paper.get("categories_raw") or [],
    )
    return len(SYSTEM_PROMPT) + len(prompt)


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    if "ratelimit" in name or "rate_limit" in name:
        return True
    return "429" in str(exc)


def _is_retryable_server_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None and int(status) >= 500:
        return True
    name = type(exc).__name__.lower()
    if any(x in name for x in ("timeout", "apiconnection", "internalserver", "serviceunavailable")):
        return True
    return False


def _is_refusal(exc: Exception | None, payload: dict | None = None) -> bool:
    if exc is not None:
        name = type(exc).__name__.lower()
        if "refus" in name or "contentfilter" in name or "moderation" in name:
            return True
    if isinstance(payload, dict) and payload.get("refusal"):
        return True
    return False


def _usage_tokens_chat(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    return int(inp or 0), int(out or 0)


def _call_openrouter_chat_completion(client, *, model: str, user_prompt: str):
    """OpenRouter chat/completions + json_schema structured output."""
    # OpenAI SDK does not accept `reasoning=` on chat.completions.create;
    # OpenRouter expects it in the request body via extra_body.
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"reasoning": {"effort": SEMANTIC_REASONING_EFFORT}},
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_paper_assessment",
                "strict": True,
                "schema": SEMANTIC_RESPONSE_SCHEMA,
            },
        },
    )


def call_semantic_assessment(
    *,
    title: str,
    abstract: str,
    categories,
    client=None,
) -> dict:
    """
    Call OpenRouter semantic assessment with structured JSON output.
    Returns dict with scores, reasons, labels, tokens, cost, response_id.
    """
    require_scoring_enabled()
    require_api_key()

    user_prompt = build_user_prompt(title=title, abstract=abstract, categories=categories)
    assert_prompt_is_paper_only(user_prompt)

    model = resolve_model_name()

    if client is None:
        client = create_llm_client()

    last_exc: Exception | None = None
    for attempt in range(1, SEMANTIC_MAX_RETRIES + 1):
        try:
            if SEMANTIC_REQUEST_SLEEP > 0:
                time.sleep(SEMANTIC_REQUEST_SLEEP)
            response = _call_openrouter_chat_completion(
                client, model=model, user_prompt=user_prompt
            )
            text = (response.choices[0].message.content or "").strip()
            inp, out = _usage_tokens_chat(response)
            response_id = getattr(response, "id", None)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SemanticParseError(f"invalid JSON from model: {exc}") from exc

            if _is_refusal(None, payload if isinstance(payload, dict) else None):
                raise SemanticAPIError("model refused assessment", status="REFUSED", retryable=False)

            scores, reasons, labels = parse_structured_assessment(payload)
            semantic = compute_semantic_score(scores)
            cost = estimate_cost_usd(inp, out)
            return {
                "scores": scores,
                "reasons": reasons,
                "industry_labels": labels,
                "semantic_score": semantic,
                "input_tokens": inp,
                "output_tokens": out,
                "estimated_cost_usd": cost,
                "response_id": response_id,
                "status": "COMPLETED",
                "error_message": None,
                "raw_payload": payload,
                "provider": LLM_PROVIDER,
                "model_name": model,
            }
        except SemanticParseError:
            raise
        except SemanticAPIError:
            raise
        except Exception as exc:
            last_exc = exc
            if _is_refusal(exc):
                raise SemanticAPIError(str(exc), status="REFUSED", retryable=False) from exc
            retryable = _is_rate_limit_error(exc) or _is_retryable_server_error(exc)
            if retryable and attempt < SEMANTIC_MAX_RETRIES:
                wait = min(60.0, (2 ** (attempt - 1)) * 1.0)
                log.warning(
                    "OpenRouter semantic retry attempt=%s/%s wait=%.1fs err=%s",
                    attempt,
                    SEMANTIC_MAX_RETRIES,
                    wait,
                    exc,
                )
                time.sleep(wait)
                continue
            if _is_rate_limit_error(exc):
                raise SemanticAPIError(str(exc), status="RATE_LIMITED", retryable=False) from exc
            raise SemanticAPIError(str(exc), status="ERROR", retryable=False) from exc

    raise SemanticAPIError(str(last_exc or "OpenRouter request failed"), status="ERROR")


# Backward-compatible alias for tests/imports
call_openai_semantic_assessment = call_semantic_assessment


def assessment_exists(conn, content_id: int) -> bool:
    """True only when a COMPLETED assessment already exists (ERROR/RATE_LIMITED are retryable)."""
    model = resolve_model_name()
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_score_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
          AND status = 'COMPLETED'
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, model, SEMANTIC_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_assessment(
    conn,
    *,
    content_id: int,
    sample_group: str | None,
    result: dict,
    force: bool = False,
):
    scores = result.get("scores") or {}
    reasons = result.get("reasons") or {}
    status = result.get("status") or "ERROR"
    provider = result.get("provider") or LLM_PROVIDER
    model_name = result.get("model_name") or resolve_model_name()
    sql = """
        INSERT INTO research_radar.content_score_assessments(
            content_id, assessment_type, provider, model_name, prompt_version,
            sample_group,
            ai_relevance, technical_significance, practical_applicability,
            professional_value, student_learning_value, apparent_novelty,
            explainability, industry_relevance, semantic_score,
            reasons, industry_labels,
            input_tokens, output_tokens, estimated_cost_usd,
            response_id, status, error_message
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s::jsonb, %s::jsonb,
            %s, %s, %s,
            %s, %s, %s
        )
    """
    params = (
        content_id,
        ASSESSMENT_TYPE,
        provider,
        model_name,
        SEMANTIC_PROMPT_VERSION,
        sample_group,
        scores.get("ai_relevance"),
        scores.get("technical_significance"),
        scores.get("practical_applicability"),
        scores.get("professional_value"),
        scores.get("student_learning_value"),
        scores.get("apparent_novelty"),
        scores.get("explainability"),
        scores.get("industry_relevance"),
        result.get("semantic_score"),
        json.dumps(reasons),
        json.dumps(result.get("industry_labels") or []),
        result.get("input_tokens"),
        result.get("output_tokens"),
        result.get("estimated_cost_usd"),
        result.get("response_id"),
        status,
        result.get("error_message"),
    )
    if force:
        sql += """
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            sample_group = EXCLUDED.sample_group,
            ai_relevance = EXCLUDED.ai_relevance,
            technical_significance = EXCLUDED.technical_significance,
            practical_applicability = EXCLUDED.practical_applicability,
            professional_value = EXCLUDED.professional_value,
            student_learning_value = EXCLUDED.student_learning_value,
            apparent_novelty = EXCLUDED.apparent_novelty,
            explainability = EXCLUDED.explainability,
            industry_relevance = EXCLUDED.industry_relevance,
            semantic_score = EXCLUDED.semantic_score,
            reasons = EXCLUDED.reasons,
            industry_labels = EXCLUDED.industry_labels,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            response_id = EXCLUDED.response_id,
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            created_at = NOW()
        """
    else:
        # Retryable statuses (ERROR / RATE_LIMITED / …) may be overwritten;
        # COMPLETED rows are left untouched unless --force.
        sql += """
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            sample_group = EXCLUDED.sample_group,
            ai_relevance = EXCLUDED.ai_relevance,
            technical_significance = EXCLUDED.technical_significance,
            practical_applicability = EXCLUDED.practical_applicability,
            professional_value = EXCLUDED.professional_value,
            student_learning_value = EXCLUDED.student_learning_value,
            apparent_novelty = EXCLUDED.apparent_novelty,
            explainability = EXCLUDED.explainability,
            industry_relevance = EXCLUDED.industry_relevance,
            semantic_score = EXCLUDED.semantic_score,
            reasons = EXCLUDED.reasons,
            industry_labels = EXCLUDED.industry_labels,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            response_id = EXCLUDED.response_id,
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            created_at = NOW()
        WHERE research_radar.content_score_assessments.status <> 'COMPLETED'
        """
    conn.execute(sql, params)


def load_relevant_scored_papers(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            ci.id AS content_id,
            ci.title,
            ci.summary,
            ci.status,
            ci.categories_raw,
            cs.intrinsic_candidate_score,
            cs.ai_relevance AS det_ai_relevance,
            cs.technical_significance AS det_technical_significance,
            cs.practical_applicability AS det_practical_applicability,
            cs.notable_org_signal,
            COALESCE(pm.abstract, ci.summary, '') AS abstract,
            COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories
        FROM research_radar.content_items ci
        JOIN research_radar.content_scores cs ON cs.content_id = ci.id
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.status IN ('SCORED', 'CANDIDATE')
        ORDER BY ci.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def stage_semantic_score(
    conn,
    run_id,
    *,
    sample: int | None = 100,
    full: bool = False,
    dry_run: bool = False,
    force: bool = False,
    threshold: float | None = None,
    client=None,
):
    """
    Assessment-only GPT semantic scoring.
    Does NOT modify content_items.status or content_scores.
    """
    from research_radar.pipeline import MIN_CANDIDATE_SCORE, bump, event

    if threshold is None:
        threshold = float(MIN_CANDIDATE_SCORE)

    if not dry_run:
        require_scoring_enabled()
        require_api_key()

    papers = load_relevant_scored_papers(conn)
    if full:
        selected = []
        for r in papers:
            item = dict(r)
            item["sample_group"] = "full"
            selected.append(item)
    else:
        n = int(sample or 100)
        if n % 4 != 0:
            raise ValueError("--sample must be divisible by 4 (equal group sizes)")
        n_per = n // 4
        selected = select_semantic_sample(
            papers,
            threshold=threshold,
            n_per_group=n_per,
            seed=SAMPLE_SEED,
        )

    stats = SemanticRunStats(requested=len(selected))
    for g in ("top", "threshold", "low", "random", "full"):
        stats.sample_groups[g] = sum(1 for r in selected if r.get("sample_group") == g)

    if dry_run:
        chars = sum(estimate_input_chars(p) for p in selected)
        # rough token estimate ~4 chars/token for planning only
        approx_tokens = chars // 4
        approx_cost = estimate_cost_usd(approx_tokens, approx_tokens // 3)
        summary = {
            "dry_run": True,
            "selected": len(selected),
            "sample_groups": stats.sample_groups,
            "approx_input_chars": chars,
            "approx_input_tokens": approx_tokens,
            "approx_cost_usd_rough": approx_cost,
            "model": resolve_model_name(),
            "provider": LLM_PROVIDER,
            "prompt_version": SEMANTIC_PROMPT_VERSION,
            "seed": SAMPLE_SEED,
            "openai_calls": 0,
            "assessment_writes": 0,
        }
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"semantic_score_dry_run": summary}), run_id),
        )
        log.info("Semantic-score DRY RUN: %s", json.dumps(summary))
        print("\nSEMANTIC SCORE DRY RUN")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return stats

    workers = max(1, SEMANTIC_SCORING_WORKERS)

    def _process_one(paper: dict) -> tuple[str, dict]:
        content_id = int(paper["content_id"])
        sample_group = paper.get("sample_group")
        # Read-only existence check uses a short-lived connection in worker path
        from research_radar.pipeline import connect as _connect

        with _connect() as wconn:
            if not force and assessment_exists(wconn, content_id):
                return ("skipped", {"content_id": content_id, "sample_group": sample_group})

            try:
                result = call_semantic_assessment(
                    title=paper.get("title") or "",
                    abstract=paper.get("abstract") or paper.get("summary") or "",
                    categories=paper.get("categories") or [],
                    client=client,
                )
                result["sample_group"] = sample_group
                upsert_assessment(
                    wconn,
                    content_id=content_id,
                    sample_group=sample_group,
                    result=result,
                    force=force,
                )
                event(
                    wconn,
                    run_id,
                    content_id,
                    "semantic_score",
                    "completed",
                    True,
                    {
                        "semantic_score": result.get("semantic_score"),
                        "sample_group": sample_group,
                        "response_id": result.get("response_id"),
                    },
                )
                wconn.commit()
                return ("completed", {"content_id": content_id, **result})
            except SemanticAPIError as exc:
                err_result = {
                    "scores": {},
                    "reasons": {},
                    "industry_labels": [],
                    "semantic_score": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "response_id": None,
                    "status": exc.status,
                    "error_message": str(exc)[:1000],
                }
                try:
                    upsert_assessment(
                        wconn,
                        content_id=content_id,
                        sample_group=sample_group,
                        result=err_result,
                        force=force,
                    )
                    event(
                        wconn,
                        run_id,
                        content_id,
                        "semantic_score",
                        exc.status.lower(),
                        False,
                        {"sample_group": sample_group},
                        str(exc),
                    )
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return (exc.status.lower(), {"content_id": content_id, "error": str(exc)})
            except Exception as exc:
                wconn.rollback()
                try:
                    err_result = {
                        "scores": {},
                        "reasons": {},
                        "industry_labels": [],
                        "semantic_score": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "estimated_cost_usd": 0.0,
                        "response_id": None,
                        "status": "ERROR",
                        "error_message": str(exc)[:1000],
                    }
                    upsert_assessment(
                        wconn,
                        content_id=content_id,
                        sample_group=sample_group,
                        result=err_result,
                        force=True,
                    )
                    bump(wconn, run_id, "errors")
                    wconn.commit()
                except Exception:
                    wconn.rollback()
                return ("failed", {"content_id": content_id, "error": str(exc)})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, p) for p in selected]
        done = 0
        for fut in as_completed(futures):
            kind, payload = fut.result()
            done += 1
            if kind == "skipped":
                stats.skipped_existing += 1
            elif kind == "completed":
                stats.completed += 1
                stats.input_tokens += int(payload.get("input_tokens") or 0)
                stats.output_tokens += int(payload.get("output_tokens") or 0)
                stats.estimated_cost_usd += float(payload.get("estimated_cost_usd") or 0)
                log.info(
                    "Semantic COMPLETED id=%s score=%.2f group=%s",
                    payload.get("content_id"),
                    float(payload.get("semantic_score") or 0),
                    payload.get("sample_group"),
                )
            elif kind == "rate_limited":
                stats.rate_limited += 1
                stats.failed += 1
            elif kind == "refused":
                stats.refused += 1
                stats.failed += 1
            else:
                stats.failed += 1
            if done % 10 == 0 or done == len(selected):
                log.info(
                    "Semantic progress %d/%d completed=%d failed=%d skipped=%d",
                    done,
                    len(selected),
                    stats.completed,
                    stats.failed,
                    stats.skipped_existing,
                )

    summary = stats.to_dict()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"semantic_score_stats": summary}), run_id),
    )
    log.info("Semantic score stats: %s", json.dumps(summary))
    print("\nSEMANTIC SCORE SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return stats


def print_semantic_compare(conn, *, limit: int = 20, prompt_version: str | None = None):
    prompt_version = prompt_version or SEMANTIC_PROMPT_VERSION
    provider = LLM_PROVIDER
    model = resolve_model_name()
    rows = conn.execute(
        """
        SELECT
            ci.id AS content_id,
            ci.title,
            a.sample_group,
            cs.intrinsic_candidate_score AS old_intrinsic,
            a.semantic_score AS gpt_semantic,
            cs.ai_relevance AS old_ai_relevance,
            a.ai_relevance AS gpt_ai_relevance,
            cs.technical_significance AS old_technical,
            a.technical_significance AS gpt_technical,
            cs.practical_applicability AS old_practical,
            a.practical_applicability AS gpt_practical,
            a.apparent_novelty AS gpt_apparent_novelty,
            a.industry_relevance AS gpt_industry_relevance,
            cs.notable_org_signal AS org_signal,
            a.status AS assessment_status
        FROM research_radar.content_score_assessments a
        JOIN research_radar.content_items ci ON ci.id = a.content_id
        JOIN research_radar.content_scores cs ON cs.content_id = ci.id
        WHERE a.provider = %s
          AND a.model_name = %s
          AND a.prompt_version = %s
          AND a.status = 'COMPLETED'
          AND a.semantic_score IS NOT NULL
        ORDER BY a.semantic_score DESC NULLS LAST
        """,
        (provider, model, prompt_version),
    ).fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        print("No completed semantic assessments found for comparison.")
        return

    by_old = sorted(rows, key=lambda r: (-float(r["old_intrinsic"] or 0), r["content_id"]))
    by_gpt = sorted(rows, key=lambda r: (-float(r["gpt_semantic"] or 0), r["content_id"]))
    old_rank = {r["content_id"]: i + 1 for i, r in enumerate(by_old)}
    gpt_rank = {r["content_id"]: i + 1 for i, r in enumerate(by_gpt)}

    enriched = []
    for r in rows:
        cid = r["content_id"]
        item = dict(r)
        item["old_rank"] = old_rank[cid]
        item["gpt_rank"] = gpt_rank[cid]
        item["rank_delta"] = old_rank[cid] - gpt_rank[cid]  # positive = GPT moved up
        enriched.append(item)

    print(f"\nSEMANTIC COMPARE (n={len(enriched)} prompt_version={prompt_version})")
    print(
        "content_id | group | old_intr | gpt_sem | old_rk | gpt_rk | delta | "
        "old_ai | gpt_ai | old_tech | gpt_tech | old_prac | gpt_prac | "
        "gpt_nov | gpt_ind | org_sig | title"
    )
    for r in sorted(enriched, key=lambda x: x["content_id"]):
        print(
            f"{r['content_id']} | {r.get('sample_group')} | "
            f"{float(r['old_intrinsic'] or 0):.2f} | {float(r['gpt_semantic'] or 0):.2f} | "
            f"{r['old_rank']} | {r['gpt_rank']} | {r['rank_delta']:+d} | "
            f"{float(r['old_ai_relevance'] or 0):.1f} | {float(r['gpt_ai_relevance'] or 0):.1f} | "
            f"{float(r['old_technical'] or 0):.1f} | {float(r['gpt_technical'] or 0):.1f} | "
            f"{float(r['old_practical'] or 0):.1f} | {float(r['gpt_practical'] or 0):.1f} | "
            f"{float(r['gpt_apparent_novelty'] or 0):.1f} | {float(r['gpt_industry_relevance'] or 0):.1f} | "
            f"{float(r['org_signal'] or 0):.1f} | {(r.get('title') or '')[:60]}"
        )

    print(f"\nTOP {limit} BY DETERMINISTIC INTRINSIC")
    for i, r in enumerate(by_old[:limit], 1):
        print(
            f"{i}. [{r.get('sample_group')}] id={r['content_id']} "
            f"old={float(r['old_intrinsic'] or 0):.2f} gpt={float(r['gpt_semantic'] or 0):.2f} "
            f"{(r.get('title') or '')[:70]}"
        )

    print(f"\nTOP {limit} BY GPT SEMANTIC")
    for i, r in enumerate(by_gpt[:limit], 1):
        print(
            f"{i}. [{r.get('sample_group')}] id={r['content_id']} "
            f"gpt={float(r['gpt_semantic'] or 0):.2f} old={float(r['old_intrinsic'] or 0):.2f} "
            f"{(r.get('title') or '')[:70]}"
        )

    moved_up = sorted(enriched, key=lambda r: (-r["rank_delta"], r["content_id"]))[:limit]
    moved_down = sorted(enriched, key=lambda r: (r["rank_delta"], r["content_id"]))[:limit]

    print(f"\nGPT MOVED UP MOST (rank_delta = old_rank - gpt_rank)")
    for r in moved_up:
        print(
            f"  id={r['content_id']} delta={r['rank_delta']:+d} "
            f"old_rk={r['old_rank']} gpt_rk={r['gpt_rank']} {(r.get('title') or '')[:60]}"
        )

    print(f"\nGPT MOVED DOWN MOST")
    for r in moved_down:
        print(
            f"  id={r['content_id']} delta={r['rank_delta']:+d} "
            f"old_rk={r['old_rank']} gpt_rk={r['gpt_rank']} {(r.get('title') or '')[:60]}"
        )


# ===========================================================================
# Scoring v2 — quality scorer (Call A). Title + abstract + categories ONLY.
#
# This is a separate model call from independence.py (Call B), which is the
# only module that ever sees affiliation_text. That separation is a hard
# architectural guarantee, not a prompt instruction: this section must never
# import from independence.py or accept an affiliation/author/organisation
# parameter. Do not merge the two calls to save cost.
#
# v1 (above) is left untouched — its assessment rows stay under
# prompt_version="research-semantic-v1" and are never deleted, overwritten or
# migrated. stage_semantic_score (v1) is no longer wired into the pipeline;
# stage_semantic_score_v2 is what `--stage semantic-score` now runs.
# ===========================================================================

QUALITY_PROMPT_VERSION = (
    os.getenv("QUALITY_PROMPT_VERSION", "research-semantic-v3").strip() or "research-semantic-v3"
)
QUALITY_BATCH_SIZE = int(os.getenv("QUALITY_BATCH_SIZE", "5"))
QUALITY_REASONING_EFFORT = os.getenv("QUALITY_REASONING_EFFORT", "medium").strip() or "medium"
QUALITY_MAX_RETRIES = int(os.getenv("QUALITY_MAX_RETRIES", "3"))
QUALITY_REQUEST_SLEEP = float(os.getenv("QUALITY_REQUEST_SLEEP", "0.2"))
QUALITY_DEFAULT_SAMPLE = int(os.getenv("QUALITY_DEFAULT_SAMPLE", "100"))

# Fraction of screened papers (by mean of the four screen dimensions, after
# excluding ai_relevance <= 3) that proceed to pass 2 / independence. Set from
# a measured recall test (scripts/validate_scoring.py --test tier-recall),
# never chosen from a cost target — see that test's docstring.
GATE_PERCENTILE = float(os.getenv("GATE_PERCENTILE", "15"))

# Pass 1 (screen) config — a cheap, non-reasoning model; see the SCREEN_*
# section near the bottom of this file for the prompt/schema/parse/call/stage
# functions. Declared here (not there) so load_gated_quality_candidates below
# can reference SCREEN_PROMPT_VERSION without a forward-reference.
SCREEN_MODEL = os.getenv("SCREEN_MODEL", "anthropic/claude-haiku-4.5").strip() or "anthropic/claude-haiku-4.5"
SCREEN_PROMPT_VERSION = (
    os.getenv("SCREEN_PROMPT_VERSION", "research-screen-v3").strip() or "research-screen-v3"
)
SCREEN_BATCH_SIZE = int(os.getenv("SCREEN_BATCH_SIZE", "15"))
SCREEN_MAX_RETRIES = int(os.getenv("SCREEN_MAX_RETRIES", "3"))
SCREEN_REQUEST_SLEEP = float(os.getenv("SCREEN_REQUEST_SLEEP", "0.2"))

QUALITY_NUMERIC_FIELDS = [
    "ai_relevance",
    "technical_significance",
    "apparent_novelty",
    "practical_applicability",
    "professional_value",
    "learning_value",
    "evidence_strength",
    "newsletter_fit",
    "confidence",
]
QUALITY_TEXT_FIELDS = ["so_what", "reason_not_higher"]

QUALITY_SYSTEM_PROMPT = """You score AI research papers for TheNeural, a research intelligence system
that also feeds a weekly AI newsletter read by working technology and product
professionals, and by students entering the field.

You are given each paper's title, arXiv categories and abstract. You are NOT
told who wrote it or where they work. Do not guess, and do not let a familiar
research style, dataset or terminology lead you to infer a laboratory.

SEPARATING CONTRIBUTION FROM EVIDENCE

Score the apparent contribution on each quality dimension separately from how
well that contribution is evidenced.

Do NOT reduce technical_significance, apparent_novelty,
practical_applicability, professional_value or learning_value merely because
the abstract provides weak evidence. Score how important the claimed
contribution would be if it holds.

Put ALL uncertainty about whether it holds into evidence_strength and
confidence.

Never invent a contribution, result or number that is not stated. If the
abstract does not say what was achieved, the contribution you are scoring is
whatever it actually claims, which may be very little.

SCORING PRINCIPLES

Most papers are ordinary. A typical arXiv submission is competent,
incremental work advancing a narrow question. That is a 5 or 6. Expect most
papers to land there, but this is a description of the field, not a quota -
if a batch genuinely contains three strong papers, score three strong papers.

Do not reward familiar terminology. "Transformer", "state-of-the-art",
"foundation model", "agentic" and similar terms tell you nothing about
quality.

Treat novelty as APPARENT novelty from this abstract alone. You cannot know
the full prior literature.

Results reported by the authors are internally reported evidence.
Independent replication is stronger, but self-reported results are still
findings - do not dismiss them. Reflect the difference in evidence_strength.

Judge each paper on its own merits against the absolute scale below. Do NOT
score papers relative to the others in this batch. A batch of weak papers
does not make the least weak one strong.

GATE: Score ai_relevance first, then score every other dimension honestly on its own
merits. Do NOT lower other dimensions because ai_relevance is low - an off-topic paper
can still be rigorous, and the system drops off-topic papers separately. Each dimension
must remain a real measurement.

SCALE

All scores are 0.0 to 10.0 in 0.5 increments. Valid values: 0.0, 0.5, 1.0,
1.5 ... 9.5, 10.0. Do not return other values.

0-2     Weak. Trivial, unsupported, or off-topic.
3-4     Below average. Minor contribution.
5-6     Ordinary. Competent incremental work. MOST PAPERS BELONG HERE.
7-8     Strong. A clear contribution that matters.
9       Exceptional. Changes how a practitioner approaches the problem.
10      Reserved. Requires an unusually strong claim clearly stated in this
        abstract.

DIMENSIONS

ai_relevance            Is this AI, ML, or their direct application? (GATE)
technical_significance  If the claim holds, does it change what is
                        technically possible or understood?
apparent_novelty        Is the core idea new, or a known idea applied again?
practical_applicability If it holds, could a practitioner use this within a
                        year?
professional_value      Would it change a technical or product decision?
learning_value          Is it worth reading to understand where the field is
                        going?
evidence_strength       Does the abstract REPORT results, or only claim them?
                        Numbers, named baselines, named datasets, ablations
                        and stated limitations raise this. Vague superlatives
                        lower it. This is the ONLY dimension where weak
                        evidence should lower the number.

NEWSLETTER FIT - a SEPARATE question

newsletter_fit asks something different from research quality: could we tell
a busy professional something useful about this in one line, and would they
care? A rigorous incremental paper can be excellent research and a poor
newsletter item. A simple result with a striking implication can be the
reverse. Score independently of the dimensions above.

High: a concrete result a reader can act on or repeat, a finding that
contradicts a common assumption, something that changes a build-or-buy
decision.
Low: incremental benchmark gains, work meaningful only inside a narrow
subfield, results needing heavy background to understand.

ALSO RETURN PER PAPER

so_what            One line, max 20 words. What this means for a working
                   professional. Not a summary of the abstract.
reason_not_higher  The single strongest reason this is not scored higher.
                   ALWAYS give one, even for a paper you scored 9.
confidence         0-10 in 0.5 increments. How sure you are given only what
                   this abstract actually contains.

Return ONLY a JSON object with a single key "papers", whose value is an array
with one object per paper, in the order supplied. No prose, no markdown
fences.
"""

# Built from QUALITY_NUMERIC_FIELDS/QUALITY_TEXT_FIELDS so the schema can never
# drift out of sync with the parser's own field list.
QUALITY_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "integer"},
        **{k: {"type": "number"} for k in QUALITY_NUMERIC_FIELDS},
        **{k: {"type": "string"} for k in QUALITY_TEXT_FIELDS},
    },
    "required": ["paper_id", *QUALITY_NUMERIC_FIELDS, *QUALITY_TEXT_FIELDS],
}

QUALITY_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "papers": {"type": "array", "items": QUALITY_ITEM_SCHEMA},
    },
    "required": ["papers"],
}


class QualityParseError(ValueError):
    pass


def normalize_half_point(value: Any) -> tuple[float, bool]:
    """Clamp to 0-10 and round to nearest 0.5. Returns (value, was_rounded)."""
    x = float(value)
    x = max(0.0, min(10.0, x))
    rounded = round(x * 2.0) / 2.0
    was_rounded = abs(rounded - x) > 1e-9
    return rounded, was_rounded


def build_quality_paper_block(paper: dict) -> str:
    """PAPER {id} / TITLE / CATEGORIES / ABSTRACT.

    Reads ONLY title, categories, abstract from `paper` — never affiliation,
    author or organisation keys, even if present on the dict the caller
    passed in (e.g. a row that also carries paper_metadata.affiliation_text
    for the independence classifier's own use).
    """
    categories = paper.get("categories") or paper.get("categories_raw") or []
    if isinstance(categories, str):
        cat_text = categories
    elif isinstance(categories, (list, tuple)):
        cat_text = ", ".join(str(c) for c in categories if c)
    else:
        cat_text = ""
    content_id = paper.get("content_id", paper.get("id"))
    return (
        f"PAPER {content_id}\n"
        f"TITLE: {(paper.get('title') or '').strip()}\n"
        f"CATEGORIES: {cat_text.strip() or '(none)'}\n"
        f"ABSTRACT: {(paper.get('abstract') or paper.get('summary') or '').strip() or '(empty)'}\n"
    )


def build_quality_batch_user_prompt(papers: list[dict]) -> str:
    return "\n\n".join(build_quality_paper_block(p) for p in papers)


def assert_quality_prompt_is_paper_only(prompt: str):
    """
    Structural boundary only (same philosophy as v1's assert_prompt_is_paper_only):
    do NOT scan for banned words, since paper text may legitimately discuss
    affiliation networks, authorship, or organisations. What is enforced is
    that the builder never emits an AFFILIATIONS/AUTHORS/ORGANISATION section —
    those keys are never read by build_quality_paper_block in the first place.
    """
    if not isinstance(prompt, str):
        raise QualityParseError("prompt must be a string")
    if "AFFILIATIONS:" in prompt:
        raise QualityParseError("quality prompt must not contain an AFFILIATIONS section")
    if "AUTHORS:" in prompt or "ORGANISATION:" in prompt or "ORGANIZATION:" in prompt:
        raise QualityParseError("quality prompt must not contain an author/organisation section")
    if "TITLE:" not in prompt or "CATEGORIES:" not in prompt or "ABSTRACT:" not in prompt:
        raise QualityParseError("prompt missing required paper-only sections")


def parse_quality_batch(text: str, expected_ids: set[int]) -> tuple[dict[int, dict], int]:
    """Parse the model's {"papers": [...]} object.

    Returns ({content_id: {...}}, rounding_warning_count). strip_json_fences
    handles accidental markdown-fence wrapping; the object shape itself is
    enforced API-side by QUALITY_RESPONSE_SCHEMA (strict json_schema), so this
    is a defensive re-check, not the primary validation.
    """
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualityParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise QualityParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise QualityParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    rounding_warnings = 0
    for item in items:
        if not isinstance(item, dict):
            raise QualityParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise QualityParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc

        dims: dict[str, Any] = {}
        for key in QUALITY_NUMERIC_FIELDS:
            if key not in item:
                raise QualityParseError(f"paper {pid} missing dimension: {key}")
            try:
                raw_val = float(item[key])
            except (TypeError, ValueError) as exc:
                raise QualityParseError(f"paper {pid} non-numeric {key}: {item[key]!r}") from exc
            if raw_val < 0.0 or raw_val > 10.0:
                raise QualityParseError(f"paper {pid} {key} out of range 0-10: {raw_val}")
            rounded, was_rounded = normalize_half_point(raw_val)
            if was_rounded:
                rounding_warnings += 1
                log.warning(
                    "Quality score not a 0.5 increment paper=%s dim=%s raw=%s rounded=%s",
                    pid,
                    key,
                    raw_val,
                    rounded,
                )
            dims[key] = rounded

        for key in QUALITY_TEXT_FIELDS:
            val = (item.get(key) or "").strip()
            if not val:
                raise QualityParseError(f"paper {pid} missing {key}")
            dims[key] = val

        if pid in expected_ids:
            out[pid] = dims
    return out, rounding_warnings


def call_quality_batch(papers: list[dict], *, client=None, on_rate_limited=None) -> dict:
    """One OpenRouter call for a batch of papers (title+categories+abstract only)."""
    if client is None:
        client = create_llm_client()
    model = resolve_model_name()
    user_prompt = build_quality_batch_user_prompt(papers)
    assert_quality_prompt_is_paper_only(user_prompt)
    expected_ids = {int(p.get("content_id", p.get("id"))) for p in papers}

    result = call_chat_completion(
        client,
        model=model,
        system_prompt=QUALITY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=QUALITY_REASONING_EFFORT,
        temperature=0.2,
        max_retries=1,  # outer stage loop owns the batch-level retry/backoff
        request_sleep=QUALITY_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "quality_assessment",
                "strict": True,
                "schema": QUALITY_RESPONSE_SCHEMA,
            },
        },
        on_rate_limited=on_rate_limited,
    )
    parsed, rounding_warnings = parse_quality_batch(result["text"], expected_ids)
    return {
        "results": parsed,
        "rounding_warnings": rounding_warnings,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
        "estimated_cost_usd": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
    }


def call_quality_batch_with_retry(
    papers: list[dict],
    *,
    client=None,
    max_retries: int | None = None,
    on_rate_limited=None,
    batch_tag: str = "",
) -> dict:
    max_retries = max_retries if max_retries is not None else QUALITY_MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_quality_batch(papers, client=client, on_rate_limited=on_rate_limited)
        except (LLMBatchError, QualityParseError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(60.0, 2 ** (attempt - 1))
                log.warning(
                    "[batch %s] Quality batch retry attempt=%s/%s size=%s err=%s",
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
class QualityRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped_existing: int = 0
    rounding_warnings: int = 0
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
            "skipped_existing": self.skipped_existing,
            "rounding_warnings": self.rounding_warnings,
            "batches": self.batches,
            "calls": self.calls,
            "batch_size": QUALITY_BATCH_SIZE,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "model": resolve_model_name(),
            "provider": LLM_PROVIDER,
            "prompt_version": QUALITY_PROMPT_VERSION,
            "reasoning_effort": QUALITY_REASONING_EFFORT,
        }


def quality_assessment_exists(conn, content_id: int) -> bool:
    model = resolve_model_name()
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_score_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
          AND status = 'COMPLETED'
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, model, QUALITY_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_quality_assessment(
    conn,
    *,
    content_id: int,
    sample_group: str,
    batch_id,
    batch_size: int,
    batch_position: int,
    result: dict,
    scoring_tier: str = "full",
):
    sql = """
        INSERT INTO research_radar.content_score_assessments(
            content_id, assessment_type, provider, model_name, prompt_version,
            sample_group, scoring_tier,
            ai_relevance, technical_significance, apparent_novelty,
            practical_applicability, professional_value,
            learning_value, evidence_strength, newsletter_fit,
            so_what, reason_not_higher, confidence,
            batch_id, batch_size, batch_position,
            reasons, industry_labels,
            input_tokens, output_tokens, estimated_cost_usd,
            response_id, status, error_message
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s::jsonb, %s::jsonb,
            %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            sample_group = EXCLUDED.sample_group,
            scoring_tier = EXCLUDED.scoring_tier,
            ai_relevance = EXCLUDED.ai_relevance,
            technical_significance = EXCLUDED.technical_significance,
            apparent_novelty = EXCLUDED.apparent_novelty,
            practical_applicability = EXCLUDED.practical_applicability,
            professional_value = EXCLUDED.professional_value,
            learning_value = EXCLUDED.learning_value,
            evidence_strength = EXCLUDED.evidence_strength,
            newsletter_fit = EXCLUDED.newsletter_fit,
            so_what = EXCLUDED.so_what,
            reason_not_higher = EXCLUDED.reason_not_higher,
            confidence = EXCLUDED.confidence,
            batch_id = EXCLUDED.batch_id,
            batch_size = EXCLUDED.batch_size,
            batch_position = EXCLUDED.batch_position,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            response_id = EXCLUDED.response_id,
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            created_at = NOW()
    """
    params = (
        content_id,
        ASSESSMENT_TYPE,
        LLM_PROVIDER,
        resolve_model_name(),
        QUALITY_PROMPT_VERSION,
        sample_group,
        scoring_tier,
        result.get("ai_relevance"),
        result.get("technical_significance"),
        result.get("apparent_novelty"),
        result.get("practical_applicability"),
        result.get("professional_value"),
        result.get("learning_value"),
        result.get("evidence_strength"),
        result.get("newsletter_fit"),
        result.get("so_what"),
        result.get("reason_not_higher"),
        result.get("confidence"),
        batch_id,
        batch_size,
        batch_position,
        json.dumps({}),
        json.dumps([]),
        result.get("input_tokens"),
        result.get("output_tokens"),
        result.get("estimated_cost_usd"),
        result.get("response_id"),
        result.get("status") or "COMPLETED",
        result.get("error_message"),
    )
    conn.execute(sql, params)


def load_quality_candidates(conn, limit: int | None = None) -> list[dict]:
    """
    ENTITY_RESOLVED, SCORED or CANDIDATE papers, title+categories+abstract only.
    SCORED/CANDIDATE are included alongside ENTITY_RESOLVED because papers
    ingested before the deterministic `score` stage was removed from `all`
    already advanced past ENTITY_RESOLVED; both populations are equally
    eligible for v2 (correction to the original brief, which said
    ENTITY_RESOLVED *instead of* SCORED/CANDIDATE — that would have made this
    stage select nothing for the entire pre-existing corpus).
    """
    rows = conn.execute(
        """
        SELECT
            ci.id AS content_id,
            ci.title,
            COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
            COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.status IN ('ENTITY_RESOLVED', 'SCORED', 'CANDIDATE')
        ORDER BY ci.id
        LIMIT %s
        """,
        (limit or 10_000,),
    ).fetchall()
    return [dict(r) for r in rows]


# Code gate: screen rows with ai_relevance <= 3 never reach pass 2 (prompt no longer
# collapses other dimensions when ai_relevance is low).
SCREEN_AI_RELEVANCE_FLOOR = 3.0
# ai_relevance gates only; ranking mean matches radar-v2 quality mean (excludes ai_relevance).
SCREEN_RANKING_FIELDS = ["technical_significance", "apparent_novelty", "evidence_strength"]


def select_gated_content_ids(conn, *, gate_percentile: float | None = None) -> list[int]:
    """
    Rank COMPLETED screen assessments by the mean of technical_significance,
    apparent_novelty and evidence_strength. Drop ai_relevance <= 3 outright, and
    return the content_ids in the top `gate_percentile` percent. Pure Python
    ranking over a DB read — same style as final_score.py's org/person boosts.
    Makes no API calls.
    """
    gate_percentile = GATE_PERCENTILE if gate_percentile is None else gate_percentile
    rows = conn.execute(
        """
        SELECT content_id, ai_relevance, technical_significance, apparent_novelty, evidence_strength
        FROM research_radar.content_score_assessments
        WHERE prompt_version = %s AND scoring_tier = 'screen' AND status = 'COMPLETED'
        """,
        (SCREEN_PROMPT_VERSION,),
    ).fetchall()

    ranked: list[tuple[int, float]] = []
    for r in rows:
        ai_rel = r.get("ai_relevance")
        if ai_rel is None or float(ai_rel) <= SCREEN_AI_RELEVANCE_FLOOR:
            continue
        dims = [r.get(k) for k in SCREEN_RANKING_FIELDS]
        if any(d is None for d in dims):
            continue
        mean = sum(float(d) for d in dims) / len(dims)
        ranked.append((int(r["content_id"]), mean))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    k = math.ceil(len(ranked) * gate_percentile / 100.0)
    return [cid for cid, _ in ranked[:k]]


def load_gated_quality_candidates(
    conn, *, gate_percentile: float | None = None, limit: int | None = None
) -> list[dict]:
    """Papers that passed the screen gate — the pool pass 2 actually scores."""
    gated_ids = select_gated_content_ids(conn, gate_percentile=gate_percentile)
    if not gated_ids:
        return []
    rows = conn.execute(
        """
        SELECT
            ci.id AS content_id,
            ci.title,
            COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
            COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.id = ANY(%s)
        ORDER BY ci.id
        LIMIT %s
        """,
        (gated_ids, limit or 10_000),
    ).fetchall()
    return [dict(r) for r in rows]


def estimate_quality_prompt_tokens(papers: list[dict]) -> int:
    prompt = QUALITY_SYSTEM_PROMPT + build_quality_batch_user_prompt(papers)
    return max(1, len(prompt) // 4)


def stage_semantic_score_v2(
    conn,
    run_id,
    *,
    sample: int | None = None,
    full: bool = False,
    dry_run: bool = False,
    force: bool = False,
    gate_percentile: float | None = None,
    client=None,
):
    """
    Pass 2 — full quality scoring. Batch 5, randomly composed, random order
    within batch. Runs ONLY on papers that passed the screen gate (top
    `gate_percentile` percent of pass-1 scores, ai_relevance > 3) — see
    load_gated_quality_candidates. Writes scoring_tier='full'.
    """
    gate_percentile = GATE_PERCENTILE if gate_percentile is None else gate_percentile

    if not dry_run:
        require_scoring_enabled()
        require_api_key()

    candidates = load_gated_quality_candidates(conn, gate_percentile=gate_percentile)
    if not force:
        candidates = [c for c in candidates if not quality_assessment_exists(conn, c["content_id"])]

    if full:
        selected = candidates
        sample_group = "full"
    else:
        n = int(sample or QUALITY_DEFAULT_SAMPLE)
        selected = candidates if len(candidates) <= n else random.sample(candidates, n)
        sample_group = "random"

    stats = QualityRunStats(requested=len(selected))
    batches = random_batches(selected, QUALITY_BATCH_SIZE)
    stats.batches = len(batches)

    if dry_run:
        est_in = sum(estimate_quality_prompt_tokens(b) for b in batches)
        est_out = 220 * len(selected)  # structured-JSON-per-paper heuristic
        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.calls = len(batches)
        summary = stats.to_dict()
        summary["gate_percentile"] = gate_percentile
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"semantic_score_v2_dry_run": summary}), run_id),
        )
        log.info("Quality v2 DRY RUN: %s", json.dumps(summary))
        print("\nSEMANTIC SCORE V2 (PASS 2) DRY RUN")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(cost_summary_line("Pass 2:", stats.requested, stats.calls, stats.estimated_cost_usd))
        return stats

    if client is None:
        client = create_llm_client()

    gate = AdaptiveConcurrencyGate(SCORING_CONCURRENCY)

    def _process_batch(batch: list[dict]) -> list[tuple[str, dict, int]]:
        from research_radar.pipeline import bump, connect as _connect, event

        batch_id = uuid.uuid4()
        batch_tag = str(batch_id)[:8]
        batch_size = len(batch)
        positioned = {int(p["content_id"]): (i + 1, p) for i, p in enumerate(batch)}
        outcomes: list[tuple[str, dict, int]] = []

        gate.acquire()
        try:
            try:
                result = call_quality_batch_with_retry(
                    batch, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                )
                stats.calls += 1
            except (LLMBatchError, QualityParseError) as exc:
                log.warning("[batch %s] Quality batch failed entirely; falling back to individual calls: %s", batch_tag, exc)
                result = None

            parsed: dict[int, dict] = {}
            rounding_warnings = 0
            if result is not None:
                parsed = result["results"]
                rounding_warnings = result["rounding_warnings"]
                stats.input_tokens += int(result["input_tokens"] or 0)
                stats.output_tokens += int(result["output_tokens"] or 0)
                stats.estimated_cost_usd += float(result["estimated_cost_usd"] or 0)

            missing_ids = set(positioned.keys()) - set(parsed.keys())
            for pid in missing_ids:
                pos, paper = positioned[pid]
                try:
                    single = call_quality_batch_with_retry(
                        [paper], client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                    )
                    stats.calls += 1
                    r = single["results"].get(pid)
                    if r is None:
                        raise QualityParseError(f"paper {pid} missing from individual retry response too")
                    stats.input_tokens += int(single["input_tokens"] or 0)
                    stats.output_tokens += int(single["output_tokens"] or 0)
                    stats.estimated_cost_usd += float(single["estimated_cost_usd"] or 0)
                    rounding_warnings += single["rounding_warnings"]
                    with _connect() as wconn:
                        upsert_quality_assessment(
                            wconn,
                            content_id=pid,
                            sample_group=sample_group,
                            batch_id=batch_id,
                            batch_size=batch_size,
                            batch_position=pos,
                            scoring_tier="full",
                            result={
                                **r,
                                "input_tokens": single["input_tokens"],
                                "output_tokens": single["output_tokens"],
                                "estimated_cost_usd": single["estimated_cost_usd"],
                                "response_id": single["response_id"],
                                "status": "COMPLETED",
                            },
                        )
                        event(
                            wconn, run_id, pid, "semantic_score", "completed", True,
                            {"individual_retry": True, "ai_relevance": r.get("ai_relevance"), "batch": batch_tag},
                        )
                        wconn.commit()
                    outcomes.append(("completed", {"content_id": pid}, pos))
                except Exception as exc:
                    with _connect() as wconn:
                        upsert_quality_assessment(
                            wconn,
                            content_id=pid,
                            sample_group=sample_group,
                            batch_id=batch_id,
                            batch_size=batch_size,
                            batch_position=pos,
                            scoring_tier="full",
                            result={"status": "ERROR", "error_message": str(exc)[:1000]},
                        )
                        event(wconn, run_id, pid, "semantic_score", "error", False, {"batch": batch_tag}, str(exc))
                        bump(wconn, run_id, "errors")
                        wconn.commit()
                    outcomes.append(("failed", {"content_id": pid, "error": str(exc)}, pos))

            if parsed:
                with _connect() as wconn:
                    for pid, r in parsed.items():
                        pos, _paper = positioned[pid]
                        upsert_quality_assessment(
                            wconn,
                            content_id=pid,
                            sample_group=sample_group,
                            batch_id=batch_id,
                            batch_size=batch_size,
                            batch_position=pos,
                            scoring_tier="full",
                            result={**r, "status": "COMPLETED", "response_id": result.get("response_id") if result else None},
                        )
                        event(
                            wconn, run_id, pid, "semantic_score", "completed", True,
                            {"ai_relevance": r.get("ai_relevance"), "batch_size": batch_size, "batch": batch_tag},
                        )
                    wconn.commit()
                for pid in parsed:
                    pos, _ = positioned[pid]
                    outcomes.append(("completed", {"content_id": pid}, pos))

            stats.rounding_warnings += rounding_warnings
            return outcomes
        finally:
            gate.release()

    workers = max(1, SCORING_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_batch, b) for b in batches]
        done_batches = 0
        for fut in as_completed(futures):
            for kind, payload, _pos in fut.result():
                if kind == "completed":
                    stats.completed += 1
                else:
                    stats.failed += 1
            done_batches += 1
            log.info("Quality v2 (pass 2) batches progress %d/%d concurrency_ceiling=%d", done_batches, len(batches), gate.ceiling)

    summary = stats.to_dict()
    summary["gate_percentile"] = gate_percentile
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"semantic_score_v2_stats": summary}), run_id),
    )
    log.info("Quality v2 stats: %s", json.dumps(summary))
    print("\nSEMANTIC SCORE V2 (PASS 2) SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(cost_summary_line("Pass 2:", stats.completed, stats.calls, stats.estimated_cost_usd))
    return stats


# ===========================================================================
# Pass 1 — screen. Cheap, non-reasoning model, four dimensions, no prose.
#
# Same architectural guarantee as pass 2: title+categories+abstract only,
# never affiliation (build_quality_paper_block is reused verbatim). Writes to
# the SAME content_score_assessments table as pass 2, distinguished by
# prompt_version=SCREEN_PROMPT_VERSION and scoring_tier='screen'.
# ===========================================================================

SCREEN_NUMERIC_FIELDS = ["ai_relevance", "technical_significance", "apparent_novelty", "evidence_strength"]


def resolve_screen_model() -> str:
    model = SCREEN_MODEL
    return model if "/" in model else f"openai/{model}"


def _extract_gate_and_scale(system_prompt: str) -> str:
    """Pull the GATE + SCALE sections out of QUALITY_SYSTEM_PROMPT verbatim —
    by slicing the shared source text rather than retyping it, the two
    prompts' anchors cannot drift apart (brief §1: "same absolute anchors")."""
    start = system_prompt.index("GATE: Score ai_relevance first,")
    end = system_prompt.index("\n\nDIMENSIONS", start)
    return system_prompt[start:end].strip()


SCREEN_SYSTEM_PROMPT = (
    "You are a fast, cheap first-pass screen for AI research papers for "
    "TheNeural, a research intelligence system. Score four dimensions only, "
    "with no explanation, so the strongest papers can be sent to a slower, "
    "more careful second pass.\n\n"
    "You are given each paper's title, arXiv categories and abstract. You are "
    "NOT told who wrote it or where they work. Do not guess, and do not let "
    "a familiar research style, dataset or terminology lead you to infer a "
    "laboratory.\n\n"
    "Judge each paper on its own merits against the absolute scale below. Do "
    "NOT score papers relative to the others in this batch.\n\n"
    + _extract_gate_and_scale(QUALITY_SYSTEM_PROMPT)
    + "\n\n"
    "DIMENSIONS (score these four only)\n\n"
    "ai_relevance            Is this AI, ML, or their direct application? (GATE)\n"
    "technical_significance  If the claim holds, does it change what is\n"
    "                        technically possible or understood?\n"
    "apparent_novelty        Is the core idea new, or a known idea applied again?\n"
    "evidence_strength       Does the abstract REPORT results, or only claim them?\n"
    "                        Numbers, named baselines, named datasets, ablations\n"
    "                        and stated limitations raise this. Vague superlatives\n"
    "                        lower it.\n\n"
    "No per-dimension reasons, no so_what, no reason_not_higher. Output tokens "
    "dominate cost and screening does not need explanations.\n\n"
    "Return ONLY a JSON object with a single key \"papers\", whose value is an "
    "array with one object per paper, in the order supplied. No reasons, no "
    "prose, no markdown fences.\n"
    '{"papers": [{"paper_id": <int>, "ai_relevance": <0-10>, '
    '"technical_significance": <0-10>, "apparent_novelty": <0-10>, '
    '"evidence_strength": <0-10>}]}\n'
)

SCREEN_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_id": {"type": "integer"},
        **{k: {"type": "number"} for k in SCREEN_NUMERIC_FIELDS},
    },
    "required": ["paper_id", *SCREEN_NUMERIC_FIELDS],
}

SCREEN_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "papers": {"type": "array", "items": SCREEN_ITEM_SCHEMA},
    },
    "required": ["papers"],
}


class ScreenParseError(ValueError):
    pass


def build_screen_batch_user_prompt(papers: list[dict]) -> str:
    """Same paper-only block as pass 2 — reused, not reimplemented, so the
    blind-prompt guarantee has exactly one code path to audit."""
    return build_quality_batch_user_prompt(papers)


def parse_screen_batch(text: str, expected_ids: set[int]) -> tuple[dict[int, dict], int]:
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScreenParseError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScreenParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise ScreenParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    rounding_warnings = 0
    for item in items:
        if not isinstance(item, dict):
            raise ScreenParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise ScreenParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc

        dims: dict[str, float] = {}
        for key in SCREEN_NUMERIC_FIELDS:
            if key not in item:
                raise ScreenParseError(f"paper {pid} missing dimension: {key}")
            try:
                raw_val = float(item[key])
            except (TypeError, ValueError) as exc:
                raise ScreenParseError(f"paper {pid} non-numeric {key}: {item[key]!r}") from exc
            if raw_val < 0.0 or raw_val > 10.0:
                raise ScreenParseError(f"paper {pid} {key} out of range 0-10: {raw_val}")
            rounded, was_rounded = normalize_half_point(raw_val)
            if was_rounded:
                rounding_warnings += 1
                log.warning("Screen score not a 0.5 increment paper=%s dim=%s raw=%s rounded=%s", pid, key, raw_val, rounded)
            dims[key] = rounded

        if pid in expected_ids:
            out[pid] = dims
    return out, rounding_warnings


def call_screen_batch(papers: list[dict], *, client=None, on_rate_limited=None) -> dict:
    if client is None:
        client = create_llm_client()
    model = resolve_screen_model()
    user_prompt = build_screen_batch_user_prompt(papers)
    assert_quality_prompt_is_paper_only(user_prompt)  # same blind-prompt guarantee as pass 2
    expected_ids = {int(p.get("content_id", p.get("id"))) for p in papers}

    result = call_chat_completion(
        client,
        model=model,
        system_prompt=SCREEN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=None,  # reasoning disabled — cheap non-reasoning model
        temperature=0.0,
        max_retries=1,  # outer stage loop owns the batch-level retry/backoff
        request_sleep=SCREEN_REQUEST_SLEEP,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "screen_assessment",
                "strict": True,
                "schema": SCREEN_RESPONSE_SCHEMA,
            },
        },
        on_rate_limited=on_rate_limited,
    )
    parsed, rounding_warnings = parse_screen_batch(result["text"], expected_ids)
    return {
        "results": parsed,
        "rounding_warnings": rounding_warnings,
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
        "estimated_cost_usd": estimate_cost_usd(result["input_tokens"], result["output_tokens"]),
    }


def call_screen_batch_with_retry(
    papers: list[dict],
    *,
    client=None,
    max_retries: int | None = None,
    on_rate_limited=None,
    batch_tag: str = "",
) -> dict:
    max_retries = max_retries if max_retries is not None else SCREEN_MAX_RETRIES
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_screen_batch(papers, client=client, on_rate_limited=on_rate_limited)
        except (LLMBatchError, ScreenParseError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(60.0, 2 ** (attempt - 1))
                log.warning(
                    "[batch %s] Screen batch retry attempt=%s/%s size=%s err=%s",
                    batch_tag, attempt, max_retries, len(papers), exc,
                )
                time.sleep(wait)
                continue
    raise last_exc


@dataclass
class ScreenRunStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped_existing: int = 0
    rounding_warnings: int = 0
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
            "skipped_existing": self.skipped_existing,
            "rounding_warnings": self.rounding_warnings,
            "batches": self.batches,
            "calls": self.calls,
            "batch_size": SCREEN_BATCH_SIZE,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_total_cost_usd": round(self.estimated_cost_usd, 6),
            "model": resolve_screen_model(),
            "provider": LLM_PROVIDER,
            "prompt_version": SCREEN_PROMPT_VERSION,
            "reasoning": "disabled",
        }


def screen_assessment_exists(conn, content_id: int) -> bool:
    model = resolve_screen_model()
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_score_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
          AND status = 'COMPLETED'
        LIMIT 1
        """,
        (content_id, LLM_PROVIDER, model, SCREEN_PROMPT_VERSION),
    ).fetchone()
    return bool(row)


def upsert_screen_assessment(
    conn,
    *,
    content_id: int,
    batch_id,
    batch_size: int,
    batch_position: int,
    result: dict,
):
    """Only the four screen dimensions + status/tokens/cost. No prose fields,
    no sample_group (screen isn't sampled — every eligible paper is screened)."""
    sql = """
        INSERT INTO research_radar.content_score_assessments(
            content_id, assessment_type, provider, model_name, prompt_version,
            scoring_tier,
            ai_relevance, technical_significance, apparent_novelty, evidence_strength,
            batch_id, batch_size, batch_position,
            reasons, industry_labels,
            input_tokens, output_tokens, estimated_cost_usd,
            response_id, status, error_message
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s::jsonb, %s::jsonb,
            %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO UPDATE SET
            scoring_tier = EXCLUDED.scoring_tier,
            ai_relevance = EXCLUDED.ai_relevance,
            technical_significance = EXCLUDED.technical_significance,
            apparent_novelty = EXCLUDED.apparent_novelty,
            evidence_strength = EXCLUDED.evidence_strength,
            batch_id = EXCLUDED.batch_id,
            batch_size = EXCLUDED.batch_size,
            batch_position = EXCLUDED.batch_position,
            input_tokens = EXCLUDED.input_tokens,
            output_tokens = EXCLUDED.output_tokens,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd,
            response_id = EXCLUDED.response_id,
            status = EXCLUDED.status,
            error_message = EXCLUDED.error_message,
            created_at = NOW()
    """
    params = (
        content_id,
        ASSESSMENT_TYPE,
        LLM_PROVIDER,
        resolve_screen_model(),
        SCREEN_PROMPT_VERSION,
        "screen",
        result.get("ai_relevance"),
        result.get("technical_significance"),
        result.get("apparent_novelty"),
        result.get("evidence_strength"),
        batch_id,
        batch_size,
        batch_position,
        json.dumps({}),
        json.dumps([]),
        result.get("input_tokens"),
        result.get("output_tokens"),
        result.get("estimated_cost_usd"),
        result.get("response_id"),
        result.get("status") or "COMPLETED",
        result.get("error_message"),
    )
    conn.execute(sql, params)


def estimate_screen_prompt_tokens(papers: list[dict]) -> int:
    prompt = SCREEN_SYSTEM_PROMPT + build_screen_batch_user_prompt(papers)
    return max(1, len(prompt) // 4)


def stage_screen(
    conn,
    run_id,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    client=None,
):
    """
    Pass 1 — screen. Batch 15, randomly composed. Every ENTITY_RESOLVED/
    SCORED/CANDIDATE paper is screened (no sampling — the gate, not a sample,
    decides who reaches pass 2).
    """
    if not dry_run:
        require_scoring_enabled()
        require_api_key()

    candidates = load_quality_candidates(conn, limit=limit)
    if not force:
        candidates = [c for c in candidates if not screen_assessment_exists(conn, c["content_id"])]

    stats = ScreenRunStats(requested=len(candidates))
    batches = random_batches(candidates, SCREEN_BATCH_SIZE)
    stats.batches = len(batches)

    if dry_run:
        est_in = sum(estimate_screen_prompt_tokens(b) for b in batches)
        est_out = 40 * len(candidates)  # four bare numbers, no prose
        stats.input_tokens = est_in
        stats.output_tokens = est_out
        stats.estimated_cost_usd = estimate_cost_usd(est_in, est_out)
        stats.calls = len(batches)
        summary = stats.to_dict()
        conn.execute(
            "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
            (json.dumps({"screen_dry_run": summary}), run_id),
        )
        log.info("Screen DRY RUN: %s", json.dumps(summary))
        print("\nSCREEN (PASS 1) DRY RUN")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(cost_summary_line("Pass 1:", stats.requested, stats.calls, stats.estimated_cost_usd))
        return stats

    if client is None:
        client = create_llm_client()

    gate = AdaptiveConcurrencyGate(SCORING_CONCURRENCY)

    def _process_batch(batch: list[dict]) -> list[tuple[str, dict]]:
        from research_radar.pipeline import bump, connect as _connect, event

        batch_id = uuid.uuid4()
        batch_tag = str(batch_id)[:8]
        batch_size = len(batch)
        positioned = {int(p["content_id"]): (i + 1, p) for i, p in enumerate(batch)}
        outcomes: list[tuple[str, dict]] = []

        gate.acquire()
        try:
            try:
                result = call_screen_batch_with_retry(
                    batch, client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                )
                stats.calls += 1
            except (LLMBatchError, ScreenParseError) as exc:
                log.warning("[batch %s] Screen batch failed entirely; falling back to individual calls: %s", batch_tag, exc)
                result = None

            parsed: dict[int, dict] = {}
            rounding_warnings = 0
            if result is not None:
                parsed = result["results"]
                rounding_warnings = result["rounding_warnings"]
                stats.input_tokens += int(result["input_tokens"] or 0)
                stats.output_tokens += int(result["output_tokens"] or 0)
                stats.estimated_cost_usd += float(result["estimated_cost_usd"] or 0)

            missing_ids = set(positioned.keys()) - set(parsed.keys())
            for pid in missing_ids:
                pos, paper = positioned[pid]
                try:
                    single = call_screen_batch_with_retry(
                        [paper], client=client, on_rate_limited=gate.report_rate_limited, batch_tag=batch_tag
                    )
                    stats.calls += 1
                    r = single["results"].get(pid)
                    if r is None:
                        raise ScreenParseError(f"paper {pid} missing from individual retry response too")
                    stats.input_tokens += int(single["input_tokens"] or 0)
                    stats.output_tokens += int(single["output_tokens"] or 0)
                    stats.estimated_cost_usd += float(single["estimated_cost_usd"] or 0)
                    rounding_warnings += single["rounding_warnings"]
                    with _connect() as wconn:
                        upsert_screen_assessment(
                            wconn, content_id=pid, batch_id=batch_id, batch_size=batch_size, batch_position=pos,
                            result={**r, "input_tokens": single["input_tokens"], "output_tokens": single["output_tokens"],
                                    "estimated_cost_usd": single["estimated_cost_usd"], "response_id": single["response_id"],
                                    "status": "COMPLETED"},
                        )
                        event(wconn, run_id, pid, "screen", "completed", True,
                              {"individual_retry": True, "ai_relevance": r.get("ai_relevance"), "batch": batch_tag})
                        wconn.commit()
                    outcomes.append(("completed", {"content_id": pid}))
                except Exception as exc:
                    with _connect() as wconn:
                        upsert_screen_assessment(
                            wconn, content_id=pid, batch_id=batch_id, batch_size=batch_size, batch_position=pos,
                            result={"status": "ERROR", "error_message": str(exc)[:1000]},
                        )
                        event(wconn, run_id, pid, "screen", "error", False, {"batch": batch_tag}, str(exc))
                        bump(wconn, run_id, "errors")
                        wconn.commit()
                    outcomes.append(("failed", {"content_id": pid, "error": str(exc)}))

            if parsed:
                with _connect() as wconn:
                    for pid, r in parsed.items():
                        pos, _paper = positioned[pid]
                        upsert_screen_assessment(
                            wconn, content_id=pid, batch_id=batch_id, batch_size=batch_size, batch_position=pos,
                            result={**r, "status": "COMPLETED", "response_id": result.get("response_id") if result else None},
                        )
                        event(wconn, run_id, pid, "screen", "completed", True,
                              {"ai_relevance": r.get("ai_relevance"), "batch_size": batch_size, "batch": batch_tag})
                    wconn.commit()
                for pid in parsed:
                    outcomes.append(("completed", {"content_id": pid}))

            stats.rounding_warnings += rounding_warnings
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
            log.info("Screen (pass 1) batches progress %d/%d concurrency_ceiling=%d", done_batches, len(batches), gate.ceiling)

    summary = stats.to_dict()
    conn.execute(
        "UPDATE research_radar.pipeline_runs SET notes = notes || %s::jsonb WHERE run_id = %s",
        (json.dumps({"screen_stats": summary}), run_id),
    )
    log.info("Screen stats: %s", json.dumps(summary))
    print("\nSCREEN (PASS 1) SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(cost_summary_line("Pass 1:", stats.completed, stats.calls, stats.estimated_cost_usd))
    return stats


def count_scoring_pool(conn) -> dict:
    """How many papers were screened vs fully scored — for `report`'s pool line."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) FILTER (WHERE scoring_tier = 'screen') AS screened,
            COUNT(*) FILTER (WHERE scoring_tier = 'full') AS full_scored
        FROM research_radar.content_score_assessments
        WHERE status = 'COMPLETED'
          AND prompt_version IN (%s, %s)
        """,
        (SCREEN_PROMPT_VERSION, QUALITY_PROMPT_VERSION),
    ).fetchone()
    return {"screened": int(row["screened"] or 0), "full": int(row["full_scored"] or 0)}


def project_full_run_costs(conn, *, gate_percentile: float | None = None, sample_size: int = 50) -> dict:
    """
    Zero-API-call cost projection for a full scoring run over the currently
    eligible (not-yet-screened) pool: pass 1 over all of it, pass 2 + independence
    over the gated top `gate_percentile` percent. Token estimates are extrapolated
    from a small real sample's actual title/categories/abstract lengths, not a
    synthetic paper, so the projection reflects the real corpus.
    """
    from research_radar.independence import SYSTEM_PROMPT as INDEPENDENCE_SYSTEM_PROMPT

    gate_percentile = GATE_PERCENTILE if gate_percentile is None else gate_percentile

    all_eligible = load_quality_candidates(conn)
    already_screened = {
        r["content_id"]
        for r in conn.execute(
            "SELECT content_id FROM research_radar.content_score_assessments "
            "WHERE prompt_version = %s AND scoring_tier = 'screen' AND status = 'COMPLETED'",
            (SCREEN_PROMPT_VERSION,),
        ).fetchall()
    }
    pass1_pool = [c for c in all_eligible if c["content_id"] not in already_screened]

    sample = pass1_pool[: max(1, sample_size)] if pass1_pool else all_eligible[: max(1, sample_size)]
    if not sample:
        zero = {"n": 0, "calls": 0, "cost": 0.0}
        return {"pass1": zero, "pass2": dict(zero), "independence": dict(zero)}

    sample_batches_p1 = random_batches(sample, SCREEN_BATCH_SIZE)
    avg_in_p1 = sum(estimate_screen_prompt_tokens(b) for b in sample_batches_p1) / len(sample)
    avg_out_p1 = 40  # four bare numbers, no prose

    sample_batches_p2 = random_batches(sample, QUALITY_BATCH_SIZE)
    avg_in_p2 = sum(estimate_quality_prompt_tokens(b) for b in sample_batches_p2) / len(sample)
    avg_out_p2 = 220  # structured JSON incl. so_what/reason_not_higher

    avg_in_ind = (len(INDEPENDENCE_SYSTEM_PROMPT) // 4) / 20 + 140  # amortised system + per-paper body, batch 20
    avg_out_ind = 30  # status + short reason

    n_pass1 = len(pass1_pool)
    n_pass2 = math.ceil(n_pass1 * gate_percentile / 100.0)
    n_independence = n_pass2  # independence runs only on pass-2 papers (brief §1)

    def calls_for(n, batch_size):
        return math.ceil(n / batch_size) if n else 0

    pass1 = {
        "n": n_pass1,
        "calls": calls_for(n_pass1, SCREEN_BATCH_SIZE),
        "cost": estimate_cost_usd(int(avg_in_p1 * n_pass1), int(avg_out_p1 * n_pass1)),
    }
    pass2 = {
        "n": n_pass2,
        "calls": calls_for(n_pass2, QUALITY_BATCH_SIZE),
        "cost": estimate_cost_usd(int(avg_in_p2 * n_pass2), int(avg_out_p2 * n_pass2)),
    }
    independence = {
        "n": n_independence,
        "calls": calls_for(n_independence, 20),
        "cost": estimate_cost_usd(int(avg_in_ind * n_independence), int(avg_out_ind * n_independence)),
    }
    return {"pass1": pass1, "pass2": pass2, "independence": independence}
