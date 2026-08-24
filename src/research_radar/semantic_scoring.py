"""GPT semantic paper scoring via OpenRouter (assessment-only; does not mutate content_scores).

Phase 1: reproducible 100-paper experiment using openai/gpt-5.6-sol through OpenRouter.
Organisation / person / deterministic scores are intentionally NOT sent to the model.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("research-radar")

# ---------------------------------------------------------------------------
# Configuration — OpenRouter only (same pattern as Newsletter5)
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1").strip()
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://theneural.ai/").strip()
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "TheNeural Research Radar").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-sol").strip() or "openai/gpt-5.6-sol"
SEMANTIC_MAX_RETRIES = int(os.getenv("SEMANTIC_MAX_RETRIES", "5"))
SEMANTIC_REQUEST_SLEEP = float(os.getenv("SEMANTIC_REQUEST_SLEEP", "0.2"))
SEMANTIC_PROMPT_VERSION = os.getenv("SEMANTIC_PROMPT_VERSION", "research-semantic-v1").strip()
SEMANTIC_SCORING_WORKERS = int(os.getenv("SEMANTIC_SCORING_WORKERS", "2"))
LLM_PROVIDER = "openrouter"
ASSESSMENT_TYPE = "llm_semantic"
SAMPLE_SEED = int(os.getenv("SEMANTIC_SAMPLE_SEED", "20260824"))

# Pricing (USD per 1M tokens). Override via env — provisional placeholders.
OPENROUTER_INPUT_COST_PER_MILLION = float(os.getenv("OPENROUTER_INPUT_COST_PER_MILLION", "1.25"))
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
    """Guard: prompt must not contain organisational or deterministic-score leakage."""
    banned = [
        "organisation",
        "organization",
        "notable_org",
        "notable_person",
        "intrinsic_candidate_score",
        "CANDIDATE",
        "OpenAlex",
        "employer",
        "watchlist",
    ]
    low = prompt.lower()
    for term in banned:
        if term.lower() in low:
            raise SemanticParseError(f"prompt contains forbidden term: {term}")


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
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
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
    assert_prompt_is_paper_only(SYSTEM_PROMPT)

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
    model = resolve_model_name()
    row = conn.execute(
        """
        SELECT 1 AS ok
        FROM research_radar.content_score_assessments
        WHERE content_id = %s
          AND provider = %s
          AND model_name = %s
          AND prompt_version = %s
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
        sql += """
        ON CONFLICT (content_id, provider, model_name, prompt_version) DO NOTHING
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
