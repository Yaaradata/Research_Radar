"""Classification model bake-off — shared classify prompt, measured on our papers.

Uses the production classify prompt/schema from classify.py. Bake-off output
lives in bakeoff_* tables only — never content_classifications.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import requests
import yaml

from research_radar.classification_vocab import (
    APPLICATION_DOMAINS,
    AUDIENCE_RELEVANCE,
    GEOGRAPHY_FOCUS,
    PAPER_KINDS,
)
from research_radar.classify import (
    CLASSIFY_BATCH_SIZE,
    CLASSIFY_INPUT_KIND,
    CLASSIFY_PROMPT_VERSION,
    CLASSIFY_RESPONSE_SCHEMA,
    CLASSIFY_SYSTEM_PROMPT,
    ClassifyParseError,
    parse_classify_batch,
)
from research_radar.llm_batch import LLMBatchError, call_chat_completion, random_batches, strip_json_fences
from research_radar.semantic_scoring import build_quality_batch_user_prompt, create_llm_client

log = logging.getLogger("research-radar")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "bakeoff_models.yaml"

BASELINE_CANDIDATE_ID = "haiku"
LABELLERS = ("subhashini", "urmila", "ranjith")

NON_STANDARD_PAPER_KINDS = frozenset(
    {"survey_review", "benchmark_dataset", "theory", "negative_result"}
)
STRATUM_TARGETS = {
    "general_method": 150,
    "specific_sector": 150,
    "non_standard_kind": 50,
    "remainder": 50,
}


def scaled_stratum_targets(sample_size: int) -> dict[str, int]:
    """Scale the default 400-paper strata to a smaller (or larger) sample size."""
    base_total = sum(STRATUM_TARGETS.values())
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_size == base_total:
        return dict(STRATUM_TARGETS)
    # Largest-remainder so targets always sum to sample_size.
    raw = {k: (v * sample_size) / base_total for k, v in STRATUM_TARGETS.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainders = sorted(
        ((raw[k] - floors[k], k) for k in floors),
        reverse=True,
    )
    out = dict(floors)
    for i in range(sample_size - sum(floors.values())):
        out[remainders[i % len(remainders)][1]] += 1
    return out


@dataclass
class BakeoffCandidate:
    id: str
    model: str
    reasoning: str
    input_cost_per_million: float = 1.0
    output_cost_per_million: float = 5.0


@dataclass
class BakeoffConfig:
    candidates: list[BakeoffCandidate]
    batch_size: int = CLASSIFY_BATCH_SIZE
    sample_size: int = 400
    sample_seed: int = 20260907
    baseline_prompt_version: str = CLASSIFY_PROMPT_VERSION
    baseline_date_from: str = "2026-09-01"
    baseline_date_until: str = "2026-09-04"
    baseline_candidate_id: str = BASELINE_CANDIDATE_ID


@dataclass
class PaperValidation:
    raw_response: str
    json_valid: bool
    schema_valid: bool
    dropped_values: list[str]
    application_domain: list[str] | None = None
    audience_relevance: list[str] | None = None
    paper_kind: str | None = None
    geography_focus: str | None = None
    domain_confidence: float | None = None
    exclusivity_violation: bool = False


@dataclass
class BakeoffCallResult:
    content_id: int
    validation: PaperValidation
    retries: int
    json_valid_first_try: bool
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


def load_bakeoff_config(path: Path | None = None) -> BakeoffConfig:
    cfg_path = path or DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    candidates = [
        BakeoffCandidate(
            id=c["id"],
            model=c["model"],
            reasoning=str(c.get("reasoning", "disabled")),
            input_cost_per_million=float(c.get("input_cost_per_million", 1.0)),
            output_cost_per_million=float(c.get("output_cost_per_million", 5.0)),
        )
        for c in raw["candidates"]
    ]
    return BakeoffConfig(
        candidates=candidates,
        batch_size=int(raw.get("batch_size", CLASSIFY_BATCH_SIZE)),
        sample_size=int(raw.get("sample_size", 400)),
        sample_seed=int(raw.get("sample_seed", 20260907)),
        baseline_prompt_version=str(raw.get("baseline_prompt_version", CLASSIFY_PROMPT_VERSION)),
        baseline_date_from=str(raw.get("baseline_date_from", "2026-09-01")),
        baseline_date_until=str(raw.get("baseline_date_until", "2026-09-04")),
        baseline_candidate_id=str(raw.get("baseline_candidate_id", BASELINE_CANDIDATE_ID)),
    )


def new_run_id() -> UUID:
    return uuid4()


def reasoning_effort_for(candidate: BakeoffCandidate) -> str | None:
    if candidate.reasoning == "disabled":
        return None
    if candidate.reasoning == "minimum":
        return "low"
    return None


def cost_from_tokens(
    tokens_in: int,
    tokens_out: int,
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    return round(
        (tokens_in / 1_000_000.0) * input_cost_per_million
        + (tokens_out / 1_000_000.0) * output_cost_per_million,
        6,
    )


def cost_per_thousand_from_measured(
    rows: list[dict],
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    tin = sum(int(r.get("tokens_in") or 0) for r in rows)
    tout = sum(int(r.get("tokens_out") or 0) for r in rows)
    total = cost_from_tokens(
        tin,
        tout,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    return round(total / n * 1000.0, 4)


def verify_openrouter_models(model_ids: list[str], *, api_key: str | None = None) -> dict[str, bool]:
    """Return {model_id: resolves} for each requested id. Does not substitute."""
    api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {m: False for m in model_ids}
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        available = {item["id"] for item in resp.json().get("data", [])}
    except Exception as exc:
        log.warning("OpenRouter model list failed: %s", exc)
        return {m: False for m in model_ids}
    return {m: m in available for m in model_ids}


def load_baseline_classified_papers(conn, config: BakeoffConfig | None = None) -> list[dict]:
    config = config or load_bakeoff_config()
    rows = conn.execute(
        """
        SELECT
            cc.content_id,
            ci.title,
            COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
            COALESCE(pm.abstract, ci.summary, '') AS abstract,
            cc.application_domain,
            cc.audience_relevance,
            cc.paper_kind,
            cc.geography_focus,
            cc.domain_confidence,
            cc.model AS baseline_model
        FROM research_radar.content_classifications cc
        JOIN research_radar.content_items ci ON ci.id = cc.content_id
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE cc.prompt_version = %s
          AND cc.classify_input_kind = %s
          AND ci.published_at >= %s::date
          AND ci.published_at < (%s::date + INTERVAL '1 day')
        ORDER BY cc.content_id
        """,
        (
            config.baseline_prompt_version,
            CLASSIFY_INPUT_KIND,
            config.baseline_date_from,
            config.baseline_date_until,
        ),
    ).fetchall()
    return [dict(r) for r in rows]


def _stratum_key(paper: dict) -> str | None:
    domains = paper.get("application_domain") or []
    if "general_method" in domains:
        return "general_method"
    if any(d for d in domains if d != "general_method"):
        return "specific_sector"
    if paper.get("paper_kind") in NON_STANDARD_PAPER_KINDS:
        return "non_standard_kind"
    return None


def select_stratified_sample(
    papers: list[dict],
    *,
    seed: int,
    targets: dict[str, int] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Draw stratified sample without replacement. Returns (sample, counts_by_stratum)."""
    targets = targets or STRATUM_TARGETS
    rng = random.Random(seed)
    by_id = {int(p["content_id"]): p for p in papers}
    selected_ids: set[int] = set()
    counts: dict[str, int] = {}

    pools: dict[str, list[dict]] = {
        "general_method": [],
        "specific_sector": [],
        "non_standard_kind": [],
        "remainder": [],
    }
    for p in papers:
        cid = int(p["content_id"])
        domains = p.get("application_domain") or []
        if "general_method" in domains:
            pools["general_method"].append(p)
        elif any(d for d in domains if d != "general_method"):
            pools["specific_sector"].append(p)
        if p.get("paper_kind") in NON_STANDARD_PAPER_KINDS:
            pools["non_standard_kind"].append(p)

    for stratum in ("general_method", "specific_sector", "non_standard_kind"):
        n = targets[stratum]
        pool = [p for p in pools[stratum] if int(p["content_id"]) not in selected_ids]
        rng.shuffle(pool)
        picked = pool[: min(n, len(pool))]
        for p in picked:
            selected_ids.add(int(p["content_id"]))
        counts[stratum] = len(picked)

    remainder_target = targets["remainder"]
    remainder_pool = [p for p in papers if int(p["content_id"]) not in selected_ids]
    rng.shuffle(remainder_pool)
    picked = remainder_pool[: min(remainder_target, len(remainder_pool))]
    for p in picked:
        selected_ids.add(int(p["content_id"]))
    counts["remainder"] = len(picked)

    sample = [by_id[cid] for cid in sorted(selected_ids)]
    return sample, counts


def reconstruct_raw_response(paper: dict) -> str:
    """Rebuild classify JSON from stored classification fields (Haiku baseline import)."""
    payload = {
        "papers": [
            {
                "paper_id": int(paper["content_id"]),
                "application_domain": list(paper.get("application_domain") or []),
                "audience_relevance": list(paper.get("audience_relevance") or []),
                "paper_kind": paper.get("paper_kind"),
                "geography_focus": paper.get("geography_focus"),
                "domain_confidence": float(paper["domain_confidence"])
                if paper.get("domain_confidence") is not None
                else None,
            }
        ]
    }
    return json.dumps(payload, separators=(",", ":"))


def analyze_raw_classify_response(raw_text: str, expected_ids: set[int]) -> dict[int, PaperValidation]:
    """Validate raw model text without raising — invalid output remains measurable."""
    out: dict[int, PaperValidation] = {}
    json_valid = False
    items: list[Any] = []
    dropped_global: list[str] = []

    try:
        payload = json.loads(strip_json_fences(raw_text))
        json_valid = isinstance(payload, dict)
        if json_valid:
            items = payload.get("papers") if isinstance(payload.get("papers"), list) else []
    except json.JSONDecodeError:
        json_valid = False

    if not json_valid:
        for pid in expected_ids:
            out[pid] = PaperValidation(
                raw_response=raw_text,
                json_valid=False,
                schema_valid=False,
                dropped_values=["json_parse_failed"],
            )
        return out

    parsed_ids: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            dropped_global.append("non_object_item")
            continue
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError):
            dropped_global.append(f"invalid_paper_id:{item.get('paper_id')!r}")
            continue
        if pid not in expected_ids:
            continue
        parsed_ids.add(pid)
        dropped: list[str] = list(dropped_global)
        exclusivity = False
        schema_valid = True

        app_dom = item.get("application_domain")
        audience = item.get("audience_relevance")
        paper_kind = (item.get("paper_kind") or "").strip() if item.get("paper_kind") else ""
        geography = (item.get("geography_focus") or "").strip() if item.get("geography_focus") else ""

        if not isinstance(app_dom, list):
            schema_valid = False
            dropped.append("application_domain:not_list")
            app_dom = None
        elif len(app_dom) > 3:
            schema_valid = False
            dropped.append("application_domain:too_many")
        else:
            for v in app_dom:
                if v not in APPLICATION_DOMAINS:
                    schema_valid = False
                    dropped.append(f"application_domain:invalid:{v}")
            if "general_method" in app_dom and len(app_dom) > 1:
                exclusivity = True
                schema_valid = False
                dropped.append("application_domain:general_method_exclusivity")

        if not isinstance(audience, list) or not (1 <= len(audience) <= 4):
            schema_valid = False
            dropped.append("audience_relevance:invalid")
            audience = None
        elif audience is not None:
            for v in audience:
                if v not in AUDIENCE_RELEVANCE:
                    schema_valid = False
                    dropped.append(f"audience_relevance:invalid:{v}")

        if paper_kind not in PAPER_KINDS:
            schema_valid = False
            dropped.append(f"paper_kind:invalid:{paper_kind!r}")
            paper_kind = None
        if geography not in GEOGRAPHY_FOCUS:
            schema_valid = False
            dropped.append(f"geography_focus:invalid:{geography!r}")
            geography = None

        conf = None
        try:
            conf_raw = float(item["domain_confidence"])
            if conf_raw < 0.0 or conf_raw > 10.0:
                schema_valid = False
                dropped.append("domain_confidence:out_of_range")
            else:
                conf = round(conf_raw * 2) / 2.0
        except (TypeError, ValueError, KeyError):
            schema_valid = False
            dropped.append("domain_confidence:missing")

        out[pid] = PaperValidation(
            raw_response=raw_text,
            json_valid=True,
            schema_valid=schema_valid,
            dropped_values=dropped,
            application_domain=list(app_dom) if isinstance(app_dom, list) else None,
            audience_relevance=list(audience) if isinstance(audience, list) else None,
            paper_kind=paper_kind or None,
            geography_focus=geography or None,
            domain_confidence=conf,
            exclusivity_violation=exclusivity,
        )

    for pid in expected_ids - parsed_ids:
        out[pid] = PaperValidation(
            raw_response=raw_text,
            json_valid=True,
            schema_valid=False,
            dropped_values=["missing_paper_id"],
        )
    return out


def count_exclusivity_violations(application_domain: list[str] | None) -> int:
    if not application_domain:
        return 0
    return 1 if "general_method" in application_domain and len(application_domain) > 1 else 0


def is_general_method(domains: list[str] | None) -> bool:
    return bool(domains) and len(domains) == 1 and domains[0] == "general_method"


def is_force_fit(model_domains: list[str] | None, human_general: bool) -> bool:
    """Model assigned a specific sector but human said general_method."""
    if not human_general:
        return False
    if not model_domains:
        return False
    return not is_general_method(model_domains)


def compute_force_fit_rate(model_by_id: dict[int, dict], human_by_id: dict[int, dict]) -> float:
    hits = 0
    n = 0
    for cid, human in human_by_id.items():
        if cid not in model_by_id:
            continue
        human_general = is_general_method(human.get("application_domain"))
        n += 1
        if is_force_fit(model_by_id[cid].get("application_domain"), human_general):
            hits += 1
    return hits / n if n else 0.0


def self_consistency_rate(pass1: dict[int, dict], pass2: dict[int, dict]) -> float:
    """Agreement on application_domain between passes, keyed by content_id."""
    common = set(pass1.keys()) & set(pass2.keys())
    if not common:
        return 0.0
    agree = sum(
        1
        for cid in common
        if (pass1[cid].get("application_domain") or []) == (pass2[cid].get("application_domain") or [])
    )
    return agree / len(common)


def call_classify_batch_bakeoff(
    papers: list[dict],
    *,
    candidate: BakeoffCandidate,
    client=None,
    on_rate_limited=None,
) -> dict:
    if client is None:
        client = create_llm_client()
    user_prompt = build_quality_batch_user_prompt(papers)
    expected_ids = {int(p["content_id"]) for p in papers}
    result = call_chat_completion(
        client,
        model=candidate.model,
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort=reasoning_effort_for(candidate),
        temperature=0.0,
        max_retries=1,
        request_sleep=0.2,
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
    return {
        "text": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "response_id": result["response_id"],
    }


def classify_batch_with_retries(
    papers: list[dict],
    *,
    candidate: BakeoffCandidate,
    client=None,
    max_retries: int = 3,
) -> tuple[dict[int, PaperValidation], int, bool, int, int, int]:
    """Returns validations, retries, json_valid_first_try, tokens_in, tokens_out, latency_ms."""
    last_raw = ""
    retries = 0
    json_valid_first_try = False
    start = time.perf_counter()
    tokens_in = tokens_out = 0
    expected_ids = {int(p["content_id"]) for p in papers}

    for attempt in range(1, max_retries + 1):
        try:
            result = call_classify_batch_bakeoff(papers, candidate=candidate, client=client)
            last_raw = result["text"]
            tokens_in = int(result["input_tokens"] or 0)
            tokens_out = int(result["output_tokens"] or 0)
            if attempt == 1:
                try:
                    json.loads(strip_json_fences(last_raw))
                    json_valid_first_try = True
                except json.JSONDecodeError:
                    json_valid_first_try = False
            validations = analyze_raw_classify_response(last_raw, expected_ids)
            if all(v.schema_valid for v in validations.values()):
                latency_ms = int((time.perf_counter() - start) * 1000)
                return validations, retries, json_valid_first_try, tokens_in, tokens_out, latency_ms
            retries = attempt
        except (LLMBatchError, ClassifyParseError) as exc:
            retries = attempt
            log.warning("bakeoff batch attempt %s failed: %s", attempt, exc)
            if attempt >= max_retries:
                break
            time.sleep(min(60.0, 2 ** (attempt - 1)))

    latency_ms = int((time.perf_counter() - start) * 1000)
    validations = analyze_raw_classify_response(last_raw or "", expected_ids)
    return validations, retries, json_valid_first_try, tokens_in, tokens_out, latency_ms


def persist_bakeoff_run(conn, run_id: UUID, *, seed: int, sample_size: int, prompt_version: str) -> None:
    conn.execute(
        """
        INSERT INTO research_radar.bakeoff_runs (run_id, sample_seed, sample_size, prompt_version)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING
        """,
        (str(run_id), seed, sample_size, prompt_version),
    )


def insert_bakeoff_result(
    conn,
    *,
    run_id: UUID,
    candidate: BakeoffCandidate,
    content_id: int,
    pass_index: int,
    validation: PaperValidation,
    retries: int,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
) -> None:
    cost = cost_from_tokens(
        tokens_in,
        tokens_out,
        input_cost_per_million=candidate.input_cost_per_million,
        output_cost_per_million=candidate.output_cost_per_million,
    )
    conn.execute(
        """
        INSERT INTO research_radar.bakeoff_results (
            run_id, candidate_id, model, content_id, pass_index,
            application_domain, audience_relevance, paper_kind, geography_focus, domain_confidence,
            raw_response, json_valid, schema_valid, dropped_values,
            retries, tokens_in, tokens_out, cost_usd, latency_ms
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON CONFLICT (run_id, candidate_id, content_id, pass_index) DO UPDATE SET
            model = EXCLUDED.model,
            application_domain = EXCLUDED.application_domain,
            audience_relevance = EXCLUDED.audience_relevance,
            paper_kind = EXCLUDED.paper_kind,
            geography_focus = EXCLUDED.geography_focus,
            domain_confidence = EXCLUDED.domain_confidence,
            raw_response = EXCLUDED.raw_response,
            json_valid = EXCLUDED.json_valid,
            schema_valid = EXCLUDED.schema_valid,
            dropped_values = EXCLUDED.dropped_values,
            retries = EXCLUDED.retries,
            tokens_in = EXCLUDED.tokens_in,
            tokens_out = EXCLUDED.tokens_out,
            cost_usd = EXCLUDED.cost_usd,
            latency_ms = EXCLUDED.latency_ms
        """,
        (
            str(run_id),
            candidate.id,
            candidate.model,
            content_id,
            pass_index,
            validation.application_domain,
            validation.audience_relevance,
            validation.paper_kind,
            validation.geography_focus,
            validation.domain_confidence,
            validation.raw_response,
            validation.json_valid,
            validation.schema_valid,
            validation.dropped_values,
            retries,
            tokens_in,
            tokens_out,
            cost,
            latency_ms,
        ),
    )


def import_haiku_baseline(
    conn,
    run_id: UUID,
    sample: list[dict],
    *,
    candidate_id: str = BASELINE_CANDIDATE_ID,
    model: str = "anthropic/claude-haiku-4.5",
) -> int:
    """Import existing content_classifications rows into bakeoff_results — no API calls."""
    n = 0
    for paper in sample:
        raw = reconstruct_raw_response(paper)
        validation = analyze_raw_classify_response(raw, {int(paper["content_id"])})[int(paper["content_id"])]
        candidate = BakeoffCandidate(id=candidate_id, model=model, reasoning="disabled")
        insert_bakeoff_result(
            conn,
            run_id=run_id,
            candidate=candidate,
            content_id=int(paper["content_id"]),
            pass_index=1,
            validation=validation,
            retries=0,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
        )
        n += 1
    return n


def load_sample_papers(conn, run_id: UUID) -> list[dict]:
    rows = conn.execute(
        """
        SELECT DISTINCT br.content_id, ci.title,
               COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories,
               COALESCE(pm.abstract, ci.summary, '') AS abstract
        FROM research_radar.bakeoff_results br
        JOIN research_radar.content_items ci ON ci.id = br.content_id
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE br.run_id = %s AND br.candidate_id = %s AND br.pass_index = 1
        ORDER BY br.content_id
        """,
        (str(run_id), BASELINE_CANDIDATE_ID),
    ).fetchall()
    return [dict(r) for r in rows]


def load_results_for_run(conn, run_id: UUID, candidate_id: str | None = None) -> list[dict]:
    params: list[Any] = [str(run_id)]
    sql = """
        SELECT * FROM research_radar.bakeoff_results
        WHERE run_id = %s
    """
    if candidate_id:
        sql += " AND candidate_id = %s"
        params.append(candidate_id)
    sql += " ORDER BY candidate_id, content_id, pass_index"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def load_human_labels(conn, run_id: UUID) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM research_radar.bakeoff_labels
        WHERE run_id = %s
        ORDER BY content_id, labeller
        """,
        (str(run_id),),
    ).fetchall()
    return [dict(r) for r in rows]


def estimate_bakeoff_run_cost(config: BakeoffConfig, n_papers: int, n_passes: int = 2) -> dict[str, Any]:
    """Token-based estimate for non-baseline candidates (no API calls)."""
    per_paper_in = 550
    per_paper_out = 80
    batches_per_pass = max(1, (n_papers + config.batch_size - 1) // config.batch_size)
    by_candidate = {}
    total = 0.0
    for cand in config.candidates:
        if cand.id == config.baseline_candidate_id:
            continue
        tin = per_paper_in * n_papers * n_passes
        tout = per_paper_out * n_papers * n_passes
        cost = cost_from_tokens(
            tin,
            tout,
            input_cost_per_million=cand.input_cost_per_million,
            output_cost_per_million=cand.output_cost_per_million,
        )
        by_candidate[cand.id] = {
            "model": cand.model,
            "papers": n_papers,
            "passes": n_passes,
            "batches_per_pass": batches_per_pass,
            "estimated_cost_usd": round(cost, 2),
        }
        total += cost
    return {
        "n_papers": n_papers,
        "n_passes": n_passes,
        "non_baseline_candidates": len(by_candidate),
        "by_candidate": by_candidate,
        "total_estimated_cost_usd": round(total, 2),
    }


def compute_candidate_metrics(
    results: list[dict],
    labels: list[dict],
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> dict[str, Any]:
    if not results:
        return {}
    n = len(results)
    general_method_n = sum(1 for r in results if is_general_method(r.get("application_domain")))
    exclusivity = sum(count_exclusivity_violations(r.get("application_domain")) for r in results)
    json_valid = sum(1 for r in results if r.get("json_valid"))
    schema_valid = sum(1 for r in results if r.get("schema_valid"))
    first_try = sum(1 for r in results if r.get("json_valid") and int(r.get("retries") or 0) == 0)
    invalid_raw = sum(
        1
        for r in results
        if not r.get("schema_valid") and any(
            "invalid" in (dv or "") for dv in (r.get("dropped_values") or [])
        )
    )
    latencies = [int(r["latency_ms"]) for r in results if r.get("latency_ms")]
    cost_per_1k = cost_per_thousand_from_measured(
        results,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )

    human_by_id: dict[int, dict] = {}
    for row in labels:
        cid = int(row["content_id"])
        human_by_id.setdefault(cid, row)

    model_by_id = {int(r["content_id"]): r for r in results}
    force_fit = compute_force_fit_rate(model_by_id, human_by_id) if human_by_id else None

    accuracy = None
    if human_by_id:
        agree = 0
        labelled = 0
        for cid, human in human_by_id.items():
            if cid not in model_by_id:
                continue
            labelled += 1
            if (model_by_id[cid].get("application_domain") or []) == (human.get("application_domain") or []):
                agree += 1
        accuracy = agree / labelled if labelled else None

    return {
        "n": n,
        "general_method_rate": round(general_method_n / n, 4) if n else 0.0,
        "force_fit_rate": round(force_fit, 4) if force_fit is not None else None,
        "exclusivity_violations": exclusivity,
        "invalid_rate": round(invalid_raw / n, 4) if n else 0.0,
        "valid_json_rate": round(first_try / n, 4) if n else 0.0,
        "schema_valid_rate": round(schema_valid / n, 4) if n else 0.0,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "cost_per_1000": cost_per_1k,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }


def candidate_disagreement_rows(results: list[dict]) -> set[int]:
    """Papers where candidates disagree on general_method vs specific, or on domains."""
    by_paper: dict[int, list[set[str]]] = {}
    for r in results:
        if int(r.get("pass_index") or 1) != 1:
            continue
        cid = int(r["content_id"])
        domains = tuple(sorted(r.get("application_domain") or []))
        by_paper.setdefault(cid, []).append(set(domains))

    disagreements: set[int] = set()
    for cid, domain_sets in by_paper.items():
        if len(domain_sets) < 2:
            continue
        gm_flags = {is_general_method(sorted(s)) for s in domain_sets}
        if len(gm_flags) > 1:
            disagreements.add(cid)
            continue
        if len({tuple(sorted(s)) for s in domain_sets}) > 1:
            disagreements.add(cid)
    return disagreements


def agreement_control_ids(results: list[dict], *, seed: int, n: int = 30) -> list[int]:
    by_paper: dict[int, list[set[str]]] = {}
    for r in results:
        if int(r.get("pass_index") or 1) != 1:
            continue
        cid = int(r["content_id"])
        by_paper.setdefault(cid, []).append(set(r.get("application_domain") or []))
    unanimous = [cid for cid, sets in by_paper.items() if len({tuple(sorted(s)) for s in sets}) == 1]
    rng = random.Random(seed)
    rng.shuffle(unanimous)
    return unanimous[:n]
