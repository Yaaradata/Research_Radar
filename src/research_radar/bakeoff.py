"""Classification / screening model bake-off — shared prompt, schema, metrics.

High-volume fixed-schema calls: domain, subdomains, application_domains,
primary_audience, ai_relevance. Reasoning disabled for all candidates.
Quality pass is out of scope.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml

from research_radar.classification_vocab import AUDIENCE_RELEVANCE
from research_radar.llm_batch import LLMBatchError, call_chat_completion, random_batches, strip_json_fences
from research_radar.semantic_scoring import build_quality_batch_user_prompt, create_llm_client, require_api_key
from research_radar.topics import (
    TOPICS_SYSTEM_PROMPT,
    build_vocabulary_block,
    load_topic_vocabulary,
)

log = logging.getLogger("research-radar")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "bakeoff_models.yaml"

BAKEOFF_PROMPT_KIND = "classification-screening"
BAKEOFF_PROMPT_VERSION = (
    os.getenv("BAKEOFF_PROMPT_VERSION", "classification-bakeoff-v1").strip() or "classification-bakeoff-v1"
)

GENERAL_METHOD_VALUE = "general-method"

APPLICATION_DOMAIN_INSTRUCTION = """
general-method    The paper presents a method, model or theoretical result
                  with no specific application domain. Most methods papers
                  are this. Choosing general-method is a correct, expected
                  answer - do NOT reach for a sector when none is addressed.
                  If you find yourself constructing an argument for why a
                  method "could apply" to a domain, the answer is
                  general-method.

general-method is mutually exclusive with every other application domain.
Return general-method alone when it applies; never combine it with a sector.
"""

AUDIENCE_INSTRUCTION = """
PRIMARY AUDIENCE — return exactly ONE value for primary_audience:

practitioner           An engineer or data scientist could act on this.
technical_leadership   Bears on architecture, platform or build-or-buy decisions.
enterprise_adoption    Bears on deploying AI in an organisation.
student                Good entry point for someone still learning the subfield.
"""

AI_RELEVANCE_INSTRUCTION = """
AI RELEVANCE (ai_relevance) — score 0-10 in 0.5 increments:

Is this AI, ML, or their direct application? This is a gate dimension only.
Use the same absolute scale as the screen pass: 0 = not AI-related, 10 = core AI.
"""

BAKEOFF_SYSTEM_PROMPT = (
    TOPICS_SYSTEM_PROMPT.replace(
        "applications ZERO to FOUR, chosen from the APPLICATIONS list below. Only\n"
        "            include an application the paper explicitly addresses or\n"
        "            evaluates. An abstract that never mentions education must not be\n"
        "            tagged education.",
        "application_domains ZERO to FOUR, chosen from the APPLICATIONS list below.\n"
        "            When no sector is explicitly addressed, return [\"general-method\"]\n"
        "            — never an empty list. Only include a sector application the paper\n"
        "            explicitly addresses or evaluates.",
    )
    + "\n"
    + APPLICATION_DOMAIN_INSTRUCTION
    + "\n"
    + AUDIENCE_INSTRUCTION
    + "\n"
    + AI_RELEVANCE_INSTRUCTION
    + "\n\n"
    "Return ONLY a JSON object with a single key \"papers\", whose value is an array\n"
    "with one object per paper, in the order supplied. No prose, no markdown fences.\n\n"
    '{"papers": [{"paper_id": <int>, "domain": "...", "subdomains": ["..."], '
    '"application_domains": ["..."], "primary_audience": "...", '
    '"ai_relevance": <0-10>}]}\n'
)

# Fallback when DB vocabulary is unavailable (unit tests).
FALLBACK_DOMAINS = [
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
    "Machine Learning Theory",
    "Other",
]
FALLBACK_SUBDOMAINS = {
    "Natural Language Processing": ["Text Classification", "Question Answering"],
    "Computer Vision": ["Object Detection", "Image Classification"],
    "Reinforcement Learning": ["Deep Reinforcement Learning"],
    "Machine Learning Theory": ["Optimization Theory"],
    "Other": [],
}
FALLBACK_APPLICATIONS = [
    "general-method",
    "education",
    "healthcare",
    "finance",
    "security",
    "defense",
]


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
    batch_size: int = 10
    sample_size: int = 400
    sample_seed: int = 20260905


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
        batch_size=int(raw.get("batch_size", 10)),
        sample_size=int(raw.get("sample_size", 400)),
        sample_seed=int(raw.get("sample_seed", 20260905)),
    )


def fallback_vocabulary() -> dict:
    return {
        "domains_list": list(FALLBACK_DOMAINS),
        "subdomains_by_domain": {k: list(v) for k, v in FALLBACK_SUBDOMAINS.items()},
        "applications_list": list(FALLBACK_APPLICATIONS),
    }


def vocabulary_from_conn(conn) -> dict:
    try:
        return load_topic_vocabulary(conn)
    except Exception:
        return fallback_vocabulary()


def build_bakeoff_user_prompt(papers: list[dict], vocab: dict) -> str:
    block = build_vocabulary_block(vocab)
    return block + "\n\n" + build_quality_batch_user_prompt(papers)


def bakeoff_response_schema(vocab: dict) -> dict:
    domains = vocab.get("domains_list") or FALLBACK_DOMAINS
    all_subdomains: list[str] = []
    for subs in (vocab.get("subdomains_by_domain") or {}).values():
        all_subdomains.extend(subs)
    applications = vocab.get("applications_list") or FALLBACK_APPLICATIONS
    if GENERAL_METHOD_VALUE not in applications:
        applications = [GENERAL_METHOD_VALUE, *applications]

    item_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "paper_id": {"type": "integer"},
            "domain": {"type": "string", "enum": domains + ["Other"]},
            "subdomains": {
                "type": "array",
                "items": {"type": "string", "enum": all_subdomains or ["Other"]},
                "maxItems": 3,
            },
            "application_domains": {
                "type": "array",
                "items": {"type": "string", "enum": applications},
                "maxItems": 4,
            },
            "primary_audience": {"type": "string", "enum": list(AUDIENCE_RELEVANCE)},
            "ai_relevance": {"type": "number"},
        },
        "required": [
            "paper_id",
            "domain",
            "subdomains",
            "application_domains",
            "primary_audience",
            "ai_relevance",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"papers": {"type": "array", "items": item_schema}},
        "required": ["papers"],
    }


class BakeoffParseError(ValueError):
    pass


def count_general_method_violations(application_domains: list[str]) -> int:
    """general-method must not co-occur with any other application domain."""
    if not application_domains:
        return 0
    has_gm = GENERAL_METHOD_VALUE in application_domains
    if has_gm and len(application_domains) > 1:
        return 1
    return 0


def is_general_method(application_domains: list[str] | None) -> bool:
    if not application_domains:
        return False
    return GENERAL_METHOD_VALUE in application_domains


def normalize_application_domains(raw: list[str] | None) -> list[str]:
    return [str(x).strip() for x in (raw or []) if str(x).strip()]


def parse_bakeoff_batch(text: str, expected_ids: set[int]) -> dict[int, dict]:
    raw = strip_json_fences(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BakeoffParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BakeoffParseError("response must be a JSON object with a 'papers' array")
    items = payload.get("papers")
    if not isinstance(items, list):
        raise BakeoffParseError("response object missing 'papers' array")

    out: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise BakeoffParseError("each item must be an object")
        try:
            pid = int(item.get("paper_id"))
        except (TypeError, ValueError) as exc:
            raise BakeoffParseError(f"invalid paper_id: {item.get('paper_id')!r}") from exc
        if pid not in expected_ids:
            continue

        app_domains = normalize_application_domains(item.get("application_domains"))
        if count_general_method_violations(app_domains):
            raise BakeoffParseError(f"paper {pid} general-method combined with other domains")
        if not app_domains:
            raise BakeoffParseError(f"paper {pid} application_domains must not be empty")

        audience = (item.get("primary_audience") or "").strip()
        if audience not in AUDIENCE_RELEVANCE:
            raise BakeoffParseError(f"paper {pid} invalid primary_audience: {audience!r}")

        try:
            ai_rel = float(item["ai_relevance"])
        except (TypeError, ValueError, KeyError) as exc:
            raise BakeoffParseError(f"paper {pid} missing ai_relevance") from exc
        ai_rel = round(ai_rel * 2) / 2.0
        if ai_rel < 0.0 or ai_rel > 10.0:
            raise BakeoffParseError(f"paper {pid} ai_relevance out of range")

        domain = str(item.get("domain") or "").strip()
        subdomains = [str(s).strip() for s in (item.get("subdomains") or []) if str(s).strip()]

        out[pid] = {
            "domain": domain,
            "subdomains": subdomains,
            "application_domains": app_domains,
            "primary_audience": audience,
            "ai_relevance": ai_rel,
        }
    return out


def estimate_tokens_for_papers(papers: list[dict], vocab: dict) -> tuple[int, int]:
    """Rough token estimate: chars/4 for prompt, fixed output per paper."""
    prompt = BAKEOFF_SYSTEM_PROMPT + build_bakeoff_user_prompt(papers, vocab)
    est_in = max(1, len(prompt) // 4)
    est_out = 60 * len(papers)
    return est_in, est_out


def cost_from_tokens(tokens_in: int, tokens_out: int, candidate: BakeoffCandidate) -> float:
    return round(
        (tokens_in / 1_000_000.0) * candidate.input_cost_per_million
        + (tokens_out / 1_000_000.0) * candidate.output_cost_per_million,
        6,
    )


def cost_per_thousand_from_measured(
    total_cost: float, paper_count: int
) -> float:
    if paper_count <= 0:
        return 0.0
    return round((total_cost / paper_count) * 1000.0, 4)


def estimate_full_bakeoff_cost(config: BakeoffConfig, vocab: dict, paper_count: int | None = None) -> dict:
    """Estimate cost for sample_size × candidates × 3 passes (pass1, pass2, batch-B)."""
    n = paper_count if paper_count is not None else config.sample_size
    batches = max(1, math.ceil(n / config.batch_size))
    dummy_papers = [
        {
            "content_id": i + 1,
            "title": "Sample paper title for token estimation",
            "abstract": "A" * 1200,
            "categories": ["cs.AI", "cs.LG"],
        }
        for i in range(config.batch_size)
    ]
    est_in, est_out = estimate_tokens_for_papers(dummy_papers, vocab)
    calls_per_pass = batches
    passes = 3
    per_candidate = {
        "calls": calls_per_pass * passes,
        "papers": n * passes,
        "tokens_in": est_in * calls_per_pass * passes,
        "tokens_out": est_out * calls_per_pass * passes,
    }
    lines = []
    total_cost = 0.0
    for cand in config.candidates:
        cost = cost_from_tokens(per_candidate["tokens_in"], per_candidate["tokens_out"], cand)
        total_cost += cost
        lines.append(
            {
                "candidate_id": cand.id,
                "model": cand.model,
                "estimated_cost_usd": cost,
                "calls": per_candidate["calls"],
                "papers_scored": per_candidate["papers"],
            }
        )
    return {
        "sample_size": n,
        "candidates": len(config.candidates),
        "passes_per_candidate": passes,
        "total_estimated_cost_usd": round(total_cost, 2),
        "per_candidate": lines,
    }


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

STRATA = (
    ("ai_core", 100, ("cs.AI", "cs.CL", "cs.LG", "cs.NE")),
    ("cs_cv", 100, ("cs.CV",)),
    ("non_ai_cs", 100, ("cs.CR", "cs.NI", "cs.DB", "cs.SE", "cs.DS")),
    ("other", 100, ()),
)


def _categories_match(categories: list[str], prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    for cat in categories:
        c = (cat or "").strip()
        for p in prefixes:
            if c == p or c.startswith(p + "."):
                return True
    return False


def _parse_categories(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            return [raw]
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def load_eligible_papers(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT ci.id AS content_id,
               ci.title,
               COALESCE(pm.abstract, ci.summary, '') AS abstract,
               COALESCE(pm.categories, ci.categories_raw, '[]'::jsonb) AS categories
        FROM research_radar.content_items ci
        LEFT JOIN research_radar.paper_metadata pm ON pm.content_id = ci.id
        WHERE ci.status IN ('RELEVANT', 'ENRICHED', 'ENTITY_RESOLVED', 'SCORED', 'CANDIDATE')
          AND COALESCE(pm.abstract, ci.summary, '') <> ''
        """
    ).fetchall()
    out = []
    for r in rows:
        cats = _parse_categories(r["categories"])
        out.append(
            {
                "content_id": int(r["content_id"]),
                "title": r["title"] or "",
                "abstract": r["abstract"] or "",
                "categories": cats,
            }
        )
    return out


def select_stratified_sample(
    papers: list[dict],
    *,
    seed: int,
    strata: tuple[tuple[str, int, tuple[str, ...]], ...] = STRATA,
) -> tuple[list[dict], dict[str, int]]:
    rng = random.Random(seed)
    shuffled = list(papers)
    rng.shuffle(shuffled)

    used: set[int] = set()
    selected: list[dict] = []
    counts: dict[str, int] = {}

    for name, target, prefixes in strata:
        pool = []
        for p in shuffled:
            if p["content_id"] in used:
                continue
            cats = p.get("categories") or []
            if name == "other":
                if _categories_match(cats, ("cs.AI", "cs.CL", "cs.LG", "cs.NE", "cs.CV")):
                    continue
                if _categories_match(cats, ("cs.CR", "cs.NI", "cs.DB", "cs.SE", "cs.DS")):
                    continue
                pool.append(p)
            elif name == "ai_core" or name == "cs_cv" or name == "non_ai_cs":
                if _categories_match(cats, prefixes):
                    pool.append(p)
            else:
                pool.append(p)
        take = pool[:target]
        for p in take:
            used.add(p["content_id"])
            selected.append(p)
        counts[name] = len(take)

    return selected, counts


def persist_sample(conn, run_id: UUID, seed: int, sample: list[dict], prompt_version: str):
    conn.execute(
        """
        INSERT INTO research_radar.bakeoff_runs (run_id, sample_seed, sample_size, prompt_kind, prompt_version)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (str(run_id), seed, len(sample), BAKEOFF_PROMPT_KIND, prompt_version),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

PROBE_PAPER = {
    "content_id": 0,
    "title": "A General Attention Mechanism for Sequence Modelling",
    "abstract": (
        "We propose a new attention variant and evaluate on standard language modelling "
        "benchmarks. Results show modest improvements over baselines."
    ),
    "categories": ["cs.CL", "cs.LG"],
}


def reasoning_effort_for(candidate: BakeoffCandidate) -> str | None:
    if candidate.reasoning in ("disabled", "none", "off"):
        return None
    return "low"


def test_structured_output_gate(candidate: BakeoffCandidate, vocab: dict, *, client=None) -> dict:
    """Returns {passed: bool, json_valid_first_try: bool, error: str|None}."""
    require_api_key()
    if client is None:
        client = create_llm_client()
    schema = bakeoff_response_schema(vocab)
    user_prompt = build_bakeoff_user_prompt([PROBE_PAPER], vocab)
    try:
        result = call_chat_completion(
            client,
            model=candidate.model,
            system_prompt=BAKEOFF_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning_effort=reasoning_effort_for(candidate),
            temperature=0.0,
            max_retries=1,
            request_sleep=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "bakeoff_probe", "strict": True, "schema": schema},
            },
        )
        parse_bakeoff_batch(result["text"], {0})
        return {"passed": True, "json_valid_first_try": True, "error": None}
    except (LLMBatchError, BakeoffParseError, json.JSONDecodeError) as exc:
        return {"passed": False, "json_valid_first_try": False, "error": str(exc)}


def agreement_rate(rows_a: dict[int, dict], rows_b: dict[int, dict], *, fields: tuple[str, ...]) -> float:
    common = set(rows_a) & set(rows_b)
    if not common:
        return 0.0
    agree = 0
    for cid in common:
        a, b = rows_a[cid], rows_b[cid]
        if all(a.get(f) == b.get(f) for f in fields):
            agree += 1
    return agree / len(common)


def batch_stability_passes(rows_a: dict[int, dict], rows_b: dict[int, dict], threshold: float = 0.95) -> bool:
    fields = ("domain", "application_domains", "primary_audience", "ai_relevance")
    return agreement_rate(rows_a, rows_b, fields=fields) >= threshold


def self_consistency_passes(rows_1: dict[int, dict], rows_2: dict[int, dict], threshold: float = 0.95) -> bool:
    return batch_stability_passes(rows_1, rows_2, threshold=threshold)


# ---------------------------------------------------------------------------
# Scoring run (single candidate, one pass)
# ---------------------------------------------------------------------------


@dataclass
class BakeoffCallResult:
    content_id: int
    parsed: dict | None
    json_valid_first_try: bool
    retries: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    raw_response: str | None


def score_papers_batch(
    papers: list[dict],
    candidate: BakeoffCandidate,
    vocab: dict,
    *,
    client=None,
    batch_arrangement: str | None = None,
) -> list[BakeoffCallResult]:
    if client is None:
        client = create_llm_client()
    schema = bakeoff_response_schema(vocab)
    user_prompt = build_bakeoff_user_prompt(papers, vocab)
    expected_ids = {int(p["content_id"]) for p in papers}
    retries = 0
    json_valid_first_try = False
    t0 = time.monotonic()
    last_text = ""
    parsed: dict[int, dict] = {}

    for attempt in range(1, 4):
        try:
            result = call_chat_completion(
                client,
                model=candidate.model,
                system_prompt=BAKEOFF_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                reasoning_effort=reasoning_effort_for(candidate),
                temperature=0.0,
                max_retries=1,
                request_sleep=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "bakeoff_batch", "strict": True, "schema": schema},
                },
            )
            last_text = result["text"]
            parsed = parse_bakeoff_batch(last_text, expected_ids)
            json_valid_first_try = attempt == 1
            latency_ms = int((time.monotonic() - t0) * 1000)
            tokens_in = int(result["input_tokens"] or 0)
            tokens_out = int(result["output_tokens"] or 0)
            cost = cost_from_tokens(tokens_in, tokens_out, candidate)
            out = []
            for p in papers:
                pid = int(p["content_id"])
                row = parsed.get(pid)
                out.append(
                    BakeoffCallResult(
                        content_id=pid,
                        parsed=row,
                        json_valid_first_try=json_valid_first_try and row is not None,
                        retries=attempt - 1,
                        tokens_in=tokens_in // max(1, len(papers)),
                        tokens_out=tokens_out // max(1, len(papers)),
                        cost_usd=cost / max(1, len(papers)),
                        latency_ms=latency_ms // max(1, len(papers)),
                        raw_response=last_text if row is None else None,
                    )
                )
            return out
        except (LLMBatchError, BakeoffParseError) as exc:
            retries = attempt
            if attempt >= 3:
                latency_ms = int((time.monotonic() - t0) * 1000)
                return [
                    BakeoffCallResult(
                        content_id=int(p["content_id"]),
                        parsed=None,
                        json_valid_first_try=False,
                        retries=retries,
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms // max(1, len(papers)),
                        raw_response=str(exc),
                    )
                    for p in papers
                ]
            time.sleep(min(30.0, 2 ** (attempt - 1)))
    return []


def shuffle_batch_arrangement(papers: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = list(papers)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Metrics (report)
# ---------------------------------------------------------------------------


def is_force_fit(model_apps: list[str] | None, human_general: bool) -> bool:
    """Model assigned a specific sector but human said general-method."""
    if human_general:
        if not model_apps:
            return False
        return not is_general_method(model_apps)
    return False


def compute_force_fit_rate(
    model_rows: dict[int, dict],
    human_labels: dict[int, dict],
) -> float:
    n = 0
    hits = 0
    for cid, human in human_labels.items():
        if cid not in model_rows or model_rows[cid] is None:
            continue
        n += 1
        human_gm = bool(human.get("is_general_method"))
        model_apps = model_rows[cid].get("application_domains") or []
        if is_force_fit(model_apps, human_gm):
            hits += 1
    return (hits / n) if n else 0.0


def compute_accuracy(
    model_rows: dict[int, dict],
    human_labels: dict[int, dict],
    *,
    field: str = "domain",
) -> float:
    n = 0
    hits = 0
    for cid, human in human_labels.items():
        if cid not in model_rows or model_rows[cid] is None:
            continue
        n += 1
        if model_rows[cid].get(field) == human.get(field):
            hits += 1
    return (hits / n) if n else 0.0


def compute_general_method_rate(model_rows: dict[int, dict]) -> float:
    n = len(model_rows)
    if not n:
        return 0.0
    hits = sum(1 for r in model_rows.values() if r and is_general_method(r.get("application_domains")))
    return hits / n


def compute_valid_json_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("json_valid_first_try")) / len(results)


def inter_labeller_agreement(labels: list[dict], field: str = "domain") -> float:
    """Pairwise agreement across labellers for the same content_id."""
    by_cid: dict[int, list[str]] = {}
    for row in labels:
        cid = int(row["content_id"])
        val = row.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        by_cid.setdefault(cid, []).append(str(val))
    if not by_cid:
        return 0.0
    scores = []
    for vals in by_cid.values():
        if len(vals) < 2:
            continue
        agree = sum(1 for i in range(len(vals)) for j in range(i + 1, len(vals)) if vals[i] == vals[j])
        pairs = len(vals) * (len(vals) - 1) // 2
        scores.append(agree / pairs if pairs else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def group_disagreements_by_pattern(disagreements: list[dict]) -> dict[str, list[dict]]:
    """Group human reasoning by failure pattern keyword buckets."""
    buckets: dict[str, list[dict]] = {
        "force_fitting": [],
        "domain_mismatch": [],
        "audience_mismatch": [],
        "other": [],
    }
    for row in disagreements:
        reason = (row.get("reasoning") or "").lower()
        if "general" in reason or "force" in reason or "sector" in reason:
            buckets["force_fitting"].append(row)
        elif "domain" in reason:
            buckets["domain_mismatch"].append(row)
        elif "audience" in reason:
            buckets["audience_mismatch"].append(row)
        else:
            buckets["other"].append(row)
    return buckets


def new_run_id() -> UUID:
    return uuid4()
