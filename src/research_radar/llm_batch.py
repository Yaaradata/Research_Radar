"""Shared OpenRouter batch-call plumbing for scoring v2.

Generic HTTP/JSON mechanics only (retry-with-backoff, fence stripping, random
batch composition). Domain prompts, schemas and parsing stay in
semantic_scoring.py (quality) and independence.py (independence) — those two
modules must never share a prompt or a call. This module does not know what a
"paper" is.
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Any, Callable

# Shared concurrency dial for pass 1 (screen), pass 2 (full) and independence.
# `enrich` deliberately does NOT use this — arXiv's ~3s-between-requests limit
# is a separate, hardcoded throttle (see pipeline.py's _arxiv_throttle) and
# must never be widened by this knob.
SCORING_CONCURRENCY = int(os.getenv("SCORING_CONCURRENCY", "4"))


class LLMBatchError(RuntimeError):
    def __init__(self, message: str, *, status: str = "ERROR", retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class AdaptiveConcurrencyGate:
    """Shared concurrency limiter for a single scoring run.

    Bounds how many workers may be mid-API-call at once (up to `max_concurrency`
    threads still exist in the pool, but only `_ceiling` may hold the gate at a
    time). Every reported 429 permanently drops the ceiling by one (floor 1) for
    the rest of the run, so a burst of rate-limiting backs the whole run off
    instead of every worker hammering the API independently. Thread-safe.
    """

    def __init__(self, max_concurrency: int):
        self._lock = threading.Lock()
        self._ceiling = max(1, int(max_concurrency))
        self._sem = threading.Semaphore(self._ceiling)

    def acquire(self):
        self._sem.acquire()

    def release(self):
        self._sem.release()

    def report_rate_limited(self):
        """Call when a worker observes a 429. Idempotent-safe to call often."""
        with self._lock:
            if self._ceiling > 1:
                self._ceiling -= 1
                # Permanently remove one permit — acquire without a matching
                # release so the pool's effective concurrency stays reduced.
                self._sem.acquire(blocking=False)

    @property
    def ceiling(self) -> int:
        return self._ceiling


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    inp = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    out = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    return int(inp or 0), int(out or 0)


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        try:
            if int(status) == 429:
                return True
        except (TypeError, ValueError):
            pass
    name = type(exc).__name__.lower()
    return "ratelimit" in name or "rate_limit" in name


def _is_retryable(exc: Exception) -> bool:
    if _is_rate_limit_error(exc):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        try:
            if int(status) >= 500:
                return True
        except (TypeError, ValueError):
            pass
    name = type(exc).__name__.lower()
    return any(
        x in name
        for x in ("timeout", "apiconnection", "internalserver", "serviceunavailable")
    )


def call_chat_completion(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None,
    temperature: float = 0.2,
    max_retries: int = 3,
    request_sleep: float = 0.2,
    response_format: dict | None = None,
    on_rate_limited: Callable[[], None] | None = None,
) -> dict:
    """One logical chat.completions.create call with HTTP-level retry/backoff.

    `reasoning_effort=None` omits the `reasoning` extra_body entirely (used by
    the screen pass, which runs with reasoning disabled — it is a cheap
    non-reasoning model, not a reasoning model told to think less).

    `response_format` is passed straight through to the SDK — pass an OpenAI
    strict json_schema block (object root) to get schema-level validation on
    the API side instead of relying only on parsing after the fact.

    `on_rate_limited`, if given, is called synchronously the instant a 429 is
    observed on any attempt (whether or not this call ultimately succeeds on a
    later attempt) — the caller uses this to shrink a shared concurrency gate.

    Returns {"text", "input_tokens", "output_tokens", "response_id"}.
    Raises LLMBatchError on unrecoverable failure (never returns partial junk).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if request_sleep > 0:
                time.sleep(request_sleep)
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            if reasoning_effort is not None:
                kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            inp, out = _usage_tokens(response)
            return {
                "text": text,
                "input_tokens": inp,
                "output_tokens": out,
                "response_id": getattr(response, "id", None),
            }
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc) and on_rate_limited is not None:
                on_rate_limited()
            if _is_retryable(exc) and attempt < max_retries:
                wait = min(60.0, (2 ** (attempt - 1)) * 1.0)
                time.sleep(wait)
                continue
            raise LLMBatchError(str(exc), status="ERROR", retryable=False) from exc
    raise LLMBatchError(str(last_exc or "LLM call failed"), status="ERROR")


def strip_json_fences(text: str) -> str:
    """Model is prompt-instructed to return bare JSON; strip ```/```json fences if present anyway."""
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def cost_summary_line(label: str, n: int, calls: int, cost: float) -> str:
    """One line of the standard Pass 1 / Pass 2 / Independence cost block."""
    return f"{label:<15}{n} papers, {calls} calls, ${cost:.2f}"


def print_scoring_cost_summary(*, pass1: dict, pass2: dict, independence: dict):
    """pass1/pass2/independence: {"n": int, "calls": int, "cost": float}."""
    total_cost = float(pass1["cost"]) + float(pass2["cost"]) + float(independence["cost"])
    total_n = int(pass1["n"]) + int(pass2["n"]) + int(independence["n"])
    avg = (total_cost / total_n) if total_n else 0.0
    print("\nSCORING COST SUMMARY")
    print(cost_summary_line("Pass 1:", pass1["n"], pass1["calls"], pass1["cost"]))
    print(cost_summary_line("Pass 2:", pass2["n"], pass2["calls"], pass2["cost"]))
    print(cost_summary_line("Independence:", independence["n"], independence["calls"], independence["cost"]))
    print(f"TOTAL:         ${total_cost:.2f}   (avg ${avg:.4f}/paper)")


def random_batches(items: list, batch_size: int, *, rng: random.Random | None = None) -> list[list]:
    """Shuffle then chunk — random composition AND random order within each batch.

    Never group by date/category/prior score: the model would calibrate to the
    batch instead of the absolute scale. Uses a fresh unseeded RNG by default so
    repeated calls on the same input produce different groupings.
    """
    rng = rng or random.Random()
    shuffled = list(items)
    rng.shuffle(shuffled)
    size = max(1, batch_size)
    return [shuffled[i : i + size] for i in range(0, len(shuffled), size)]
