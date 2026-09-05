# Active — 2026-09-05

## Current focus

Scoring v3 implementation complete on branch `cursor/scoring-v3-all-bbc7`.

## Changes shipped (code)

| # | Change | Status |
|---|---|---|
| 1 | New `classify` stage + `content_classifications` table | **DONE** (migration `sql/014_scoring_v3.sql` unapplied) |
| 2 | Remove gate-collapse from screen + quality prompts | **DONE** (v3 prompt versions) |
| 3 | Drop `ai_relevance` from screen ranking mean | **DONE** |
| 4 | Independence emits `paper_kind` (`independence-v2`) | **DONE** |

## Blockers

- Human must apply `sql/014_scoring_v3.sql` before running `classify` or independence v2 writes.
- Paid stages require `--allow-paid` and `OPENROUTER_API_KEY`.

## Verified

- `python -m pytest tests/test_scoring_v2.py tests/test_classify.py -q` → 33 passed

## Next

1. Apply migration 014
2. `./scripts/run_stage.sh classify --dry-run --allow-paid` then live classify on a limit
3. Re-screen with v3 prompts if stored v2 screen scores need refresh
