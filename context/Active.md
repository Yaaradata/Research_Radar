# Active — 2026-09-05

## Current focus

Cursor agent definitions installed under `.cursor/agents/` and `.cursor/rules/`.
Beginning Scoring v3 changes (uploaded brief).

## Branch

TBD — create `cursor/scoring-v3-*-bbc7` per change.

## Agent layout

```text
.cursor/agents/engineer.md   — implementation
.cursor/agents/scout.md      — read-only facts
.cursor/agents/verifier.md   — artifact validation
.cursor/rules/architect.mdc  — parent orchestration
.cursor/rules/research-radar.mdc — project traps and stage order
```

## Scoring v3 queue (one change at a time)

1. **NEXT:** Remove gate-collapse instruction from screen + quality prompts
2. Drop `ai_relevance` from screen ranking mean in `select_gated_content_ids()`
3. Independence prompt emits `paper_kind`
4. Migration `content_classifications` + new `classify` stage

## Blockers

None for prompt/ranking changes (no DDL). Classify pass blocked on migration apply by human.

## Verified

- Repo structure: `src/research_radar/`, `sql/001`–`013`, tests present
- Gate-collapse text at `semantic_scoring.py` lines ~1176–1178 (quality) and screen prompt
