# Research Radar — agent rules

Standing rules for agents working in this repo. The always-applied Cursor rule
is `.cursor/rules/research-radar.mdc` (same content, IDE-integrated).

## Pipeline

```text
ingest → relevance → enrich → entities → [affiliation-gpt] → classify → screen → semantic-score → independence → final-score
```

Parallel (not in `all`): `arxiv-backfill`, `topics`, `corpus-search`.

**Cron `all`:** ingest → relevance → enrich → entities only. Never add a paid stage to `all`.

**Paid stages:** `affiliation-gpt`, `classify`, `screen`, `semantic-score`, `independence`, `topics` — require `--allow-paid`.

## Classification vs quality

- **Classification / screening** — fixed enum, reasoning **off**. Includes classify, screen gate (`ai_relevance`), topics domain/application tagging in bake-off.
- **Quality pass** — `semantic-score`, reasoning **on**. Never sees affiliations, authors, or organisations.

## Bake-off

- Candidates in `config/bakeoff_models.yaml`.
- Winner = accuracy vs human labels, subject to structured-output + batch-stability gates.
- Human labelling workbooks must not show model outputs on labelling sheets.
- Migrations `015`–`016` unapplied until human runs `scripts/setup_db.sh`.

## Engineering rules

- New stages select a **status set**, never a single status.
- Prompts are versioned; same version string must not change meaning.
- No DDL by agents — numbered `sql/` migrations only.
- Claims must ground to literal source text.
- Prefer NULL to plausible defaults.
