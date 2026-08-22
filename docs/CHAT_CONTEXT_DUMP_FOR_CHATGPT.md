# TheNeural AI Research Radar — Full Chat Context Dump (for ChatGPT)

**Purpose of this document:** Paste into ChatGPT so another model has full project + conversation context: product intent, code that was written, every major user request, bugs hit, fixes applied, current DB results, and open issues.

**Project path:** `/home/ubuntu/Research_Radar1`  
**Date of dump:** 21 August 2026  
**Related sibling project (DO NOT MODIFY):** Newsletter5 (existing Inoreader/newsletter pipeline)

---

## 0. One-line product

Build a **separate** research corpus pipeline (not newsletter selection) that ingests AI papers from Inoreader, enriches them (arXiv + affiliations), matches a **30-org watchlist** as a **boost not a gate**, scores them deterministically, and stores reusable candidates in Postgres schema `research_radar` on the **same RDS** as Newsletter5.

---

## 1. Hard product rules (from user’s architecture brief)

1. **Separate from Newsletter5** — reuse infra/keys conceptually, but **never edit Newsletter5**.
2. Primary scope: AI / LLMs / agents / ML / multimodal / evaluation / safety / RAG / systems / applied AI / HAI / AI product.
3. **Notable orgs/people are ranking signals, not gates.** Strong paper from unknown author must not disappear.
4. Audiences: (A) tech/product professionals, (B) learners/students — opportunities tagged separately.
5. Affiliation resolution waterfall:
   1. Explicit affiliation text in paper  
   2. Corporate/institutional email domain  
   3. OpenAlex paper → author → institution (backup)  
   4–5. Expensive web resolution only when relevant + unresolved + high potential (MVP did not fully implement 4–5).
6. Domain→org mappings live in **organisation table**, not hardcoded in app logic.
7. **Critical:** paper affiliation ≠ current employer. Store relationships with evidence/provenance. LLM is **not** factual affiliation evidence.
8. Keep score **dimensions separate** (not one permanent newsletter score). Intrinsic candidate score is a blend for labeling only.
9. Store `published_at` for date-based analysis.
10. Watchlist priority guidance: 10 strong boost, 8 material, 6 secondary, 3 specialist — never override intrinsic quality alone.

---

## 2. What the user asked for (chronological, Research Radar thread)

> Note: early unrelated turns in same transcript (SSH key, newsletter selection prompts, SQL restore, git commands, tmux) are **not** part of Research Radar. Radar work starts ~20 Aug 2026.

### Setup / scaffolding
- Write scripts in `Research_Radar1`, reuse Inoreader/RDS keys from Newsletter5 conceptually, **no Newsletter5 changes**.
- Attached full architecture brief (ingest → enrich → entities → score → corpus).
- Create `.env` in Research_Radar1 with needed keys.
- Asked where org/author importance lists live → `sql/002_seed_watchlists.sql`.
- Wanted **staged runs** (not one-shot) to see progress/logs.
- Removed/disliked `bootstrap_from_newsletter5.sh` because env is local in Research_Radar1.

### Inoreader targeting
- Asked which Inoreader folder is used.
- Sent screenshot of folder **`10 RESEARCH - T3 Firehose`** and said: **pipeline must see this folder only**.
- Confused when ingest returned only ~10 items (Inoreader page size / limit); wanted last 7 days of that folder.
- Hit bash error when stream name had spaces (unquoted env) → fixed by quoting `INOREADER_STREAM=...`.
- Hit expired Inoreader tokens → re-auth via Newsletter5 oauth helper, copy tokens into Research_Radar1 `.env`.

### OpenAlex
- Asked if OpenAlex is needed and its role → backup affiliation when email/affiliation miss.
- Confirmed code already had OpenAlex path but no key initially.
- Provided OpenAlex API key + mailto `commontech@theneural.ai` to add to `.env`.

### Watchlist + dating (major product change)
- Replace seed watchlist with **30 organisations of interest** (full table provided by user: org_id, name, aliases, type, domains, priority 10/8/6/3, rationale, example signal, tags, evidence sources).
- Pipeline must run for **last 7 days**.
- DB must store article dates for analysis.
- Later asked to verify all 30 orgs were added properly (they were: 30 active, correct priority mix).

### Running last 7 days
- Asked for exact commands + explanation of rules/steps.
- Confirmed ingest pulled ~2149 items for 7 days from firehose folder.
- Relevance: ENRICHED path for ≥5.0 AI relevance → ~1351 kept, ~798 REJECTED.

### Enrich stage pain (major ops issues)
- Massive arXiv **429 Too Many Requests** during enrich.
- Asked if rerun resumes remaining or restarts from first.
- Asked what `id` and emails in enrich logs mean.
- Confused that logs showed `id` starting from 1 again after a fix/rerun — thought resume was broken.
  - Reality: content_ids are sequential DB ids; enrich **skips already ENRICHED**, but logs still print ids as workers pick remaining work; early run also lost progress when commits were only at end / crash.
- Asked to **parallelize enrich with 4 workers**, **commit every 10 papers**, stop slow sequential run (~100/10min).
- Enrich eventually completed: **ENRICHED=1351**.

### Entities + score parallelization
- Asked if entities works with 4 workers → initially enrich-only; then asked to make **entities and score also 4 parallel workers**.
- Entities FK/run error fixed by committing `pipeline_runs` immediately in `start_run()`.
- OpenAlex during entities: flood of **429 Too Many Requests** under 4 workers (non-fatal; papers finish with `orgs=0`).
- OpenAlex query hardening (sanitize special chars; prefer `title.search` filter) + later shared OpenAlex throttle (`OPENALEX_REQUEST_SLEEP`).
- Entities completed: all 1351 → `ENTITY_RESOLVED`. Re-run showed `processing 0 items`.
- Score completed: **CANDIDATE=2, SCORED=1349, REJECTED=798**.
- User reaction: “from the 1351, we got only 2 as top candidate” — explained threshold ≥6.0 + scoring weights + empty people watchlist + sparse org matches (169/1351 with any org link).

### This request
- Full context dump for ChatGPT including issues, user statements, and how code was written.

---

## 3. Repository layout (as built)

```text
/home/ubuntu/Research_Radar1/
  .env                    # local secrets (gitignored) — DO NOT commit
  .env.example
  .gitignore
  README.md
  docs/CHAT_CONTEXT_DUMP_FOR_CHATGPT.md   # this file
  sql/
    001_schema.sql        # research_radar schema + tables + v_candidates
    002_seed_watchlists.sql  # 30-org watchlist + topics (replaced earlier 20-org seed)
    003_org_watchlist_extend.sql
  src/research_radar/
    __init__.py
    pipeline.py           # main MVP pipeline (~1250 lines)
    query.py              # candidate queries (--days, --org, etc.)
  scripts/
    setup_db.sh           # apply SQL to existing RDS
    run_stage.sh          # staged CLI: ingest|relevance|enrich|entities|score|show|all
    run_pipeline.sh
    install_ec2.sh
    install_cron.sh
    golden_test.py
  logs/                   # stage logs timestamped
  .venv/
```

**Removed:** `scripts/bootstrap_from_newsletter5.sh` (user didn’t want Newsletter5 bootstrap).

---

## 4. Infrastructure / config (conceptual — redact secrets)

Uses **same Amazon RDS Postgres** as Newsletter5, schema **`research_radar`** (not newsletter schema).

Key `.env` settings (values redacted here):

```env
DATABASE_URL=postgresql://...@...rds.../...
INOREADER_ACCESS_TOKEN=...
INOREADER_CLIENT_ID=...
INOREADER_CLIENT_SECRET=...
INOREADER_REFRESH_TOKEN=...
INOREADER_STREAM="user/-/label/10 RESEARCH - T3 Firehose"
INOREADER_LOOKBACK_DAYS=7
INOREADER_BATCH_SIZE=100
INOREADER_MAX_PAGES=50

OPENALEX_ENABLED=true
OPENALEX_API_KEY=<user-provided>
OPENALEX_MAILTO=commontech@theneural.ai
OPENALEX_REQUEST_SLEEP=1.2

ARXIV_WORKERS=4
PIPELINE_WORKERS=4
ARXIV_COMMIT_EVERY=10
ARXIV_REQUEST_SLEEP=1.0

MIN_AI_RELEVANCE_FOR_ENRICHMENT=5.0
MIN_INTRINSIC_CANDIDATE_SCORE=6.0
```

---

## 5. Pipeline stages (actual implemented behavior)

### Stage 1 — ingest
- Pull Inoreader stream `10 RESEARCH - T3 Firehose` only.
- Paginate with continuation + lookback cutoff (`INOREADER_LOOKBACK_DAYS=7`).
- Upsert into `research_radar.content_items` with `published_at`, title, summary, authors, categories, canonical_url, raw metadata.
- Latest successful 7-day pull: **~2149 items**.

### Stage 2 — relevance
- Deterministic keyword/category AI relevance scoring.
- Keep if score ≥ `MIN_AI_RELEVANCE_FOR_ENRICHMENT` (5.0) → status path toward enrichment.
- Else → `REJECTED`.
- Result: **~1351 relevant / ENRICHED path**, **~798 REJECTED**.

### Stage 3 — enrich
- For arXiv items: fetch Atom/API metadata (title, abstract, categories, doi, links).
- Parse HTML for emails/affiliations when available.
- Shared arXiv throttle + retries/backoff on 429.
- Parallel workers (`ARXIV_WORKERS=4`), commit every `ARXIV_COMMIT_EVERY=10`.
- Skip already `ENRICHED` on resume.
- Writes `paper_metadata` (+ enrichment fields).

### Stage 4 — entities
- Match watchlist orgs via:
  1. email domain → organisation domains
  2. affiliation text → aliases/names
  3. OpenAlex exact/near title fallback (optional)
- Insert `content_organisations` with evidence_type, evidence_text, confidence, relationship_type=`paper_author_affiliation`.
- People watchlist currently **empty** → always `people=0` in logs.
- Parallel `PIPELINE_WORKERS=4`, commit every 10.
- Result: **1351 ENTITY_RESOLVED**; only **169** papers got any org link; **103** strong org signal (≥8).

### Stage 5 — score
- Multi-dimension deterministic proxies in `content_scores`.
- Intrinsic blend (approx):
  - 30% ai_relevance
  - 15% technical_significance
  - 15% practical_applicability
  - 10% professional_value
  - 10% student_learning_value
  - 10% notable_org_signal
  - 10% notable_person_signal
- If `intrinsic_candidate_score >= 6.0` → status `CANDIDATE`, else `SCORED`.
- Also writes opportunity tags (notable_research, professional_learning, weekend_read, etc.).
- Latest result: **CANDIDATE=2**, **SCORED=1349**.

### Stage 6 — show / query
- `run_stage.sh show --top N` prints top candidates from view/query.
- `query.py` supports filtering by days/org etc.

---

## 6. Scoring math (why only 2 candidates)

From `intrinsic_scores()` in `pipeline.py`:
- Keyword heuristics for technical/practical/professional/learner/explainability.
- `novelty` stored as constant 5.0 but **not** in intrinsic blend.
- Org/person priorities come from matched watchlist rows (0 if none).
- With empty people list, person term always 0.
- Without org boost, strong papers cluster ~4–5.5.
- With org=10 + strong AI relevance, scores scrape ~5.9–6.3.

Observed distribution (content_scores):
- median ~4.1, p95 ~5.15, p99 ~5.74, max 6.29
- ≥6.0: **2**
- ≥5.5: **41**
- ≥5.0: **229**

Two CANDIDATEs (both Meta-affiliated):
1. Robust Checkpoint Selection for Multimodal LLMs… intrinsic≈6.29  
2. AI Research Preference Models… intrinsic≈6.14  

If user wants more candidates without changing formula much: set `MIN_INTRINSIC_CANDIDATE_SCORE=5.5` and re-run score.

---

## 7. Organisation watchlist (30 orgs)

Stored in `research_radar.organisations` via `sql/002_seed_watchlists.sql`.

Fields include: `org_key` / organisation_id, display name, aliases[], domains[], org_type, priority, rationale, watchlist_tags, evidence_sources, active flag.

Priority bands present: 10 / 8 / 6 / 3 across frontier labs, Big Tech, infra, universities, startups, gov/standards (exact 30-row list as user provided in chat; DB verified 30 active).

People table watchlist not seeded yet.

---

## 8. All major issues encountered (and fixes)

| # | Issue | Symptom | Fix / outcome |
|---|--------|---------|---------------|
| 1 | Don’t touch Newsletter5 | Constraint | Work only in Research_Radar1 |
| 2 | Unwanted bootstrap script | User rejected Newsletter5 bootstrap | Removed/stopped using bootstrap script; local `.env` |
| 3 | Stream with spaces | bash: `RESEARCH: command not found` | Quote `INOREADER_STREAM="user/-/label/10 RESEARCH - T3 Firehose"` |
| 4 | Wrong/default Inoreader stream | Not the firehose folder | Pointed to label `10 RESEARCH - T3 Firehose` |
| 5 | Only ~10 items ingested | Page size / limit confusion | Pagination + lookback days; full 7-day pull ~2149 |
| 6 | Expired Inoreader OAuth | Auth failures | Refresh via Newsletter5 oauth helper; copy tokens |
| 7 | `feedparser.loads` bug | AttributeError | Use `feedparser.parse` |
| 8 | OpenAlex missing keys | Fallback weak/unpolite | Added API key + mailto |
| 9 | OpenAlex 400 on special titles | Bad query chars | Sanitize title; safer `title.search` filter |
| 10 | arXiv 429 during enrich | Rate limit flood | Sleep/retry, shared throttle, batching, 4 workers carefully |
| 11 | Enrich progress lost on interrupt | Commit only at end | Commit every N items; skip ENRICHED on resume |
| 12 | User thought resume broken because ids start at 1 | Logs show low content_ids | Explained: sequential IDs; skip ENRICHED; workers may process leftover low IDs |
| 13 | Enrich too slow | ~100 papers / 10 min | Parallel 4 workers + commit every 10 |
| 14 | Entities FK / pipeline_runs | Workers insert before run row visible | `start_run()` commits immediately |
| 15 | OpenAlex 429 under entity parallelism | Many `orgs=0` | Non-fatal; later shared OpenAlex throttle; optional disable |
| 16 | Entities re-run after complete | `processing 0 items` | Correct — all already ENTITY_RESOLVED |
| 17 | Only 2 CANDIDATES | Threshold + weights | Expected with ≥6.0; explain distribution; optional lower threshold |
| 18 | People watchlist empty | Always people=0 | Not seeded yet — residual gap |
| 19 | README still mentions older “20 orgs” in places | Doc drift | Seed is 30 orgs; README partially outdated |

---

## 9. How to run (current recommended)

```bash
cd /home/ubuntu/Research_Radar1
source .venv/bin/activate

./scripts/setup_db.sh                 # once / after SQL changes

./scripts/run_stage.sh ingest         # 7-day firehose
./scripts/run_stage.sh relevance
./scripts/run_stage.sh enrich
./scripts/run_stage.sh entities
./scripts/run_stage.sh score
./scripts/run_stage.sh show --top 20
```

Logs go under `logs/YYYYMMDD-HHMMSS-stage.log`.

---

## 10. Current DB state (after full 7-day run)

```text
REJECTED:          798
SCORED:           1349
CANDIDATE:           2
ENTITY_RESOLVED:  (all enriched set processed; statuses moved to SCORED/CANDIDATE)
ENRICHED input set: 1351 papers scored
Papers with any org link: 169
Strong org signal (≥8): 103
People watchlist matches: 0 (watchlist empty)
```

---

## 11. Code design notes (for continuing work)

- Main logic monolith: `src/research_radar/pipeline.py` (ingest, relevance, enrich, entities, score, show, HTTP helpers, throttles).
- Parallelism via ThreadPoolExecutor for enrich/entities/score.
- Shared locks for arXiv and OpenAlex request pacing.
- Status machine roughly: ingested → RELEVANT/REJECTED → ENRICHED → ENTITY_RESOLVED → SCORED/CANDIDATE (exact transitions in code).
- Evidence-backed org links required; boost-only.
- Query helpers in `query.py` for analysis by date/org/score.
- Golden test exists (`scripts/golden_test.py`) including a specific arXiv id from brief.

---

## 12. Open issues / next work the user may want

1. **Tune candidate volume** — lower `MIN_INTRINSIC_CANDIDATE_SCORE` to 5.5 (≈41 candidates) or reweight org/person terms.
2. **Seed people watchlist** (still empty).
3. **OpenAlex reliability** — throttle is in place; may still want sequential OpenAlex-only queue or disable for MVP speed.
4. **Re-entity papers with orgs=0** after OpenAlex cool-down (need status reset; currently ENTITY_RESOLVED/SCORED won’t auto-retry OpenAlex).
5. **Affiliation steps 4–5** (official sources / web) not fully built.
6. **Doc sync** — README still partially reflects older seed size / install paths.
7. **Do not commit `.env` secrets**.
8. Product evaluation: is intrinsic threshold too strict for “research radar corpus” vs “top notable shortlist”?

---

## 13. User’s explicit preferences / tone of requests

- Wants **clear run commands** and **step-by-step** visibility.
- Prefers staged pipeline over opaque all-in-one.
- Keeps secrets in local `.env`, not bootstrap from Newsletter5.
- Targets one specific Inoreader folder only.
- Cares about **dates** (7-day window + published_at in DB).
- Pushes for **speed** (parallel workers, frequent commits) when rate limits allow.
- Asks diagnostic questions from terminal logs frequently (what does this mean?).
- Surprised by low candidate count (2/1351) — needs scoring explanation / possible threshold change.

---

## 14. Suggested prompt starter for ChatGPT

Paste this whole dump, then ask e.g.:

> Given this Research Radar context, propose a better scoring threshold/weights so we get a useful weekly shortlist (~20–50) without turning org fame into a hard gate. Also outline how to re-run OpenAlex safely for papers that currently have orgs=0, and how to seed a people watchlist consistent with the affiliation evidence rules.

---

## 15. Security note for paste destinations

This dump intentionally **redacts** live API tokens / DB passwords. If you also paste `.env`, remove secrets first. An OpenAlex key and Inoreader tokens were shared in chat historically — rotate if this dump is shared broadly.
