# AGENTS.md

Research repository for a Master's thesis on news-driven liquidity dynamics in WTI crude oil futures. Read this file before doing any work here.

## Project overview

Master's thesis in Artificial Intelligence at Radboud University, supervised by Dr. Lejla Batina, hosted at Hammer Market Intelligence.

**Title**: "News-Driven Liquidity Dynamics in WTI Crude Oil Futures: A Channel-Decomposition Approach"

**Deadline**: submission targeted for late July 2026.

**Research questions**:

- **RQ1**: At what temporal lag do news events have their strongest impact on WTI liquidity?
- **RQ2**: Is there directional asymmetry in the response (bearish vs bullish news of comparable magnitude)?
- **RQ3** (deferred to future work): cross-commodity spillovers.

**Methodology in one paragraph**: two phases. Phase 1 uses FinBERT sentiment on regex-filtered articles (`body_valid=1`) and closes with the classical statistical models: contemporaneous OLS, distributed-lag OLS, VAR/IRF, and a headline-bias experiment. Phase 2 replaces FinBERT with Claude Haiku v2 structured extraction (sentiment, magnitude, certainty, three channel scores: `supply_impact`, `demand_impact`, `risk_premium`, event type, time horizon, entities, `usable` flag) and does the deep-learning modeling: it begins with TFT v1 (trained on the Phase 1 corpus with the initial v1 schema) and, after an inter-model calibration finding motivates the channel decomposition, culminates in TFT v2, the canonical model reported in the thesis.

## Canonical sources of truth

Do not infer from memory; verify against these files. Exact paths:

- `05_reports/development-decisions/project_logbook.md` — operational record of runs, data decisions, numerical results. Bullet format.
- `05_reports/development-decisions/thesis_decisions_log.md` — methodological rationale for each significant decision.
- `05_reports/development-decisions/TFTv2-specs.md` — locked TFT v2 design specification.
- `05_reports/development-decisions/project_plan.md` — overall plan.
- `05_reports/v2_training_results.md` — per-run record of TFT v2 experiments with configurations and headline metrics. Canonical source for all v2 numbers.
- `05_reports/thesis/draft/0-outline.md` — thesis outline with figures and tables registry.
- `05_reports/thesis/draft/3-methods.md` — Chapter 3 draft (Methods).
- `05_reports/thesis/draft/4-experiments.md` — Chapter 4 draft (Experiments and Results).

For any TFT v2 result (metrics, feature importance, attention patterns), read `05_reports/v2_training_results.md` or the artifacts in `04_outputs/experiment_tracking/TFT/TFTv2/tft_v2_outputs/v2.2/`. Numbers are not duplicated here because they go stale.

## Repository map

```
news-driven-liquidity/
├── 01_data/
│   ├── wti_thesis.db          # SQLite database, the analytical core
│   ├── raw/                   # Original inputs (macro/, news/, price/), never modified
│   ├── processed/, features/  # Intermediate data
│   └── models/                # TFT v1 checkpoint (tft_wti.ckpt) + training_dataset.pkl
├── 02_notebooks/              # Numbered pipeline, run order = number order
│   ├── 00_setup_database … 05_alignment      # Data acquisition and alignment
│   ├── 06_llm_features        # Haiku v2 extraction (current)
│   ├── 07_headline_bias … 09_parallel_features
│   ├── 10_tft_analysis, 11_calibration, 12_entity_normalization
│   ├── 13_tft_v2_training     # Canonical v2 training (runs on Colab)
│   └── deprecated_notebooks/  # Superseded: 06_finbert_sentiment (Phase 1), 06_llm_features_v1
├── 03_src/
│   ├── tft/config.py          # Locked constants + entity column mapping + verify_against_db()
│   ├── tft/config-for-colab.py# Colab copy of the above; keep in sync manually
│   ├── nlp/llm_features.py    # CANONICAL_ENTITIES list + extraction code
│   ├── acquisition/           # gdelt_client, yfinance_client, eia_downloader
│   ├── features/, models/     # Event windows, OLS baseline
│   └── config/                # Project config
├── 04_outputs/
│   ├── figures/, tables/, models/
│   └── experiment_tracking/TFT/TFTv2/tft_v2_outputs/{v2.0,v2.1,v2.2}  # Ablation runs (60/20/20); v2.2 = canonical "TFT v2"
├── 05_reports/                # See canonical sources above; also month1/, literature/, calibration/
├── old_stuff/                 # Pre-thesis exploration, ignore unless asked
└── requirements.txt
```

## Environment

- Python 3.13 venv at `.venv/` inside this repo (`.venv/bin/python`). There is no shared venv outside the repo.
- Local dev on Apple Silicon; TFT training runs on Google Colab (T4 GPU) via `13_tft_v2_training.ipynb` and `config-for-colab.py`.
- API keys in `.env` at repo root (anthropic key for Haiku extraction).
- Key installed libraries: torch 2.11, lightning 2.6 (import as `lightning.pytorch`), pytorch-forecasting 1.7.0, transformers, anthropic. Colab versions may differ; for the reproducibility appendix, verify versions from the training run logs, not from the local venv.

## Data

**Time grid**: May 2024 to May 2026, 11,232 hourly rows in `market_context`, indexed `time_idx = [0, 11231]`. War onset (28 Feb 2026 attack) at `WAR_ONSET_IDX = 10056`, inside the test set by design.

**Database**: `01_data/wti_thesis.db` (SQLite). Modeling tables:

- `articles` — raw scraped articles (GDELT headlines + BeautifulSoup bodies), `body_valid` is the Phase 1 regex filter.
- `llm_features` — Haiku v2 structured extraction per article. Holds `usable` and the derived `usable_strict` (= usable AND at least one non-zero channel score). **`usable_strict` lives only here.**
- `article_entities` — normalized entity mentions (71 canonical entities from `03_src/nlp/llm_features.py`), source of the entity flag columns for TFT v2.
- `market_context` — hourly grid: liquidity targets (`log_volume`, `amihud`, `price_range`), `log_return`, macro (`dxy`, `vix`), EIA fundamentals.
- `liquidity` — one row per article, joins article + LLM features + market context at the assigned hour. Does NOT contain `usable_strict`; JOIN `llm_features` on `article_id` to filter by it.
- `eia_events` — weekly EIA inventory releases.

Supporting tables (`raw_entity_counts`, `calibration_sample`, `llm_features_v1_backup`, `opec_events`) are not part of the modeling pipeline.

Inspect schemas with `sqlite3 01_data/wti_thesis.db '.schema <table>'` instead of trusting any written copy.

## Locked constants

Defined in `03_src/tft/config.py`; import from there, never hardcode:

```python
TOTAL_HOURS = 11232
ENCODER_LENGTH = 48
MAX_PREDICTION_LENGTH = 12
TRAIN_END = 6739         # 60%
VAL_START = 6787         # +48h buffer
VAL_END = 9014           # 20%
TEST_START = 9062        # +48h buffer
TEST_END = 11232         # 20%
WAR_ONSET_IDX = 10056    # "2026-03-01 23:00:00+00:00"
```

Do not modify without documenting in `thesis_decisions_log.md`. Call `config.verify_against_db()` at the top of any training or evaluation notebook; it asserts the DB row count still matches the locked split.

`config.py` also provides `entity_to_column_name()` and the `ENTITY_COL_MAP` / `COL_TO_ENTITY` dicts for the `ent_*` feature columns.

## TFT v2 canonical model

The reported model is variant **v2.2** (`04_outputs/experiment_tracking/TFT/TFTv2/tft_v2_outputs/v2.2/`): trained on `usable_strict=1` articles, multi-target (`log_volume`, `amihud`, `price_range`), multi-horizon `[1, 3, 6, 12]`h, encoder length 48h, with 71 entity flag columns. Variants v2.0 and v2.1 exist only for the ablation (Appendix C). Full configuration in `TFTv2-specs.md`; results in `v2_training_results.md`.

## Working rules (non-negotiable)

### Source verification

Every numerical claim or methodological specific in any output must trace to a notebook cell output, a SQL query against `wti_thesis.db`, a source CSV, or a canonical `.md` in `05_reports/`. Never infer numbers from memory. Never invent references to files, sections, or papers; if unsure whether a reference is real, ask before including it.

### Correct filter for v2

The canonical filter for TFT v2 training and any v2-related query is `usable_strict=1`, not `usable=1`. Using `usable=1` silently changes the sample (~11.7k vs ~11.4k articles) and invalidates comparisons against reported results.

### Lightning namespace

Use `lightning.pytorch`, never `pytorch_lightning`. Both packages are installed (pytorch-forecasting pulls in the legacy one), so the wrong import works until it fails silently: mixing namespaces makes `isinstance` checks on callbacks/trainers fail without error messages. Correct pattern:

```python
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
```

### Temporal alignment (causality)

Align articles to trading hours with `dt.ceil`, never `dt.round`. Off-hours articles are forward-assigned to the next trading hour, not discarded. `dt.round` would assign some articles to a past hour and introduce look-ahead bias.

### Documentation style

- Logbook entries: condensed bullets with subheaders, matching existing structure. Never dense paragraphs.
- Chapter drafts contain only verified numbers; referenced sections must exist with the described content.
- All code, comments, notebook prints, and markdown in English. Chat may be Spanish or English.
- Prose: no em dashes; use commas or parentheses.

## Common pitfalls

- Importing `pytorch_lightning` instead of `lightning.pytorch` (both are installed; see above).
- Hardcoding split indices instead of importing from `03_src/tft/config.py`.
- Filtering on `usable=1` when `usable_strict=1` is required, or looking for `usable_strict` in the `liquidity` table (it is only in `llm_features`).
- Editing `config.py` without also updating `config-for-colab.py` (they are synced manually).
- Trusting stale copies of results or schemas in docs instead of the DB and `v2_training_results.md`.
- Confusing report paths: decision logs live under `05_reports/development-decisions/`, thesis drafts under `05_reports/thesis/draft/`.

## When to update this file

Update when: the canonical v2 model changes, a locked constant changes (with a decisions-log entry), a canonical file moves or is added, a working rule changes, or the repo structure changes. Do not update for chapter prose revisions, notebook bug fixes, or anything that does not change the project's structure or rules. Keep changes minimal; replace inaccurate sections rather than appending contradictory notes.
