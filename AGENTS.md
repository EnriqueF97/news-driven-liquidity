# AGENTS.md

This is a research repository for a Master's thesis on news-driven liquidity dynamics in WTI crude oil futures. Read this file before doing any work on the project.

## Project overview

Master's thesis in Artificial Intelligence at Radboud University, supervised by Dr. Lejla Batina, hosted at Hammer Market Intelligence. The thesis investigates the causal relationship between news events and WTI crude oil futures liquidity through a two-phase methodological approach.

**Title**: "News-Driven Liquidity Dynamics in WTI Crude Oil Futures: A Channel-Decomposition Approach"

**Deadline**: submission targeted for ~20 days from current date.

## Research questions

- **RQ1**: At what temporal lag do news events have their strongest impact on WTI crude oil futures liquidity?
- **RQ2**: Is there directional asymmetry in the response (bearish vs bullish news of comparable magnitude)?
- **RQ3** (deferred to future work): Cross-commodity spillovers between WTI and related instruments.

## Files to read first

These are the canonical sources of truth. Do not infer from memory; verify against these:

- `05_reports/project_logbook.md` — Operational record of experimental runs, data preparation decisions, and numerical results. Bullet-point format.
- `05_reports/thesis_decisions_log.md` — Methodological rationale for each significant decision.
- `05_reports/v2_training_runs.md` — Per-run record of all TFT v2 training experiments with configurations and headline metrics.
- `05_reports/thesis/3-methods.md` — Chapter 3 draft (Methods).
- `05_reports/thesis/4-experiments.md` — Chapter 4 draft (Experiments and Results).
- `05_reports/thesis/0-outline.md` — Full thesis outline with figures and tables registry.

## Data sources and coverage

**Time range**: May 2024 to May 2026, 11,232 hourly observations indexed as `time_idx = [0, 11231]`.

**Regime boundary**: War onset at 28 February 2026 corresponds to `WAR_ONSET_IDX = 10056`. This falls within the test set by design.

**News corpus**:

- Phase 1: 13,690 articles from March 2024 through February 2026, regex-filtered
- Phase 2: 11,675 articles under `usable=1` LLM filter; ~11,433 under stricter `usable_strict=1` (canonical for TFT v2 training)

**Market data**: WTI crude oil futures hourly OHLC via yfinance

**Macro covariates**: DXY, VIX

**Fundamentals**: EIA weekly inventory releases (Wednesday)

**News source**: GDELT for headline scraping; BeautifulSoup for article body scraping

## Two-phase methodology

### Phase 1 (regression-based, FinBERT features)

- FinBERT sentiment extraction (3-class: bearish, neutral, bullish)
- Regex-based article filter (`body_valid`)
- Analytical methods:
  - Contemporaneous OLS
  - Distributed lag OLS (peak identified at +6h lag, bearish > bullish, p<0.001)
  - Vector autoregression (VAR) with impulse response functions
  - Headline bias experiment (title-only vs title+body sentiment: 41.6% divergence, χ²=2050, p<0.001)
- TFT v1 closes Phase 1

### Phase 2 (deep-learning, LLM features)

- Claude Haiku v2 for LLM-based feature extraction with structured schema
- Channel decomposition: `supply_impact`, `demand_impact`, `risk_premium` each on `[-1, +1]`
- Categorical: `event_type_primary` (8 categories), `time_horizon` (4 categories)
- 71 canonical entities normalized (see Appendix A)
- LLM-based `usable=1` filter; `usable_strict=1` variant requires non-zero channel scores
- TFT v2 closes Phase 2

## LLM extraction schema (Haiku v2)

Each article produces:

- `sentiment_score` on `[-1, +1]`
- `magnitude` (0-1)
- `certainty` (0-1)
- Three channel scores: `supply_impact`, `demand_impact`, `risk_premium`
- `event_type_primary` (categorical: no_news, geopolitical, supply, demand, macro, technical, other, structural)
- `time_horizon` (categorical: no_news, immediate, short_term, structural)
- `entities` (list of canonical entities mentioned)
- `usable` (binary judgment on whether article is topical)
- `usable_strict` (derived: usable AND at least one non-zero channel score)

## TFT v2 canonical model (currently reported in §4.3.7)

**Configuration**:

- Architecture: `hidden_size=32`, `attention_head_size=4`, `hidden_continuous_size=16`, `dropout=0.15`
- 298,329 trainable parameters
- Multi-target: `log_volume`, `amihud`, `price_range`
- Multi-horizon: `[1, 3, 6, 12, 28]` hours
- Encoder length: 48 hours
- Loss: `MultiLoss([QuantileLoss()] * 3)`
- Target normalizer: `MultiNormalizer([GroupNormalizer(groups=['asset']) for _ in range(3)])`
- Optimizer: Adam, lr=1e-3, `reduce_on_plateau_patience=3`, gradient_clip=0.1
- Batch size: 128
- Early stopping: patience=10, min_delta=1e-4
- Filter: `usable_strict=1`
- Compute: Google Colab T4 GPU
- Determinism: `pytorch_lightning.seed_everything(42, workers=True)` and `torch.use_deterministic_algorithms(True, warn_only=True)`

**Split (canonical, 70/15/15)**:

- Train: time_idx [0, 7862), 7,788 samples after decoder buffer
- Val: [7910, 9547), 1,610 samples
- Test: [9595, 11232), 1,610 samples (461 pre-war + 1149 war)
- 48h buffer between splits

**Best result**: val_loss 0.408 at epoch 26, early-stopped epoch 36

**Test set MAE reduction over persistence (log_volume)**:

- 1h: 41%
- 3h: 57%
- 6h: 67%
- 12h: 73% (peak)
- 28h: 63%

**Feature importance top 10** (VSN output):

1. vix (0.236)
2. demand_impact (0.136)
3. ent_us (0.045)
4. log_return (0.034)
5. dxy (0.029)
6. amihud (0.028)
7. ent_yemen (0.022)
8. ent_lebanon (0.020)
9. ent_oman (0.020)
10. ent_brent (0.019)

**Attention pattern**: population mean peak at -17h, with cluster -15h to -18h, secondary peak at -1h. Bearish attention peaks at -17h, bullish at -12h.

**Success criteria (locked pre-training)**:

1. Channels economically interpretable: PASS
2. Directional asymmetry p<0.05: PARTIAL PASS (+3h pre-war, p=0.013, bearish > bullish, matches lag OLS)
3. Multi-horizon error structure matches lag OLS peak: PARTIAL PASS

## Repository structure

```
hammer-market-intelligence-internship/
├── 00_data_raw/ # Original CSVs, never modified
├── 01_data/ # Working database
│ ├── wti_thesis.db # SQLite database
│ └── models/ # TFT checkpoints (.ckpt)
├── 02_notebooks/ # All notebooks numbered sequentially
├── 03_src/ # Shared Python code
│ ├── tft/config.py # Locked TFT v2 training constants
│ └── nlp/llm_features.py # Entity normalization data
├── 04_outputs/ # Generated artifacts
│ └── tft_v2/ # Per-variant outputs
│ ├── v2.0/ # ablation variant
│ ├── v2.1/ # ablation variant
│ └── v2.2/ # canonical v2 model
└── 05_reports/ # Documentation
├── project_logbook.md
├── thesis_decisions_log.md
├── v2_training_runs.md
└── thesis/ # Chapter drafts
```

## Key locked constants (`03_src/tft/config.py`)

```python
TOTAL_HOURS = 11232
ENCODER_LENGTH = 48
MAX_PREDICTION_LENGTH = 28
TRAIN_END = 7862         # 70%
VAL_START = 7910         # +48h buffer
VAL_END = 9547           # 15%
TEST_START = 9595        # +48h buffer
TEST_END = 11232         # 15%
WAR_ONSET_IDX = 10056    # 2026-03-01 23:00 UTC
```

Do not modify these without documenting the change in the decisions log.

## Environment

- Python 3.13 venv at `~/Documents/Code/.venv`
- Local dev: M1 Apple Silicon
- Training: Google Colab T4 GPU
- Database: SQLite at `01_data/wti_thesis.db`

**Key libraries**:

- torch 2.3.1
- pytorch-forecasting 1.7.0
- lightning.pytorch (NOT `pytorch_lightning` — different namespace)
- transformers (FinBERT via ProsusAI/finbert)
- anthropic (for Haiku v2 extraction)

## Working rules (non-negotiable)

### Source verification

Every numerical claim or methodological specific in any output must trace to:

- A notebook cell output
- A SQL query result against `01_data/wti_thesis.db`
- A source CSV
- A canonical `.md` file in `05_reports/`

Do not infer numbers from memory. Do not invent details.

**Never invent references** to files, sections, papers, or documentation that has not been verified to exist. If unsure whether a reference is real, ask before including it.

### Logbook and documentation

The logbook uses condensed bullet-point format with subheaders. New entries follow existing structure. Never write dense paragraphs.

Chapter drafts include only verified numbers. Referenced sections must exist with the described content.

### Language conventions

All code, comments, notebook prints, and markdown in English. Conversational chat may be in Spanish or English.

Prose style: no em dashes. Use commas or parentheses instead.

### Lightning namespace

Use `lightning.pytorch` (modern unified package), not `pytorch_lightning` (legacy). Mixing them causes silent `isinstance` failures. Correct pattern:

```python
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
```

### Data filter

The canonical filter for TFT v2 training is `usable_strict=1`, not `usable=1`. The `usable_strict` column lives in the `llm_features` table; the `liquidity` table only has `usable`.

### Temporal alignment (causality)

Use `dt.ceil` (not `dt.round`) to align articles to trading hours. Off-hours articles forward-assigned to next trading hour, not discarded. This prevents look-ahead bias.

## Database schema and examples

The SQLite database at `01_data/wti_thesis.db` contains 11 tables. The following are the key tables used in modeling; supporting tables (`raw_entity_counts`, `sqlite_sequence`, `calibration_sample`, `llm_features_v1_backup`, `opec_events`) exist but are not part of the modeling pipeline.

### `articles` (22,795 rows)

Raw scraped articles with metadata. Source for LLM feature extraction.

| Column     | Type    | Notes                                                   |
| ---------- | ------- | ------------------------------------------------------- |
| id         | INTEGER | Primary key, referenced by other tables as `article_id` |
| title      | TEXT    | Article headline                                        |
| datetime   | TEXT    | ISO 8601 timestamp with timezone                        |
| domain     | TEXT    | Source website domain                                   |
| url        | TEXT    | Full article URL                                        |
| body       | TEXT    | Scraped article body                                    |
| body_valid | INTEGER | Phase 1 regex-based filter (1 = passes)                 |

Example row: `id=1`, `title="2023 Was a Bad Year for Commodities"`, `datetime="2024-01-01 00:30:00+00:00"`, `domain="oilprice.com"`, `body_valid=1`.

### `llm_features` (19,619 rows)

Structured extraction from Claude Haiku v2. One row per article that passed extraction.

| Column          | Type    | Notes                                                                              |
| --------------- | ------- | ---------------------------------------------------------------------------------- |
| article_id      | INTEGER | Foreign key to `articles.id`                                                       |
| sentiment_score | REAL    | On `[-1, +1]`                                                                      |
| magnitude       | REAL    | On `[0, 1]`                                                                        |
| certainty       | REAL    | On `[0, 1]`                                                                        |
| event_type      | TEXT    | One of: no_news, geopolitical, supply, demand, macro, technical, other, structural |
| entities        | TEXT    | JSON list of raw entity names                                                      |
| price_direction | TEXT    | bearish, bullish, neutral                                                          |
| time_horizon    | TEXT    | immediate, short_term, structural                                                  |
| usable          | BOOLEAN | LLM's binary judgment on topical relevance                                         |
| supply_impact   | REAL    | Channel score on `[-1, +1]`                                                        |
| demand_impact   | REAL    | Channel score on `[-1, +1]`                                                        |
| risk_premium    | REAL    | Channel score on `[-1, +1]`                                                        |
| usable_strict   | INTEGER | Derived: `usable=1 AND at least one non-zero channel score`                        |

Example row for an article with no LLM output: `article_id=1`, all feature columns `None`, `usable=0`, `usable_strict=0`.

Example row from v1 backup with populated features: `article_id=3610`, `sentiment_score=-0.15`, `magnitude=0.2`, `event_type="macro"`, `certainty=0.85`, `price_direction="bearish"`, `time_horizon="short_term"`.

### `article_entities` (29,473 rows)

Normalized entity mentions per article. Many-to-many relationship: each article can mention multiple canonical entities.

| Column           | Type    | Notes                                                    |
| ---------------- | ------- | -------------------------------------------------------- |
| article_id       | INTEGER | Foreign key to `articles.id`                             |
| canonical_entity | TEXT    | Normalized entity name from the 71-entity canonical list |

Example row: `article_id=2`, `canonical_entity="Russia"`.

This table is the source for the 71 entity flag columns fed to TFT v2. Each hour receives a binary vector indicating which canonical entities were mentioned by the dominant article of that hour.

### `market_context` (11,232 rows)

Hourly market data for WTI crude oil futures. One row per hour of the modeling grid.

| Column            | Type    | Notes                                             |
| ----------------- | ------- | ------------------------------------------------- |
| datetime_hour     | TEXT    | ISO 8601 hourly timestamp with timezone           |
| log_volume        | REAL    | Natural log of hourly volume                      |
| price_range       | REAL    | High - low for the hour                           |
| log_return        | REAL    | Log return: `ln(close[t] / close[t-1])`           |
| amihud            | REAL    | Illiquidity measure: `return / dollar_volume`     |
| dxy               | REAL    | US Dollar Index value                             |
| vix               | REAL    | VIX volatility index value                        |
| cushing_inventory | REAL    | Cushing crude oil inventory (weekly interpolated) |
| eia_surprise      | REAL    | Deviation from expected EIA weekly inventory      |
| is_eia_release    | INTEGER | 1 on Wednesday EIA release hours, 0 otherwise     |

Example row: `datetime_hour="2024-05-13 11:00:00+00:00"`, `log_volume=0.0`, `price_range=0.330002`, `dxy=105.199`, `vix=13.31`, `eia_surprise=-1915.0`, `is_eia_release=0`.

Note: `log_volume=0.0` at the very first row indicates opening hour with zero-baseline volume. `log_return=None` for first row because no previous price exists.

### `liquidity` (22,795 rows)

Joined table combining articles + LLM features + hourly market context. This is the analytical source for Phase 1 methods and the input to TFT v1.

Contains one row per article, with the article's LLM features and the market context at the article's assigned hour. Note: `usable_strict` is NOT in this table; that flag lives only in `llm_features`. To query with `usable_strict`, JOIN through:

```sql
SELECT l.article_id, l.datetime_hour, ...
FROM liquidity l
JOIN llm_features f ON l.article_id = f.article_id
WHERE f.usable_strict = 1
```

Key columns: `article_id`, `datetime_hour`, `assignment_gap` (hours between article publication and assigned trading hour, used for causality validation), the market context columns (`log_volume`, `price_range`, etc.), the LLM feature columns (mirroring `llm_features`), and `has_llm_features` (1 if LLM extraction succeeded).

### `eia_events` (331 rows)

Weekly EIA crude oil inventory events. Wednesday releases with market-relevant surprise metrics.

| Column           | Type    | Notes                                            |
| ---------------- | ------- | ------------------------------------------------ |
| date             | TEXT    | Date of report                                   |
| inventory_mbbl   | INTEGER | Reported inventory in thousand barrels           |
| inventory_change | REAL    | Week-over-week change                            |
| news_direction   | TEXT    | bearish, bullish, neutral inferred from surprise |
| datetime_et      | TEXT    | Timestamp adjusted to Eastern Time               |

Example row: `date="2020-01-03"`, `inventory_mbbl=1066027`, `news_direction="neutral"`.

## Data pipeline overview

Articles flow through the following pipeline:

1. **Scraping**: `articles.body` populated from GDELT + BeautifulSoup
2. **Phase 1 filter**: `articles.body_valid = 1` if regex heuristic passes
3. **Phase 2 filter**: `llm_features.usable = 1` if LLM judges topical
4. **Feature extraction**: `llm_features` populated with structured schema
5. **Entity normalization**: `article_entities` populated with canonical entities
6. **Alignment**: articles assigned to trading hours via `dt.ceil` (never `round`)
7. **Aggregation**: hourly features computed from articles in that hour
8. **Modeling table**: `liquidity` joins articles + LLM features + market context

The TFT v2 modeling table is constructed by joining `market_context` (hourly grid, 11,232 rows) with aggregated features from articles that pass `usable_strict=1`.

## Ablation variants

TFT v2 was designed through a controlled ablation. All three variants are preserved in `04_outputs/tft_v2/`:

- **v2.0**: `usable=1`, int-encoded categoricals, single target (log_volume), single horizon (1h), no entity flags
- **v2.1**: `usable=1`, proper categorical encoding, single target/horizon, no entity flags
- **v2.2**: `usable_strict=1`, proper categoricals, multi-target, multi-horizon, 71 entity flags (canonical)

The ablation appears in Appendix C. Only v2.2 is reported in the main text as "TFT v2".

## Appendices planned

Registry of appendices for the thesis:

- **Appendix A**: Canonical entity list (71 entities with aliases)
- **Appendix B**: LLM extraction schema (Haiku v2)
- **Appendix C**: TFT v2 ablation study (v2.0, v2.1, v2.2 comparison)
- **Appendix D**: Extended metrics tables (full 60-row table: 3 targets × 5 horizons × 4 slices)
- **Appendix E**: Reproducibility statement (library versions, hardware, seeds)

## When to update this file

Update AGENTS.md when any of the following changes occur:

1. **The canonical v2 model changes**: if a new experiment produces the reported v2 configuration, update the "TFT v2 canonical model" section.
2. **A locked constant changes**: update the "Key locked constants" section and add an entry to the decisions log explaining why.
3. **A new key file is added**: add it to "Files to read first" or the repository structure.
4. **A rule is added or removed**: update "Working rules" section.
5. **Success criteria evaluation changes**: update the results in "TFT v2 canonical model" section.
6. **A new appendix is planned**: update "Appendices planned" registry.
7. **New Chapter drafts are created**: add reference to "Files to read first".
8. **Deadline changes significantly**: update the "Project overview" section.

Do not update AGENTS.md for:

- Draft revisions of chapter prose
- Minor bug fixes in notebooks
- Adding or removing individual references
- Any change that does not affect the project's overall structure or state

When updating, keep changes minimal and preserve the existing structure. If a section is no longer accurate, replace it rather than adding contradictory notes.

## Common pitfalls to avoid

- Using `pytorch_lightning` instead of `lightning.pytorch`
- Hardcoding paths instead of importing from `03_src/tft/config.py`
- Using `usable=1` instead of `usable_strict=1` for v2 training
- Writing logbook entries as prose instead of bullets
- Inferring numerical values from memory instead of verifying
- Proposing solutions without running diagnostics first
- Referencing sections, files, or papers without verifying they exist
