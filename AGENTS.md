# AGENTS.md

This is a research repo for a Master's thesis on news-driven liquidity dynamics in WTI crude oil futures. Read this file before doing any work.

## Project overview

The thesis investigates how news events influence WTI crude oil futures liquidity. It is hosted at Hammer Market Intelligence and supervised by Dr. Lejla Batina at Radboud University. The work spans two phases:

- **Phase 1** (March 2024 – February 2026): FinBERT-derived sentiment on a regex-filtered corpus. Closes with TFT v1.
- **Phase 2** (May 2024 – May 2026): Channel-decomposed Claude Haiku-extracted features on an LLM-filtered corpus, with entity normalization. Closes with TFT v2.

Three research questions:

- **RQ1**: Lag structure of news impact on liquidity
- **RQ2**: Bearish vs bullish asymmetry in news-liquidity response
- **RQ3** (deferred): Cross-commodity spillovers

## Read these files before starting any work

These are the canonical sources of truth. Do not infer from memory; verify against these:

- `05_reports/project_logbook.md` — Operational record. Every numerical result, every experimental run, every data prep decision. Bullet-point format, not prose.
- `05_reports/thesis_decisions_log.md` — Methodological rationale. Every decision and why.
- `05_reports/v2_training_runs.md` — Per-run record of TFT v2 trainings with config and headline metrics.
- `05_reports/thesis/3-methods.md` — Chapter 3 (complete).
- `05_reports/thesis/4-experiments.md` — Chapter 4 (in progress).

## Working rules

These are non-negotiable. Following them prevents the most common failure modes in this project.

### Source verification

Every numerical claim or methodological specific in any output (notebook, report, thesis section) must trace to:

- A notebook cell output, or
- A SQL query result against `01_data/wti_thesis.db`, or
- A source CSV, or
- A canonical `.md` file in `05_reports/`

Do not infer numbers from memory. Do not invent details that "should be correct." When in doubt, run a query.

### Logbook updates

The logbook uses condensed bullet-point format with subheaders. Never write dense paragraphs. New entries follow the same structure as existing ones.

### Language

All code, comments, notebook prints, and markdown in English. Chat conversation may be in Spanish or English depending on user preference.

### Lightning namespace

Use `lightning.pytorch` (the modern unified package), not `pytorch_lightning` (the legacy package). They are different namespaces and mixing them causes silent `isinstance` failures. Specifically:

```python
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
```

### Data filter

The canonical filter for TFT v2 training is `usable_strict=1`, not `usable=1`. This is documented in the decisions log. The `usable_strict` column lives in the `llm_features` table; the `liquidity` table only has `usable`.

## Environment

- Python 3.13 venv at `~/Documents/Code/.venv`
- Compute: M1 Apple Silicon (local) + Google Colab T4 (GPU training)
- Database: SQLite at `01_data/wti_thesis.db`
- Project memory directory: `05_reports/`
- Source code: `03_src/`

### Project structure

```
news-driven-liquidity/
├── 00_data_raw/          # Original CSVs, never modified
├── 01_data/              # Working database and checkpoints
│   ├── wti_thesis.db
│   └── models/           # TFT checkpoints (.ckpt)
├── 02_notebooks/         # All notebooks numbered sequentially
├── 03_src/               # Shared Python code
│   ├── tft/config.py     # Locked TFT v2 training constants
│   └── nlp/llm_features.py  # Entity normalization data
├── 04_outputs/           # Generated artifacts (predictions, metrics, importance)
└── 05_reports/           # Documentation, logbook, decisions, draft thesis
```

### Key constants (already locked in `03_src/tft/config.py`)

- `TOTAL_HOURS = 11232`
- `ENCODER_LENGTH = 48`
- `MAX_PREDICTION_LENGTH = 28`
- Split: train [0, 7862), val [7910, 9547), test [9595, 11232)
- War onset boundary: `WAR_ONSET_IDX = 10056` (2026-03-01 23:00 UTC)

## TFT v2 model status

The canonical v2 model is the configuration documented in run #6 of `v2_training_runs.md`:

- `usable_strict=1` filter
- Patience 10, dropout 0.15
- Multi-horizon `[1, 3, 6, 12, 28]`, multi-target `[log_volume, amihud, price_range]`
- Best val_loss 0.408 at epoch 26
- Architecture matches v1 (hidden_size=32, etc.)

Three success criteria for v2:

1. Channels economically interpretable in attention/importance — **PASS**
2. Directional asymmetry at p<0.05 — **PARTIAL PASS** (+3h pre-war replicated)
3. Multi-horizon error structure matches lag OLS peak — **PARTIAL PASS** (+6h to +12h range)

The v2.0 and v2.1 ablation runs are preserved in `v2_training_runs.md` and will appear in Appendix X of the thesis. The main thesis text reports only the canonical v2 model.

## Where to put new work

- New notebooks: `02_notebooks/`, numbered sequentially (next would be 14)
- New source code: `03_src/<module>/`
- New analysis outputs: `04_outputs/<topic>/`
- New documentation: `05_reports/`

## Common pitfalls to avoid

- Using `pytorch_lightning` instead of `lightning.pytorch` (see Lightning namespace rule above)
- Hardcoding paths instead of importing from `03_src/tft/config.py`
- Using `usable=1` filter instead of `usable_strict=1` for v2 training
- Writing logbook entries as prose instead of bullets
- Inferring numerical values from memory instead of verifying
- Proposing solutions without running diagnostics first
