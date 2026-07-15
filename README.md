# News-Driven Liquidity Dynamics in WTI Crude Oil Futures

**A Channel-Decomposition Approach**

**Author:** Enrique Favila Martínez\
**Program:** MSc Artificial Intelligence & Cybersecurity, Radboud University\
**Host company:** Hammer Market Intelligence\
**Supervisor:** Dr. Lejla Batina

---

## Project overview

This project studies how news propagates into the **liquidity** of WTI crude oil futures at **hourly** resolution, along two axes:

- **RQ1 (lag structure):** at what delay does news sentiment most strongly move trading liquidity?
- **RQ2 (directional asymmetry):** does the liquidity response differ between bearish and bullish news?

**Headline results.** The liquidity response is gradual and peaks over a **+6 to +12 hour** window (not contemporaneous). There is a robust **bearish-over-bullish** asymmetry in the marginal sensitivity of volume to negative-tone news; at the level of predicted volume the model organizes around **risk and salience rather than price direction**. Cross-commodity propagation (WTI, Brent, natural gas) is left as future work, and the data notebooks are written to generalize by swapping the ticker and query configuration.

Liquidity is measured hourly, primarily by **log trading volume**, alongside the **Amihud** illiquidity ratio and the **Parkinson** high-low range.

---

## Two-phase design

The same questions run through two independent methodologies, so the second corroborates the first rather than replacing it.

- **Phase 1 (interpretable baseline).** FinBERT 3-class sentiment over a regex-filtered corpus, analyzed with contemporaneous and lag OLS. Produces the primary, transparent answer to RQ1 and RQ2.
- **Phase 2 (expressive deep learning).** Structured LLM extraction (Claude Haiku, Schema v2) that decomposes each article into three orthogonal economic channels (supply, demand, risk premium) plus entity flags, fed to a **Temporal Fusion Transformer (TFT)** that predicts three liquidity targets at four horizons and is interpretable through variable selection and attention.

Everything is **database-first**: the notebooks read from and write to a single SQLite database (`01_data/wti_thesis.db`). CSVs are kept only as backups and restore points.

---

## Repository structure

```text
news-driven-liquidity/
├── 01_data/
│   ├── wti_thesis.db            # canonical SQLite database (single source of truth)
│   ├── raw/
│   │   ├── price/               # wti_hourly_raw.csv (yfinance OHLCV)
│   │   ├── macro/               # eia_inventories_raw.csv
│   │   └── news/                # gdelt_wti_raw.csv, gdelt_wti_with_body_raw.csv
│   ├── processed/               # gdelt_wti_aligned.csv, market_context.csv
│   ├── features/                # gdelt_wti_sentiment.csv (FinBERT), headline_bias_summary.csv
│   └── models/                  # tft_wti.ckpt, training_dataset.pkl (TFT v1)
├── 02_notebooks/                # the numbered pipeline, 00 -> 13 (see below)
│   └── deprecated_notebooks/    # superseded exploration
├── 03_src/
│   └── nlp/
│       └── llm_features.py      # locked prompt, tool schema, entity maps, canonicalize_entities
├── 04_outputs/
│   ├── figures/                 # generated PNGs (lag coefficients, TFT attention, importance...)
│   └── tables/                  # generated CSVs (lag OLS results...)
└── 05_reports/
    ├── thesis/                  # LaTeX thesis (thesis.tex, sections/, references.bib, figures/) + draft/
    ├── presentation/            # defense.md (Marp deck) + exported pptx/pdf/html
    ├── calibration/             # llm_calibration_v2.json, llm_batch_id.txt, batch_errors.json
    ├── development-decisions/   # project_logbook.md (dated run log)
    └── v2_training_results.md   # per-run TFT v2 record
```

### The database (`wti_thesis.db`)

Core tables created in notebook 00: `articles`, `liquidity`, `llm_features`, `market_context`, `opec_events`, `eia_events`. Later notebooks add `calibration_sample` (30-article sample), `raw_entity_counts`, and `article_entities`.

---

## What each notebook does (in plain words)

Run in numeric order; each one builds on the tables the previous ones wrote.

| #   | Notebook                     | What it does                                                                                                                                                                                                                                           |
| --- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 00  | `setup_database`             | Creates the SQLite database and its six core tables, migrates the pre-aligned Phase 1 CSVs, and attaches the EIA features. The CSV migrations are optional, so the notebook does not break when adapting it to a new commodity with no local CSVs.     |
| 01  | `yfinances_prices`           | Downloads two years of hourly WTI futures prices (`CL=F`) from Yahoo Finance, plus DXY and VIX, and computes the four liquidity metrics into `market_context`. Change the ticker map to retarget another commodity.                                    |
| 02  | `eia_inventories`            | Downloads the weekly EIA crude inventory series, computes the week-on-week surprise, timestamps each Wednesday release, and merges `eia_surprise` / `is_eia_release` onto the hourly grid.                                                             |
| 03  | `gdelt_headlines`            | Queries GDELT for oil-news headlines (metadata only), deduplicates, keeps English, and appends new rows to `articles`. Incremental and safe to re-run.                                                                                                 |
| 04  | `gdelt_scrapper`             | Scrapes the body text for each headline URL, cleans it, and validates it (the `body_valid` flag). Resume-capable, with a DB checkpoint every 50 articles.                                                                                              |
| 05  | `alignment`                  | Assigns each article to its next available trading hour (causal, never before publication), joins the market snapshot and LLM features, and writes the `liquidity` table with hard consistency assertions.                                             |
| 06  | `llm_features`               | The full LLM extraction. Sends every article body to Claude Haiku through the Batches API, forcing the Schema v2 tool (the `usable` flag, composite sentiment, the three supply/demand/risk channels, entities), and writes `llm_features`. Resumable. |
| 07  | `headline_bias`              | Statistical test of title-only vs title+body FinBERT sentiment (the ~41.6% divergence finding), motivating title+body as the primary sentiment input.                                                                                                  |
| 08  | `lag_and_asymmetry_analysis` | **Phase 1 core.** Contemporaneous and lag OLS of FinBERT sentiment on `log_volume`, producing the +6h peak (RQ1) and the bearish-over-bullish asymmetry (RQ2). Includes an exploratory VAR that was dropped from the thesis.                           |
| 09  | `parallel_features`          | Trains **TFT v1** (single target, one hour ahead) to validate that a deep-learning sequence model beats the OLS / persistence baseline.                                                                                                                |
| 10  | `tft_analysis`               | Interpretability of TFT v1: attention over the 48-hour window (RQ1, lag), Variable Selection Network importance, and a bearish-vs-bullish comparison (RQ2, asymmetry).                                                                                 |
| 11  | `calibration`                | One-shot Schema v2 calibration on a stratified 30-article sample; produces the Haiku side of the inter-model agreement check (the GPT reference is scored separately).                                                                                 |
| 12  | `entity_normalization`       | Normalizes the thousands of raw LLM entity strings down to 71 canonical entities and builds the long-format `article_entities` table.                                                                                                                  |
| 13  | `tft_v2_training` / `_colab` | **The canonical model.** Trains **TFT v2** (three targets: log_volume, amihud, price_range; four horizons: 1/3/6/12h; channels plus 71 entity flags) on Google Colab, producing the reported results, feature importance, and attention figures.       |

> The FinBERT sentiment used by notebooks 07 and 08 comes from an earlier sentiment-scoring step (see `deprecated_notebooks/`) and is stored in `01_data/features/gdelt_wti_sentiment.csv`.

---

## Data sources

| Source                                | Data                                        | Frequency  |
| ------------------------------------- | ------------------------------------------- | ---------- |
| yfinance (`CL=F`, `DX-Y.NYB`, `^VIX`) | WTI OHLCV, plus DXY and VIX                 | Hourly     |
| EIA API (`WCRSTUS1`)                  | U.S. commercial crude oil inventories       | Weekly     |
| GDELT Project                         | News article metadata (energy, geopolitics) | Continuous |

## Liquidity variables

| Variable      | Definition                                    | Role                              |
| ------------- | --------------------------------------------- | --------------------------------- |
| `log_volume`  | Log-transformed hourly trading volume         | primary liquidity target          |
| `amihud`      | Absolute return / volume (Amihud, 2002)       | price-impact target               |
| `price_range` | ln(high) − ln(low) per hour (Parkinson proxy) | volatility target                 |
| `log_return`  | ln(close*t / close*{t-1})                     | price variable, input to `amihud` |

---

## Key modeling notes

- **Canonical model:** TFT v2, variant **v2.2 exp2** (60/20/20 split, `usable_strict=1` corpus, encoder 48h, hidden size 32, four heads, 298,329 parameters). Selected through a three-variant ablation (v2.0 int-encoded -> v2.1 proper categoricals -> v2.2 multi-target + entities); the pivotal step is fixing the categorical encoding, which is what makes the channels predictive. Adding EIA features (v2.3) made it worse, so v2.2 exp2 stays canonical.
- **`usable_strict`:** the downstream training filter, defined as `usable=1 AND (supply_impact != 0 OR demand_impact != 0 OR risk_premium != 0)`. It drops topical-but-channel-neutral articles.
- **LLM-as-filter:** the LLM `usable` flag replaces the Phase 1 regex `body_valid`. It is more accurate on both of the regex's failure modes (long off-topic keyword matches, and short but substantive briefs).

---

## Reproducing the pipeline

1. Create a Python environment (a local `.venv` is used) and install the dependencies (pandas, numpy, yfinance, requests, beautifulsoup4, anthropic, python-dotenv, statsmodels, scipy, matplotlib, seaborn, torch, lightning, pytorch-forecasting, transformers for FinBERT).
2. Put an `ANTHROPIC_API_KEY` in a `.env` file at the project root (needed for notebooks 06 and 11). An EIA API key is optional (notebook 02 falls back to `DEMO_KEY`, which is rate-limited).
3. Run the notebooks in order (00 -> 13). Notebook 13 (TFT v2) is intended for Google Colab with a GPU.

---

## Outputs

- **Thesis:** `05_reports/thesis/` (LaTeX, ~88 pages, self-contained for Overleaf).
- **Defense deck:** `05_reports/presentation/defense.md` (Marp), exported to `defense.{pptx,pdf,html}`.
- **Figures and tables:** `04_outputs/`.

---

## Key references

- Amihud, Y. (2002). Illiquidity and stock returns. _Journal of Financial Markets_.
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. _arXiv:1908.10063_.
- Kilian, L. (2009). Not all oil price shocks are alike. _American Economic Review_.
- Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. _International Journal of Forecasting_.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. _Journal of Business_.
- Tetlock, P. C. (2007). Giving content to investor sentiment. _The Journal of Finance_.
