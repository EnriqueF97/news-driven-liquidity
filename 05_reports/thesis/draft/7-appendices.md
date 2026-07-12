# Appendices

Appendices are a top-level section following the Conclusion (Chapter 6), not part of any chapter. The full appendix registry (A-G) is maintained in `0-outline.md`; this file drafts the appendix content.

## Appendix A: Extraction prompt

This appendix reproduces the verbatim Haiku v2 extraction interface: the system prompt and the `extract_article_features` tool schema passed to the tool-use API. The conceptual field-level schema and its revision history are in Appendix B.

**Verbatim Haiku v2 system prompt and tool schema.** *(To insert: the system prompt and the `extract_article_features` tool schema from the extraction code, `03_src/nlp/llm_features.py` and the extraction notebook.)*

## Appendix B: LLM extraction schema (v1 vs v2)

The Haiku extraction schema was revised once during Phase 2, from Schema v1 (used for TFT v1, §4.3.5) to Schema v2 (the canonical extraction used for all reported Phase 2 results, §4.3.6). The revision was the direct response to the inter-model calibration finding of §4.3.4: the composite `sentiment_score` was unreliable on high-magnitude geopolitical events because it conflated event valence with directional price impact.

**Field-by-field comparison.**

| Field             | Schema v1                 | Schema v2                                   |
| ----------------- | ------------------------- | ------------------------------------------- |
| `sentiment_score` | continuous [-1, +1]       | continuous [-1, +1] (retained)              |
| `supply_impact`   | —                         | continuous [-1, +1] (new)                   |
| `demand_impact`   | —                         | continuous [-1, +1] (new)                   |
| `risk_premium`    | —                         | continuous [-1, +1] (new)                   |
| `magnitude`       | [0, 1]                    | [0, 1]                                       |
| `certainty`       | [0, 1]                    | [0, 1]                                       |
| `event_type`      | single categorical label  | array of one to three salience-ordered labels |
| `time_horizon`    | categorical label         | categorical label                           |
| `price_direction` | categorical label         | dropped                                     |
| `entities`        | list of canonical actors  | list of canonical actors                    |
| `usable`          | implicit (prompt only)    | required boolean field                      |

**Changes and rationale.**

1. **Three orthogonal economic channels added** (`supply_impact`, `demand_impact`, `risk_premium`, each on [-1, +1]). This decomposes the single sentiment judgment into its separable economic drivers, mirroring the standard structural decomposition of oil-price shocks into supply, aggregate-demand, and precautionary (risk) components [CITE: Kilian 2009]. Re-calibration lifted the composite `sentiment_score` cross-model correlation from 0.39 to 0.88 and gave the channels correlations of 0.94 (supply), 0.96 (demand), and 0.82 (risk) (§4.3.6).
2. **`usable` promoted to a required field** (from an implicit prompt instruction), enabling the LLM-based filter and the derived `usable_strict` variant (§4.3.3).
3. **`event_type` changed from a single label to an array of one to three salience-ordered labels**, so an article that is jointly geopolitical and supply-related (e.g. Iran sanctions) need not be forced into a single category.
4. **`price_direction` dropped**: it agreed with the sign of `sentiment_score` on all but roughly 16 of 12,024 articles and carried about 0.008 feature importance in TFT v1, and the three channels provide a richer directional substrate.
5. **`sentiment_score` retained** (not replaced by the channels), for continuity with the headline-bias experiment (§4.2.2) and as a hedge against individually noisy channels.

TFT v1 (§4.3.5) was trained on Schema v1; TFT v2 (§4.3.7) and all reported Phase 2 results use Schema v2.

## Appendix C: TFT v2 ablation and hyperparameters

The reported TFT v2 (variant v2.2) was selected through a three-variant ablation, all on the 60/20/20 split with `usable_strict=1`, dropout 0.15, and early-stopping patience 10; only the design axis changes between rows.

| Variant          | Categorical encoding | Targets × horizons        | Entity flags | Best val_loss | Test MAE@1h | Feature #1            |
| ---------------- | -------------------- | ------------------------- | ------------ | ------------: | ----------: | -------------------- |
| v2.0             | int-encoded          | log_volume × 1h           | no           |         0.257 |       0.455 | sentiment_score 0.274 |
| v2.1             | proper categoricals  | log_volume × 1h           | no           |         0.258 |       0.424 | demand_impact 0.485   |
| v2.2 (canonical) | proper categoricals  | 3 × [1, 3, 6, 12]         | 71           |       0.427\* |       0.585 | vix 0.188             |

\* v2.2's val_loss (`MultiLoss` over three targets) is not comparable to the single-target v2.0/v2.1. The key finding is that proper categorical encoding (v2.0 -> v2.1) is what unlocks the channels' predictive role: `demand_impact` jumps to 0.485. Full per-run record in `05_reports/v2_training_results.md`.

**Canonical TFT v2 hyperparameters.** hidden_size 32, attention_head_size 4, hidden_continuous_size 16, dropout 0.15, learning_rate 1e-3 (Adam, on-plateau patience 3), gradient clip 0.1, early-stopping patience 10, seed 42; 298,329 trainable parameters; encoder length 48h; targets `[log_volume, amihud, price_range]`; horizons `[1, 3, 6, 12]`; split 60/20/20 (TRAIN_END 6739, VAL 6787-9014, TEST 9062-11232).

## Appendix D: Canonical entity list

The 71 canonical entities used as TFT v2 entity flags (from `03_src/nlp/llm_features.py`), each mapped from raw article mentions via the normalization of §4.3.2 (alias mappings omitted here for brevity):

US, Iran, Russia, China, Israel, India, Saudi Arabia, Ukraine, Venezuela, Canada, Iraq, Nigeria, UAE, Kazakhstan, Qatar, Oman, Japan, Kuwait, Pakistan, Libya, Mexico, Azerbaijan, Yemen, Lebanon, Brazil, South Korea, Guyana, UK, Algeria, Germany, Australia, Hungary, Egypt, Türkiye, Strait of Hormuz, Middle East, Gaza, Red Sea, Persian Gulf, Gulf of Mexico, Permian Basin, Europe, Asia, Trump, Maduro, Putin, Powell, Bessent, Biden, OPEC+, OPEC, Fed, EU, EIA, IEA, API, ECB, Hamas, Hezbollah, Houthis, Saudi Aramco, Chevron, Shell, BP, ExxonMobil, Rosneft, Lukoil, TotalEnergies, WTI, Brent, S&P 500.

## Appendix G: Reproducibility statement

The Phase 2 training was run on Google Colab (T4 GPU); the LLM extraction and the Phase 1 analyses were run locally (Apple Silicon, MPS backend). Software versions with a paper of record are cited; the rest are attributed here by name, version, and source.

**Software libraries.** Version constraints are pinned in the training notebook `02_notebooks/13_tft_v2_training_colab.ipynb`.

| Library                    | Version        | Role                                                   |
| -------------------------- | -------------- | ------------------------------------------------------ |
| Python                     | 3.x            | runtime                                                |
| `numpy`                    | < 2.0          | numerics                                               |
| `pandas`                   | —              | data processing and hourly aggregation                 |
| PyTorch (`torch`)          | >= 2.3, < 2.4  | deep-learning backend                                  |
| Lightning (`lightning`)    | >= 2.2, < 2.4  | training loop                                          |
| `pytorch-forecasting`      | >= 1.0, < 1.2  | TFT implementation and `TimeSeriesDataSet`             |
| HuggingFace `transformers` | —              | FinBERT inference [CITE: Wolf et al. 2020]             |
| `yfinance`                 | —              | market-data retrieval                                  |
| `beautifulsoup4`           | —              | article-body HTML parsing                              |
| `scipy`                    | —              | statistical tests (Welch t-test, correlations)         |
| `sqlite3` (stdlib)         | —              | database                                               |

**Models and APIs.**

- Claude Haiku 4.5 (Anthropic), via the tool-use API, for Phase 2 feature extraction (§3.3).
- GPT-5.5 (OpenAI), via the ChatGPT interface, as the inter-model calibration reference (§3.7).

**Data sources.**

- GDELT 2.0 Document API [CITE: Leetaru & Schrodt 2013], for news article metadata (`https://www.gdeltproject.org`).
- U.S. Energy Information Administration (EIA) v2 API, weekly crude inventory series `WCRSTUS1` (`https://www.eia.gov/opendata/`).
- yfinance / Yahoo Finance, hourly OHLCV for `CL=F`, `DX-Y.NYB`, and `^VIX`.

**Determinism.** The reported TFT v2 run fixes the random seed to 42 with deterministic algorithms; its exact configuration is stored in `run_metadata.json`. Exact pinned library versions and data-source access dates are to be recorded from the environment at finalisation.
