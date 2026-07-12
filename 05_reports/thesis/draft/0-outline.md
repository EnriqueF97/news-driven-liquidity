# Thesis Outline

**Title:** News-Driven Liquidity Dynamics in WTI Crude Oil Futures: A Channel-Decomposition Approach

**Author:** Enrique Favila\
**Supervisor:** Dr. Lejla Batina\
**Institution:** Radboud University\
**Host:** Hammer Market Intelligence

## Purpose of this document

Structural reference for the thesis. The work investigates how news propagates into WTI crude oil futures liquidity at hourly resolution, using a channel decomposition of LLM-extracted news features (supply, demand, risk premium) to characterize the lag structure (RQ1) and the bearish/bullish asymmetry (RQ2) of the liquidity response.

## Chapter 1 - Introduction

**Purpose:** What is the problem, current research questions, contributions.

- 1.1 Motivation
  - News shapes commodity markets, but most empirical work operates at daily resolution, missing the intra-day dynamics where information actually propagates
  - WTI crude oil specifically has a rich event structure (geopolitics, OPEC, EIA inventories, macro) that makes it well-suited for hourly-resolution analysis
- 1.2 Problem statement
  - How does news propagate into WTI liquidity at hourly resolution?
  - What is the lag structure, the directional asymmetry, the role of macro and event-type features?
- 1.3 Research questions
  - **RQ1**: lag structure of news impact on liquidity
  - **RQ2**: bearish vs bullish asymmetry in liquidity response
- 1.4 Contributions
  - (a) Empirical lag-and-asymmetry finding for WTI at hourly resolution, with peak news impact at lag +6h and a consistent bearish > bullish asymmetry. Supported by lag OLS results (4.2.3) and confirmed by TFT v2 attention patterns (4.3.7).
  - (b) Methodological contribution: a decomposition of LLM-extracted news features into orthogonal economic channels (supply, demand, risk premium), validated by inter-model calibration. The decomposition improves cross-model agreement on news sentiment by roughly 50 percentage points relative to a single composite score. Supported by 4.3.2–4.3.5.
  - (c) Comparative finding on news content representation: title-only and full-body sentiment extraction disagree on 41.6% of articles, with systematic asymmetry. Supported by 4.2.2.
- 1.5 Structure of the thesis

## Chapter 2 - Background

**Purpose:** Describe everything needed to follow Chapters 3 and 4.

- 2.1 Terminology
  - 2.1.1 WTI crude oil futures and the EIA inventory cycle
  - 2.1.2 Liquidity measures: log volume, Amihud illiquidity, Parkinson range
  - 2.1.3 NLP for finance: sentiment classification, structured extraction, prompt engineering
  - 2.1.4 Temporal Fusion Transformer: variable selection network, multi-head attention
- 2.2 State of the Art
  - 2.2.1 News impact studies on commodity markets (mostly daily-resolution)
  - 2.2.2 Financial NLP: FinBERT and its limitations
  - 2.2.3 LLM-based feature extraction in finance
  - 2.2.4 Temporal Fusion Transformer applications in financial time series

# Chapter 3 - Methods

## 3.1 Data sources and acquisition

Describes what data was collected, from where, and why. Establishes the foundation for all downstream analysis.

### 3.1.1 Commodity selection

Justifies the choice of WTI crude oil futures over other commodities. Discusses liquidity, geopolitical sensitivity, and news volume as selection criteria.

### 3.1.2 Market data

yfinance for WTI hourly OHLC data. Time range covered, aggregation to hourly grid, boundary null handling.

### 3.1.3 EIA inventory data

Weekly EIA inventory reports as fundamental context. Wednesday releases and their integration into the hourly time series.

### 3.1.4 News data

GDELT for headline scraping. Article body scraping methodology (BeautifulSoup). Corpus sizes: 13,690 (Phase 1) → ~11,433 (Phase 2 with usable_strict).

### 3.1.5 Persistence

Describes the SQLite database structure. Table schemas for liquidity, llm_features, article_entities. Rationale for using SQLite for reproducibility.

## 3.2 Temporal alignment

Explains how news articles (variable timing) are aligned with the hourly market grid. Central to the modeling approach.

### 3.2.1 Ceiling assignment to the next trading hour

Uses `dt.ceil` (not `dt.round`) to prevent leakage: an article published at 14:32 is assigned to hour 15:00, not hour 14:00. Preserves temporal causality.

### 3.2.2 Forward-assignment of off-hours articles

Articles published outside trading hours (weekends, holidays) are assigned to the next trading hour. Prevents these articles from being lost or leaked backwards.

### 3.2.3 Joined output

Describes the final joined output: hourly grid with market data + aggregated LLM features + calendar covariates. Documents the boundary null handling for DXY and VIX (holidays), and zero-fill for first-row log_return and amihud.

## 3.3 News feature extraction approaches

Compares two approaches to extracting features from news articles. Central methodological contribution of Phase 2.

### 3.3.1 FinBERT sentiment extraction

Phase 1 approach. Three-class sentiment labels (bearish/neutral/bullish) per article, aggregated hourly. Simplicity vs limitations of the categorical output.

### 3.3.2 Structured LLM extraction with Claude Haiku

Phase 2 approach. LLM-based extraction with structured schema. Richer signal per article than FinBERT.

### 3.3.3 Tool-use API and schema enforcement

Uses the Anthropic tool-use API to enforce structured output. Guarantees valid JSON, prevents hallucinated fields, enables schema validation.

### 3.3.4 Channel decomposition

Introduces the three-channel decomposition: supply_impact, demand_impact, risk_premium. Each on `[-1, +1]`. Motivation: FinBERT collapses multiple economic dimensions into one axis.

### 3.3.5 Magnitude, event type, entities, certainty, and time horizon

Additional fields extracted by Haiku v2. Magnitude and certainty for weight, event_type_primary and time_horizon as categoricals, entity list for entity normalization.

## 3.4 Filtering strategies

Two approaches to deciding which articles are usable for modeling.

### 3.4.1 Regex/keyword heuristic (Phase 1)

`body_valid` regex-based filter. Fast, deterministic, false negative-heavy.

### 3.4.2 LLM-judged usable flag (Phase 2)

`usable=1` flag produced by the LLM. Semantically-grounded judgment. Introduces `usable_strict=1` variant that additionally requires non-zero channel scores.

### 3.4.3 Comparative analysis

Cross-validation of regex vs LLM filter on a manual audit sample. Confusion matrices, agreement rates, false positive/negative analysis.

## 3.5 Statistical models

Phase 1's regression-based methods for RQ1 and RQ2 evidence.

### 3.5.1 Lag OLS specification

Distributed lag regression: `log_volume(t) = β_0 + Σ β_k * sentiment(t-k) + controls`. Motivation, specification, testing procedure. Brief note that an exploratory VAR was fitted and abandoned (sparse sentiment signal), kept as a one-paragraph mention in §4.2.4 with no standalone method section.

## 3.6 Temporal Fusion Transformer

Central architecture for both TFT v1 and TFT v2. Describes the model at the level of design choices rather than per-variant specifics.

### 3.6.1 Architecture overview

High-level description of the TFT: Variable Selection Networks, LSTM encoder-decoder, attention mechanism, quantile prediction. Why chosen over alternatives.

### 3.6.2 Input typing

How features are typed in pytorch-forecasting: `time_varying_unknown_reals`, `time_varying_known_reals`, `time_varying_unknown_categoricals`. Consequences for the VSN processing.

### 3.6.3 Forecast configuration

Encoder length, prediction horizon, quantile levels. Rationale for 48-hour encoder window. Single-horizon (v1) vs multi-horizon (v2) prediction.

### 3.6.4 Architectural hyperparameters

hidden_size, attention_head_size, hidden_continuous_size, dropout. Rationale for the specific values chosen. Trade-offs against overfitting.

### 3.6.5 Training procedure

Adam optimizer, learning rate schedule, early stopping. Loss functions (QuantileLoss for v1, MultiLoss for v2). Split methodology, buffer windows to prevent leakage.

### 3.6.6 Evaluation and interpretation outputs

Persistence-relative MAE reduction as the headline metric (raw val_loss is not comparable across instances, per §4.3.8), evaluated per target/horizon/slice (val, test, pre-war, war). Interpretability: VSN feature importance and attention weights. Research-question diagnostics: attention for RQ1; a bearish-vs-bullish Welch t-test on predicted volume for RQ2 (a group-mean estimand, deliberately distinct from the lag OLS marginal coefficients).

## 3.7 Inter-model calibration methodology

Methodology for cross-validating LLM outputs across model revisions. Applied in §4.3.4.

### 3.7.1 The validation problem and our approach

Why calibration is needed: different LLM revisions may produce different outputs on same articles. Approach: run the same articles through both versions and compare.

### 3.7.2 Sample design

How to select articles for the calibration. Random sampling, event type stratification, size of sample.

### 3.7.3 Extraction procedure

How to run both LLM versions on the calibration sample. Handling of temperature, seed, prompt versions.

### 3.7.4 Agreement metrics

Metrics used to measure agreement: correlation coefficients for continuous features,

# Chapter 4 - Experiments

## 4.1 Overview

This chapter includes a series of experiments that involve Phase 1, changes that enable Phase 2 and the 2 versions of the TFT.

### 4.2.1 Phase 1 data and feature setup

Describes the Phase 1 corpus (13,690 articles, regex-filtered), the FinBERT sentiment extraction, hourly aggregation, and the market/macro/calendar features. Establishes the input data for all Phase 1 methods.

### 4.2.2 Headline bias experiment

Compares sentiment predicted from title-only vs title+body inputs, demonstrating that FinBERT sentiment is substantially different when body is included. Motivates the LLM-based approach in Phase 2.

### 4.2.3 Lag OLS: RQ1 and RQ2

Distributed-lag regression of log_volume on the FinBERT class probabilities across lags k = 0 (contemporaneous) through 12. Identifies the +6h lag as the peak effect (RQ1) and establishes bearish > bullish directional asymmetry (RQ2). This is the primary Phase 1 evidence for both research questions.

### 4.2.4 Phase 1 summary

Consolidates the RQ1 and RQ2 findings from Phase 1. Notes the abandoned exploratory VAR (sparse sentiment signal) and motivates the transition to Phase 2's richer feature extraction and deep-learning approach.

## 4.3 Phase 2 - Temporal Fusion Transformer

### 4.3.1 Phase 2 data and feature setup

Introduces the Phase 5 batch and the expanded corpus through May 2026. Documents the production LLM extraction pipeline, hourly aggregation for LLM features, and the temporal split adjustments for the new dataset.

### 4.3.2 LLM feature extraction

Describes the Haiku v2 schema, including channel decomposition (supply/demand/risk_premium), magnitude, certainty, event type, time horizon, and entity extraction. Documents the canonical entity normalization (71 entities, Appendix D).

### 4.3.3 LLM `usable` flag and filter comparison

Compares regex filter (Phase 1) against LLM filter (Phase 2). Reports agreement rate, false positive analysis, and manual audit. Introduces the `usable_strict=1` variant used for v2 training.

### 4.3.4 Inter-model calibration of LLM features

Cross-validates Haiku v1 vs Haiku v2 outputs. Reports the calibration process for consistency across LLM revisions.

### 4.3.5 Temporal Fusion Transformer v1

First deep-learning model using Phase 2 features. 113k parameters, single target (log_volume), 1h horizon, val_loss=0.204. Reports feature importance, attention pattern (peak -4h, secondary -27/-28h), and directional asymmetry analysis.

### 4.3.6 Channel decomposition response

Discusses how the v2 schema iteration was motivated by v1's findings. Explains why channels were introduced and how they affect subsequent modeling choices.

### 4.3.7 Temporal Fusion Transformer v2

Main Phase 2 model. Multi-horizon multi-target TFT with entity flags and channel decomposition. Trained on `usable_strict=1` corpus.

#### 4.3.7.1 Success criteria

Declares the three criteria TFT v2 was designed to satisfy, prior to presenting model design and results.

- **Criterion 1**: channels and entities are visible drivers of prediction (qualitative).
- **Criterion 2**: prediction accuracy substantially beats persistence baseline in the 1-12h window (quantitative).
- **Criterion 3**: multi-horizon error curve consistent with Phase 1's lag OLS peak at +6h (quantitative). Pre-registration prevents post-hoc rationalization; evaluation is deferred to §4.3.7.7.

#### 4.3.7.2 Model design

Architecture (hidden_size=32, dropout=0.15), 298k parameters, feature set (channels, entity flags, categoricals, market, macro, calendar), aggregation rules, training procedure (patience=10, best val_loss 0.427 at epoch 10), 60/20/20 split with test set straddling war onset.

#### 4.3.7.3 Predictive performance

Three tables (log_volume, amihud, price_range) with MAE by horizon and slice. Persistence baseline comparison shows 41-73% reduction on log_volume, 52-54% on amihud, failure on price_range war slice. Justifies focus on log_volume for RQ1/RQ2 analysis.

#### 4.3.7.4 Feature contributions

Top 10 features by VSN importance. VIX at 23.6%, demand_impact at 13.6%, five entity flags in top 10. Includes ablation reference paragraph explaining that proper categorical encoding was needed to unlock channel-driven prediction (full ablation in Appendix C).

#### 4.3.7.5 Lag structure analysis: evidence for RQ1

Per-horizon prediction error curve peaks at +12h (73% reduction). Attention pattern peaks at -17h with bearish/bullish divergence (bearish -17h, bullish -12h). Partial corroboration of lag OLS peak at +6h through the +6h to +12h range.

#### 4.3.7.6 Directional asymmetry analysis: evidence for RQ2

Twenty t-tests (5 horizons × 4 slices) on bearish vs bullish predicted log_volume. One significant at +3h pre-war (p=0.013, +0.660 log_volume, matches lag OLS direction). Interpretation: TFT and lag OLS answer different questions; the asymmetry is real in the data but does not fully survive TFT's integrated prediction.

#### 4.3.7.7 Success criteria evaluation

Scorecard: Criterion 1 (channels interpretable) PASS, Criterion 2 (asymmetry) PARTIAL PASS, Criterion 3 (multi-horizon peak) PARTIAL PASS. Designates v2 configuration as canonical for the thesis.

### 4.3.8 Methodological comparison: Phase 1 versus Phase 2

Comparative analysis of the two phases as complementary methodologies rather than competing approaches.

#### 4.3.8.1 What Phase 2 changed methodologically

Four levels of change: corpus expansion (Phase 1 pre-war regex-filtered vs Phase 2 LLM-filtered with war coverage), feature extraction (FinBERT sentiment scalar vs Haiku channels + entities + categoricals), entity normalization (Phase 2 addition), analytical framework (regression vs multi-horizon forecasting). Feature extraction includes a "tone versus price impact" deep-dive: FinBERT measures surface tone while Haiku scores net WTI price impact; they agree overall (Pearson 0.54) but sign-flip on geopolitical news (negative tone, bullish price), shown with example headlines, the empirical fingerprint of FinBERT's limitation and a reconciliation of the RQ2 divergence.

#### 4.3.8.2 Why cross-phase numbers are not comparable

Direct v1-vs-v2 loss comparison rejected: different loss functions, validation periods, horizons, feature counts, and upstream corpora (so v1 val_loss 0.204 vs v2 0.427 is meaningless). Fair comparison is persistence-relative reduction on a shared target/horizon, where both substantially beat persistence. Phase 2's contribution is analytical richness, not a numerical win. (Merges the former 4.3.8.2 and 4.3.8.3.)

## Chapter 5 - Discussion

**Purpose:** Interpret the findings, contextualize them in the literature, identify limitations.

- 5.1 Findings on RQ1 (lag structure)
  - The lag +6h peak in OLS, confirmed by TFT v2 per-horizon error curve and direction-conditioned attention (bearish -6h)
  - What this means: news propagates over a +6 to +12h window, not instantaneously
- 5.2 Findings on RQ2 (asymmetry)
  - Bearish > bullish in OLS is statistically robust and remains the primary evidence (a marginal-coefficient sensitivity)
  - The TFT group-mean asymmetry test is null, but this is a DIFFERENT, coarser estimand (group means of predicted volume vs OLS marginal coefficients), NOT underpowered and NOT a failed replication: verified robust to the sentiment threshold (0.0 to 0.5), groups hold hundreds of samples, and the ground-truth target is itself near-symmetric over the test period
  - Reconciliation via tone versus price impact: FinBERT (tone) and Haiku (net price impact) agree overall (Pearson 0.54) but sign-flip on geopolitical news (the high-volume subset), so the two phases' sentiment-sign analyses can point differently
  - The model keys on risk/salience features (VIX, channels, Hormuz-region entities), direction-agnostic, which is why no bearish>bullish separation appears in its point predictions
- 5.3 Methodological contributions
  - Channel decomposition as a response to inter-model calibration failure (grounded in Kilian 2009's supply/demand/risk shock decomposition)
  - Tone versus price impact: tone classifiers (FinBERT) and LLM price-direction reasoning (Haiku) measure different constructs; the LLM captures the asset-specific causal step (supply threat to higher price) that a tone classifier cannot
  - LLM-as-filter as an alternative to regex heuristics for noisy news corpora
  - Phase-based pipeline progression as a reproducible template for similar empirical work
- 5.4 Limitations
  - Inter-LLM calibration is methodologically weaker than human ground truth
  - price_range regime-extrapolation failure: trained only on the pre-war regime, the model reverts toward the mean and underperforms persistence on the unseen war-regime volatility (§4.3.7.3)
  - Sentiment findings are construct-dependent (tone vs price impact); RQ2 conclusions are conditional on which sentiment definition is used
  - GDELT corpus skews retail-aimed news rather than institutional-grade feeds
  - 24-month dataset with regime change (Iran war) introduces non-stationarity
  - Single-commodity scope; results may not generalize to other commodities or asset classes
- 5.5 Future work
  - Cross-commodity spillovers (out of scope for this thesis)
  - Larger calibration with human raters at small scale
  - Robustness across commodities (gold, copper, gas)
  - Extending the entity vocabulary with domain-aware embeddings

## Chapter 6 - Conclusion

**Purpose:** Wrap up. Restate the contributions, summarize the findings, point to future work.

- 6.1 Summary of contributions
- 6.2 Summary of findings
- 6.3 Closing remarks

## Appendices

Appendices are a top-level section following Chapter 6 (Conclusion), drafted in `6-appendices.md`. Letters are ordered by first mention in the text.

| ID  | Title                               | Scope                                                                                             | Status                     | First referenced |
| --- | ----------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------- | ---------------- |
| A   | Extraction prompt                   | Verbatim Haiku v2 system prompt and the `extract_article_features` tool schema                    | Drafted (prompt to insert) | §3.3.2           |
| B   | LLM extraction schema (v1 vs v2)    | Field-by-field Schema v1 vs v2 comparison and the rationale for each change                       | Drafted                    | §3.3.2           |
| C   | TFT v2 ablation and hyperparameters | Three-variant ablation (v2.0/v2.1/v2.2) with val_loss, test MAE@1h, top feature; canonical config | Drafted                    | §4.3.3           |
| D   | Canonical entity list               | The 71 canonical entities (from `03_src/nlp/llm_features.py`); alias mappings to add              | Drafted                    | §4.3.7.4         |
| E   | Inter-model calibration comparison  | Article-by-article Haiku vs GPT scores on the 30-article calibration sample                       | Not started                | §4.3.4           |
| F   | Extended metrics tables             | Full TFT v2 metrics (3 targets × 4 horizons × 4 slices), MAE and RMSE                             | Not started                | §4.3.7.3         |
| G   | Reproducibility statement           | Software libraries and versions, models/APIs, data sources (GDELT/EIA/yfinance), hardware, seed 42 | Drafted                    | §3.1, §4.3.7     |

Optional extras (add if space allows): a sample of LLM-rejected articles illustrating the body_valid-vs-usable disagreement (§4.3.3); the full per-lag OLS regression tables with confidence intervals (§4.2.3).

## Tables

Central registry of tables with their location and content. Numbered sequentially by first appearance in the text.

| ID   | Title                                         | Chapter/Section | Status              |
| ---- | --------------------------------------------- | --------------- | ------------------- |
| 4.1  | Phase 1 modeling dataset construction         | §4.2.1          | Complete (in draft) |
| 4.2  | Lag OLS regression by horizon (RQ1/RQ2)       | §4.2.3          | Complete (in draft) |
| 4.3  | Phase 1 vs Phase 2 corpus contrast            | §4.3.1          | Complete (in draft) |
| 4.4  | Regex vs LLM filter 2x2 contingency           | §4.3.3          | Complete (in draft) |
| 4.5  | Inter-model re-calibration metrics (v1 vs v2) | §4.3.6          | Complete (in draft) |
| 4.6  | TFT v2 log_volume MAE vs persistence          | §4.3.7.3        | Complete (in draft) |
| 4.7  | TFT v2 amihud MAE vs persistence              | §4.3.7.3        | Complete (in draft) |
| 4.8  | TFT v2 price_range MAE vs persistence         | §4.3.7.3        | Complete (in draft) |
| 4.9  | TFT v2 feature importance (VSN top 10)        | §4.3.7.4        | Complete (in draft) |
| 4.10 | Directional asymmetry t-tests (16 tests)      | §4.3.7.6        | Complete (in draft) |

## Figures

Central registry of figures with their location and content.

| ID  | Title                                                                                                  | Type                     | Chapter/Section | Status                                               |
| --- | ------------------------------------------------------------------------------------------------------ | ------------------------ | --------------- | ---------------------------------------------------- |
| 1   | Placeholder for first figure in Chapter 1 (Introduction)                                               | TBD                      | §1.X            | Not planned                                          |
| 2   | Placeholder for first figure in Chapter 2 (Background)                                                 | TBD                      | §2.X            | Not planned                                          |
| 3.1 | Data pipeline overview (Phase 1 vs Phase 2 flow)                                                       | Diagram                  | §3.X            | Not started                                          |
| 3.2 | Corpus size comparison (Phase 1 vs Phase 2, articles per month)                                        | Bar chart                | §3.X            | Not started                                          |
| 4.1 | Lag OLS coefficients                                                                                   | Line plot                | §4.2.3          | Complete (`lag_coefficients.png`)                    |
| 4.2 | TFT v1 feature importance (news vs market features)                                                    | Horizontal bar chart     | §4.3.5          | Complete (`tft_feature_importance.png`)              |
| 4.3 | TFT v1 encoder attention by lag                                                                        | Line plot                | §4.3.5          | Complete (`tft_attention_weights.png`)               |
| 4.4 | TFT v2 attention pattern (mean encoder attention across 48 lags, disaggregated by sentiment direction) | Line plot                | §4.3.7.5        | Complete (`attention_by_sentiment_tftv2.2-exp2.png`) |
| 4.5 | TFT v2 feature importance (top 10 features from Variable Selection Network)                            | Horizontal bar chart     | §4.3.7.4        | Not started                                          |
| 4.6 | Per-horizon prediction error curve (MAE and persistence reduction across horizons for log_volume)      | Line plot with dual axis | §4.3.7.5        | Not started                                          |
| 4.7 | Directional asymmetry (bearish vs bullish predicted log_volume by horizon and slice)                   | Grouped bar chart        | §4.3.7.6        | Not started                                          |
| 5.1 | Placeholder for discussion synthesis figure                                                            | TBD                      | §5.X            | Not planned                                          |
| 6.1 | Placeholder for future work overview                                                                   | TBD                      | §6.X            | Not planned                                          |
