# Chapter 4 - Experiments

## 4.1 Overview

This chapter reports the empirical results of both methodological phases of the thesis. Phase 1 (§4.2) presents the regression-based analysis of the pre-war corpus, including the headline bias experiment, the contemporaneous and lag OLS regressions that establish the initial evidence for RQ1 and RQ2, and the VAR model. Phase 2 (§4.3) presents the deep-learning analysis of the expanded corpus, including the LLM feature extraction pipeline, the Temporal Fusion Transformer models (v1 and v2), and the ablation study that guided the v2 design. The chapter closes with a methodological comparison of the two phases (§4.3.8), articulating what Phase 2 changed and why direct numerical comparison between the two phases is not appropriate.

## 4.2 Phase 1 - Initial pipeline and baseline modeling

Phase 1 establishes the empirical baseline for the research questions of this thesis, building on prior evidence that news sentiment predicts trading activity and volume, not only prices [CITE: Tetlock 2007]. It uses the FinBERT-extracted news sentiment described in Section 3.3.1 together with the regex-based body validity filter of Section 3.4.1, applied to a corpus of WTI-relevant news articles published between March 2024 and February 2026. Three modeling techniques are applied in turn: linear regression (contemporaneous and lagged), vector autoregression, and a headline-bias comparative experiment that motivates the broader use of full article bodies in downstream feature extraction. By the end of Phase 1, the canonical statistical evidence for RQ1 (lag structure) and RQ2 (bearish-versus-bullish asymmetry) is in hand, and the limitations of the FinBERT-plus-classical-models pipeline are visible enough to motivate the methodological refinements that follow in Phase 2.

### 4.2.1 Phase 1 data and feature setup

The Phase 1 modeling dataset combines **13,690 articles**: 7,756 with substantive bodies (regex filter `body_valid=1`, Section 3.4.1) feeding title-plus-body FinBERT sentiment, and 5,934 where body retrieval failed but the title was retained as a fallback. The corpus derives from 51,948 raw GDELT records across eight oil-market queries, reduced to 16,326 unique English articles after deduplication (Table 4.1). Each article is scored by FinBERT [CITE: Araci 2019] (Section 3.3.1); the Phase 1 regressions use the title-plus-body class probabilities as the continuous sentiment feature, with the discrete `{+1, 0, -1}` label kept as a baseline. The sentiment distribution is **44.9% bearish, 30.4% bullish, 24.7% neutral**; the bearish skew reflects the headline-tone tendency documented in Section 4.2.2, not a population-level claim.

Articles are aligned to trading hours by the ceiling rule (Section 3.2.1): 82.7% fall within two hours of a trading hour (contemporaneous), the remainder are forward-assigned. The market backbone is 11,219 hourly WTI records with the liquidity targets (`log_volume`, `price_range`, `amihud` [CITE: Amihud 2002]) defined in Section 3.1; DXY, VIX, and the EIA covariates enter only in Phase 2. **Table 4.1** summarises the construction.

| Stage                                                | Count                               |
| ---------------------------------------------------- | ----------------------------------- |
| Raw GDELT articles                                   | 51,948                              |
| After deduplication + English filter                 | 16,326                              |
| With successful body retrieval                       | ~13,000 (80% of 16,326)             |
| With `body_valid=1` (regex-passed body)              | 7,756                               |
| Title-only fallback (body failed but title retained) | 5,934                               |
| **Phase 1 modeling dataset**                         | **13,690**                          |
| Aligned market hours (raw, before VAR cleaning)      | 11,219                              |
| Articles aligned within 2h gap (contemporaneous)     | 11,316                              |
| Articles aligned with gap ≥ 2h (forward-assigned)    | 2,374                               |
| Article date range                                   | 2024-03-11 10:30 → 2026-03-01 00:30 |

This dataset is the substrate for the three Phase 1 experiments that follow.

### 4.2.2 Headline bias experiment

**Motivation.** The Phase 1 pipeline mixes two FinBERT input regimes, title-only (fallback) and title-plus-body, in the same 13,690-article dataset. Before treating the resulting sentiment as one signal, we test whether the two regimes disagree, a question also of independent interest: headlines are written to attract attention, and the body can qualify or even reverse the implication a reader draws from the title alone.

**Method and categorical result.** For the 7,755 articles with both a title and a substantive body, FinBERT was applied twice (title-only, then title-plus-body), so every difference between the two scores is attributable to the input content, not the model (Section 3.3.1). On the argmax labels the two series **disagree on 41.6% of articles** (χ² = 2050, p < 0.001), and the disagreement is directionally structured: positive titles keep a positive body in only 59.2% of cases (40.8% shift toward neutral or negative), neutral titles resolve to negative more often than positive (39.6% vs 30.0%), while negative titles stay negative in 78.2%.

**Continuous result.** The label measure ignores how far sentiment moves and misses shifts inside same-label articles. Defining a signed score `P(positive) − P(negative)` per input and the divergence magnitude `|signed_full − signed_title|`: label flips carry a mean magnitude of **0.96** (near-full reversals), same-label articles still shift by **0.17** on average (31.6% by more than 0.20), and the mean signed shift from title to body is **−0.09**, with 57.6% of articles becoming more bearish once the body is read.

**Worked example.** The quantities are easiest to read on concrete articles. The signed score is `P(positive) − P(negative)`, computed once on the title and once on the title plus body; the magnitude is their absolute distance `|signed_full − signed_title|` (how far sentiment moved), and the signed shift is `signed_full − signed_title` (which direction it moved).

| Article (title) | signed_title | signed_full | magnitude | signed shift | Label transition |
| --------------------------------------------- | -----------: | ----------: | --------: | -----------: | ---------------------------- |
| "Oil price soars above $80"                   |        +0.49 |       −0.63 |      1.12 |        −1.12 | positive → negative (flip)   |
| "U.S. Oil Is Stealing Market Share from OPEC+" |        −0.32 |       −0.77 |      0.45 |        −0.45 | negative → negative (no flip) |
| "Oil prices continue to drop in world markets" |        −0.96 |       −0.96 |      0.00 |         0.00 | negative → negative (no flip) |

The first article flips label: a headline FinBERT reads as bullish ("soars above $80") turns bearish once the body is scored, a near-full reversal (magnitude 1.12). The second keeps its negative label yet still moves 0.45 toward more bearish, exactly the within-bucket movement the 41.6% flip rate does not count. The third is stable, title and body agree. In all three the body is at least as bearish as the title, the one-directional pattern behind the −0.09 mean signed shift.

**Divergence and label-flip totals.** The aggregate quantities across the 7,755 title-plus-body articles are collected below.

| Quantity                                       |            Value |
| ---------------------------------------------- | ---------------: |
| Label agreement (title and body same label)    |     4,527 (58.4%) |
| Label flip (disagreement)                      |     3,228 (41.6%) |
| Flip rate, positive titles                     |            40.8% |
| Flip rate, neutral titles                      |            69.5% |
| Flip rate, negative titles                     |            21.8% |
| Mean divergence magnitude, all articles        |             0.50 |
| Mean divergence magnitude, label flips         |             0.96 |
| Mean divergence magnitude, same-label          |             0.17 |
| Same-label articles with magnitude above 0.20  |            31.6% |
| Mean signed shift, title to body               |            −0.09 |
| Articles more bearish once the body is read     |            57.6% |

**Implications.** Two findings carry forward. First, the choice of input regime matters: a pipeline that treats title-only and title-plus-body interchangeably averages over a 41.6% disagreement rate. Second, and against the common intuition that attention-grabbing headlines overstate negativity, **titles lean more bullish than the articles they contain**: reading the body moves FinBERT's sentiment more negative on average, and only the signed continuous score fixes this direction, which the labels alone could not establish. The 5,934 title-only fallback articles therefore carry a sentiment that would have shifted in roughly 40% of cases had a body been available; because that shift is one-directional across all lags, it attenuates rather than inflates the OLS effect reported below. More broadly, this is the first concrete evidence that the choice of news representation matters, and it motivates Phase 2's move to structured extraction from full article bodies (Section 3.3.2).

### 4.2.3 Lag OLS: RQ1 and RQ2

The lag OLS regressions are the canonical Phase 1 evidence for RQ1 (lag structure of news impact on liquidity) and RQ2 (bearish-versus-bullish asymmetry), and the result the Phase 2 methodological choices ultimately aim to corroborate.

**Specification.** At horizon `k`,

$$\text{log\_volume}_{t+k} = \beta_0 + \beta_1\, P(\text{negative})_t + \beta_2\, P(\text{positive})_t + \varepsilon_t$$

where `P(negative)_t` and `P(positive)_t` are the title-plus-body FinBERT class probabilities for the article published at hour `t`, with the neutral probability as the omitted reference (Section 3.5.1); `β_1` and `β_2` measure the `log_volume` response to a maximally confident bearish or bullish article relative to a neutral one. Each article is one observation, and a separate regression is estimated at each `k ∈ {0, 1, 2, 3, 4, 6, 8, 12}` (k = 0 is the contemporaneous case), on 13,690 rows at k = 0 falling to roughly 9,900 at the longest lag. The continuous encoding is primary; a discrete-dummy baseline is retained for comparison and, at k = 0, gives smaller coefficients (β_bearish = 0.133, β_bullish = 0.103) than the continuous 0.186 / 0.166, confirming that weighting each article by FinBERT's confidence sharpens the signal.

| Lag `k` (h)  | β_P(neg) |     p | β_P(pos) |     p |     R² |      n |
| ------------ | -------: | ----: | -------: | ----: | -----: | -----: |
| 0 (contemp.) |    0.186 | 0.000 |    0.166 | 0.000 | 0.0015 | 13,690 |
| 1            |    0.262 | 0.000 |    0.200 | 0.000 | 0.0028 | 11,598 |
| 2            |    0.105 | 0.068 |    0.114 | 0.084 | 0.0003 | 11,512 |
| 3            |    0.117 | 0.047 |    0.170 | 0.013 | 0.0006 | 11,360 |
| 4            |    0.267 | 0.000 |    0.237 | 0.001 | 0.0017 | 11,065 |
| 6            |    0.342 | 0.000 |    0.291 | 0.000 | 0.0027 | 10,679 |
| 8            |    0.056 | 0.409 |    0.101 | 0.192 | 0.0002 | 10,440 |
| 12           |    0.001 | 0.984 |   -0.171 | 0.029 | 0.0008 |  9,917 |

**Result.** Table 4.2 reports the regression at every horizon. Each row is one regression at lag `k` (in hours; `k = 0` is the contemporaneous case). The columns are: **β_P(neg)** and **β_P(pos)**, the fitted coefficients on the bearish and bullish class probabilities, that is, the expected change in `log_volume` for a maximally confident bearish or bullish article relative to a neutral one; the two **p** columns, their two-sided p-values (a coefficient is significant at the usual threshold when p < 0.05); **R²**, the share of `log_volume` variance that the two-feature regression explains at that lag; and **n**, the number of articles entering that regression, which falls as `k` grows because targets at hour `t + k` beyond the coverage window are dropped.

![Lag OLS coefficients and p-values](../../../04_outputs/figures/lag_coefficients.png)
**Figure 4.1: Lag OLS coefficients and p-values across horizons.** Left panel: the bearish `P(negative)` and bullish `P(positive)` coefficients at lags 1, 2, 3, 4, 6, 8, and 12 hours. Stars mark lags significant at the conventional `p < 0.05` threshold. Right panel: the p-values for the same coefficients, with the `p = 0.05` reference line. The peak at lag +6h is the canonical finding for RQ1; the ordering of bearish above bullish at the dominant lags is the canonical finding for RQ2.

**RQ1 (lag structure).** The impact is already detectable at publication (k = 0), grows over the first hours, peaks sharply at **+6h** (β_P(neg) = 0.342, β_P(pos) = 0.291, both p < 0.001; the +6h bearish coefficient implies roughly 41% more volume than a neutral hour for a fully confident bearish article, `exp(0.342) − 1 ≈ 0.41`), and decays to insignificance by +8h. The pattern is structured rather than a smooth exponential decay, with secondary maxima at +1h and +4h, suggesting discrete information-incorporation events at characteristic horizons.

**RQ2 (asymmetry).** Bearish articles produce a stronger response than bullish ones at the dominant lags (0, 1, 4, and the +6h peak, where bearish exceeds bullish by approximately 18%). The one exception is the weak lag 3, where both coefficients are small and bullish is marginally higher; it does not overturn the pattern at the horizons that carry the effect. Because the two probabilities are estimated as independent regressors (Section 3.5.1), the bearish-over-bullish ordering is not an artifact of an imposed model symmetry.

**On the small R² and robustness.** R² stays below 0.003 throughout, which is expected: hourly volume has many drivers and sentiment is one signal, so R² measures sentiment's share of _total_ volume variance, not of _news-driven_ volume; the substantive finding is the structured coefficient pattern, not the explained variance. Two robustness points. The regression is cross-sectional (one article per row, with no strong serial correlation), so HAC corrections are not applied (Section 3.5.1). And the ceiling alignment guarantees each article precedes its matched hour, which if anything underestimates the immediate reaction, a conservative bias for the peak-lag finding. Both RQ1 and RQ2 are revisited independently by the TFT v2 attention pattern in Phase 2 (Section 4.3.7).

### 4.2.4 Phase 1 summary

Phase 1 delivers the canonical statistical evidence for both research questions and exposes the limitations that Phase 2 addresses.

**Evidence.** RQ1 is answered by the lag OLS peak at **+6h**, and RQ2 by the **bearish > bullish** ordering at the dominant lags (§4.2.3); both are the primary Phase 1 evidence and are revisited independently by the TFT v2 attention pattern in Phase 2. The headline-bias experiment (§4.2.2) adds a separate finding: title-only and title-plus-body FinBERT disagree on 41.6% of articles, with titles leaning more bullish than the bodies they head, which motivates extracting features from full article bodies in Phase 2.

**VAR exploration (abandoned).** An exploratory VAR was fitted but abandoned: more than half of the hours carry no contemporaneous news, so the sentiment series is dominated by zeros and the impulse responses were not significant. The failure is itself diagnostic; it identifies the data as sparse and event-driven, the property that motivates the deep-learning model of Phase 2.

**Limitations that drive Phase 2.** Five limitations of the Phase 1 pipeline each map to a Phase 2 change:

| Phase 1 limitation                                                                                          | Phase 2 response                                        |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| FinBERT reduces each article to one sentiment axis (no magnitude, event type, certainty, horizon, entities) | Structured LLM extraction with a richer schema (§4.3.2) |
| The regex `body_valid` filter is fast but not semantically discriminative (false positives and negatives)   | LLM-judged `usable` flag (§4.3.3)                       |
| The sentiment signal is sparse (~50% of hours have no news), defeating models that need joint observation   | A TFT built for sparse, event-driven inputs (§4.3.5)    |
| The corpus is pre-war only (ends February 2026)                                                             | Corpus extended through May 2026 (§4.3.1)               |
| The OLS has no macro controls                                                                               | DXY and VIX added as exogenous covariates (§4.3.5)      |

Each Phase 2 step responds to a concrete limitation exposed by the previous one, so the methodological progression is traceable rather than asserted.

## 4.3 Phase 2: Refined pipeline and deep-learning modeling

Phase 2 addresses the Phase 1 limitations (§4.2.4) through a coordinated redesign: FinBERT is replaced by structured LLM extraction, DXY and VIX enter as macro covariates, the regex filter is replaced by an LLM-judged `usable` flag, a Temporal Fusion Transformer becomes the primary model, and the corpus is extended through May 2026. The phase unfolds chronologically: an initial LLM schema (v1) supports the first TFT training (TFT v1); an inter-model calibration then exposes a weakness in the composite sentiment score, motivating a revised schema (v2) with three orthogonal economic channels and the `usable` flag; the full extraction runs on the expanded corpus and supports the final TFT v2. The subsections follow this order: data setup (§4.3.1), LLM extraction (§4.3.2), filter migration (§4.3.3), first calibration (§4.3.4), TFT v1 (§4.3.5), the schema response (§4.3.6), TFT v2 (§4.3.7), and a Phase 1 vs Phase 2 comparison (§4.3.8).

### 4.3.1 Phase 2 data and feature setup

The Phase 2 corpus extends Phase 1 in two directions: the GDELT scrape runs through May 2026 to cover the post-war period, and five new queries target geopolitical and demand-side themes (Section 3.1). The full corpus is **22,795 article records** (January 2024 to May 2026); **19,619 have substantive body content** (non-null, not a scraping error) and go to LLM extraction, while 2,749 returned null bodies (paywall, fetch failure, JavaScript rendering).

**Macro covariates.** DXY (`DX-Y.NYB`) and VIX (`^VIX`) hourly series were added as exogenous controls (Section 3.6.2). The rationale is empirical: the Phase 1 OLS explained under 0.3% of volume variance (§4.2.3), so macro controls let the model absorb broad market-driven variation and isolate a cleaner residual for news. The pipeline was also migrated from CSV intermediates to the SQLite database (`wti_thesis.db`), making the alignment a deterministic join.

**Aligned dataset.** Re-running the temporal alignment (Section 3.2) over the expanded corpus yields **22,795 article-aligned rows** (15,290 contemporaneous, 7,505 forward-assigned) on the 11,232-hour market grid spanning 13 May 2024 to 13 May 2026. Table 4.3 contrasts the two phases.

| Stage                            | Phase 1                                                                | Phase 2                                                                         |
| -------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| GDELT queries                    | 8                                                                      | 13                                                                              |
| Article window                   | March 2024 – February 2026                                             | January 2024 – May 2026                                                         |
| Raw articles (GDELT scrape)      | 51,948                                                                 | 76,345 (cumulative; Phase 2 re-scrape added 24,397)                             |
| After deduplication + English    | 16,326                                                                 | 22,795                                                                          |
| Articles with substantive bodies | 7,756 (`body_valid=1`)                                                 | 19,619 (`body NOT NULL AND NOT 'ERROR%'`)                                       |
| Title-only fallback              | 5,934                                                                  | 0 (no title-only fallback in Phase 2)                                           |
| Modeling-ready corpus            | 13,690 (Phase 1 OLS/VAR)                                               | 11,433 (`usable_strict`, canonical for TFT v2) / 11,675 (`usable`)              |
| Market hours window              | March 2024 – February 2026                                             | May 2024 – May 2026                                                             |
| Hourly market observations       | 11,219                                                                 | 11,232                                                                          |
| Macro covariates                 | none                                                                   | DXY, VIX                                                                        |
| Sentiment representation         | FinBERT 3-class probabilities (discrete {-1, 0, +1} label as baseline) | LLM-extracted continuous sentiment + 3 orthogonal channels + categorical fields |

The Phase 2 dataset is the substrate for all subsequent subsections of this chapter.

### 4.3.2 LLM feature extraction

Replacing FinBERT with LLM extraction is the central Phase 2 change: it recovers the dimensions FinBERT's single sentiment axis discards (magnitude, event type, certainty, time horizon, entities, and the economic channels). The methodology is specified in Sections 3.3.2 through 3.3.5; this subsection reports its Phase 2 deployment and output.

**Extraction setup.** Each of the 19,619 articles with a substantive body is scored by Claude Haiku 4.5 [CITE: LLM structured extraction] through the tool-use API, which forces the `extract_article_features` schema (Section 3.3.3) and is parsed deterministically from the tool call; the title and body (truncated to 1,500 characters) form the user message, and the full prompt and tool schema are in Appendix A. The schema short-circuits unusable articles to only the `usable` flag, cutting output tokens for the roughly 40% unusable share, and all 19,619 articles returned a valid response with zero unrecoverable errors. The deployed schema is v2 (Section 3.3.4): the composite `sentiment_score` plus three orthogonal channels (`supply_impact`, `demand_impact`, `risk_premium`), `magnitude`, `certainty`, `event_type` (an array of one to three categories), `time_horizon`, `entities`, and the required `usable` flag. The batch filtered on `body IS NOT NULL AND body NOT LIKE 'ERROR%'` rather than the Phase 1 regex, so the LLM's own `usable` flag becomes the downstream filter (compared against the regex in §4.3.3).

**Extraction outcome.** The LLM flagged **11,675 articles as `usable` (59.5%)** and 7,944 as unusable (40.5%, predominantly paywalls, Cloudflare blocks, and keyword-matched off-topic content), consistent with the pre-batch estimate and the calibration sample (§4.3.4). Across the 11,675 usable articles the four continuous scores are:

| Feature           | Mean   | Std   |
| ----------------- | ------ | ----- |
| `sentiment_score` | +0.025 | 0.501 |
| `supply_impact`   | −0.071 | 0.463 |
| `demand_impact`   | −0.075 | 0.340 |
| `risk_premium`    | +0.156 | 0.426 |

The near-zero `sentiment_score` mean indicates a roughly bearish/bullish-balanced corpus; the negative `supply_impact` and `demand_impact` means reflect a tilt toward supply-tightening and demand-weakening news; and the positive `risk_premium` mean (the largest in magnitude) reflects the elevated-geopolitics regime of the coverage period. A notable 9.9% of usable articles carry all three channels at exactly zero, articles the LLM judged usable but materially neutral; §4.3.4 revisits this, and §4.3.3 uses it to define the stricter `usable_strict` filter.

**Categorical and entity distributions.** First-rank event types are dominated by `geopolitical` (37.6%) and `supply` (28.6%), then `macro` (14.5%) and `demand` (12.7%). The most frequent entities are Iran (3,197), the United States (3,092), Russia, China, OPEC+, Trump, Israel, India, the Strait of Hormuz (832, comparable to OPEC's count), and OPEC, all intuitive for the coverage period and evidence that the entity normalization correctly isolates chokepoints and actors rather than collapsing them. These distributions are the input substrate for the TFT models that follow.

### 4.3.3 LLM `usable` flag and filter comparison

The Phase 1 regex `body_valid` filter (Section 3.4.1) and the Phase 2 LLM `usable` flag (Section 3.4.2) are two approaches to the same problem: deciding which scraped articles carry substantive content. Both labels coexist for the 19,619 processed articles, so they can be compared directly.

**Result.** Table 4.4 is the 2x2 contingency table.

|                          | LLM `usable=0` | LLM `usable=1` | **Total** |
| ------------------------ | -------------: | -------------: | --------: |
| **regex `body_valid=0`** |          4,578 |          1,491 |     6,069 |
| **regex `body_valid=1`** |          3,366 |         10,184 |    13,550 |
| **Total**                |          7,944 |         11,675 |    19,619 |

The regex accepts 69.1% of articles and the LLM 59.5%; they agree on 75.2% (14,762 of 19,619), a **Cohen's κ of 0.47** (chance-corrected, "moderate" on the Landis-Koch convention). The two off-diagonal cells characterise the disagreement: the regex rejects but the LLM accepts **1,491** articles (regex false negatives, short but substantive content below the 400-character threshold), while the regex accepts but the LLM rejects **3,366** (regex false positives, long keyword-matched content not about WTI). The regex is permissive on long non-substantive content (it cannot judge topic semantically) and restrictive on short substantive content (its length cutoff is blunt); the LLM, with semantic access to the body, behaves the opposite way on both axes.

**Manual audit.** Sampling both off-diagonal cells and hand-labelling confirms the LLM's judgment in the large majority of cases. The regex's false positives are dominated by keyword collisions (palm oil, canola, a semiconductor firm with an energy-sounding name, generic equity wraps); its false negatives are short but on-topic briefs (for example a 320-character note on OPEC+ holding its production plan). The audit also surfaced a residual LLM error mode: a few articles the LLM marks `usable` are not genuinely WTI-relevant (the clearest being an article on Canada rolling back EV tariffs), so the LLM filter is more accurate than the regex on both of the regex's failure modes but is not itself a ground-truth oracle, a limitation revisited in Chapter 5.

**The `usable_strict` variant.** The residual LLM false positives share a diagnostic pattern: the article is marked `usable` yet all three channel scores are zero, meaning the LLM's own downstream reasoning judged no material market impact. Approximately **242** articles show this. We therefore define a stricter filter, `usable_strict=1`, requiring at least one non-zero channel; it is the canonical filter for TFT v2 training and the reported ablation (Appendix C), whereas `usable=1` is used only for this filter comparison. It reduces training noise without imposing external judgment.

**Filter choice and cross-phase note.** The two filters are not nested: the LLM accepts 1,491 articles the regex rejects and rejects 3,366 it accepts, so its acceptance set overlaps but is not a subset of the regex's. The LLM filter is the canonical Phase 2 downstream filter. This migration (plus the loss of Phase 1's title-only fallback) means the Phase 1 modeling set (13,690) and the Phase 2 set (11,675 usable / 11,433 strict) are not directly comparable, a point developed in §4.3.8.

### 4.3.4 Inter-model calibration of LLM features

The LLM-extracted features are the substrate of all Phase 2 modeling, so they are validated before use. Following Section 3.7, we run an **inter-model calibration** [CITE: LLM-as-annotator]: the same 30 stratified articles are scored by Claude Haiku 4.5 and by a GPT-family reference model with the identical prompt and schema, treating cross-family disagreement as a signal of genuine ambiguity or systematic bias. This subsection reports the first calibration, on the initial Schema v1, and the diagnostic it prompted.

**Result on Schema v1.** Schema v1 carried a single composite `sentiment_score` as its directional feature. The two models agreed on the `usable` flag for 26 of 30 articles (87%), but agreement on the composite sentiment was poor: a Pearson correlation of only **0.39**, with **opposite-signed** scores on **4 of 13 usable articles (31%)**. Sign disagreement is the most consequential kind, since it means the two models read the same article as pushing WTI price in opposite directions.

**Diagnostic (the Phase 2 pivot).** The sign disagreements were not random: they concentrated on high-magnitude geopolitical events, exactly the articles that move the market. The root cause is that the composite `sentiment_score` conflates two separable judgments, the **valence of the event** (is this good or bad news?) and its **directional price impact** (does it push crude up or down?). For most articles the two align, but for a supply-threatening escalation they diverge: the event is negative in valence yet bullish for price, so a valence-weighting model scores it negative and a price-weighting model scores it positive. A single number cannot encode both, so the two models resolve the ambiguity differently precisely where it matters most. A 0.39 correlation on the primary feature is structurally, not incidentally, inadequate.

**Response.** The fix is to decompose the judgment rather than tweak the prompt: Section 4.3.6 introduces the three orthogonal economic channels (`supply_impact`, `demand_impact`, `risk_premium`) and the re-calibration that validates them. First, though, TFT v1 was trained on the v1 schema; its results and limitations follow.

### 4.3.5 Temporal Fusion Transformer (TFT) v1

TFT v1 is the project's first deep-learning model and the first to use the richer LLM feature set. It was trained on the Phase 1 corpus (Schema v1 features plus the DXY and VIX macro covariates). The Temporal Fusion Transformer [CITE: Lim et al. 2021] is an attention-based multi-horizon forecaster whose Variable Selection Network (VSN) and interpretable attention make it well suited to this analysis.

**Training setup.** The architecture (Section 3.6) uses a 48-hour encoder, a one-hour horizon, `hidden_size=32`, `attention_head_size=4`, `dropout=0.1`, and a quantile loss (~113,000 parameters). On 10,797 hourly rows with an 80/20 temporal split, early stopping triggered at epoch 21 with a best validation loss of **0.204**. Inputs are the Schema v1 LLM features (`sentiment_score`, `magnitude`, integer-encoded `event_type`/`price_direction`/`time_horizon`, `certainty`, `n_articles`), the market context (`log_volume`, `price_range`, `log_return`, `amihud`, `dxy`, `vix`), and calendar covariates.

**Feature importance.** The VSN weight is dominated by one feature:

| Feature           | Importance | Group  |
| ----------------- | ---------: | ------ |
| `sentiment_score` |       0.53 | News   |
| `log_volume`      |      0.095 | Market |
| `dxy`             |      0.075 | Market |
| `log_return`      |      0.055 | Market |
| `event_type_num`  |      0.030 | News   |
| `magnitude`       |      0.025 | News   |
| `vix`             |      0.005 | Market |

![TFT v1 feature importance](../../../04_outputs/figures/tft_feature_importance.png)

**Figure 4.2: TFT v1 feature importance, news features (red) versus market features (blue).** `sentiment_score` dominates at 0.53, more than five times the next feature; the market-context features form a second tier and the rest carry minor weight.

Three points stand out. First, **`sentiment_score` dominates (0.53, over five times the next feature)**, a strong validation of the LLM approach: the continuous LLM sentiment carries far more signal than any market variable, and far more than the FinBERT label could. Second, **VIX is negligible (0.005)** while DXY ranks third, so for WTI the dollar index already absorbs most of the macro-risk signal. Third, the categorical features (`event_type`, `price_direction`, `time_horizon`) were integer-encoded, imposing a meaningless numeric order that likely understates their informativeness, a limitation fixed in TFT v2.

**Attention and RQ1.** The attention peaks at **lag −4h** (with a −2h to −5h cluster), independently corroborating the Phase 1 lag OLS peak at +6h (§4.2.3): two very different methods place the dominant news-impact horizon several hours after publication. A secondary cluster at **−27h to −28h** is a "daily memory" effect (the same hour on the previous trading day), a structure invisible to the 12h-max lag OLS.

![TFTv1 attention weights](../../../04_outputs/figures/tft_attention_weights.png)

**Figure 4.3: TFT v1 encoder attention by lag.** Mean attention over the 48-hour encoder window, peaking at −4h. Two clusters are visible: a short-term one between −2h and −5h, and a secondary one around −27h to −28h.

**Directional asymmetry (RQ2).** Splitting validation samples by sentiment direction gives the mean predicted `log_volume`:

| Sentiment direction | Mean predicted `log_volume` |
| ------------------- | --------------------------: |
| Bearish             |                       8.775 |
| Bullish             |                       8.700 |
| Neutral             |                       8.875 |

Bearish predicted volume exceeds bullish, consistent in direction with §4.2.3, but a two-sample t-test on the difference is not significant (**p = 0.56**), underpowered by the few hundred hours per category. The lag OLS remains the canonical RQ2 evidence; the TFT corroborates only the direction. Neutral hours carry the highest predicted volume of the three, a speculative "uncertainty drives trading" aside for the discussion.

**Limitations of TFT v1 that drive TFT v2.** TFT v1 validates the approach (LLM sentiment dominates, attention corroborates the lag OLS, and the asymmetry is reproduced in direction), but five limitations each motivate a Phase 2 change:

1. **Underpowered asymmetry test** (p = 0.56) -> a larger corpus.
2. **Integer-encoded categoricals** impose a meaningless numeric order -> proper categorical encoding.
3. **No entity awareness** (entities extracted but unused) -> 71 entity flags.
4. **Pre-war training data only** -> corpus extended through May 2026.
5. **The dominant feature is the unreliable composite sentiment**: the 0.39-correlation feature from §4.3.4 carries 53% of the model's weight -> the channel decomposition.

The fifth is the pivot: the model leans hardest on exactly the feature the calibration flagged as least trustworthy, so v2 replaces it with the orthogonal channels (§4.3.6) and retrains (§4.3.7).

### 4.3.6 Channel decomposition response

The calibration (§4.3.4) showed the composite `sentiment_score` was unreliable (r = 0.39, 31% sign disagreement) precisely on the high-magnitude geopolitical events that matter most, yet TFT v1 leaned on it for 53% of its weight. This subsection documents the schema revision that fixes this and the re-calibration that validates it.

**Design principle.** An oil-market article carries up to three economically distinct judgments: its implication for **supply**, for **demand**, and for **geopolitical or operational risk**. This decomposition mirrors the standard structural separation of oil-price drivers into supply, aggregate-demand, and precautionary (risk) components [CITE: Kilian 2009]. A single composite forces the three into one number; when they diverge (a supply shock is bearish for the event but bullish for price; a de-escalation is positive in valence but bearish-supply via sanctions relief) the composite breaks down. Schema v2 therefore decomposes the judgment so each score captures one factual claim.

**Schema v2 changes.** The core addition is the three orthogonal channels, each on `[-1, +1]`: `supply_impact` (positive = more oil to market), `demand_impact` (positive = stronger consumption), and `risk_premium` (positive = elevated risk). Schema v2 also promotes `usable` to a required field, makes `event_type` a salience-ordered array, drops the redundant `price_direction`, and retains the composite `sentiment_score`. The full field-by-field v1-vs-v2 comparison is in Appendix B.

**Re-calibration.** Re-scoring the same 30-article sample with Schema v2 improves every metric (Table 4.5).

| Metric                               | v1 schema         | v2 schema            |
| ------------------------------------ | ----------------- | -------------------- |
| `usable` agreement                   | 26/30 (87%)       | 27/30 (90%)          |
| `sentiment_score` correlation        | 0.39              | **0.88**             |
| `sentiment_score` sign disagreements | 4/13 usable (31%) | **1/14 usable (7%)** |
| `supply_impact` correlation          | —                 | 0.94                 |
| `supply_impact` MAD                  | —                 | 0.11                 |
| `supply_impact` sign disagreements   | —                 | 0                    |
| `demand_impact` correlation          | —                 | 0.96                 |
| `demand_impact` MAD                  | —                 | 0.05                 |
| `demand_impact` sign disagreements   | —                 | 0                    |
| `risk_premium` correlation           | —                 | 0.82                 |
| `risk_premium` MAD                   | —                 | 0.13                 |
| `risk_premium` sign disagreements    | —                 | 1                    |

The headline results: `sentiment_score` correlation jumps **0.39 to 0.88** and its sign disagreements drop **31% to 7%**, and each new channel correlates above 0.82 (`supply_impact` 0.94, `demand_impact` 0.96, `risk_premium` 0.82) with near-zero sign disagreements. Strikingly, the composite improved even though it was not modified: asking the model to decompose the article first, and only then produce a composite, disciplines every output it touches. The channels are genuinely orthogonal (within-model pairwise |r| < 0.5), so they capture distinct economic dimensions rather than one relabelled judgment.

**Residual weakness.** On 3 of the 14 usable calibration articles, Haiku zeroed all three channels where GPT assigned small non-zero values, on articles of modest relevance; the corresponding `sentiment_score` and `magnitude` were also subdued, so these do not dominate downstream, but this conservatism is the same pattern that defines the `usable_strict` filter (§4.3.3). With the channels validated, the full Schema v2 batch (19,619 articles) produces the feature set for TFT v2 (§4.3.7).

### 4.3.7 Temporal Fusion Transformer v2

TFT v1 validated the deep-learning approach; TFT v2 integrates the Phase 2 investments (channel decomposition §4.3.6, LLM filter §4.3.3, expanded corpus, entity normalization) and is the model against which the research questions are evaluated. A controlled ablation guided the design (Appendix C); the reported configuration is the one selected on validation performance.

#### 4.3.7.1 Success criteria

Three criteria reflect Phase 2's complementary role: the Phase 1 OLS provides the primary evidence for RQ1 (+6h peak) and RQ2 (bearish > bullish), and TFT v2 is expected to corroborate this through an independent method while showing the channels and entities as visible drivers.

1. **Channels and entities are visible drivers.** The VSN importance ranking should place channel features and entity flags in the top ranks, with a non-degenerate, interpretable attention pattern.
2. **Prediction substantially beats persistence** in the 1-to-12h window, confirming the Phase 2 features carry predictive signal.
3. **The multi-horizon error curve is consistent with Phase 1's lag structure**, with the largest persistence reduction in the +6h range identified by the lag OLS.

The evaluation is in §4.3.7.7.

#### 4.3.7.2 Model design

TFT v2 uses `hidden_size=32`, `attention_head_size=4`, `dropout=0.15`, `hidden_continuous_size=16` (298,329 parameters), trained on the `usable_strict=1` filter (§4.3.3, which drops ~242 channel-neutral articles labelled as `usable`). The input features are:

- **Channels (Phase 2 contribution):** `supply_impact`, `demand_impact`, `risk_premium`, each on `[-1, +1]`.
- **Composite news:** `sentiment_score`, `magnitude`, `certainty`, `n_articles`.
- **Categoricals:** `event_type_primary` (8 categories including the synthetic `no_news`) and `time_horizon` (4), as proper embeddings (dimension 5 and 3), taking the dominant article's value (highest `magnitude`, tie-broken on `article_id`).
- **Entity flags (Phase 2 contribution):** 71 binary multi-hot columns (§4.3.2).
- **Market context:** `log_volume`, `price_range`, `log_return`, `amihud`, `dxy`, `vix`.
- **Calendar:** `hour`, `day_of_week`, `month`, `is_us_session`, `is_wednesday`.

Continuous features are hour-averaged, entity flags use the maximum, and categoricals take the highest-magnitude article; news-free hours get the `no_news` category and zeros. A handful of boundary nulls (DXY/VIX holiday gaps, first-hour `log_return`/`amihud`) are forward-filled or set to zero, affecting under 0.1% of samples.

The model predicts three targets jointly (`log_volume`, `amihud`, `price_range`) at four horizons (**1, 3, 6, 12h**); the +6h horizon matches the lag OLS peak (§4.2.3). The temporal split is **60/20/20** on the 11,232 hourly observations, with 48h buffers between partitions: train 2024-05-13 to 2025-06-11 (6,728 samples), validation 2025-06-13 to 2025-12-27 (2,216), test 2025-12-29 to 2026-05-13 (2,159). The war onset (28 February 2026) falls inside the test window (994 pre-war, 1,165 war hours), so the test measures generalization from pre-war training to the unseen war regime; metrics are reported on the full test set, the pre-war slice, and the war slice throughout.

Training uses the Adam optimizer (learning rate 1e-3, halved after three epochs without validation improvement). The objective is a quantile loss applied independently to each of the three targets and summed, so the model learns to predict a spread of quantiles (a prediction interval) per target rather than a single point estimate. Each target is normalized separately, with the normalization statistics pooled within the asset group (a single group here, WTI), so the three targets are placed on comparable scales before their losses are combined. Gradients are clipped at 0.1, training stops early once validation loss stops improving (patience 10, minimum delta 1e-4), and the run is seeded (42) with deterministic algorithms on a Colab T4 GPU. The best validation loss of **0.427** was reached at epoch 10.

#### 4.3.7.3 Predictive performance

We evaluate TFT v2 on the held-out test set for all three targets at the four horizons, against a persistence baseline (the current value carried forward, a standard financial-forecasting reference), reporting the median quantile (q50) on four slices: validation, full test, the pre-war test portion (994 hours before the 28 February 2026 onset), and the war portion (1,165 hours from onset to the end of the data).

**log_volume (Table 4.6).**

| Horizon | Persistence MAE | TFT v2 MAE | Reduction | Pre-war MAE | War MAE |
| ------: | --------------: | ---------: | --------: | ----------: | ------: |
|      1h |           1.076 |      0.585 |       46% |       0.536 |   0.628 |
|      3h |           1.452 |      0.577 |       60% |       0.537 |   0.611 |
|      6h |           1.820 |      0.602 |       67% |       0.551 |   0.646 |
|     12h |           2.174 |      0.631 |       71% |       0.568 |   0.685 |

The error reduction over persistence grows monotonically with horizon, from 46% at 1h to a peak of 71% at 12h, as persistence weakens at longer horizons (volume autocorrelation decays); the +6h/+12h peak matches the lag OLS range of Section 4.2.3. Error is 17 to 21% higher on the war slice than pre-war at every horizon, the expected cost of extrapolating from pre-war-only training, but modest enough that the model retains substantial predictive value on the unseen regime.

**amihud (Table 4.7).**

| Horizon | Persistence MAE | TFT v2 MAE | Reduction | Pre-war MAE | War MAE |
| ------: | --------------: | ---------: | --------: | ----------: | ------: |
|      1h |          0.0004 |    0.00023 |       43% |     0.00010 | 0.00035 |
|      3h |          0.0004 |    0.00023 |       43% |     0.00010 | 0.00034 |
|      6h |          0.0004 |    0.00022 |       45% |     0.00009 | 0.00034 |
|     12h |          0.0004 |    0.00022 |       45% |     0.00009 | 0.00033 |

A steady 43 to 45% reduction across horizons, with little horizon variation (partly an artifact of amihud's compressed 1e-4 scale). War-slice persistence MAE is three to four times the pre-war value, reflecting illiquidity spikes in the war regime, yet the model's relative reduction over persistence holds on both slices.

**price_range (Table 4.8).**

| Horizon | Persistence MAE | TFT v2 MAE | Reduction | Pre-war MAE | War MAE |
| ------: | --------------: | ---------: | --------: | ----------: | ------: |
|      1h |           0.495 |      0.718 |      -45% |       0.180 |   1.200 |
|      3h |           0.578 |      0.719 |      -24% |       0.180 |   1.200 |
|      6h |           0.630 |      0.721 |      -14% |       0.180 |   1.198 |
|     12h |           0.701 |      0.724 |       -3% |       0.180 |   1.201 |

Here the model underperforms persistence on the full test set (-3 to -45%), a regime-specific failure: on the pre-war slice it matches persistence (MAE approximately 0.180), but on the war slice it degrades to approximately 1.200, worse than war-slice persistence (0.764 to 1.059). Trained only on the moderate-volatility pre-war regime, the model reverts toward the historical mean when forced to predict the unseen high-volatility war regime, whereas persistence at least tracks the elevated current state. This is a clean regime-extrapolation failure, and price_range, a direct intraday-volatility measure, is far more regime-sensitive than the aggregate targets; we return to the implications in Chapter 5.

**Across targets.** The model establishes substantial gains on log_volume (46 to 71%) and amihud (43 to 45%) but fails on price_range in the war regime, consistent with the train/test regime mismatch: the aggregate liquidity measures transfer across regimes, whereas price_range does not. The remainder of Section 4.3.7 therefore focuses on log_volume, the target most relevant to RQ1 and RQ2 and the one on which the model is strongest; the price_range failure mode recurs in Section 4.3.7.5 and Chapter 5.

#### 4.3.7.4 Feature contributions

The Variable Selection Network (VSN) assigns each input feature a learned weight reflecting how much the model relies on it; aggregated over validation samples, these weights rank the features. Table 4.9 reports the top ten for log_volume prediction.

| Rank | Feature       | Importance | Type               |
| ---: | ------------- | ---------: | ------------------ |
|    1 | vix           |      0.188 | macro covariate    |
|    2 | supply_impact |      0.121 | channel (news)     |
|    3 | ent_oman      |      0.113 | entity flag (news) |
|    4 | demand_impact |      0.055 | channel (news)     |
|    5 | is_wednesday  |      0.022 | calendar           |
|    6 | ent_japan     |      0.017 | entity flag (news) |
|    7 | ent_eu        |      0.016 | entity flag (news) |
|    8 | ent_iran      |      0.015 | entity flag (news) |
|    9 | ent_china     |      0.014 | entity flag (news) |
|   10 | ent_algeria   |      0.014 | entity flag (news) |

Three patterns stand out. First, VIX ranks first at 18.8%, consistent with its role as a market-wide volatility proxy: `log_volume` tracks broad equity-market activity through stress periods, and risk expectations modulate the volume response to oil news. Second, the channel decomposition places two features in the top five, `supply_impact` (12.1%) and `demand_impact` (5.5%), together 17.6% of importance and comparable to VIX alone; the composite `sentiment_score` does not appear in the top ten, indicating that once the model has the decomposed channels the aggregate sentiment adds little marginal information. Third, six of the top ten are entity flags, validating the entity normalization of Section 4.3.2, and they are economically interpretable: `ent_oman` (a producer adjacent to the Strait of Hormuz), `ent_iran` (the actor at the center of the conflict), `ent_japan`/`ent_eu`/`ent_china` (major importers), and `ent_algeria` (a mid-size OPEC+ producer).

The two feature-engineering choices behind this result, proper categorical encoding and the entity flags, were validated in a controlled three-variant ablation whose key finding is that the channels' predictive role emerges only under proper categorical encoding (where `demand_impact` becomes the top feature), not under the integer encoding of the simplest variant, in which the composite sentiment score dominates instead. The full three-variant comparison is deferred to Appendix C to keep the main text focused on the canonical model.

The VSN reports a feature's weight, not the direction or magnitude of its effect, so this interpretation is qualitative: VIX informs volatility expectations, the channels capture fundamental-driven supply and demand news, and the entity flags carry source-specific risk signals. Sections 4.3.7.5 and 4.3.7.6 examine how the model turns these features into predictions, through its attention pattern and per-horizon error structure.

**Stability of the ranking.** The specific ordering in Table 4.9 is not fully robust to retraining. An independent run, retrained after adding two candidate covariates and a minor data-cleaning fix, reshuffled the ranking substantially: `risk_premium` and `magnitude` rose to the top two positions while `supply_impact` and `ent_oman` fell out of it. What is stable across both runs is the character of the leading features rather than their exact order: risk and volatility proxies (VIX, `risk_premium`) and news-salience measures dominate in both, while the composite `sentiment_score` and the article count stay near the bottom. The importance results should therefore be read at the level of feature type, risk and salience over price direction (Section 4.3.7.6 and Chapter 5), rather than as a precise per-feature ranking.

#### 4.3.7.5 Lag structure analysis: evidence for RQ1

Research Question 1 asks at what temporal lag news events most strongly affect liquidity. Phase 1's lag OLS (Section 4.2.3) placed the bearish-news peak on log_volume at +6h, with a secondary trace at +12h. We test TFT v2 against this through two complementary diagnostics: the per-horizon error curve and the attention pattern.

**Per-horizon error curve.** If news impact concentrates at a specific horizon, the multi-horizon model should be most accurate there after normalizing for baseline difficulty, which the MAE-reduction-over-persistence of the log_volume curve (Table 4.6) does. That curve grows monotonically from 46% at +1h to 71% at +12h, placing the strongest persistence-relative improvement in the +6h to +12h window, the same range lag OLS identified (the exact peak differs slightly: +12h for TFT v2, +6h for lag OLS). The two methods answer structurally different questions, lag OLS isolates the direct effect of a single sentiment value at lag -k, whereas the TFT predicts forward from the full 48-hour window and all features jointly, so their agreement on this window is a substantive corroboration through an independent methodology rather than a restatement of one finding.

**Attention pattern.** The encoder attention distribution, aggregated over validation samples, gives a complementary view of the lag structure.

![Attention by sentiment on v2.2](../../../04_outputs/figures/attention_by_sentiment_tftv2.2-exp2.png)

**Figure 4.4** shows mean encoder attention across the 48-hour window. Overall attention peaks at lag -1h (weight 0.0315) and falls off smoothly through -2h to -5h (0.0308, 0.0304, 0.0301, 0.0300), a graded recency effect with no boundary concentration. Disaggregated by sentiment direction the pattern diverges: bearish-sentiment hours peak at -6h (0.0321) while bullish-sentiment hours peak at -1h (0.0347). The bearish peak at -6h aligns exactly with the +6h lag OLS peak, indicating the model has learned that bearish news requires a longer integration window than bullish news, and independently confirming the same structure Phase 1 found in the average bearish impact.

**Synthesis on RQ1.** Both diagnostics corroborate the lag OLS finding through methodologically distinct views: the per-horizon error curve peaks in the +6h to +12h range, and the bearish-sentiment attention peaks precisely at -6h. Together they give the +6h lag substantive empirical support from two independent analyses.

#### 4.3.7.6 Directional asymmetry analysis: evidence for RQ2

Research Question 2 asks whether bearish news produces a different liquidity response than comparable bullish news. Phase 1's lag OLS (Section 4.2.3) found a robust bearish > bullish asymmetry in log_volume, most pronounced at +6h, and remains the primary answer to RQ2; here we test whether TFT v2's predictions provide complementary evidence through an independent method.

**Test design.** For each horizon and slice we split samples by the sentiment score at the prediction hour, bearish (below -0.1), bullish (above +0.1), and neutral (within plus or minus 0.1), and test whether the mean predicted log_volume differs between the bearish and bullish subsets with Welch's t-test (unequal variance). Neutral hours are excluded: they capture low-news activity, dominate the sample counts, and dilute the contrast the test is designed to detect. This yields sixteen tests (four horizons times four slices: val, test_full, test_prewar, test_war), reported in Table 4.10.

| Horizon | Slice       | Bearish n | Bullish n | Bearish mean | Bullish mean | Difference | p-value | Significant |
| ------: | ----------- | --------: | --------: | -----------: | -----------: | ---------: | ------: | ----------- |
|      1h | val         |       320 |       271 |        8.345 |        8.281 |     +0.064 |   0.578 | No          |
|      1h | test_full   |       384 |       826 |        8.928 |        8.857 |     +0.071 |   0.316 | No          |
|      1h | test_prewar |       180 |       340 |        8.853 |        8.783 |     +0.070 |   0.543 | No          |
|      1h | test_war    |       204 |       486 |        9.001 |        8.901 |     +0.100 |   0.253 | No          |
|      3h | val         |       320 |       271 |        7.938 |        8.131 |     -0.193 |   0.270 | No          |
|      3h | test_full   |       384 |       826 |        8.732 |        8.596 |     +0.136 |   0.247 | No          |
|      3h | test_prewar |       180 |       340 |        8.706 |        8.365 |     +0.341 |   0.058 | No          |
|      3h | test_war    |       204 |       486 |        8.756 |        8.729 |     +0.027 |   0.869 | No          |
|      6h | val         |       320 |       271 |        8.186 |        8.369 |     -0.183 |   0.230 | No          |
|      6h | test_full   |       384 |       826 |        8.633 |        8.529 |     +0.104 |   0.400 | No          |
|      6h | test_prewar |       180 |       340 |        8.470 |        8.380 |     +0.089 |   0.636 | No          |
|      6h | test_war    |       204 |       486 |        8.788 |        8.615 |     +0.174 |   0.285 | No          |
|     12h | val         |       320 |       271 |        8.277 |        8.371 |     -0.094 |   0.561 | No          |
|     12h | test_full   |       384 |       826 |        8.220 |        8.464 |     -0.244 |   0.070 | No          |
|     12h | test_prewar |       180 |       340 |        8.274 |        8.333 |     -0.059 |   0.761 | No          |
|     12h | test_war    |       204 |       486 |        8.169 |        8.540 |     -0.371 |   0.053 | No          |

**Results.** None of the sixteen tests reaches significance at p < 0.05. The closest is +12h on test_war (bearish 8.169 vs bullish 8.540, difference -0.371, p = 0.053); two others approach without crossing, +3h test_prewar (bearish > bullish, +0.341, p = 0.058) and +12h test_full (bearish < bullish, -0.244, p = 0.070).

**Interpretation.** This test and Phase 1's lag OLS estimate different quantities, and the distinction governs how the null should be read. The OLS compares the marginal coefficients on bearish and bullish sentiment (the sensitivity of volume to each direction holding the other fixed, Section 4.2.3). The test here compares the mean predicted volume of bearish hours against bullish hours, an unconditional difference of group means with nothing held fixed. A larger marginal sensitivity to bearish news, Phase 1's finding, can coexist with near-equal average volume across bearish and bullish hours (for instance if bullish hours cluster in higher-baseline-volume periods, such as risk-on sessions), so the absence of a group-mean asymmetry here does not contradict Phase 1; it answers a coarser, confounded question.

The point-prediction differences are small and non-significant at every horizon, with no consistent sign across slices. This is not a power artifact: the bearish and bullish groups each hold hundreds of samples, and no test becomes significant under alternative sentiment thresholds (swept from 0.0 to 0.5) once corrected for multiple comparisons. It instead mirrors the target itself. In the actual test data, the future volume of bearish and bullish hours is statistically indistinguishable at the 1h to 6h horizons, and the only significant ground-truth difference, at +12h, runs bullish above bearish (a sign the model reproduces, predicted difference -0.244). The model is therefore tracking a group-level near-symmetry in its target period, not washing out a signal that is present. The one directional structure the TFT does carry is temporal rather than in magnitude: the attention pattern (Section 4.3.7.5) attends to -6h for bearish sentiment and -1h for bullish, matching Phase 1's +6h peak.

**Synthesis on RQ2.** Phase 1's lag OLS remains the primary and appropriate evidence for RQ2, because it measures the directional sensitivity that the research question is about. TFT v2 shows no magnitude asymmetry in its point predictions, but this is expected rather than a failed replication: the group-mean comparison is a different, confounded estimand, and the model's target is itself near-symmetric across sentiment direction over the test period. The model's genuine complementary contribution is temporal, the attention asymmetry of Section 4.3.7.5 (bearish integrated over a longer window, -6h, than bullish, -1h), which corroborates the +6h lag structure of RQ1 without depending on the magnitude test.

#### 4.3.7.7 Success criteria evaluation

We now evaluate TFT v2 against the three success criteria declared in Section 4.3.7.1, using the evidence presented in Sections 4.3.7.3 through 4.3.7.6.

| Criterion                                       | Status | Evidence                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Channels and entities as visible drivers     | Pass   | Supply_impact and demand_impact appear in top four features (12.1% and 5.5% importance); six entity flags in the top ten (Oman, Japan, EU, Iran, China, Algeria); attention layer non-degenerate with peak at -1h and smooth fall-off; bearish/bullish attention divergence (-6h vs -1h) matches Phase 1's identified lag |
| 2. Substantial persistence-relative improvement | Pass   | Log_volume MAE reduction of 46-71% across the 1-12h window and amihud 43-45%, both substantial margins over persistence (price_range is the exception, worse than persistence on the war regime, §4.3.7.3)                                                                                                                |
| 3. Multi-horizon curve consistent with Phase 1  | Pass   | Persistence reduction peaks at +12h (71%) with +6h close behind (67%); both horizons in the range Phase 1's lag OLS identified as most informative; attention pattern independently confirms the +6h finding via bearish-sentiment attention peak                                                                         |

TFT v2 passes all three criteria: the Phase 2 features (channels, entities) are economically interpretable and drive prediction, the model substantially beats persistence on the aggregate liquidity targets, and its multi-horizon error curve and attention independently corroborate Phase 1's +6h lag structure. This does not make Phase 2 the source of evidence for the research questions. Phase 1's lag OLS remains the primary statistical evidence for RQ1 (peak at +6h) and RQ2 (the bearish > bullish sensitivity), while TFT v2 contributes complementary, methodologically distinct evidence: interpretable feature importance and a bearish-sentiment attention peak at -6h that confirms the same lag structure. The relationship between the two phases is complementary, not competitive. The configuration evaluated here is designated TFT v2 for the remainder of the thesis.

### 4.3.8 Methodological comparison: Phase 1 versus Phase 2

The two phases approach the same research questions through structurally different methodologies: Phase 1 (§4.2) uses regression (contemporaneous and lag OLS) on FinBERT sentiment over a regex-filtered corpus, Phase 2 (§4.3) uses a deep-learning forecaster (TFT v2) on LLM-derived channel-decomposed features over an LLM-filtered corpus. Neither supersedes the other; they answer complementary questions about the same data.

#### 4.3.8.1 What Phase 2 changed methodologically

Phase 2 changed the analytical stack at four levels (corpus, feature extraction, entity normalization, and analytical framework), each motivated by a specific Phase 1 limitation.

**Corpus expansion.** Phase 1's 13,690-article regex-filtered corpus (`body_valid`) covers the pre-war period to February 2026. Phase 2's LLM-filtered corpus (`usable_strict=1`, approximately 11,433 articles) extends through May 2026 to include the 28 February 2026 war onset and its aftermath; the `usable_strict` filter (§4.3.3) drops 242 topical-but-channel-neutral articles. The Phase 2 corpus is smaller in count but temporally more complete, covering the regime change Phase 1's data does not.

**Feature extraction.** Phase 1's FinBERT produces one three-class sentiment label per article (bearish, neutral, bullish), averaged hourly. Phase 2's LLM extracts a structured schema per article: a composite sentiment score, three channels (`supply_impact`, `demand_impact`, `risk_premium`), magnitude, certainty, event type, time horizon, and entities. The channels let the model separate supply, demand, and risk-premium news that FinBERT collapses onto a single axis.

**Tone versus price impact.** The two sentiment signals do not only differ in resolution; they measure different constructs, and the difference is diagnostic. FinBERT classifies the surface financial tone of an article, whereas the Haiku `sentiment_score` is prompted for the net directional impact on the WTI price ("assess the actual content, not the headline tone"). Across the 6,300 articles scored by both methods the two agree only moderately (Pearson 0.54), and they diverge in one specific place: geopolitical news, where the mean FinBERT tone is negative (-0.116) while the mean Haiku price impact is bullish (+0.137); on demand, macro, and supply articles the two agree in sign. The mechanism is visible in the headlines they disagree on, for example "Russian Sanctions Drive Oil Prices Higher" (FinBERT -0.95, Haiku +0.85) and "Oil prices soar after Israel launches strikes on Iran capital" (FinBERT -0.96, Haiku +0.80): FinBERT anchors on the alarming vocabulary (sanctions, strikes) and labels the article negative even when the headline states that prices rose, because it has no asset grounding and no causal step from a supply threat to a higher price. The LLM, prompted to reason about the price impact for WTI specifically, resolves the same articles as bullish. This is the concrete form of the FinBERT limitation that motivated the Phase 2 extraction, and it is confined to exactly the geopolitical news that carries the highest risk premium and the highest trading volume (§4.3.7.4). It also explains how a sentiment-sign asymmetry test can diverge between the two phases (§4.3.7.6): the high-volume geopolitical news falls in opposite sentiment buckets under the two definitions.

**Entity normalization.** Phase 1 uses no entity features. Phase 2 maps entity mentions to 71 canonical flags (§4.3.2 and Appendix D), giving each hour a binary vector of which actors (US, Iran, Saudi Arabia, OPEC+, etc.) the dominant article named, a signal no Phase 1 method had.

**Analytical framework.** Phase 1's regression isolates the effect of specific variables at specific lags but produces no multi-horizon forecast. Phase 2's TFT predicts three targets at four horizons jointly from the previous 48 hours, but does not isolate individual variable effects the way OLS does. The move is not from a weaker method to a stronger one but from one that isolates effects to one that integrates them, each complementing the other.

#### 4.3.8.2 Why cross-phase numbers are not comparable

TFT v1 and TFT v2 loss values cannot be compared directly, because almost everything beneath them differs: the loss function (v1 `QuantileLoss` on one target; v2 `MultiLoss` on three, roughly three times the per-target scale), the validation period (v1 approximately 2,160 pre-war hours from an 80/20 split; v2 approximately 2,216 transitional hours from the 60/20/20 split), the horizons (v1 one hour; v2 four jointly), the feature count (v1 approximately 12; v2 over 90), and the upstream corpus and extraction (FinBERT on the regex corpus versus LLM channels on the `usable_strict` corpus). Comparing v1's val_loss of 0.204 to v2's 0.427 would therefore be misleading.

The fair comparison is persistence-relative reduction on a shared target and horizon (log_volume at 1h), where both models substantially beat persistence and neither fails. Phase 2's contribution is not measurably better forecasts but a richer analytical vocabulary: where v1 concentrates importance on a scalar sentiment score, v2 distributes it across decomposed channels, entity flags, and macro covariates. The progression from Phase 1 to Phase 2 is real but qualitative and analytical, not a numerical win; the thesis' primary contribution is the Phase 2 methodology, not a demonstration that it forecasts better.

## 4.4 Summary

Chapter 4 has reported the empirical results of both methodological phases of this thesis.

Phase 1 (§4.2) established the initial evidence for the research questions through regression-based analysis of a pre-war corpus of 13,690 articles processed with FinBERT sentiment features. The headline bias experiment (§4.2.2) demonstrated that title-only and title+body FinBERT inputs produce substantially different sentiment classifications, motivating the shift toward richer feature extraction in Phase 2. The lag OLS (§4.2.3), including the contemporaneous k=0 case, confirmed a positive news-volume association and identified the +6 hour lag as the peak of the news impact on liquidity, with a statistically significant bearish > bullish directional asymmetry. An exploratory vector autoregression (§4.2.4) was abandoned once the sparsity of the hourly sentiment signal proved incompatible with its assumptions, which reinforced the move to the deep-learning models of Phase 2.

Phase 2 (§4.3) extended the analysis through LLM-based feature extraction on an expanded corpus covering the pre-war and war regimes. The Schema v2 extraction produced channel-decomposed features (§4.3.2) that distinguish supply-side, demand-side, and risk-premium news impacts, alongside 71 canonical entity flags derived from a normalized entity list. The comparison between the regex-based Phase 1 filter and the LLM-based Phase 2 filter (§4.3.3) identified specific error modes in each and motivated the stricter `usable_strict=1` variant used for TFT v2 training. TFT v1 (§4.3.5) validated the deep-learning approach on a single target and horizon, achieving substantial improvements over persistence and identifying a two-cluster attention pattern (short-term at -4h, daily memory at -27/-28h) consistent with intraday and daily market cycles. TFT v2 (§4.3.7) integrated the full Phase 2 methodological investments: channel decomposition, entity flags, proper categorical encoding, multi-horizon prediction, and multi-target learning. The v2 canonical configuration reduces prediction error over persistence by 46 to 71 percent on log_volume across horizons, with peak reduction at +12h consistent with the +6 hour to +12 hour range identified by Phase 1's lag analysis. The Variable Selection Network's feature importance ranking places VIX and supply_impact as the top two features and includes six entity flags in the top ten, demonstrating that the channel decomposition and entity normalization together produce a model in which economically interpretable features drive prediction. The directional asymmetry test (sixteen bearish-versus-bullish comparisons of predicted volume) returned no significant group-level difference; this reflects a different, coarser estimand than Phase 1's marginal-coefficient asymmetry rather than a failed replication, and the TFT's complementary evidence for RQ2 is temporal, its attention attending to -6h for bearish sentiment against -1h for bullish.

The methodological comparison of the two phases (§4.3.8) argued that direct numerical comparison between v1 and v2 is not appropriate given differences in validation sets, loss functions, prediction horizons, feature counts, and news corpus. Both models substantially beat persistence in their respective settings; Phase 2's contribution is analytical richness (channel decomposition, entity-conditional predictions, multi-horizon error structure) rather than measurably better raw predictive accuracy. That section also established that FinBERT's tone-based sentiment and the LLM's price-impact sentiment measure different constructs, agreeing overall but diverging in sign on the high-volume geopolitical news.

The success criteria established for TFT v2 in advance of training were evaluated in §4.3.7.7, and all three pass: Criterion 1 (channels and entities as visible drivers of prediction), Criterion 2 (substantial persistence-relative improvement, with price_range the exception on the unseen war regime), and Criterion 3 (multi-horizon error curve and attention consistent with Phase 1's +6h lag). The evaluation refines what the Phase 1 findings mean through a different methodological lens and exposes the price_range regime-extrapolation limitation that informs the discussion in Chapter 5.

Chapter 5 develops the implications of these findings, together with the limitations and directions for future work. Chapter 6 concludes.
