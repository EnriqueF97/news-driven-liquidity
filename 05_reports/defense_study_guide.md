# Defense Study Guide - News-Driven Liquidity Dynamics in WTI Crude Oil Futures

**Purpose.** A single self-contained reference to study the project before showing it to an expert and before the thesis defense. It explains both phases end to end (data, alignment, models, what each was expected to show, what it actually showed), defines the technical terms you may be asked about, anticipates likely questions with defensible answers, and lists the open weak points to resolve. Everything here traces to the thesis draft (`05_reports/thesis/draft/`), the executed notebooks, and the SQLite database. Where the draft is internally inconsistent, it is flagged in the "Loose ends" section rather than papered over.

Read the "One-paragraph pitch", "Research questions", and "Loose ends" sections first. The rest is reference.

---

## 1. One-paragraph pitch

This thesis measures how news events move the _liquidity_ of WTI crude oil futures (mainly hourly trading volume), and specifically at what time lag the effect is strongest (RQ1) and whether bad news moves the market more than good news (RQ2). It does this in two methodological phases on the same underlying news corpus. Phase 1 is classical and interpretable: it turns each article into a FinBERT sentiment signal and runs ordinary least squares (OLS) regressions of volume on sentiment at different lags. Phase 2 is modern and richer: it replaces FinBERT with a structured Claude Haiku extraction (a composite sentiment plus three orthogonal economic "channels" and named entities) and feeds those features to a Temporal Fusion Transformer (TFT), a deep-learning forecaster. Phase 1 provides the primary statistical answers; Phase 2 corroborates them through an independent method and shows the richer features are economically meaningful drivers of the model.

---

## 2. Research questions

- **RQ1 (lag structure).** At what temporal lag do news events have their strongest impact on WTI liquidity? **Answer: peak at +6 hours** (lag OLS), corroborated by the TFT (per-horizon error curve peaks +6h to +12h; bearish attention peaks at -6h).
- **RQ2 (directional asymmetry).** Do bearish news events produce a different (larger) liquidity response than bullish news of comparable magnitude? **Answer: yes, bearish > bullish**, statistically robust in the lag OLS; the TFT reproduces the _direction_ and a _temporal_ version of it (bearish news is integrated over a longer window) but does not reach significance in its point predictions.
- **RQ3 (cross-commodity spillovers).** Explicitly deferred to future work; not investigated in this thesis.

The key framing to repeat in the defense: **Phase 1 (OLS) is the primary evidence for both RQs; Phase 2 (TFT) is complementary corroboration through a different methodology.** The two are not competitors.

---

## 3. Data

### 3.1 Sources

- **Market data:** WTI crude oil futures, hourly OHLCV via `yfinance`. Targets derived from it: `log_volume` (natural log of hourly contracts traded), `price_range` (high minus low, a Parkinson-style intraday volatility proxy), `log_return` (`ln(close_t / close_{t-1})`), `amihud` (Amihud illiquidity = |return| / dollar volume, higher = less liquid).
- **News:** headlines scraped from **GDELT**, article bodies scraped with **BeautifulSoup**.
- **Macro covariates (Phase 2 only):** DXY (US Dollar Index, `DX-Y.NYB`) and VIX (`^VIX`), hourly via `yfinance`.
- **Fundamentals:** EIA weekly crude inventory releases (Wednesdays), giving `eia_surprise` and `is_eia_release`.

### 3.2 Coverage and the two corpora

- **Full modeling grid:** 13 May 2024 to 13 May 2026, **11,232 hourly observations** (`time_idx` 0 to 11231).
- **Regime boundary:** the Iran war onset on **28 February 2026** sits inside the test window by design. It is the reason Phase 2 test metrics are split into "pre-war" and "war" slices.
- **Phase 1 corpus:** ~13,690 articles, roughly March 2024 to February 2026 (pre-war), regex-filtered.
- **Phase 2 corpus:** 22,795 article records (Jan 2024 to May 2026); 19,619 have substantive bodies and go to the LLM; 11,675 are judged `usable`; ~11,433 pass the stricter `usable_strict`.

### 3.3 Temporal alignment (this is a favourite defense topic)

- Each article is assigned to a trading hour with **`dt.ceil`** (ceiling), not `round`. Ceiling pushes every article _forward_ to the _next_ trading hour, which guarantees the article strictly precedes the hour it is matched to. **This prevents look-ahead bias** (you never let information from the future leak into a prediction of the past).
- **Off-hours articles** (published overnight, weekends, holidays) are **forward-assigned to the next trading hour, not discarded**. The gap between publication and assigned hour is stored as `assignment_gap` and used to distinguish _contemporaneous_ news (gap < 2h) from _forward-assigned_ news (gap >= 2h).
- Defense one-liner: _"We ceil rather than round so an article can only ever be attached to a trading hour that comes after it, which rules out look-ahead."_

---

## 4. Phase 1 - Classical, interpretable models (FinBERT + OLS)

Phase 1 uses FinBERT sentiment on regex-filtered articles and closes with the OLS regressions. It provides the canonical statistical evidence.

### 4.1 FinBERT sentiment extraction

- **FinBERT** = a BERT language model fine-tuned on financial text for 3-class sentiment (positive / neutral / negative). Checkpoint: `ProsusAI/finbert`, run via HuggingFace Transformers on Apple Silicon (MPS backend).
- For each article it returns **softmax probabilities** over the three classes; the label is the `argmax`, and `confidence = max(P_pos, P_neg, P_neu)`.
- FinBERT is run **twice per article**: title-only, and title-plus-body (truncated to 512 tokens with the title kept at the front). If body retrieval failed, the title-only score is the fallback.
- **Important update to how sentiment is used:** the OLS now uses the **continuous class probabilities** `P(negative)` and `P(positive)` as two separate regressors (neutral omitted as reference), not the old discrete `{-1, 0, +1}` label. The discrete mapping is kept only as a baseline. Rationale: the probabilities weight each article by FinBERT's confidence (a 0.95-bearish article should count more than a 0.51-bearish one), and using _two_ regressors (rather than one signed score) keeps the bearish and bullish coefficients independent, which is exactly what the asymmetry question (RQ2) needs. A single signed score would force bullish to be the mirror image of bearish and destroy the asymmetry test.

### 4.2 Headline bias experiment (7,755 articles with valid bodies)

Tests whether sentiment from the headline alone differs from sentiment of the full article. Two lenses:

- **Categorical (labels):** the title-only and title+body labels disagree on **41.6%** of articles; chi-square rejects independence (chi2 = 2050, p < 0.001).
- **Continuous (full distribution):** define a signed score `P(pos) - P(neg)` for each input and measure `|signed_full - signed_title|`. Label flips carry mean magnitude **0.96** (near-full reversals); articles that keep the same label still shift by **0.17** on average, and **31.6%** of them move more than 0.20, so the flip rate understates the effect.
- **Direction of the bias (corrected finding):** the mean signed shift from title to body is **-0.09**, i.e. reading the body makes sentiment _more bearish_. **Titles lean more bullish than the articles they head.** (An earlier draft said the opposite; the data are unambiguous here.)
- **Why it matters:** it proves the choice of news representation (title vs body) materially changes the signal, which motivates Phase 2's move to full-body structured extraction.

### 4.3 Contemporaneous OLS (n = 13,690)

- Spec: `log_volume_t = b0 + b1*P(negative)_t + b2*P(positive)_t + e`.
- Result: `b_P(neg) = 0.186` (p < 0.001), `b_P(pos) = 0.166` (p < 0.001), R2 = 0.0015. Discrete baseline: bearish 0.133, bullish 0.103, R2 = 0.0012.
- **What was expected:** a small effect at best (news acting within the same hour is a strong requirement). It turned out small but significant, which establishes that sentiment carries signal from the moment of publication and motivates the lag analysis.
- The asymmetry (bearish coefficient > bullish) is already visible here.

### 4.4 Lag OLS - the canonical evidence for RQ1 and RQ2

- Spec at lag k: `log_volume_{t+k} = b0 + b1*P(neg)_t + b2*P(pos)_t + e`, run separately for k in {1,2,3,4,6,8,12}. Each row is one article; n per lag ranges ~9,900 to ~11,600 (coverage falls as lag grows).

| Lag k (h) |  b_P(neg) |         p |  b_P(pos) |         p |     R2 |
| --------: | --------: | --------: | --------: | --------: | -----: |
|         1 |     0.262 |     0.000 |     0.200 |     0.000 | 0.0028 |
|         2 |     0.105 |     0.068 |     0.114 |     0.084 | 0.0003 |
|         3 |     0.117 |     0.047 |     0.170 |     0.013 | 0.0006 |
|         4 |     0.267 |     0.000 |     0.237 |     0.001 | 0.0017 |
|     **6** | **0.342** | **0.000** | **0.291** | **0.000** | 0.0027 |
|         8 |     0.056 |     0.409 |     0.101 |     0.192 | 0.0002 |
|        12 |     0.001 |     0.984 |    -0.171 |     0.029 | 0.0008 |

- **RQ1:** clear peak at **+6h** for both directions (highly significant), decaying to insignificance by +8h. Structured, not a smooth decay (secondary local maxima at +1h and +4h).
- **RQ2:** bearish > bullish at the dominant lags (1, 4, 6, including the +6h peak). The one exception is the weak lag 3 (both small, bullish marginally higher); it does not overturn the pattern where the effect actually lives.
- **On the tiny R2 (be ready for this):** R2 < 0.003 throughout. This is expected and _not_ a weakness: hourly volume is driven by many simultaneous forces (macro releases, positioning, options expiry, time-of-day), and sentiment is one signal among many. R2 measures how much of _total_ volume variance sentiment captures, not how much of _news-driven_ volume it captures. The finding is the _structured, replicable coefficient pattern across lags_, not the explained variance.
- Coefficient interpretation: continuous coefficients are on the probability scale, so 0.342 is the effect of a _maximally confident_ bearish article (P(neg)=1) versus a neutral one: `exp(0.342)-1 ~ 41%` higher volume at +6h. Typical articles imply proportionally less.

### 4.4b Reading a lag coefficient - full walkthrough (study this one)

**What the coefficient actually is.** In `log_volume_{t+k} = b0 + b1*P(neg)_t + b2*P(pos)_t + e`, the number `b1` is a **slope**: how much the outcome moves when the regressor moves by one unit, holding the other regressor fixed. It is not a correlation, not a probability, and not a percentage on its own.

Three facts fix its meaning:

1. **The regressor runs 0 to 1.** `P(neg)` is a probability, so "one unit" means going from *no bearish tone at all* to *maximum bearish confidence*. That is the full range, not a small nudge. So `b1` is the effect of the most extreme article the model can produce, which is why it looks large.
2. **The outcome is logged.** `log_volume`, not volume. So a coefficient is roughly a *proportional* change, and you convert with `exp(b) - 1`.
3. **The baseline is a neutral article.** `P(neu)` is deliberately left out of the equation. The three probabilities sum to 1, so including all three plus an intercept would be perfectly collinear and unsolvable (the same trap as including every dummy category). Dropping neutral makes it the reference point, so every coefficient reads as *"versus a fully neutral article"*.

**Worked example, the +6h peak.** `b_neg = 0.342`.

- Literal reading: an article with `P(neg) = 1` is associated with `log_volume` 0.342 higher, six hours later, than a fully neutral article.
- Converted: `exp(0.342) - 1 = 0.408`, so about **41% more volume**.
- **Realistic reading:** almost no article scores 1.0. A fairly confident article at `P(neg) = 0.6` gets `0.6 x 0.342 = 0.205`, i.e. `exp(0.205) - 1 = 23%` more volume. The relationship is linear in the probability, so you scale the coefficient by the actual score.

Do not quote the 41% without saying it is the maximum-confidence case. That is the single easiest thing to be caught overstating.

**Why the log matters.** For small coefficients `exp(b) - 1` is almost exactly `b` (lag 8: b = 0.056, effect = 5.8%). The gap only opens up as the coefficient grows (lag 6: b = 0.342, effect = 40.8%, not 34.2%). If you say "a 0.342 coefficient means 34% more volume" you are wrong by 7 points.

**The whole table converted to plain volume terms:**

| Lag | b_neg | bearish effect | b_pos | bullish effect | significant? |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 0 | 0.186 | +20.4% | 0.166 | +18.1% | both yes |
| 1 | 0.262 | +30.0% | 0.200 | +22.1% | both yes |
| 2 | 0.105 | +11.1% | 0.114 | +12.1% | neither |
| 3 | 0.117 | +12.4% | 0.170 | +18.5% | both yes |
| 4 | 0.267 | +30.6% | 0.237 | +26.7% | both yes |
| **6** | **0.342** | **+40.8%** | **0.291** | **+33.8%** | both yes |
| 8 | 0.056 | +5.8% | 0.101 | +10.6% | neither |
| 12 | 0.001 | +0.1% | -0.171 | **-15.7%** | bullish only |

**How to narrate the shape.** The effect is present immediately (+20% at publication), jumps at +1h, sags through +2h and +3h, climbs again to +4h, peaks hard at **+6h**, and is gone by +8h. It is *not* a smooth decay. Two honest readings, and you should offer the first:

- **Cautious:** with R2 this small and only eight lags, the dip at +2h/+3h may be noise. What is solid is that the effect is large and significant early, largest at +6h, and absent by +8h.
- **Speculative:** the twin peaks at +1h and +4h to +6h could reflect two different reactions, an immediate algorithmic one and a slower discretionary one. Offer this only if asked, and label it as speculation.

**What the p-values say.** Each p is the probability of seeing a coefficient this large *if the true effect were zero*. Two cautions. First, you ran 16 tests (8 lags x 2 coefficients); at p < 0.05 you would expect roughly one false positive by chance. Your defence is not any single p but the *pattern*: several adjacent lags significant, with coherent magnitudes. Second, p tells you nothing about size. Lag 3 is significant and small; that is normal with ~11,000 observations, where tiny effects clear the threshold easily.

**What R2 says, and does not.** R2 < 0.003 means sentiment explains under 0.3% of the *total* variance in hourly volume. That is expected: volume is dominated by time of day, session, scheduled releases, and positioning. R2 answers "how much of all volume movement does news explain" (almost none) and not "does news move volume" (yes, measurably). The claim rests on the coefficient pattern, not on fit.

**The bearish-versus-bullish comparison, stated carefully.** At the peak, `0.342 / 0.291 = 1.175`, so the bearish coefficient is about **18% larger** than the bullish one. Same ordering at lags 0, 1 and 4.

Now the weakness you should know before someone finds it: **the thesis compares the two coefficients descriptively, it does not formally test that they differ.** A proper test would be a Wald or F test of `b1 - b2 = 0` on the joint covariance. Without it, "bearish exceeds bullish" is an observed ordering, not an established inequality, and overlapping confidence intervals would not settle it either way. If asked, say exactly that: the ordering is consistent across the four lags that carry the effect, which is suggestive, but a formal equality test was not run. Do not claim statistical significance for the *difference*.

**Two rows that invite awkward questions:**

- **Lag 3.** Both coefficients are significant (p = 0.047 and 0.013) and bullish is *higher* (0.170 vs 0.117). This is the genuine exception to RQ2 and it is visible on the plot, so name it before you are asked. The honest line: the ordering holds at the four lags carrying the largest effects, and reverses at one weak lag.
- **Lag 12 bullish, -0.171, p = 0.029.** A maximally confident bullish article is associated with about **16% less** volume twelve hours later, and it is significant. There is no strong story for this and you should not invent one. Say it is a single isolated coefficient at the edge of the tested range, with bearish flat at 0.001 there, and that it is not part of any pattern the analysis relies on. Reaching for an explanation here is riskier than admitting you do not have one.

**Standard errors.** Estimated with the default formula, treating the data as a cross-section of articles. No HAC or clustered correction was applied. If articles published close together share unobserved shocks, the true uncertainty is wider than reported, so p-values are optimistic. The pattern is robust to moderately wider intervals; individual borderline lags (2, 3, 8) are not.

**One-sentence version to memorise:** *"At +6h, a maximally confident bearish article is associated with about 41% higher trading volume than a neutral one; the effect is significant, peaks there, and disappears by +8h."*

### 4.5 VAR (kept only as a brief mention)

- A vector autoregression was fitted as an exploratory joint model of sentiment and volume, then **abandoned**.
- **Why it failed (this is the defense answer, and it is about the data, not the VAR):** more than half of all hourly observations have no contemporaneous news, so the sentiment series is dominated by zeros; the impulse responses were not statistically significant. A VAR needs regularly-observed joint series, and this signal is sparse and event-driven.
- **Why that is useful ("productive failure"):** it diagnoses the data as event-driven rather than regularly observed, which is exactly why a model built for sparse, event-driven inputs (the TFT) is the right next step.
- Defense one-liner: _"With more than half the hours carrying no news, the sentiment series is mostly zeros, the impulse responses were not significant, so we moved to a model designed for sparse event-driven inputs."_ Do not volunteer VAR internals (lag order, orthogonalisation); they are deliberately not in the thesis.

### 4.6 Phase 1 limitations that motivate Phase 2

1. FinBERT is coarse (one sentiment axis; no magnitude, event type, certainty, horizon, or entities).
2. The regex `body_valid` filter has false positives (long off-topic content) and false negatives (short substantive content).
3. The sentiment signal is sparse (the VAR problem).
4. Phase 1 covers only the pre-war period.
5. Phase 1 OLS has no macro controls (DXY, VIX).

---

## 5. Phase 2 - Structured LLM extraction + deep learning (TFT)

Phase 2 does the deep-learning modeling. It begins with TFT v1 and, after a calibration finding, culminates in TFT v2.

### 5.1 LLM feature extraction (Claude Haiku 4.5)

- Each article (title + body truncated to 1,500 chars) is scored by **Claude Haiku 4.5** via the **tool-use API** with `tool_choice` forcing a single `extract_article_features` tool. Tool-use enforces the output schema at the API boundary (enumerated categories, numeric ranges, required fields), eliminating JSON parsing errors.
- **Schema (v2) per article:** `sentiment_score` [-1,+1]; `magnitude` [0,1]; `certainty` [0,1]; three **channels** `supply_impact`, `demand_impact`, `risk_premium` each [-1,+1]; `event_type` (array of 1-3 of: geopolitical, supply, demand, macro, technical, other); `time_horizon` (immediate / short_term / structural); `entities` (list); `usable` (boolean filter).
- **Cost trick:** when an article is unusable, the model returns only `usable=false` (~10 tokens instead of ~200), cutting cost on the ~40% unusable articles.
- Outcome on 19,619 articles: **11,675 usable (59.5%)**, 7,944 unusable (40.5%). Channel means on the usable set: sentiment +0.025, supply -0.071, demand -0.075, risk_premium +0.156 (the positive risk mean reflects the elevated-geopolitics coverage period). Top entities: Iran (3,197), US (3,092), Russia, China, OPEC+, Trump, Israel, India, Strait of Hormuz, OPEC.

### 5.2 Filter migration: `usable` vs regex, and `usable_strict`

- Regex `body_valid` accepts 69.1%; LLM `usable` accepts 59.5%; they agree on 75.2% of articles; **Cohen's kappa = 0.47** ("moderate", chance-corrected agreement).
- The filters are **not nested**: the LLM accepts 1,491 articles the regex rejected (short but substantive) and rejects 3,366 the regex accepted (long but off-topic, e.g. palm oil, canola, a semiconductor firm with an energy-sounding name).
- **`usable_strict`:** ~242 articles are `usable=1` but have all three channel scores zero (the LLM's own reasoning says "no material impact"). `usable_strict=1` additionally requires at least one non-zero channel. **This is the canonical filter for TFT v2 training** (reduces training noise); plain `usable=1` is kept for Phase 1 comparability and the ablation.

### 5.3 Inter-model calibration (why the channels exist)

- **The validation problem:** no expert human annotator was available, and the author is not a domain expert, so human ground truth was not feasible.
- **The approach:** score the same 30 stratified articles with **Claude Haiku** and an **OpenAI GPT reference** via the ChatGPT interface, same prompt/schema. Two independently-trained models from different families agreeing is a meaningful (if weaker than human) signal; disagreeing flags ambiguity or bias. Metrics: agreement rate + Cohen's kappa (binary), Pearson correlation + mean absolute difference + **sign disagreements** (continuous), exact match (categorical). Sign disagreement is the key metric: it catches cases where the two models read opposite directions.
- **v1 result (composite sentiment only):** usable agreement 26/30 (87%); but **sentiment correlation only 0.39** and **sign disagreements 4/13 usable (31%)**, concentrated on high-magnitude geopolitical events.
- **Diagnosis (memorize this):** the composite `sentiment_score` conflates two different judgments: the **valence of the event** (is it good or bad news?) versus the **directional price impact** (does it push WTI up or down?). For a supply-threatening escalation these diverge: the event is negative in valence but bullish for price. One number cannot encode both, so the two models resolve it differently, exactly on the market-moving articles.
- **Response = channel decomposition.** Split the judgment into three orthogonal economic channels (`supply_impact`, `demand_impact`, `risk_premium`), each a single factual claim. Re-run calibration on the same 30 articles.
- **v2 recalibration result:** sentiment correlation jumps **0.39 -> 0.88**; sentiment sign disagreements drop 31% -> 7%; channels reach correlations 0.94 (supply), 0.96 (demand), 0.82 (risk); channel pairwise correlations all |r| < 0.5 (genuinely orthogonal). The composite improved even though it was not changed, because asking the model to decompose first disciplines the composite it then produces.

### 5.4 TFT background (terms you must be able to define)

- **TFT (Temporal Fusion Transformer):** an attention-based deep-learning model for multi-horizon time-series forecasting. Key parts:
  - **Variable Selection Network (VSN):** learns a weight per input feature (per time step), giving an interpretable feature-importance ranking.
  - **LSTM encoder** over the history window + **interpretable multi-head attention** over past time steps, which yields a per-lag attention map (which past hours the model looked at).
  - **Quantile loss:** predicts several quantiles (e.g. median q50) in one pass, giving calibrated uncertainty without distributional assumptions and robustness to outliers.
- **Input typing:** time-varying known reals (calendar, session flags), time-varying unknown reals (the targets, market context, news features), static categoricals (asset = WTI).
- **Persistence baseline:** the naive forecast "next value = current value". All TFT skill is reported as MAE reduction versus persistence.

### 5.5 TFT v1 (the first deep model, trained on the Phase 1 corpus + schema v1)

- Setup: 48h encoder, 1h horizon, single target `log_volume`, hidden_size 32, heads 4, dropout 0.1, ~113k params, quantile loss, 80/20 temporal split on 10,797 hourly rows, best validation loss **0.204** at epoch 21. Adds DXY and VIX.
- **Feature importance:** `sentiment_score` dominates at **0.53** (5x the next feature), validating the LLM-sentiment approach. `log_volume` 0.095, `dxy` 0.075, `log_return` 0.055; **`vix` negligible at 0.005** (DXY already captures the macro-risk signal for WTI).
- **Attention (RQ1 corroboration):** peaks at **-4h** with a short-term cluster -2h to -5h, matching the OLS +6h window (the two methods are very different yet agree on a several-hour post-publication window). A secondary cluster at **-27h/-28h** = a "daily memory" effect (same hour on the previous day), which the 12h-max OLS could not see.
- **Asymmetry (RQ2):** bearish predicted volume (8.775) > bullish (8.700), consistent in direction, but the difference is **not significant (p = 0.56)** because each sentiment bucket has only a few hundred validation hours (underpowered). Neutral is highest (8.875), a speculative "uncertainty drives trading" aside.
- **v1's five limitations that lead to v2:** underpowered asymmetry test; categoricals integer-encoded (imposes fake ordering); no entity features; pre-war data only; and it leans on the composite sentiment the calibration flagged as unreliable.

### 5.6 TFT v2 (the canonical reported model)

- **Config:** hidden_size 32, heads 4, dropout 0.15, hidden_continuous_size 16, **298,329 parameters**. Filter `usable_strict=1`.
- **Multi-target:** predicts `log_volume`, `amihud`, `price_range` jointly. **Multi-horizon:** 1, 3, 6, 12 hours.
- **Features:** three channels + composite sentiment/magnitude/certainty/n_articles + proper categoricals (`event_type_primary` 8 cats, `time_horizon` 4 cats, with embeddings) + **71 binary entity flags** + market context (log_volume, price_range, log_return, amihud, dxy, vix) + calendar (hour, day_of_week, month, is_us_session, is_wednesday).
- **Aggregation to hourly:** continuous features averaged across the hour's articles; entity flags use max (any mention sets the flag); categoricals take the highest-magnitude article; news-free hours get a synthetic `no_news` category and zero continuous features.
- **Training:** `MultiLoss([QuantileLoss()]*3)`, `MultiNormalizer([GroupNormalizer(groups=['asset'])]*3)`, Adam lr 1e-3 with on-plateau reduction (patience 3), gradient clip 0.1, early stopping patience 10, seed 42, deterministic algorithms, Colab T4 GPU. Best validation loss reported as 0.427 at epoch 21 (see Loose ends).
- **Split:** temporal with 48h buffers between train/val/test so encoder windows do not leak across boundaries. The war onset (28 Feb 2026) falls inside the test set (test is ~994 pre-war + ~1,165 war hours). Metrics are reported on val, full test, pre-war, and war slices.

**Predictive performance (MAE reduction vs persistence, test set):**

| Target      |   1h |   3h |   6h | 12h | Notes                                                                         |
| ----------- | ---: | ---: | ---: | --: | ----------------------------------------------------------------------------- |
| log_volume  |  46% |  60% |  67% | 71% | grows with horizon (persistence weakens); strongest in the +6h to +12h window |
| amihud      |  43% |  43% |  45% | 45% | flat; absolute MAE ~1e-4                                                      |
| price_range | -45% | -24% | -14% | -3% | **worse than persistence**: regime-extrapolation failure                      |

- **The price_range failure (be ready to explain it honestly):** the model was trained only on the moderate-volatility pre-war regime and never saw the war regime; forced to predict intraday range in an unseen high-volatility regime, it reverts toward the historical mean while persistence at least tracks the elevated current level. log_volume and amihud are more transferable aggregate measures; price_range is a direct volatility measure and is the most regime-specific.
- **Feature importance (log_volume):** vix 0.188 (top), **supply_impact 0.121**, ent_oman 0.113, **demand_impact 0.055**, is_wednesday 0.022, then five more entity flags (Japan, EU, Iran, China, Algeria). The two supply/demand channels together (~17.6%) rival VIX; the **composite sentiment_score does not appear in the top ten** (the decomposed channels supersede it); six of the top ten are entity flags. This is the concrete payoff of Phase 2's channels + entities.
- **Attention (RQ1):** overall peak at **-1h** (recency) with smooth fall-off. Disaggregated by direction, **bearish attention peaks at -6h** (matching the OLS +6h peak exactly) while bullish peaks at -1h: the model learned that bearish news is absorbed over a longer window.
- **Asymmetry (RQ2):** 16 tests (4 horizons x 4 slices); **none significant**; direction is mixed (positive/bearish>bullish at short horizons, flipping at +12h). Closest: +12h war slice p = 0.053. Interpretation: by the time the TFT integrates 48h of many features non-linearly, the isolated directional magnitude the OLS sees does not survive into the point predictions; but the _temporal_ asymmetry survives in the attention.
- **Success criteria (declared before results, all PASS):** (1) channels and entities are visible drivers (top features + non-degenerate attention); (2) substantial persistence-relative improvement (log_volume 46-71%, amihud 43-45%); (3) multi-horizon curve consistent with Phase 1 (peaks +6h to +12h, bearish attention at -6h).

---

## 6. How the two phases connect

| Dimension     | Phase 1                                      | Phase 2                                                                                     |
| ------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| News features | FinBERT 3-class probabilities                | Haiku: sentiment + 3 channels + magnitude/certainty + event type + horizon + 71 entities    |
| Filter        | regex `body_valid`                           | LLM `usable` / `usable_strict`                                                              |
| Model         | contemporaneous + lag OLS (VAR abandoned)    | TFT v1 (schema v1) then TFT v2 (schema v2)                                                  |
| Data          | pre-war, ~13,690 articles                    | through May 2026, ~11,433 (strict)                                                          |
| Role          | **primary statistical evidence** for RQ1/RQ2 | **complementary corroboration** + shows richer features matter                              |
| RQ1 result    | lag peak +6h                                 | per-horizon peak +6h to +12h; bearish attention -6h                                         |
| RQ2 result    | bearish>bullish, significant                 | direction reproduced, temporal asymmetry in attention, not significant in point predictions |

The intellectual arc: Phase 1 finds the structure and exposes its own limits (coarse features, sparse signal, no regime coverage); each Phase 2 change (LLM extraction, channels, entities, TFT, corpus extension) answers one specific limit; the calibration failure of the composite sentiment is the pivot that produces the channel decomposition, which is the thesis's main methodological contribution.

---

## 7. Glossary (quick definitions you may be quizzed on)

- **Liquidity proxies:** `log_volume` (activity), `amihud` (price impact per dollar traded; illiquidity), `price_range` (intraday volatility).
- **Amihud illiquidity:** |return| / dollar volume. High = a given trade moves price a lot = illiquid.
- **Parkinson range:** an intraday volatility estimator from the high-low range (here `price_range` = high - low).
- **FinBERT:** finance-tuned BERT; 3-class sentiment with softmax probabilities.
- **OLS / dummy vs continuous encoding:** dummies are 0/1 indicators (hard label); the continuous encoding feeds the class probabilities so confidence is retained. Two separate regressors keep bearish and bullish independent (needed for asymmetry).
- **VAR / IRF:** vector autoregression jointly models several series on their own lags; the impulse response function traces the response of one variable to a shock in another. Abandoned here due to sparsity.
- **Cohen's kappa:** agreement between two labelers corrected for chance. 0.47 = moderate.
- **Pearson correlation / MAD / sign disagreement:** the calibration metrics; sign disagreement (opposite signs) is the most consequential.
- **Channel decomposition:** splitting one sentiment number into supply / demand / risk-premium components, each a single economic claim; the main contribution.
- **TFT, VSN, attention, quantile loss, persistence baseline:** see 5.4.
- **GroupNormalizer / MultiNormalizer:** per-group target scaling (per asset) applied to each of the three targets.
- **usable / usable_strict:** LLM topical filter; strict also requires a non-zero channel.
- **Look-ahead bias:** letting future information into a past prediction; avoided via `dt.ceil` alignment and 48h split buffers.

---

## 8. Anticipated defense questions with answers

- **"Your R2 values are tiny; is the effect real?"** Yes. R2 measures share of _total_ volume variance; volume has many drivers. The evidence is the _structured, significant coefficient pattern across lags_ peaking at +6h and replicating across the contemporaneous and lag regressions and (in direction) in the TFT. Statistical significance with n ~ 11,000 to 13,000, not R2, is the relevant test.
- **"Why did you abandon the VAR?"** The hourly sentiment series is >50% zeros (event-driven data), so the VAR could not identify significant dynamics; the impulse responses were not significant. That diagnosis (event-driven, sparse) is exactly why a TFT is appropriate.
- **"Two LLMs agreeing is weaker than human ground truth."** Agreed, and it is stated as a limitation. It was infeasible to get expert human labels; cross-_family_ LLM agreement (Claude vs GPT) is a meaningful, if imperfect, external check, and the sign-disagreement metric specifically targets the failure mode that matters.
- **"Isn't using an LLM to both extract features and filter circular / a black box?"** The filter comparison is audited manually (Section 4.3.3) and the extraction is calibrated against an independent model family. Residual LLM false positives are acknowledged; `usable_strict` removes the self-inconsistent ones (usable but all-zero channels).
- **"The TFT does not confirm RQ2."** Correct, and it is not claimed to. Phase 1 OLS is the primary RQ2 evidence. The TFT integrates 48h of many features non-linearly, so the isolated magnitude asymmetry does not survive into point predictions; the _temporal_ asymmetry does survive (bearish attention at -6h vs bullish at -1h).
- **"Why does price_range underperform persistence?"** Regime extrapolation: trained only on pre-war moderate volatility, it cannot predict war-regime intraday range and reverts toward the mean; persistence tracks the elevated current level. It is disclosed as a limitation, not hidden.
- **"Why continuous FinBERT probabilities instead of the label?"** The label discards confidence; probabilities keep it, and two probability regressors preserve the asymmetry test. It also raised R2 modestly at every lag while keeping the same +6h peak.
- **"Why did the composite sentiment improve after adding channels if it was unchanged?"** Prompt-structure effect: forcing the model to decompose into supply/demand/risk first disciplines the composite it then outputs (0.39 -> 0.88 correlation).
- **"Isn't VIX ranking first suspicious for a news thesis?"** No; VIX is a market-wide volatility proxy and volume tracks broad risk. The _news_ contribution shows up right behind it: the two channels (~17.6% combined) rival VIX, and six entity flags are in the top ten. The point of Phase 2 was that the _engineered_ news features become visible drivers, which they do.

---

## 9. Loose ends to resolve before the defense (do not skip)

These are internal inconsistencies and honest weak points. An expert may find them, so know them.

1. **TFT v1 vs v2 parameter/dropout framing.** Methods 3.6.4 describes a single shared architecture (~113k params, dropout 0.1), but v2 is multi-target/multi-horizon with 298,329 params and dropout 0.15. Make sure you present v1 (~113k, single target, dropout 0.1) and v2 (298k, three targets, dropout 0.15) distinctly.
2. **RQ2 honesty.** The bearish>bullish asymmetry is significant only in the lag OLS, and even there lag 3 is a minor counterexample; the TFT does not confirm it in point predictions. Frame RQ2 as "robust in the primary OLS evidence, corroborated in direction/temporal structure by the TFT, not independently significant in the TFT."
3. **Corpus counts vary by analysis.** Phase 1 = 13,690 (regex + title fallback); Phase 2 usable = 11,675; usable_strict ~ 11,433; TFT v1 rows 10,797; hourly grid 11,232. Know which number belongs to which analysis so you are not caught by a "how many articles?" question.
4. **LLM filter false positives** (e.g. the Canada-EV-tariff article) are a genuine limitation; `usable_strict` mitigates but does not eliminate them.

---

## 10. One-page number sheet (for last-minute review)

- Modeling grid: 11,232 hourly obs, 13 May 2024 to 13 May 2026. War onset 28 Feb 2026 (in test).
- Phase 1 corpus 13,690 (44.9% bearish / 30.4% bullish / 24.7% neutral). Headline bias: 41.6% label disagreement, mean signed title->body shift -0.09 (titles lean bullish).
- Contemporaneous OLS: P(neg) 0.186, P(pos) 0.166 (p<0.001), R2 0.0015.
- **Lag OLS peak +6h: P(neg) 0.342, P(pos) 0.291 (p<0.001).** Bearish>bullish at lags 1,4,6.
- VAR: abandoned, sparse (>50% zero-news hours), impulse responses not significant.
- Phase 2: 19,619 bodies -> 11,675 usable (59.5%) -> ~11,433 strict. Regex vs LLM filter kappa 0.47.
- Calibration: sentiment correlation 0.39 -> 0.88 after channels; sign disagreement 31% -> 7%; channels 0.94 / 0.96 / 0.82.
- TFT v1: sentiment_score importance 0.53; attention peak -4h + daily memory -27/-28h; asymmetry p=0.56 (underpowered).
- TFT v2: 298,329 params; log_volume MAE reduction 46/60/67/71% (1/3/6/12h); amihud 43-45%; price_range worse than persistence (war regime). Top features VIX 0.188, supply_impact 0.121, ent_oman 0.113, demand_impact 0.055. Attention: overall -1h, bearish -6h, bullish -1h. Asymmetry: 16 tests, none significant.
- Canonical model config (per config.py): encoder 48h, `usable_strict=1`, targets [log_volume, amihud, price_range], seed 42. (Split/horizon numbers: see Loose ends #1.)
