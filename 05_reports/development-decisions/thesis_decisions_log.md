# Thesis Decisions Log

A record of design and methodology decisions made during the project, captured to ensure they make it into the thesis writeup. Each entry includes the **decision**, the **reasoning**, and **what to write up in the thesis**.

This document is meant to be a working draft — refine, reorder, and rewrite as the thesis takes shape. Bullets are for capture, not final prose.

---

## Project trajectory (high level)

The project evolved through several methodological pivots, each driven by what the data could and could not support. The narrative of the thesis should make these pivots explicit rather than presenting only the final approach — the reasoning behind the pivots is itself a methodological contribution.

### Commodity selection: sugar → WTI

- **Decision:** Pivoted from sugar to WTI crude oil as the target commodity early in the project.
- **Reasoning:** Sugar showed limited variability and weak geopolitical sensitivity. WTI is meaningfully more responsive to news shocks, has richer event structure (OPEC, geopolitics, EIA inventory releases), and aligns better with the research questions on news-driven liquidity dynamics.
- **Thesis writeup:** A paragraph in the data section justifying commodity choice. Frame the WTI selection as the result of preliminary screening, not as a starting assumption.

### Sentiment extraction: FinBERT → Claude Haiku

- **Decision:** Replaced FinBERT (3-class classification with confidence score) with Claude Haiku LLM extraction of a richer feature schema.
- **Reasoning:**
  - FinBERT's discrete 3-class output (positive/neutral/negative) collapses information that is naturally continuous.
  - FinBERT confidence scores correlate weakly with downstream signal quality.
  - LLM extraction enables structured features beyond sentiment: event type, named entities, certainty, time horizon, magnitude.
  - Richer signal addresses the sparsity problem that ultimately killed the VAR model.
- **Thesis writeup:** Method section needs a comparative justification. Cite FinBERT's role in financial NLP literature, explain its limitations for this specific signal-extraction task, justify the move to a generative LLM with structured outputs. This is a methodological contribution — explicit comparison of two NLP regimes for the same task.

### Modeling approach: lag OLS → VAR → TFT

- **Decision (1):** VAR model was built and abandoned in favor of a Temporal Fusion Transformer.
- **Reasoning:** VAR suffered from signal sparsity (~50% zero-sentiment hours). The estimator struggled to identify dynamics from a target time series dominated by zeros. The problem was not implementation but data structure.
- **Decision (2):** TFT chosen as the primary deep-learning model.
- **Reasoning:**
  - Handles mixed continuous and categorical inputs natively.
  - Variable Selection Networks provide interpretability — feature importance per time step.
  - Multi-head attention exposes which historical hours drove a prediction (directly addresses RQ1 lag structure).
  - Designed for irregular event-driven signal mixed with continuous market data.
- **Decision (3):** Lag OLS retained as the canonical evidence for RQ1 (lag structure of news impact) and RQ2 (bearish vs bullish asymmetry).
- **Reasoning:** TFT's directional asymmetry test was underpowered (p=0.56 on the pre-war model). Lag OLS produces clean, publishable p-values for both RQs and provides a transparent baseline. TFT's role is to (a) confirm patterns at the model level, (b) expose feature importance, (c) test whether richer features capture interactions OLS cannot.
- **Thesis writeup:** Methods chapter should present the analytical stack as a sequence: OLS for hypothesis testing → TFT for nonlinear modeling and interpretability. Frame the choice as complementary rather than as a search for the "best" model.

### Headline bias finding

- **Status:** Methodological contribution, already partly written up.
- **Finding:** Title-only vs title+body inputs to FinBERT produce sentiment disagreement on 41.6% of articles (χ²=2050, p<0.001).
- **Thesis writeup:** This belongs as its own subsection in the methods or results chapter. It motivates the decision to use full article bodies (when available) for LLM feature extraction. Worth a figure (already exists: `headline_bias_comparison.png`).

### Temporal alignment

- **Decision:** Articles are assigned to the next trading hour via ceiling, with forward-assignment for off-hours articles.
- **Reasoning:**
  - Ceiling (rather than rounding) ensures the article's publication time strictly precedes the trading hour it's matched to, eliminating reverse-causal contamination.
  - Forward-assignment (rather than dropping off-hours articles) preserves overnight and weekend news, which traders read at market open and act on. Dropping these would discard exactly the most informative articles (geopolitical events, weekend supply shocks).
- **Thesis writeup:** A methods subsection on temporal alignment. Be explicit about the bias trade-off: ceiling may underestimate intra-hour reactions but eliminates causal leakage. Consider including a robustness check with floor alignment to show the lag-peak finding is not an artifact of the alignment rule.

### Data storage: CSV → SQLite

- **Decision:** Migrated the pipeline from CSV files to a SQLite database (`wti_thesis.db`).
- **Reasoning:**
  - CSVs duplicated state across notebooks, causing drift between intermediate outputs.
  - DB-backed pipeline allows the alignment step to be a simple join rather than a merge of separately-saved files.
  - `datetime_hour` is computed at write time, ensuring all consumers see the same canonical alignment.
- **Thesis writeup:** Brief mention in the methods chapter — primarily an engineering decision, not a methodological one, but worth noting for reproducibility.

---

## Phase 5 decisions (current — May 2026)

This section captures the in-progress decisions for the second LLM extraction pass, which expands and refines the feature pipeline.

### Drop `price_direction` from the feature schema

- **Decision:** Remove `price_direction` from the Haiku extraction. Derive it post-hoc from `sentiment_score` if ever needed.
- **Reasoning:**
  - Empirical analysis of the first 12,024 extractions shows `price_direction` is essentially a discretization of `sentiment_score`. Of 12,024 articles, only 16 (0.13%) showed meaningful disagreement between the two fields.
  - Keeping it costs output tokens and increases the surface area for schema drift (the existing dataset already contains a `slightly_bullish` artifact from prompt non-compliance).
  - A downstream discrete direction feature can be computed deterministically from `sentiment_score` via thresholding.
- **Thesis writeup:** A short methodological note in the feature engineering section. Demonstrates that feature reduction is justified by analysis of pilot extractions, not arbitrary.

### Add three orthogonal economic channel scores

- **Decision:** Augment the LLM extraction with three new continuous fields alongside `sentiment_score`: `supply_impact`, `demand_impact`, and `risk_premium`. All three on a -1.0 to +1.0 scale.
- **Reasoning:**
  - Inter-model calibration (Haiku vs GPT on a 30-article stratified sample) revealed 4 out of 13 usable articles (31%) had sign disagreement on `sentiment_score`. Pearson correlation was 0.39 — too noisy for downstream modeling.
  - The disagreements concentrated on high-magnitude geopolitical events (Iran-US tensions, UAE-OPEC), which are exactly the cases the model needs to handle correctly.
  - Root cause: `sentiment_score` conflates two distinct judgments — sentiment of the event vs. direction of WTI price impact. For ambiguous cases (e.g., cartel breakup driving a price surge), the prompt does not disambiguate which channel to weight.
  - Solution: decompose into orthogonal economic channels. Each channel is a single factual judgment (does supply rise or fall? does demand rise or fall? does the risk premium escalate or de-escalate?) and can be combined into an implied price direction via a fixed economic identity: `implied_direction = -supply_impact + demand_impact + risk_premium`.
- **`sentiment_score` retained**, not replaced. Three reasons: (a) preserves continuity with the FinBERT headline bias experiment, which compared a single sentiment value, (b) allows ablation studies comparing composite vs. decomposed signals in the TFT, (c) provides a hedge if the channels turn out to be noisy individually.
- **Calibration plan:**
  - Re-run the 30-article calibration with the updated prompt (output: `llm_calibration_v2.json`).
  - Compare inter-model agreement on the three channels (Haiku vs. GPT, same prompt).
  - Expectation: channels show higher agreement than the composite `sentiment_score`, because each channel is a more constrained factual judgment. Sign disagreements on `sentiment_score` may persist — that is acceptable, as the channels carry the cleaner signal.
- **Modeling implications:**
  - The TFT-v2 feature set grows by 3 continuous reals.
  - Variable Selection Network feature importance will indicate whether the model uses the channels, the composite, or both.
  - Optional ablation in the results chapter: compare TFT performance with sentiment_score only, channels only, and all four. Not required for the thesis, but a clean addendum if time permits.
- **Thesis writeup:** Methods chapter — the feature engineering section gets a subsection on the decomposition. Frame the move as a response to a documented calibration failure: _"Initial extraction used a single composite sentiment score, but inter-model calibration revealed systematic sign disagreement on high-magnitude geopolitical events (X/Y articles, correlation Z). The score was decomposed into three orthogonal economic channels — supply, demand, and geopolitical risk — to isolate LLM judgment to factual extraction rather than economic reasoning. The composite score was retained for backward compatibility with the headline bias experiment."_ This positions the decomposition as a principled methodological refinement, not an afterthought.

### `event_type` as a list

- **Decision:** Change `event_type` from a single string to an array of 1-3 categories ordered by salience.
- **Reasoning:** Articles often span multiple event types ("Iran threatens shipments while OPEC+ debates production cuts ahead of Fed meeting"). Forcing a single category collapses real information. The TFT can consume multi-hot encodings cleanly.
- **Implementation:** DB stores the JSON array as a TEXT column (no schema change needed; `entities` already uses this pattern). The model-prep code multi-hot encodes at dataset construction time, keeping the DB human-readable.
- **Thesis writeup:** Feature engineering section. Explain the tension between human-readable storage and model-ready encoding, and the design choice to handle the transformation in code rather than in the data.

### Remove `inventory` from event_type categories

- **Decision:** `inventory` is no longer a valid `event_type`. Inventory-related articles now fall under `supply` or `macro`.
- **Reasoning:** EIA inventory releases are already a structured signal in the `market_context` table (`eia_surprise`, `is_eia_release`). News articles _about_ inventory data are not a separate methodological category — they're supply news with a structured-data counterpart.
- **Thesis writeup:** Minor — mention in the feature engineering section when listing event categories.

### Refine `time_horizon` to immediate / short_term / structural

- **Decision:** Replace `long_term` with `structural`. Drop the implicit "could be 1 year, could be 2030" ambiguity.
- **Reasoning:** "Long-term" was being applied to two distinct cases: news that affects the next few months (relevant to the model) and news about structural multi-year themes (mostly irrelevant to an hourly liquidity model). `structural` cleanly tags the latter, allowing the model to down-weight it via the Variable Selection Network.
- **Thesis writeup:** Feature definition table or list. Briefly explain the three buckets and why `structural` exists as its own category.

### Entity normalization: post-extraction, in code

- **Decision:** Do not enforce entity name normalization in the prompt. Normalize variants (e.g. "Fed" / "Federal Reserve" / "U.S. Federal Reserve") in a post-extraction Python dictionary.
- **Reasoning:**
  - The LLM is poor at remembering long lists of canonical names.
  - The space of entities is not knowable in advance; new ones appear as news evolves.
  - Post-hoc normalization is easy to iterate on without re-running extractions.
- **Thesis writeup:** Feature engineering section. Mention the normalization step and report the canonical entity list used in the final binary flags.

### Entity exclusion: ambiguous and below-threshold candidates

Some candidate entities were excluded because their raw extractions contained ambiguous references (for example, "Department of Energy" appeared in both U.S. and Philippines contexts), and the clean variants did not reach the 25-article threshold required to support a stable feature column.

### Entity encoding: top-N binary flags

- **Decision:** Encode entities as binary flags for the top 15 entities (by frequency) plus a `has_other_entity` catchall, rather than as a single categorical embedding.
- **Reasoning:**
  - Articles routinely mention multiple entities; a single categorical can only represent one.
  - The TFT's Variable Selection Network handles sparse binary features well — it learns to down-weight uninformative flags automatically.
  - Top-15 captures the bulk of signal; the long tail (746 entities appearing only once) is noise.
- **Thesis writeup:** Feature engineering section. Justify the top-N cutoff with the entity frequency distribution.

### Tool-use API with strict JSON schema

- **Decision:** Use Anthropic's tool-use API with `tool_choice` forcing the extraction tool, rather than relying on the LLM to produce valid JSON via prompt instruction.
- **Reasoning:**
  - The first extraction pass produced rare but real schema leakage (`slightly_bullish`, `medium_term`).
  - Tool-use enforces enum constraints, type constraints, and required fields at the API level.
  - Eliminates JSON parsing brittleness in the pipeline.
- **Thesis writeup:** Methods — pipeline architecture. Mention as a quality control measure for LLM-extracted features.

### `usable` flag as the canonical filter

- **Decision:** Add a `usable` boolean to the extraction schema. Send all articles with a real body (regardless of `body_valid`) to the LLM. Let the LLM decide which articles are usable for modeling. `usable` becomes the canonical downstream filter.
- **Reasoning:**
  - Inspection of `body_valid=0` articles revealed legitimate news being rejected by the regex/keyword heuristic (false positives).
  - Inspection of `body_valid=1` articles revealed boilerplate slipping through (false negatives).
  - The LLM is a stronger judge of content usability than a regex heuristic.
  - With Batches API + prompt caching, the marginal cost of sending more articles is small relative to the methodological gain.
- **Implementation:**
  - The `usable` field is the only `required` field in the tool schema. When `usable=false`, the LLM returns only that field; other fields are omitted (saving output tokens and avoiding hallucinated values).
  - Downstream pipelines filter on `WHERE usable = 1`.
  - The old `body_valid` heuristic is retained for comparison but no longer the canonical filter.
- **Pending validation (calibration):**
  - 30 articles will be hand-scored by the author across three strata: `body_valid=1` (10), `body_valid=0` with substantial content (10), `body_valid=0` with weaker content (10).
  - Human scores compared to LLM `usable` flag to estimate agreement and identify systematic biases before committing to the full batch.
- **Pending analysis (post-batch):**
  - Compute disagreement between `body_valid` and `usable` across the full dataset.
  - Quantify false-positive rate (legit articles `body_valid` rejected) and false-negative rate (boilerplate `body_valid` accepted).
  - Compute inter-rater agreement (Cohen's κ) between the regex filter and the LLM filter.
  - Audit a sample of cases where the LLM says `usable=false` but the article appears legitimate to a human — characterize the LLM's failure modes.
- **Thesis writeup:** This is a methodological contribution in its own right. Frame it as: _"Two filtering regimes were compared — a regex/keyword heuristic and an LLM-based usability judgment. The disagreement rate was X%, with the LLM recovering Y articles incorrectly rejected by the heuristic and rejecting Z articles incorrectly accepted by the heuristic. Inter-rater agreement was κ=K, indicating substantial but imperfect overlap. Manual audit of LLM rejections suggests its failure modes are primarily [characterize]."_ This is publishable material — comparison of filtering approaches for noisy financial news corpora is genuinely useful to the literature.

### `usable_strict` as an additional channel inconsistent flag

- **Decision:** Add `usable_strict` to the llm_features entries that have zero on the 3 decomposition channels but have no `technical` event type.
- **Reasoning:** A diagnostic query over the production extraction batch identified 242 articles where the LLM marked usable=true and tagged the article with a channel-relevant event type (supply, demand, or geopolitical) but assigned all three channel scores to exactly zero. These represent a residual punting behavior in which the LLM committed to a categorical event type but did not commit to a magnitude on the corresponding channel. The articles were flagged with a secondary usable_strict=false indicator, enabling ablation studies in TFT v2 (Section 4.3.7) to test whether their inclusion affects channel-based feature learning.
- **Implementation:** All the llm_feature articles got a new column `usable_strict` with initial NULL values, then after filtering the 242 articles with zero on the 3 channels, but no `technical`event type, were assigned `usable_strict = 0` and the others that have at least 1 channel != 0, `usable_strict = 1`.
  The 242 articles represent approximately 2% of the usable corpus; their exclusion does not substantially reduce the modeling-ready dataset size, but their inclusion or exclusion provides a clean diagnostic for the channel decomposition's robustness.

### Cost optimizations

- **Decision:** Use Anthropic Batches API + prompt caching on the system prompt + minimal-output short-circuit for `usable=false`.
- **Reasoning:**
  - Batches API: 50% discount, asynchronous processing (1-24h turnaround), no rate-limit babysitting.
  - Prompt caching: ~700 tokens of system prompt + tool schema are identical across ~20k calls; caching reduces input cost on that portion by 90%.
  - Short-circuit: when `usable=false`, output is ~10 tokens instead of ~200. Saves ~$1-2 on the batch.
- **Estimated total cost:** ~$15-20 for the full batch over ~20k articles.
- **Thesis writeup:** Brief footnote or appendix on reproducibility. State the model version (`claude-haiku-4-5`), temperature, the use of Batches API, and the cost. Reproducibility-relevant, not methodologically central.

### Source verification discipline

- **Decision:** Every numerical claim and methodological specific written into the thesis must trace to a specific notebook cell output, DB query result, source CSV, or canonical project file (logbook, decisions log). No claims from memory of conversation; no inference of "what the result should be."
- **Reasoning:** Two errors were caught during the Chapter 4 audit that originated from drafting against condensed logbook entries rather than primary sources: (1) §4.2.3 contemporaneous OLS was initially described as a null result when the actual notebook output showed statistically significant coefficients with bearish > bullish asymmetry already present at k=0; (2) §4.2.5 VAR method paragraphs invented a Cholesky ordering, bootstrap details, and a lag-7 specification when the actual code uses `orth=False` (no Cholesky), default asymptotic ±2 std-error bands (no bootstrap), and lag 24 (all four information criteria selected the maximum candidate lag).
- **Implementation:** When in doubt, run a verification query against the working DB or paste the relevant notebook cell rather than asserting. The cost of an extra check is much smaller than the cost of an unverified claim ending up in the deliverable.

### Entity normalization — decisions worth tracing

The full 71-canonical-entity set in `llm_features.py` reflects a curated triage. Decisions that may not be self-evident from the dictionary itself:

- **OPEC and OPEC+ kept separate** rather than merged. OPEC+ is the body driving modern production decisions and OPEC alone covers a narrower set; the TFT can learn the differential weight if signal exists. Defensible as preserving the analytical distinction despite some journalistic interchangeability.
- **WTI kept as canonical** despite initial concern about tautology with the target asset. The flag carries signal because not every WTI-corpus article explicitly names the benchmark; an article tagged WTI is specifically benchmark-mention-focused, distinct from articles using generic "oil prices" language. Combined with the Brent flag, the four-way (neither/WTI/Brent/both) pattern is informative.
- **Europe and Asia kept as canonical** as geographic aggregates, parallel to the Middle East canonical. The argument for dropping (too broad given country-level entities exist) was reconsidered against the consistency argument with Middle East.
- **Xi Jinping → China, Narendra Modi → India** rather than separate canonicals. Trump remains the exception because of his exceptional frequency (1,935) and the documented frequency of his direct market-moving policy interventions; Xi and Modi at ~45 each don't carry the same per-leader signal density.
- **Joe Biden as separate canonical** at 92 articles. Asymmetric with Xi/Modi but defensible: Biden's count is well above the 25-threshold and his policy actions (SPR releases, Iran/Venezuela sanctions framing) are distinct enough from generic US coverage to merit their own column.
- **Hassan Nasrallah → Hezbollah** (not Iran). Nasrallah led Hezbollah, which is a separate canonical from the Iranian state.
- **Amin Nasser → Saudi Aramco** (not Saudi Arabia). The Aramco CEO maps to the company, which is separate from the state canonical.
- **DOE dropped entirely.** The bare "Department of Energy" string is contaminated by Philippines DOE references (~20% of mentions). Clean prefixed variants ("U.S. Department of Energy", "US Department of Energy") only total 14 articles, below the 25-threshold. SPR-related signal is adequately captured by the US, Trump, and channel features for typical coverage.
- **ECB added as canonical** at 38 mentions. Parallel to the Fed canonical for the Eurozone monetary-policy signal, which the EU canonical does not capture (EU is broader, not specifically about monetary policy).
- **Israel-Palestine, Israel-Hamas, Israeli-Palestinian conflict → Israel.** Conflict descriptors mapped to the oil-market-relevant participant. Israel's military actions affect regional risk; Palestine is not an oil-market actor; Hamas and Gaza are separate canonicals capturing the specific Palestinian dimension.
- **Goldman Sachs kept in ENTITY_CANONICAL as a self-mapping but excluded from CANONICAL_ENTITIES.** Analyst commentary signal appears in both bullish and bearish coverage, muddling its predictive value. Preserves the mapping decision for traceability without producing a feature column.

---

## Open methodological questions (to revisit before final writeup)

- **TFT directional asymmetry test was underpowered (p=0.56).** Will retraining on the expanded dataset with proper categorical handling and entity flags change this? If still underpowered, frame TFT as primarily addressing RQ1 (lag structure via attention) and feature importance, with lag OLS as the canonical evidence for RQ2 (asymmetry).
- **Daily memory effect at −27/−28h in TFT attention.** Genuinely interesting but needs an interpretation. Is it traders revisiting positions at the same time the next day, overnight cycle effects, or an artifact of the encoder window interacting with `is_us_session`? Worth a paragraph in the discussion but worth thinking through _why_ first.
- **Robustness check for the alignment rule (ceiling vs floor vs round).** A 30-minute experiment that would strengthen the methods chapter. Re-run lag OLS with floor alignment; report whether the lag-peak shifts.
- **Pre-war vs post-war TFT comparison.** The first TFT was trained on pre-war (Feb 2026) data. The retraining will include the post-war expansion. The pre-war model's clean results were the original publishable finding; the post-war model is an extension. Both should be reported, with discussion of how (or whether) the geopolitical shock affected the model.
- **Cost overrun on full LLM extraction batch.** Pre-batch estimate (scaled from the 107-article cell) implied ~$8 for the full corpus; actual aggregate across three batch submissions was $38.82 (~5x). Causes: prompt-cache strategy did not fire across separate batch submissions, body truncation increased from 800 to 1,500 characters between estimate and final batch, and per-usable-article output token count was higher than assumed due to channel decomposition adding output fields. Not methodologically central — flag for the reproducibility appendix if a reader queries the cost figure.

- **TFT v1 training accelerator was Google Colab T4 GPU, not the Apple Silicon MPS used for the local pipeline.** Chapter 3 §3.6.5 reflects this correction. The MPS was used for FinBERT extraction and pipeline orchestration only.

- **Source-verification discipline.** Two Chapter 4 errors were caught during a late audit (§4.2.3 wrong-sign claim and §4.2.5 fabricated VAR method specifics). The fix is a strict rule: every numerical or methodological specific traces to a primary source (notebook cell output, DB query, source CSV) and is never inferred from condensed entries. New drafting sessions should adopt this principle.

### News Calibration V1 vs V2 for TFT V2

Initial extraction used a single composite sentiment_score field. Cross-model calibration on a stratified sample of 30 articles revealed substantial sign disagreement on high-magnitude geopolitical events (4/13 usable articles, 31%), with Pearson correlation of only 0.39 between Haiku and a GPT-family reference model. Diagnostic inspection identified the root cause: sentiment_score conflates event valence with directional price impact, and these can diverge sharply on cartel breakup, conflict de-escalation, and similar nonlinear cases. The extraction schema was extended to include three orthogonal economic channels — supply_impact, demand_impact, and risk_premium — each capturing a single factual judgment on a [-1, +1] scale. Re-calibration showed cross-model agreement substantially improved: per-channel correlations of 0.94 (supply), 0.96 (demand), and 0.82 (risk premium), with zero or one sign disagreement per channel. As a downstream effect, sentiment_score agreement also improved from r=0.39 to r=0.88, suggesting that the decomposed reasoning structure stabilized the composite judgment. Channel orthogonality was preserved (all within-model pairwise correlations |r| < 0.5), confirming that the three fields capture distinct economic dimensions rather than redundant copies of the composite."

### Phase 1 FinBERT OLS/VAR moved to continuous probabilities (2026-07-05)

- **Decision.** Re-specify the Phase 1 lag and contemporaneous OLS from two argmax dummies (`is_bearish`, `is_bullish`) to the two FinBERT class probabilities (`P(negative)`, `P(positive)`, neutral omitted as reference). Change the VAR sentiment shock from the discrete `{+1, 0, -1}` mapping to the signed score `P(positive) − P(negative)`.
- **Reasoning.** The argmax label discards FinBERT's confidence; the probabilities use the full distribution and weight each article by how sure the model was. Two separate probabilities (not a single signed score) are required so the bearish and bullish coefficients stay independent, which is what RQ2 (asymmetry) needs. The signed score is appropriate for the VAR, which already treats sentiment as one symmetric axis.
- **Effect.** Continuous OLS sharpens the same pattern rather than changing it: contemporaneous P(neg) 0.186 / P(pos) 0.166 (both p<0.001, R²=0.0015 vs discrete 0.0012); lag peak still at +6h (P(neg) 0.342, P(pos) 0.291, both p<0.001); R² higher than discrete at every lag but still <0.003. Discrete dummies retained as a reported baseline (§4.2.3). Bearish>bullish holds at lags 1, 4, 6; weak lag 3 is the one exception (bull 0.170 > bear 0.117).
- **VAR still abandoned.** Continuous signed score did not recover significance: sentiment lags jointly non-significant in the volume equation (L1 p=0.97, L2 0.59, L3 0.055), IRF spans zero. Cause unchanged (>50% of hours have no news, zero-filled). Reinforces the sparsity diagnosis, does not overturn the abandonment.
- **Headline bias enriched (notebook 07).** Added a continuous divergence magnitude `|signed_full − signed_title|` alongside the categorical chi-square: label flips mean 0.96, same-label mean 0.17 (31.6% shift >0.2), mean signed shift −0.09.
- **Two draft errors corrected in §4.2.2.** (1) Direction was stated backwards: the draft claimed titles overstate negativity, but title→body shift is −0.09 (bodies more bearish), so titles lean more bullish than the bodies they head. (2) Neutral-title transition breakdown had the negative and neutral percentages swapped (correct: 39.6% negative, 30.5% neutral, 30.0% positive).
- **Write-up.** Chapters 3 (§3.3.1, §3.5.1, §3.5.2) and 4 (§4.2.2, §4.2.3, §4.2.4, §4.2.5) updated. All numbers trace to executed notebooks 07 and 08; lag table saved to `04_outputs/tables/finbert_lag_ols_continuous.csv`.

### VAR removed from the thesis writeup (2026-07-05)

- **Decision.** Drop the detailed VAR treatment. Deleted the VAR methods subsection (§3.5.2) and the detailed §4.2.5 VAR results; kept only a brief one-paragraph honest mention in §4.2.5 (exploratory VAR fitted, impulse responses not significant because the hourly sentiment series is dominated by zeros, abandoned in favour of the TFT). IRF figure removed from Chapter 4 and the figure registry.
- **Reasoning.** Defense-risk management: anything in the document is fair game for committee questions, and the VAR internals (lag order, IRF, orthogonalisation) are not something the author is prepared to defend. The abandonment reason that matters is a data property (event-driven sparsity), defensible without VAR expertise. A brief honest mention keeps the "productive failure" argument (justifies the move to the TFT) while removing the attack surface.
- **What stays.** The sparsity limitation and the OLS -> TFT transition narrative; passing one-line references to the VAR in Phase 1 overviews.
- **Figures.** Chapter 4 renumbered to 4.1-4.6 (headline comparison, headline divergence, lag OLS, TFT v1 feature importance, TFT v1 attention, TFT v2 attention); registry now 4.1-4.9. `irf_sentiment_to_volume.png` no longer referenced in the thesis (still produced by notebook 08 as a diagnostic).

---

## Maintenance

- Update this document after each significant methodological decision.
- Keep the format: short bullets, no dense prose, decisions traceable to reasoning.
- When the thesis writing phase begins, this document becomes the skeleton of the methods chapter.
