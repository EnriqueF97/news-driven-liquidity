# Appendices

Appendices are a top-level section following the Conclusion (Chapter 6), not part of any chapter. The full appendix registry (A-G) is maintained in `0-outline.md`; this file drafts the appendix content.

## Appendix A: Extraction prompt

This appendix reproduces the verbatim Haiku v2 extraction interface: the system prompt, the `extract_article_features` tool schema passed to the tool-use API, and the request configuration. The conceptual field-level schema and its revision history are in Appendix B. The source of record is `03_src/nlp/llm_features.py`.

**A.1 System prompt.** The following text is passed as the cached system prompt (verbatim, including its own em-dash punctuation):

```
You are a financial analyst specializing in WTI crude oil markets. Your task is to analyze news articles and extract structured features for use in a quantitative liquidity impact model.

First, determine whether the article contains substantive WTI-relevant content (usable=true). Set usable=false — and return only that field — for: paywall placeholders, Cloudflare blocks, cookie notices, or articles that are clearly not about oil markets, macro, or geopolitics relevant to oil.

When usable=true, extract the remaining fields:
- sentiment_score: NET directional impact on WTI price. -1.0 = strongly bearish, +1.0 = strongly bullish. Assess the actual content, not the headline tone.
- supply_impact: Direction of impact on WTI supply. Score the article's implication for the quantity of oil available to the market.
  +1.0 = strong supply increase (OPEC adds barrels, sanctions lifted on a major producer, new pipeline online, Venezuela ramping up).
  +0.5 = moderate supply increase (single-country production gains, inventory builds).
  0.0 = no supply implication.
  -0.5 = moderate supply decrease (refinery outages, minor OPEC cuts).
  -1.0 = strong supply decrease (major OPEC cuts, infrastructure attacks, embargoes, refinery fires affecting multiple facilities).
- demand_impact: Direction of impact on WTI demand. Score the article's implication for oil consumption.
  +1.0 = strong demand increase (China stimulus, Fed pivot to easing, growth surprise, summer driving acceleration).
  +0.5 = moderate demand increase (regional growth indicators).
  0.0 = no demand implication.
  -0.5 = moderate demand decrease (mild slowdown signals).
  -1.0 = strong demand decrease (recession confirmed, Fed strongly hawkish, major economy lockdowns, demand destruction events).
- risk_premium: Change in geopolitical or operational risk premium for oil. Score the direction of risk change, not the absolute level.
  +1.0 = major escalation (military strike on oil infrastructure, war, Strait of Hormuz incident, major sanctions imposed).
  +0.5 = moderate escalation (rising tensions, threats, troop movements).
  0.0 = no change in risk environment.
  -0.5 = moderate de-escalation (talks resuming, partial sanctions lifted).
  -1.0 = major de-escalation (ceasefire signed, diplomatic breakthrough, full sanctions removal).
- magnitude: how market-moving is this event. 0.0 = negligible, 1.0 = historic. Most articles score 0.1–0.4. Major OPEC cut = 0.9. Geopolitical crisis = 1.0.
- event_type: 1–3 categories ordered by salience from: geopolitical, supply, demand, macro, technical, other. EIA inventory articles classify as supply or macro — do not use "inventory".
- entities: specific named actors central to the article (not incidentally mentioned). Include countries, organizations, oil companies, and key officials. Use the names exactly as they appear in the article.
- certainty: how confirmed is the information. 0.0 = rumor, 0.5 = analyst forecast, 0.9 = official announcement.
- time_horizon: immediate (hours to 1 day), short_term (days to weeks), or structural (months-plus or permanent themes such as energy transition).

The three impact fields (supply_impact, demand_impact, risk_premium) are intentionally orthogonal. An article can affect multiple channels independently. Score each based on what the article actually says about that specific channel. If the article has no clear implication for a channel, score it 0.0.
```

**A.2 Tool schema.** The model is forced to call a single tool, `extract_article_features`, whose `input_schema` is reproduced verbatim below. Only `usable` is required; every other field is omitted when `usable=false`.

```json
{
  "name": "extract_article_features",
  "description": "Extract structured WTI-relevant features from a news article for use in a quantitative liquidity impact model.",
  "input_schema": {
    "type": "object",
    "properties": {
      "usable":         { "type": "boolean" },
      "sentiment_score":{ "type": "number" },
      "supply_impact":  { "type": "number", "minimum": -1.0, "maximum": 1.0 },
      "demand_impact":  { "type": "number", "minimum": -1.0, "maximum": 1.0 },
      "risk_premium":   { "type": "number", "minimum": -1.0, "maximum": 1.0 },
      "magnitude":      { "type": "number" },
      "event_type":     { "type": "array",
                          "items": { "type": "string",
                                     "enum": ["geopolitical","supply","demand","macro","technical","other"] },
                          "minItems": 1, "maxItems": 3 },
      "entities":       { "type": "array", "items": { "type": "string" } },
      "certainty":      { "type": "number" },
      "time_horizon":   { "type": "string",
                          "enum": ["immediate","short_term","structural"] }
    },
    "required": ["usable"]
  }
}
```

Each numeric field carries a per-field `description` in the live schema that restates the scoring anchors of the system prompt; those are elided here for brevity and are recoverable from the source file. Note that `time_horizon` is a single categorical label, not an array (only `event_type` is multi-valued).

**A.3 Request configuration.** Requests are submitted through the Anthropic Batches API, one request per article, with the following parameters:

- `model`: `claude-haiku-4-5`
- `max_tokens`: 500
- `system`: the A.1 prompt, sent with `cache_control` ephemeral so the prompt is cached across the batch
- `tools`: the A.2 tool; `tool_choice` set to force `extract_article_features`
- user message: `"Title: {title}\n\nBody: {body}"`, where the body is UTF-8 sanitised and truncated to the first 1,500 characters (title only when no body was retrieved)

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

## Appendix E: Inter-model calibration comparison

This appendix gives the article-by-article record behind the inter-model calibration of Sections 3.7 and 4.3.4. The same 30-article stratified sample was scored independently by Claude Haiku 4.5 (the extraction model, denoted H) and by GPT-5.5 (an independent reference from a different model family, accessed through the ChatGPT interface, denoted G), using the identical Schema v2 prompt and tool of Appendix A. Two independently trained models agreeing is a defensible, if weaker than human, validation signal; disagreement flags genuine ambiguity in the article.

**Aggregate agreement (Schema v2).** The two models agree on the binary `usable` flag for 27 of 30 articles; the three disagreements (IDs 13331, 10308, 10211) are borderline single-company or forecast pieces that Haiku filtered out and GPT retained. On the 14 articles both models judged usable, the composite `sentiment_score` reaches a Pearson correlation of 0.88, with a single sign disagreement (1/14, using a 0.1 neutral band, so only scores with magnitude above 0.1 count as directional). The three channels reach Pearson correlations of 0.94 (supply), 0.96 (demand), and 0.82 (risk). For contrast, Schema v1, which scored only the composite, gave a sentiment correlation of 0.39 with 4 sign disagreements out of 13; the channel decomposition of Schema v2 is what lifted agreement (Section 4.3.6).

**Usability and composite sentiment (all 30 articles).**

| ID | Title | Usable H | Usable G | Sent. H | Sent. G |
| ---: | :--- | :---: | :---: | ---: | ---: |
| 11630 | Top Stock Movers Now : Vertex , DoorDash , Con… | ✗ | ✗ |  |  |
| 16558 | Oil prices fall as analysts forecast an increa… | ✓ | ✓ | -0.60 | -0.68 |
| 18615 | WTI Forecast : Retreats from 7 - month top , h… | ✓ | ✓ | +0.70 | +0.74 |
| 13331 | Crude Oil Weekly Forecast 10 / 08 : Tempting S… | ✗ | ✓ |  | -0.42 |
| 13772 | Focus on Nigeria Non - oil Revenue Potential | ✗ | ✗ |  |  |
| 4108 | Oil up on Opec+ meeting , summer driving seaso… | ✓ | ✓ | +0.60 | +0.81 |
| 8677 | Top 5 things to watch in markets in the week a… | ✓ | ✓ | +0.00 | -0.16 |
| 3151 | Oil recovers after surprise drawdown in US sto… | ✓ | ✓ | +0.30 | +0.37 |
| 3779 | Global supply chains now in Goldilocks Zone : … | ✓ | ✓ | +0.40 | +0.21 |
| 6068 | Oil climbs with U . S . Federal Reserve pivot | ✓ | ✓ | +0.60 | +0.58 |
| 3335 | Gran Tierra Energy Inc . Announces First Quart… | ✓ | ✓ | +0.30 | -0.06 |
| 10308 | Vermilion Energy Inc . Announces Results for t… | ✗ | ✓ |  | -0.02 |
| 7834 | Is Occidental Petroleum Corporation ( OXY ) th… | ✓ | ✓ | +0.20 | +0.17 |
| 7188 | ALPS Advisors Inc . Expands Stake in EnLink Mi… | ✗ | ✗ |  |  |
| 10989 | Why Occidental Petroleum Stock Plunged 10 % To… | ✓ | ✓ | -0.30 | +0.09 |
| 10799 | Is Atmos Energy Corp . ( ATO ) The Best Defens… | ✗ | ✗ |  |  |
| 13211 | Baytex Energy Corp .: Baytex Delivers Solid Se… | ✓ | ✓ | +0.30 | -0.12 |
| 20149 | Why Halliburton ( HAL ) Stock Is Trading Up To… | ✓ | ✓ | +0.70 | +0.86 |
| 10211 | Is Chevron Corporation ( CVX ) The Best Crude … | ✗ | ✓ |  | -0.35 |
| 6161 | 1 Growth Stock Down 18 % to Buy Right Now | ✗ | ✗ |  |  |
| 8882 | Petroleum prices increase ahead of Christmas | ✗ | ✗ |  |  |
| 6160 | Stock market today : US stocks slip as traders… | ✗ | ✗ |  |  |
| 13217 | The Latest : Market slumps , Gaza bloodbath an… | ✗ | ✗ |  |  |
| 11929 | Oil Prices Rise Amid US - China Trade Optimism… | ✗ | ✗ |  |  |
| 16355 | Iran Warns US Intervention Would Destabilize R… | ✓ | ✓ | +0.30 | +0.25 |
| 21358 | How China built an energy fortres to survive o… | ✗ | ✗ |  |  |
| 21692 | Global oil prices surge above $111 per barrel … | ✓ | ✓ | +0.70 | +0.85 |
| 1636 | Gasoline Market 2024 - 2032 : Industry Report … | ✗ | ✗ |  |  |
| 19074 | Governments Move to Shield Economies as Oil Ju… | ✗ | ✗ |  |  |
| 18 | Oil jumps 1 % in New Year after US forces repe… | ✗ | ✗ |  |  |

**Channel scores (14 articles both models judged usable).**

| ID | Supply H | Supply G | Demand H | Demand G | Risk H | Risk G |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16558 | +0.70 | +0.72 | -0.40 | -0.34 | -0.30 | +0.18 |
| 18615 | +0.00 | -0.12 | +0.00 | +0.00 | +0.60 | +0.78 |
| 4108 | -0.50 | -0.72 | +0.50 | +0.48 | +0.00 | +0.00 |
| 8677 | +0.00 | +0.00 | +0.00 | -0.22 | +0.00 | +0.08 |
| 3151 | -0.50 | -0.44 | +0.00 | -0.12 | +0.00 | +0.00 |
| 3779 | +0.00 | +0.00 | +0.50 | +0.34 | +0.00 | +0.00 |
| 6068 | +0.00 | -0.08 | +0.70 | +0.52 | +0.00 | +0.00 |
| 3335 | +0.40 | +0.29 | +0.00 | +0.00 | +0.00 | +0.00 |
| 7834 | +0.00 | -0.12 | +0.00 | +0.00 | +0.00 | +0.44 |
| 10989 | +0.00 | +0.06 | +0.00 | +0.00 | +0.00 | +0.00 |
| 13211 | +0.40 | +0.33 | +0.00 | +0.00 | +0.00 | +0.00 |
| 20149 | +0.00 | -0.48 | +0.00 | +0.00 | +0.50 | +0.91 |
| 16355 | +0.00 | +0.00 | +0.00 | +0.00 | +0.50 | +0.45 |
| 21692 | -0.60 | -0.75 | +0.00 | +0.00 | +0.40 | +0.60 |

The residual disagreement concentrates in the risk channel (r = 0.82, the lowest of the three): on IDs 16558, 20149, and 7834 the two models weight the geopolitical or operational risk implication quite differently, whereas supply and demand track closely. This is consistent with risk being the hardest channel to score, noted in Section 4.3.6.

## Appendix F: Extended metrics tables

This appendix reports the full TFT v2 test-set metrics summarised in Section 4.3.7.3. For each of the three targets it gives mean absolute error (MAE) and root mean squared error (RMSE) at each forecast horizon against the persistence baseline (`y_hat_{t+k} = y_t`, the last observed value carried forward), broken out by the full test set and by the pre-war and war regimes (split at the war onset, decoder time index 10056). "Reduction" is the percentage by which the TFT lowers MAE relative to persistence; a positive value means the model beats persistence. The test set has 2,159 forecast origins; the pre-war and war row counts differ slightly across horizons because a fixed origin's +k-step target crosses the war boundary at different k.

#### log_volume

| Horizon | Slice | n | TFT MAE | Persist. MAE | Reduction | TFT RMSE | Persist. RMSE |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| +1h | overall | 2159 | 0.585 | 1.076 | +46% | 0.915 | 2.334 |
| +1h | pre-war | 994 | 0.536 | 1.075 | +50% | 0.863 | 2.233 |
| +1h | war | 1165 | 0.628 | 1.076 | +42% | 0.957 | 2.417 |
| +3h | overall | 2159 | 0.577 | 1.452 | +60% | 0.895 | 2.734 |
| +3h | pre-war | 992 | 0.532 | 1.480 | +64% | 0.855 | 2.664 |
| +3h | war | 1167 | 0.614 | 1.428 | +57% | 0.928 | 2.793 |
| +6h | overall | 2159 | 0.602 | 1.820 | +67% | 0.932 | 2.838 |
| +6h | pre-war | 989 | 0.543 | 1.935 | +72% | 0.865 | 2.841 |
| +6h | war | 1170 | 0.652 | 1.724 | +62% | 0.985 | 2.835 |
| +12h | overall | 2159 | 0.631 | 2.174 | +71% | 0.958 | 3.062 |
| +12h | pre-war | 983 | 0.555 | 2.319 | +76% | 0.874 | 3.111 |
| +12h | war | 1176 | 0.695 | 2.053 | +66% | 1.024 | 3.021 |

#### amihud (units of 10⁻³)

| Horizon | Slice | n | TFT MAE | Persist. MAE | Reduction | TFT RMSE | Persist. RMSE |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| +1h | overall | 2159 | 0.23 | 0.41 | +42% | 2.73 | 3.86 |
| +1h | pre-war | 994 | 0.10 | 0.15 | +31% | 0.60 | 0.87 |
| +1h | war | 1165 | 0.35 | 0.62 | +45% | 3.67 | 5.20 |
| +3h | overall | 2159 | 0.23 | 0.41 | +44% | 2.72 | 3.86 |
| +3h | pre-war | 992 | 0.10 | 0.15 | +36% | 0.59 | 0.87 |
| +3h | war | 1167 | 0.34 | 0.62 | +46% | 3.65 | 5.19 |
| +6h | overall | 2159 | 0.22 | 0.41 | +45% | 2.72 | 3.86 |
| +6h | pre-war | 989 | 0.09 | 0.15 | +37% | 0.59 | 0.87 |
| +6h | war | 1170 | 0.33 | 0.62 | +46% | 3.65 | 5.19 |
| +12h | overall | 2159 | 0.22 | 0.41 | +45% | 2.72 | 3.86 |
| +12h | pre-war | 983 | 0.09 | 0.15 | +38% | 0.60 | 0.87 |
| +12h | war | 1176 | 0.33 | 0.62 | +46% | 3.64 | 5.17 |

#### price_range

| Horizon | Slice | n | TFT MAE | Persist. MAE | Reduction | TFT RMSE | Persist. RMSE |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| +1h | overall | 2159 | 0.718 | 0.495 | -45% | 1.287 | 0.981 |
| +1h | pre-war | 994 | 0.154 | 0.180 | +14% | 0.256 | 0.278 |
| +1h | war | 1165 | 1.200 | 0.764 | -57% | 1.736 | 1.310 |
| +3h | overall | 2159 | 0.719 | 0.578 | -24% | 1.286 | 1.107 |
| +3h | pre-war | 992 | 0.155 | 0.220 | +30% | 0.257 | 0.329 |
| +3h | war | 1167 | 1.199 | 0.883 | -36% | 1.733 | 1.475 |
| +6h | overall | 2159 | 0.721 | 0.630 | -14% | 1.289 | 1.194 |
| +6h | pre-war | 989 | 0.156 | 0.233 | +33% | 0.256 | 0.353 |
| +6h | war | 1170 | 1.198 | 0.966 | -24% | 1.734 | 1.589 |
| +12h | overall | 2159 | 0.724 | 0.701 | -3% | 1.294 | 1.252 |
| +12h | pre-war | 983 | 0.157 | 0.278 | +43% | 0.257 | 0.390 |
| +12h | war | 1176 | 1.198 | 1.055 | -14% | 1.737 | 1.658 |

**Reading.** Three patterns stand out. On `log_volume`, the primary liquidity target, the TFT beats persistence at every horizon and in both regimes, and the margin widens with horizon (from 46% at +1h to 71% at +12h overall), consistent with the delayed-propagation reading of RQ1 and stable across the war boundary. On `amihud`, the TFT also beats persistence throughout (42 to 46% overall); absolute errors are on the order of 10⁻⁴ and are dominated by war-regime spikes. On `price_range`, the picture splits by regime: the TFT beats persistence in the pre-war window (reduction rising to +43% at +12h) but underperforms it overall and in the war regime (negative reductions). Trained only on pre-war data, the model reverts toward the historical mean and cannot reach the war-regime volatility spikes (actual `price_range` up to 14.4 against a predicted ceiling near 1.5), the regime-extrapolation failure discussed in Sections 4.3.7.3 and 5.4.

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
