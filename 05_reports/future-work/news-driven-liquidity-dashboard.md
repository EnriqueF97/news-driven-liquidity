# News-Driven Liquidity Dashboard — Future Project Planning

Personal portfolio project to extend the WTI crude oil thesis work into a working tool. **Not part of the thesis scope.** Start after thesis writeup and defense are complete.

## Concept

A near-real-time dashboard that ingests oil-market news as it's published, processes it through the Phase 2 Haiku v2 pipeline (channel decomposition + entity normalization), generates predictions from the TFT v2 model, and presents the resulting situational awareness in a usable interface. The user gets insight into the current news flow and its predicted impact on WTI liquidity, without the dashboard committing to a trading recommendation.

The project is presented as portfolio material, demonstrating end-to-end ML engineering: real-time pipeline construction, deployment, monitoring, and the ability to translate research output into a usable product.

## Honest framing

This is **not real-time** in the strict sense. GDELT updates every 15 minutes with ~30 min lag; yfinance has its own delays. The honest claim is "near real-time monitoring updated hourly." This is sufficient for the situational awareness use case and avoids overpromising.

The dashboard does not produce trading signals. It presents predictions as informational context. This is intentional — converting predictions to trading signals requires market microstructure considerations outside the scope of the thesis work.

## Architecture overview

```
News sources (GDELT, RSS)     Market data (yfinance, EIA)
        ↓                              ↓
   News scraper                   Market scraper
        ↓                              ↓
   Haiku v2 extraction             Hourly grid builder
        ↓                              ↓
   article_features ←──── join ────→ market_context
                            ↓
                 Hourly modeling DataFrame
                            ↓
                  TFT v2 inference service
                            ↓
                       Predictions DB
                            ↓
                Streamlit dashboard (web UI)
```

Cron-driven cadence: every hour, the pipeline pulls new articles, processes them, joins to market context, runs inference, stores predictions. The dashboard reads from the predictions store and updates on page load.

## Scope decisions

The first version is intentionally narrow:

- **One asset only**: WTI crude (same as thesis)
- **One target focus**: log_volume predictions (the strongest result from v2)
- **Historical horizon shown**: 7-30 days of past predictions vs realized values
- **Forward horizon shown**: next 28h predictions from current time
- **Update cadence**: hourly cron job, not push-based real-time

This keeps the scope contained and the cost manageable. Multi-asset and multi-target expansions are future work after the first version is stable.

## Work breakdown

The estimates assume part-time work (10-15 hours/week) and reuse of code from the thesis project where possible.

### Phase 1 — Core pipeline (~3 weeks)

Build the components that ingest, process, and store data on a recurring schedule.

**1.1 News ingestion** (1 week)

- GDELT API integration (filter for oil-market queries)
- RSS feed scrapers for selected Reuters/Bloomberg/Platts public feeds
- Article body scraping (reuse `04_gdelt_scrapper.ipynb` logic)
- Deduplication against previously-seen articles
- Error handling for paywalls, dead links, format issues

Output: a queue of new articles ready for LLM processing.

**1.2 Haiku v2 LLM extraction** (3-5 days)

- Adapt `03_src/nlp/llm_features.py` for one-article-at-a-time invocation
- Anthropic API integration (already done in thesis pipeline)
- Retry logic for rate limits and transient errors
- Cost monitoring (target: ~$0.01/article, ~50-200 articles/day)

Output: structured features per article (channels, magnitude, certainty, entities) written to DB.

**1.3 Market data ingestion** (2-3 days)

- yfinance hourly OHLC for WTI futures
- EIA inventory data (Wednesday releases)
- DXY and VIX hourly
- Forward-fill / boundary null handling (same as thesis Cell 4)

Output: updated `market_context` table with new rows.

**1.4 Hourly modeling table builder** (2-3 days)

- Join `liquidity` + `llm_features` + `article_entities` for the last 48 hours
- Hour-level aggregation (mean for continuous, max for binary entity flags, dominant article for categoricals)
- Format as a DataFrame matching the TFT v2 input schema

Output: a continuously-updated modeling DataFrame ready for inference.

**1.5 TFT v2 inference service** (3-5 days)

- Load the canonical v2 checkpoint into memory once at service startup
- For each new hour, construct an inference window (current hour + last 47 hours of context)
- Call `tft_eval.predict()` with batch_size=1
- Extract median quantile and 25/75 quantiles for confidence intervals
- Persist predictions with timestamp to a predictions table

Output: predictions table with rows like `(prediction_made_at, target, horizon, predicted_value, q25, q75, actual_value_when_known)`.

**1.6 Cron job orchestration** (2-3 days)

- Sequential pipeline: news ingestion → LLM → market data → modeling table → inference
- Logging and basic alerting on failure
- Idempotent so a failed run can be re-tried without duplicating data

Output: an end-to-end automation that runs hourly.

### Phase 2 — Dashboard UI (~1-1.5 weeks)

The frontend, presented in Streamlit. Reads from the predictions DB and the article store.

**2.1 Layout and navigation** (2-3 days)

- Sidebar: filters (time range, channels, entities)
- Main panel: current predictions + recent news feed + entity activity
- Tabs or sections: Predictions, News, Entities, Backtest

**2.2 Predictions panel** (2-3 days)

- Time series chart: realized log_volume + predictions at multiple horizons + confidence intervals
- "Next N hours" prediction summary with quantile bands
- Per-horizon accuracy track record (rolling MAE over last 30 days)

**2.3 News panel** (1-2 days)

- Recent articles table with channel breakdown (supply/demand/risk colors)
- Filterable by entity, channel, time
- Click-through to read the original (when possible)

**2.4 Entity activity panel** (2 days)

- Heatmap: entities × hours showing mention frequency
- "Trending entities" in the last N hours
- Per-entity time series of mentions

**2.5 Deployment** (1-2 days)

- Streamlit Cloud or HuggingFace Spaces (both free for portfolio use)
- Custom domain if desired
- README with setup, screenshots, methodology explanation

### Phase 3 — Engineering quality (optional, ~1 week)

Items to add if you want production-grade quality for portfolio impact.

- **Monitoring**: track inference latency, error rates, model drift indicators
- **Auto-retraining trigger**: alert when predictions diverge systematically from realized values
- **Backtest replay**: ability to "replay" the dashboard as it would have looked on a past date
- **Tests**: unit tests for the data pipeline, integration tests for the inference service

## Total estimated cost

- **Phase 1 (core)**: 3 weeks part-time
- **Phase 2 (UI)**: 1.5 weeks part-time
- **Total minimum viable**: ~4-5 weeks part-time
- **With Phase 3 polish**: ~5-6 weeks part-time

## Running costs

Hosting and API costs for a personal portfolio version:

- **Anthropic API**: $5-60/month depending on article volume
- **Hosting**: free (Streamlit Cloud, HuggingFace Spaces, Fly.io free tier)
- **News API** (if used instead of GDELT): $0-50/month
- **Total**: $5-110/month, mostly LLM extraction

This is sustainable for a side project but worth being aware of when scoping.

## Key technical risks

These are the things most likely to cost more time than expected:

1. **Article ingestion edge cases**: paywalls, dynamic content, rate limits on free sources. Plan to spend extra time on robust error handling.

2. **Model drift over time**: TFT v2 trained on data through May 2026. Performance will degrade after a few months. Decide upfront whether the dashboard retrains the model or just operates with a degrading-but-honest disclaimer.

3. **GDELT lag and gaps**: GDELT doesn't index all sources reliably. Some major oil-market news may not appear in the feed. RSS feeds fill some gaps but introduce their own issues.

4. **Inference latency**: TFT v2 with batch_size=1 takes ~1-2 seconds per prediction. For a single-hour cron job this is fine. If you want sub-minute updates, this becomes a bottleneck.

## Future extensions (after the first version is stable)

In rough order of value-to-effort ratio:

- **API endpoint** (Option B from the planning discussion): wrap the inference service in FastAPI with auth and rate limiting. Demonstrates API engineering skills. 2-3 additional weeks.
- **Multi-asset**: extend to Brent, natural gas, refined products. Mostly a data pipeline extension, model retraining required.
- **Alerting**: push notifications (email, Slack) when predictions deviate strongly from baseline or when high-magnitude news arrives.
- **Backtesting framework**: simulate strategies based on predictions, with realistic transaction costs.
- **Public-facing methodology**: blog post or technical writeup explaining the channel decomposition, the v2 architecture, the calibration findings from the thesis. Cross-promotes the dashboard.

## Decision points before starting

Three things to decide before the first day of work:

1. **News source**: GDELT only (free, broad coverage, latency) vs hybrid GDELT + paid RSS (closer to real-time, more cost). Start with GDELT only.

2. **Hosting**: Streamlit Cloud (simplest, slowest cold-start) vs HuggingFace Spaces (similar) vs Fly.io (more control, more setup). Start with Streamlit Cloud.

3. **Model serving**: load checkpoint in memory at service startup vs cache predictions to DB and serve from there. Start with cache-to-DB; simpler and decouples inference from UI.

## Pre-requisites for starting

Before opening the project:

- [ ] Thesis writeup complete (Chapters 1-6 drafted)
- [ ] Thesis defended or scheduled
- [ ] TFT v2 final checkpoint exported and documented
- [ ] Phase 2 LLM extraction pipeline cleaned and reusable (not still as research notebooks)
- [ ] Decision on news source, hosting, and serving (above) made

## Project name placeholder

Working name: **OilFlow** (placeholder; rename before deployment).

## Notes

- This document is forward-looking and may need updates as priorities shift.
- The thesis defense is the trigger to start, not just thesis writeup. Allow buffer for revisions.
- If the thesis defense is delayed, this project is on hold.
- The Haiku v2 LLM extraction pipeline is the most valuable single asset to preserve cleanly between thesis and portfolio project. Make sure the code in `03_src/nlp/` is production-grade quality before moving on.
