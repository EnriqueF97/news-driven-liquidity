# Glossary

This glossary defines terms from finance, statistics, and machine learning that recur in this thesis, in plain language, for readers who are not specialists in these fields. Terms are listed alphabetically.

**Ablation study.** Removing or changing one component of a model at a time to measure how much that component matters, like testing a recipe by leaving out one ingredient at a time.

**Amihud illiquidity ratio.** A measure of how much the price moves for a given amount of trading. A high value means even small trades push the price a lot, so the market is illiquid.

**Asymmetry (directional).** When two opposite cases produce different-sized effects. Here, whether negative news moves the market more than positive news of the same strength.

**Bearish / Bullish.** Bearish means expecting the price to fall (news that is bad for the price). Bullish means expecting it to rise (news that is good for the price).

**Channel (supply, demand, risk premium).** In this thesis, one of three separate economic dimensions a news item can affect: how much oil is available (supply), how much is consumed (demand), or how much geopolitical or operational risk it implies (risk premium).

**Chokepoint.** A narrow, critical passage in a supply route (for example the Strait of Hormuz) whose disruption can cut off a large share of the world's oil flow.

**Coefficient.** In a regression, a number that says how much the outcome changes when a given input increases by one unit.

**Contemporaneous.** Happening at the same time. A contemporaneous effect is felt in the same hour the news arrives, with no delay.

**Cross-sectional.** An analysis that treats each observation (here, each article) on its own, rather than as a connected sequence over time.

**DXY (U.S. Dollar Index).** A measure of the dollar's value against a basket of major currencies. Oil is priced in dollars, so it is a relevant control variable.

**EIA (Energy Information Administration).** The U.S. agency that publishes weekly crude oil inventory figures, a regularly scheduled and closely watched event in oil markets.

**Embedding.** A way of turning a category (such as "supply" versus "demand") into a small list of numbers the model can learn from, instead of an arbitrary code.

**Encoder / Decoder.** In a forecasting model, the encoder reads the past window of data and the decoder produces the forecast for the future steps.

**Estimand.** The specific quantity a method is actually estimating. Two methods can both study "asymmetry" yet target different estimands, so their answers need not agree.

**Front-month futures.** The futures contract closest to expiry, which is the most actively traded one for a commodity like crude oil.

**Futures contract.** An agreement to buy or sell an asset at a set price on a future date. It is the main way crude oil is traded.

**Lag.** A delay. A "+6 hour lag" effect is one that appears six hours after the news, not immediately.

**Liquidity.** How easily an asset can be traded without moving its price. It is measured here mainly by hourly trading volume.

**Logarithm (log), log volume.** A mathematical transform that compresses large numbers. Taking the log of trading volume makes its large swings easier to model.

**Marginal effect.** The change in the outcome from a one-unit increase in one input, holding everything else fixed.

**Multi-horizon forecast.** A forecast that predicts several steps ahead at once (here, 1, 3, 6, and 12 hours), rather than only the next step.

**OLS (ordinary least squares).** The standard method for fitting a straight-line (linear) relationship between inputs and an outcome.

**OPEC / OPEC+.** The group of major oil-producing countries whose production decisions strongly affect global oil supply and price.

**Orthogonal.** Independent and non-overlapping. Orthogonal channels each capture a distinct dimension without duplicating one another.

**Parkinson range.** A volatility measure based on the gap between an hour's highest and lowest prices.

**Persistence baseline.** A simple forecast that just repeats the last observed value ("the next value equals the current one"). Models are judged by how far they beat it.

**p-value.** A number indicating how likely a result could have appeared by chance. Small values (for example below 0.05) suggest the result is real rather than accidental.

**Quantile.** A cut-point in a range of possible values (the median is the 50% quantile). The model predicts a spread of quantiles to express its uncertainty.

**Regime, structural break.** A regime is a distinct market state (for example calm versus wartime). A structural break is the point where the market shifts from one regime to another.

**Risk premium.** The extra compensation markets demand for bearing risk. Here, how much a news item raises uncertainty about oil supply.

**Salience.** How attention-grabbing or prominent something is, regardless of whether it is good or bad.

**Sentiment score.** A numerical reading of how positive or negative a text is. In this thesis it specifically means the net effect the news implies for the WTI price.

**Structured extraction.** Having a model fill in a fixed set of fields (a schema) from free text, instead of returning a single label.

**Temporal Fusion Transformer (TFT).** A neural-network forecaster for time series designed to be interpretable, so one can read off which inputs and which past hours it relied on.

**Tone versus price impact.** Tone is how positive or negative the words of an article sound. Price impact is what the news actually implies for the price. The two can disagree: a supply threat sounds alarming (negative tone) yet pushes prices up (bullish).

**VAR (vector autoregression).** A time-series model of how several variables move together and influence one another over time.

**Variable Selection Network (VSN).** The part of the Temporal Fusion Transformer that learns how much weight to give each input feature.

**VIX.** A market index measuring the stock market's expected volatility, often called the "fear gauge."

**Volatility.** How much a price fluctuates. Higher volatility means larger, faster swings.

**Volume (trading volume).** The number of contracts traded in a period. It is the primary liquidity measure in this thesis.

**WTI (West Texas Intermediate).** The U.S. benchmark grade of crude oil, and the instrument this thesis studies.
