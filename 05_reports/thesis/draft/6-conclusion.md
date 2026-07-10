# Chapter 6 - Conclusion

This thesis seeks to understand and modelate how news propagates into the liquidity of WTI crude oil futures at hourly resolution, along two axes: the temporal lag of the response (RQ1) and its directional asymmetry (RQ2). It pursued the questions through a two-phase design, a regression-based Phase 1 on FinBERT sentiment and a deep-learning Phase 2 on LLM-extracted, channel-decomposed features.

This chapter states the contributions, summarizes the findings, and closes.

## 6.1 Summary of contributions

The thesis makes three empirical and four methodological contributions.

Empirically:

- **An hourly lag structure for the news-liquidity response.** The response peaks in the +6 to +12 hour window, established by the Phase 1 lag OLS and independently corroborated by the Phase 2 TFT through both its per-horizon error curve and its direction-conditioned attention. This is structure that daily-frequency work cannot observe.
- **A precise account of the directional asymmetry.** There is a robust bearish over bullish asymmetry in the marginal sensitivity of volume to negative-tone news (Phase 1). It does not appear as a difference in the level of predicted volume (Phase 2), because the model organizes its predictions around risk and salience rather than bullish or bearish price direction. The contribution is as much the precision as the result: stating the asymmetry correctly requires naming the sentiment construct and the estimand it refers to.
- **A news-representation finding.** Title-only and title-plus-body FinBERT sentiment disagree on 41.6% of articles, with titles leaning more bullish than the bodies they head, evidence that the choice of news representation is itself a modeling decision.

Methodologically (developed in §5.3):

- **Channel decomposition** as a higher-resolution feature set and, through cross-model agreement, a moderately defensible validation substrate in the absence of human ground truth.
- **The tone-versus-price distinction**, showing that a sentiment measure is defined by its construct, and that a tone classifier and a price-reasoning LLM diverge on exactly the geopolitical news that drives volume.
- **LLM-as-filter** as a semantically grounded alternative to regex heuristics for noisy web-scraped news corpora.
- **A two-phase template** that pairs an interpretable statistical baseline with an expressive deep-learning model that corroborates rather than replaces it.

## 6.2 Summary of findings

On RQ1, news affects WTI liquidity most strongly not in the same hour but over a +6 to +12 hour window. Two independent methods agree: the Phase 1 lag OLS peaks at +6 hours, and the Phase 2 TFT, trained and evaluated separately, places its strongest persistence-relative gain in the same window and attends to -6 hours for bearish news. The liquidity response to news is gradual, not instantaneous.

On RQ2, the answer depends on how the question is posed. There is a robust bearish over bullish asymmetry in the marginal sensitivity of volume to negative-tone news, and this is the thesis' primary answer. It does not survive as a difference in the average level of predicted volume, because the model organizes its forecasts around the risk and salience a news item carries rather than its price direction, and because a tone-based and a price-based sentiment measure diverge on exactly the geopolitical news that moves the market. WTI liquidity responds to how much a piece of news raises supply uncertainty more than to whether it reads as nominally good or bad.

As a forecasting exercise, TFT v2 cut log_volume error over a persistence baseline by 46 to 71 percent across horizons and amihud by 43 to 45 percent, with VIX, the supply and demand channels, and geopolitical entity flags as its leading features; it failed only on price_range in the unseen war regime, a clean case of regime extrapolation. Alongside the two headline answers, three supporting findings shaped the research development: a 41.6 percent title-versus-body sentiment disagreement, a composite-sentiment calibration failure between model families that motivated the channel decomposition, and a semantic-versus-lexical filter comparison that motivated the `usable_strict` rule.

## 6.3 Closing remarks

This thesis began from a simple observation, that news moves markets, and asked a more precise question of it: at what lag, and with what asymmetry, does news move the liquidity of WTI crude oil at hourly resolution. The answer that emerged is itself twofold. The empirical answer is that the response is gradual and peaks over a +6 to +12 hour window, and that its directional asymmetry is real but must be stated with care, because it lives in the sensitivity of volume to news, not in the average volume of nominally good or bad hours. The methodological answer is that reaching this precision required treating sentiment not as a single given quantity but as a construct to be defined, decomposed, and validated, which is where the channel decomposition, the LLM extraction, and the two-phase design earn their place.

If one idea carries beyond this study, it is that for a commodity like crude oil the liquidity response to news is organized around risk rather than direction. The market does not trade more because the news is bad; it trades more because the news raises uncertainty about supply, and in the oil market that uncertainty is often, against intuition, bullish for price. Recognizing this dissolves the apparent tension between a tone-based and a price-based reading of the same events, and reframes what a news-driven liquidity model should be built to detect.

The work is bounded: one commodity, two years, one structural break, and a validation resting on model agreement rather than human judgment. But within those bounds it offers both a concrete empirical result and a reusable way of reaching it, and the extensions of Section 5.5 are, for the most part, small steps rather than new programs. The propagation of news into market liquidity is a question of resolution as much as of theory, and at hourly resolution it shows more structure, and more nuance, than a daily view can.
