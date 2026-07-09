# Chapter 2 - Background

This chapter provides the terminology and prior work needed to follow the methods (Chapter 3) and experiments (Chapter 4): the WTI market and its liquidity measures, the NLP tools used to turn news into features, the Temporal Fusion Transformer, and the strands of literature that motivate the design choices of this thesis.

## 2.1 Terminology

*(To draft.)* This section defines the concepts used throughout:

- **WTI crude oil futures and the EIA inventory cycle** (the target instrument and its most consistently market-moving scheduled event).
- **Liquidity measures**: log trading volume, the Amihud illiquidity ratio [CITE: Amihud 2002], and the Parkinson range.
- **NLP for finance**: three-class sentiment classification, structured extraction, and prompt engineering.
- **Temporal Fusion Transformer**: the variable selection network and interpretable multi-head attention.

## 2.2 Related work

The design decisions of this thesis are grounded in prior work rather than derived in isolation. This section states the anchors; each is picked up at the point of use in Chapters 3 and 4.

**News sentiment and market activity.** The premise that news sentiment moves trading activity, and not only prices, is established in the finance literature: media pessimism has been shown to predict downward price pressure and elevated trading volume [CITE: Tetlock 2007]. This grounds the use of news sentiment as a predictor of hourly liquidity in Phase 1 (Chapter 4, §4.2).

**Financial-text sentiment models.** Domain-adapted transformer models outperform general-purpose sentiment tools on financial language; FinBERT [CITE: Araci 2019] is the de facto baseline for financial sentiment and is adopted as the Phase 1 sentiment extractor (§3.3.1). Its coarse three-class output is also the limitation that motivates the richer LLM-based extraction of Phase 2.

**Liquidity measurement.** Hourly liquidity is quantified with standard market-microstructure measures: trading volume, the Parkinson high-low range, and the Amihud illiquidity ratio [CITE: Amihud 2002], which captures price impact per unit of traded volume.

**Oil-price shock decomposition.** Oil-market movements are not a single undifferentiated shock but a composition of separable drivers: the structural decomposition of the real oil price into supply, aggregate-demand, and precautionary (risk) demand components [CITE: Kilian 2009] is the canonical framework. This grounds the Phase 2 channel decomposition (§3.3.4, §4.3.2), which extracts a `supply_impact`, `demand_impact`, and `risk_premium` from each article rather than collapsing them onto a single sentiment axis, and it clarifies why supply-risk geopolitical news is price-bullish rather than bearish.

**Interpretable multi-horizon forecasting.** The Temporal Fusion Transformer [CITE: Lim et al. 2021] is an attention-based architecture for multi-horizon time-series forecasting that couples predictive accuracy with interpretability through a Variable Selection Network and interpretable multi-head attention. These two mechanisms are what let Phase 2 (§3.6, §4.3.5, §4.3.7) read feature importance and temporal attention off the model instead of treating it as a black box, and the thesis uses them directly to corroborate the lag structure of RQ1.

**LLM-based extraction and LLM-as-annotator.** A growing body of work uses large language models for structured feature extraction from financial text [CITE: LLM structured extraction] and validates their outputs by cross-model agreement rather than costly human labels [CITE: LLM-as-annotator]. Phase 2 adopts both: Claude Haiku extracts the structured schema of §3.3.2 through the tool-use API, producing the richer, price-oriented, channel-decomposed features that FinBERT's three-class tone label cannot, and the inter-model calibration of §3.7 (§4.3.4) checks those features against a second model.

> Note: all `[CITE: ...]` tags are placeholders for the exact bibliographic entries, to be resolved during the literature pass. Verify each reference before finalising.
