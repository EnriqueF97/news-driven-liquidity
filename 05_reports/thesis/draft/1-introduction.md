# Chapter 1 - Introduction

## 1.1 Motivation

News moves markets. That media sentiment forecasts not only prices but trading activity is well established: media pessimism predicts downward price pressure and elevated trading volume [CITE: Tetlock 2007]. Yet most empirical work on news and markets operates at daily resolution, aggregating away the intraday window in which information actually propagates into trading. The market response to a scheduled release is concentrated within minutes to hours of its arrival and decays over the rest of the session [CITE: Ederington & Lee 1993]; a daily bar cannot resolve whether liquidity reacts on impact, an hour later, or over the following trading day.

WTI crude oil is an especially informative setting for closing that gap. Its price and trading activity are driven by a rich, well-dated event structure, OPEC policy, geopolitics in the major producing regions, the weekly EIA inventory cycle, and broader macroeconomic releases, each of which arrives at a known time and plausibly moves liquidity at a horizon shorter than a day. This combination of frequent, identifiable news and a deeply liquid market makes WTI well suited to an hourly-resolution study of how news propagates into liquidity, which daily-frequency work has been unable to observe.

## 1.2 Problem statement

This thesis asks how news propagates into WTI crude oil liquidity at hourly resolution. Three concerns organize the problem. First, the lag structure: when news arrives, does trading activity react in the same hour, or with a delay, and if delayed, over what horizon? Second, directional asymmetry: does negative (bearish) news move liquidity more than positive (bullish) news of comparable magnitude, as a loss-aversion account would predict? Third, the role of context: how much of the liquidity response is attributable to the news signal itself versus the macro-financial state (volatility, the dollar) and the type of event, and can that attribution be read off an interpretable model rather than assumed?

Answering these requires turning unstructured news text into numerical features at hourly resolution and aligning them with a market grid, then modeling the news-liquidity relationship with methods whose assumptions are transparent enough to support interpretation of the lag and asymmetry structure.

## 1.3 Research questions

**RQ1 (lag structure).** At what lag, if any, does news sentiment exert its strongest effect on WTI trading liquidity, measured at hourly resolution?

**RQ2 (directional asymmetry).** Does the liquidity response differ between bearish and bullish news, and if so, in what sense?

## 1.4 Contributions

This thesis makes three contributions, developed in Chapters 4 and 5.

**(a) An empirical lag-and-asymmetry characterization for WTI at hourly resolution.** The news-liquidity response is not contemporaneous: its strongest effect falls at a lag of roughly six to twelve hours, recovered independently by a lag-regression baseline (§4.2.3) and by the attention pattern of a Temporal Fusion Transformer (§4.3.7). On asymmetry, the two phases estimate different quantities and, read together, sharpen rather than contradict each other: a linear baseline finds a larger marginal coefficient on negative than on positive sentiment (§4.2.3), whereas the forecasting model shows no bearish-versus-bullish gap in predicted volume and instead keys on the risk the news implies, volatility, the extracted risk-premium channel, and supply-risk entities, rather than on the sign of the sentiment (§4.3.7). The finding is therefore not that bearish news beats bullish news, but that liquidity responds to the salience and risk content of news more than to its direction.

**(b) A channel decomposition of LLM-extracted news features, validated by cross-model agreement.** Rather than collapse each article onto a single sentiment score, Phase 2 extracts orthogonal supply, demand, and risk-premium channels with a large language model, mirroring the structural decomposition of oil-price shocks [CITE: Kilian 2009]. In the absence of human ground truth, the extraction is validated by inter-model agreement: recalibrating onto the channels raises cross-model correlation on the composite sentiment from 0.39 to 0.88 (§4.3.4). The decomposition also resolves an apparent paradox, why supply-risk geopolitical news is price-bullish rather than bearish.

**(c) A comparison of title-only and full-body news representations.** Sentiment extracted from headlines alone disagrees with sentiment from full article bodies on 41.6% of articles, and the disagreement is systematic: flipped articles shift bearish far more often than bullish (§4.2.2). This quantifies a headline bias that studies relying on titles alone silently inherit.

Underlying all three is a methodological template: a simple, interpretable baseline (FinBERT features with lag regression) and a richer, more expressive second phase (LLM channels with a Temporal Fusion Transformer) that corroborates the baseline and enriches its analytical vocabulary, with neither phase discarded (§5.3).

## 1.5 Structure of the thesis

Chapter 2 fixes terminology and situates the thesis against prior work on news and markets, financial-text models, and interpretable forecasting. Chapter 3 details the two-phase methodology: data acquisition and hourly alignment, feature extraction (FinBERT in Phase 1, LLM channel extraction in Phase 2), and the modeling and evaluation setup. Chapter 4 reports the experiments: the headline-bias comparison and lag-regression results of Phase 1, and the Temporal Fusion Transformer results of Phase 2, including feature importance, temporal attention, and the directional-asymmetry test. Chapter 5 interprets the findings against the research questions and states the study's limitations. Chapter 6 concludes. The appendices provide the extraction prompt and schema, the ablation and hyperparameters, the canonical entity list, and a reproducibility statement.
