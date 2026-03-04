# Progress Report — Month 1
## Project: Modeling News-Driven Liquidity Dynamics and Information Propagation in Commodities Markets
**Author:** Enrique Favila Martínez  
**Institution:** Radboud University — Master's in Artificial Intelligence  
**Host company:** Hammer Market Intelligence  
**Academic supervisor:** Dr. Lejla Batina  
**Reporting period:** Month 1 of 6

---

## 1. Introduction

This report documents the progress, methodological decisions, and preliminary findings corresponding to the first month of thesis development. During this period, activities included data source exploration, evaluation of natural language processing tools, commodity selection, and the construction of an initial data acquisition and analysis pipeline. The decisions made during this phase shape the methodological design of the rest of the project and are documented here to ensure traceability and reproducibility of the research.

---

## 2. Exploration of Sentiment Analysis Models

### 2.1 Models evaluated

One of the first tasks of the project involved evaluating large language models (LLMs) capable of classifying the sentiment of financial news. The goal was to determine which tool is best suited for extracting sentiment signals from unstructured text in the context of commodities markets.

The following models were evaluated:

- **FinBERT** (Araci, 2019): a BERT-based model fine-tuned on financial corpora. It classifies sentiment into three categories: positive, neutral, and negative. Its training includes data from the COVID-19 period, broadening its coverage of high-volatility informational events.

- **SEC-BERT-finetuned-finance-classification**: a variant fine-tuned on regulatory financial documents, with classification into the categories bullish, neutral, and bearish. Its focus on formal documents distinguishes it from FinBERT in terms of training domain.

- **General-purpose models**: large-scale conversational models such as ChatGPT (OpenAI) and DeepSeek were also included in the evaluation, applied directly to the same news articles through sentiment classification prompts.

### 2.2 Evaluation methodology

The news articles used for evaluation covered topics related to agriculture and international trade tariffs, in the context of the commodity initially selected for study (sugar, see Section 3). The articles came from diverse sources with varying levels of objectivity and editorial bias, which allowed for characterization of model behavior when faced with potentially slanted text.

### 2.3 Results and observations

Results were heterogeneous across models. Each tool produced different classifications for the same news articles, with some models showing a tendency toward neutral categories while others produced more polarized outputs. This dispersion is a methodologically relevant observation: the choice of sentiment model is not a secondary technical decision, but one that directly affects the labels fed into downstream models and, therefore, the conclusions of the analysis. In the absence of a consolidated standard for sentiment classification in commodities news, the selection and validation of the NLP model constitutes a methodological contribution of this project in its own right.

This observation is supported by the literature, which notes that sentiment extraction from financial text varies significantly depending on the model and its training domain (Araci, 2019; Malo et al., 2014).

---

## 3. Commodity Selection: From Sugar to Crude Oil

### 3.1 Initial commodity: sugar

The original project plan considered sugar as the commodity of study. To this end, a search for data sources was conducted across specialized platforms including TradingMap, UN Comtrade, and FAOSTAT. While these sources provide data on production, international trade, and prices, the exploration revealed structural limitations that compromise the viability of the project with this commodity.

The main limitations identified were:

- **Low temporal variability:** sugar prices change infrequently and with low amplitude compared to energy commodities.
- **Predictable seasonality:** price fluctuations are largely driven by harvest cycles, which reduces the element of informational surprise that is central to this project.
- **Scarcity of high-impact exogenous events:** unlike energy commodities, sugar rarely experiences abrupt shocks driven by geopolitical news or international policy decisions — precisely the type of events this project aims to model.

These characteristics make sugar a commodity with insufficient material for studying information propagation through news.

### 3.2 Selected commodity: WTI crude oil

Based on the above criteria, the decision was made to pivot to West Texas Intermediate (WTI) crude oil, traded on the NYMEX under the ticker CL. This decision is supported by the following reasons:

- **High sensitivity to external events:** WTI prices react immediately and measurably to geopolitical events — armed conflicts, international sanctions, tensions between producer countries — as well as to OPEC decisions and macroeconomic data releases such as the weekly inventory reports published by the U.S. Energy Information Administration (EIA). The concept of geopolitical risk (GPR) is well documented in the energy markets literature and represents one of the primary channels through which news text translates into price and liquidity movements.
- **Abundance of academic literature:** a broad body of prior research exists on the relationship between news and oil prices, enabling results to be contextualized and compared.
- **Availability of high-frequency data:** unlike sugar, WTI has intraday data accessible through multiple public sources.
- **Flexibility for extensions:** the structure of energy markets allows, in a later phase, for the analysis to be extended to related commodities such as Brent, natural gas, or gasoline, within the framework of the cross-commodity spillover study (RQ3).

---

## 4. Data Source Exploration and Selection

### 4.1 OilPriceAPI

OilPriceAPI was evaluated as a primary source of intraday price data for WTI. The platform offers real-time prices, historical data, futures curves, and inventory data. However, technical exploration revealed a structural limitation in the free tier: a hard cap of 100 records per request and 500 records for hourly historical data over a 30-day period. This restriction makes it unfeasible to build a training dataset of sufficient size for the models planned in the thesis. As a result, OilPriceAPI was discarded as the primary price data source.

### 4.2 yfinance — adopted source

As an alternative, the yfinance Python library was adopted, providing access to historical data for the WTI futures contract (ticker CL=F) at hourly resolution with approximately two years of coverage. The data includes Open, High, Low, Close, and Volume columns (OHLCV), which directly resolves the liquidity variable problem. The resulting dataset comprises **11,205 hourly records** with no volume restrictions or associated costs.

### 4.3 EIA — structured news event source

For news representation, the EIA's Weekly Petroleum Status Report was adopted as a first source, freely available through its public API. This report, published every Wednesday at 10:30 AM ET, discloses the level of commercial crude oil inventories in the United States. **321 weekly observations** were downloaded with coverage from 2020 to 2026. Each publication constitutes a structured news event with an exact timestamp, enabling the construction of event windows with hourly precision.

---

## 5. Variable Construction and Initial Pipeline

### 5.1 Liquidity variables

Since the available data does not include bid-ask spreads or order book depth, liquidity is approximated using the following metrics derived from OHLCV data:

- **Hourly volume (log-transformed):** direct measure of trading activity. Primary dependent variable.
- **Price range (High − Low):** proxy for intraday volatility, equivalent to the Parkinson (1980) estimator.
- **Amihud ratio (2002):** ratio of absolute return to volume, measuring the price impact per unit of traded volume. It is the most widely cited illiquidity measure in market microstructure literature.

### 5.2 Event windows

Event windows of ±4 hours around each EIA report were constructed, resulting in a dataset of **993 records distributed across 101 events**. Each record includes the liquidity metrics, the direction of the inventory shock (bearish if inventories rose, bullish if they fell), and the hour relative to the report publication.

---

## 6. Preliminary Results

### 6.1 Descriptive analysis

Visual analysis of the event windows reveals two notable patterns:

1. The price range reaches its maximum value in the hour immediately preceding the report (hour −1), suggesting that the market anticipates the release before its official publication.
2. Trading volume is consistently higher in bearish events than in bullish events throughout the entire observation window.

### 6.2 Statistical baseline — Volume asymmetry

Three linear regression models were estimated with log-volume as the dependent variable and inventory shock direction, shock magnitude, and their interaction as predictors:

| Model | Variable | Coefficient | p-value |
|---|---|---|---|
| M1 — Direction only | is_bearish | 0.210 | 0.030 |
| M2 — Direction + Magnitude | is_bearish | 0.217 | 0.030 |
| M2 — Direction + Magnitude | shock_size | −0.014 | 0.782 |
| M3 — With interaction | is_bearish | 0.216 | 0.031 |
| M3 — With interaction | interaction | 0.029 | 0.770 |

The effect of the bearish direction on volume is statistically significant (p < 0.05) and robust to the inclusion of controls for shock magnitude and their interaction. The coefficient of 0.21 on the log scale corresponds to approximately **23% higher trading volume** in bearish events relative to bullish events in the immediate reaction window (hours 0 to +2). Shock magnitude and its interaction with direction are not significant, suggesting that the asymmetric effect is independent of the size of the inventory change.

These results constitute the **statistical baseline of the thesis** and represent the first preliminary evidence in response to RQ1. The low R² (0.024) is expected in this context and indicates that additional contextual factors — to be captured through NLP models in subsequent phases — moderate the market's response.

---

## 7. Conclusions of Month 1

The first month of work established the empirical and methodological foundations of the project. The key decisions taken are:

- The commodity of study is **WTI Crude Oil**, with hourly OHLCV data obtained from yfinance.
- The primary liquidity variable is **log-transformed hourly volume**, complemented by price range and the Amihud ratio.
- The initial source of news events is the **EIA weekly reports**, providing exact timestamps and inventory shock direction classification.
- There is preliminary statistical evidence of **asymmetry in the liquidity response** to bearish versus bullish news (p = 0.03), empirically validating RQ1 and justifying the use of AI techniques in subsequent phases.

The next phase will focus on incorporating free-text news as input for NLP models (FinBERT), completing the feature engineering, and advancing toward formal modeling of asymmetry and temporal structure.

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns: Cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *ACL Workshop on Financial Technology and NLP*.
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61–65.
- Smales, L. A. (2015). Asymmetric volatility response to news in commodity markets. *Journal of International Financial Markets, Institutions and Money*, 34, 130–149.
- Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139–1168.
