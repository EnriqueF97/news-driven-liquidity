import anthropic

SYSTEM_PROMPT = """\
You are a financial analyst specializing in WTI crude oil markets. Your task is to \
analyze news articles and extract structured features for use in a quantitative \
liquidity impact model.

First, determine whether the article contains substantive WTI-relevant content \
(usable=true). Set usable=false — and return only that field — for: paywall \
placeholders, Cloudflare blocks, cookie notices, or articles that are clearly not \
about oil markets, macro, or geopolitics relevant to oil.

When usable=true, extract the remaining fields:
- sentiment_score: NET directional impact on WTI price. -1.0 = strongly bearish, \
+1.0 = strongly bullish. Assess the actual content, not the headline tone.
- supply_impact: Direction of impact on WTI supply. Score the article's implication \
for the quantity of oil available to the market.
  +1.0 = strong supply increase (OPEC adds barrels, sanctions lifted on a major \
producer, new pipeline online, Venezuela ramping up).
  +0.5 = moderate supply increase (single-country production gains, inventory builds).
  0.0 = no supply implication.
  -0.5 = moderate supply decrease (refinery outages, minor OPEC cuts).
  -1.0 = strong supply decrease (major OPEC cuts, infrastructure attacks, embargoes, \
refinery fires affecting multiple facilities).
- demand_impact: Direction of impact on WTI demand. Score the article's implication \
for oil consumption.
  +1.0 = strong demand increase (China stimulus, Fed pivot to easing, growth surprise, \
summer driving acceleration).
  +0.5 = moderate demand increase (regional growth indicators).
  0.0 = no demand implication.
  -0.5 = moderate demand decrease (mild slowdown signals).
  -1.0 = strong demand decrease (recession confirmed, Fed strongly hawkish, major \
economy lockdowns, demand destruction events).
- risk_premium: Change in geopolitical or operational risk premium for oil. Score the \
direction of risk change, not the absolute level.
  +1.0 = major escalation (military strike on oil infrastructure, war, Strait of \
Hormuz incident, major sanctions imposed).
  +0.5 = moderate escalation (rising tensions, threats, troop movements).
  0.0 = no change in risk environment.
  -0.5 = moderate de-escalation (talks resuming, partial sanctions lifted).
  -1.0 = major de-escalation (ceasefire signed, diplomatic breakthrough, full \
sanctions removal).
- magnitude: how market-moving is this event. 0.0 = negligible, 1.0 = historic. \
Most articles score 0.1–0.4. Major OPEC cut = 0.9. Geopolitical crisis = 1.0.
- event_type: 1–3 categories ordered by salience from: geopolitical, supply, \
demand, macro, technical, other. EIA inventory articles classify as supply or \
macro — do not use “inventory”.
- entities: specific named actors central to the article (not incidentally mentioned). \
Include countries, organizations, oil companies, and key officials. Use the names \
exactly as they appear in the article.
- certainty: how confirmed is the information. 0.0 = rumor, 0.5 = analyst \
forecast, 0.9 = official announcement.
- time_horizon: immediate (hours to 1 day), short_term (days to weeks), or \
structural (months-plus or permanent themes such as energy transition).

The three impact fields (supply_impact, demand_impact, risk_premium) are intentionally \
orthogonal. An article can affect multiple channels independently. Score each based on \
what the article actually says about that specific channel. If the article has no clear \
implication for a channel, score it 0.0.
"""

EXTRACTION_TOOL = {
    "name": "extract_article_features",
    "description": (
        "Extract structured WTI-relevant features from a news article for use "
        "in a quantitative liquidity impact model."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {
                "type": "boolean",
                "description": (
                    "True if the article contains substantive WTI-relevant content. "
                    "False for paywalls, Cloudflare blocks, cookie notices, or articles "
                    "that are clearly not about oil markets, macro, or geopolitics "
                    "relevant to oil."
                ),
            },
            "sentiment_score": {
                "type": "number",
                "description": (
                    "Net directional impact on WTI price. -1.0 = strongly bearish, "
                    "0.0 = neutral, +1.0 = strongly bullish. Omit when usable=false."
                ),
            },
            "supply_impact": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Direction of impact on WTI supply. +1=strong supply increase, -1=strong supply decrease, 0=no implication. See system prompt for anchors. Omit when usable=false.",
            },
            "demand_impact": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Direction of impact on WTI demand. +1=strong demand increase, -1=strong demand decrease, 0=no implication. See system prompt for anchors. Omit when usable=false.",
            },
            "risk_premium": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0,
                "description": "Change in geopolitical/operational risk premium for oil. +1=major escalation, -1=major de-escalation, 0=no change. See system prompt for anchors. Omit when usable=false.",
            },
            "magnitude": {
                "type": "number",
                "description": (
                    "Event importance for WTI markets. 0.0 = negligible, 1.0 = market-moving. "
                    "Routine update = 0.1, major OPEC cut = 0.9, geopolitical crisis = 1.0. "
                    "Omit when usable=false."
                ),
            },
            "event_type": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "geopolitical",
                        "supply",
                        "demand",
                        "macro",
                        "technical",
                        "other",
                    ],
                },
                "minItems": 1,
                "maxItems": 3,
                "description": (
                    "1–3 categories ordered by salience. geopolitical = sanctions/conflict/diplomacy, "
                    "supply = OPEC/production/pipelines, demand = consumption/imports, "
                    "macro = Fed/dollar/inflation, technical = price levels/chart patterns, "
                    "other = company earnings/unclassified. Omit when usable=false."
                ),
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific named actors central to the article — countries, organizations, "
                    "oil companies, key officials. Use the names exactly as they appear in the "
                    "article. Omit when usable=false."
                ),
            },
            "certainty": {
                "type": "number",
                "description": (
                    "Confidence in reported information. 0.0 = rumor/speculation, "
                    "0.5 = analyst forecast, 0.9 = confirmed fact. Omit when usable=false."
                ),
            },
            "time_horizon": {
                "type": "string",
                "enum": ["immediate", "short_term", "structural"],
                "description": (
                    "Temporal relevance. immediate = hours to 1 day, short_term = days to weeks, "
                    "structural = months-plus or permanent themes (energy transition, "
                    "OPEC long-run policy). Omit when usable=false."
                ),
            },
        },
        "required": ["usable"],
    },
}

_OPTIONAL_FIELDS = [
    "sentiment_score",
    "supply_impact",
    "demand_impact",
    "risk_premium",
    "magnitude",
    "event_type",
    "entities",
    "certainty",
    "time_horizon",
]


def _user_text(title: str, body: str | None) -> str:
    title = str(title).encode("utf-8", errors="ignore").decode("utf-8").strip()
    body_text = ""
    if isinstance(body, str):
        body_text = body.encode("utf-8", errors="ignore").decode("utf-8")[:1500].strip()
    return f"Title: {title}\n\nBody: {body_text}" if body_text else f"Title: {title}"


def _parse_tool_input(tool_input: dict) -> dict:
    result = {"usable": tool_input["usable"]}
    if result["usable"]:
        for field in _OPTIONAL_FIELDS:
            result[field] = tool_input.get(field, None)
    else:
        for field in _OPTIONAL_FIELDS:
            result[field] = None
    return result


def build_batch_request(article_id: int, title: str, body: str | None) -> dict:
    """Return one Batches API request dict for the given article."""
    return {
        "custom_id": str(article_id),
        "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 500,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": [EXTRACTION_TOOL],
            "tool_choice": {"type": "tool", "name": "extract_article_features"},
            "messages": [{"role": "user", "content": _user_text(title, body)}],
        },
    }


def parse_batch_result(result) -> dict | None:
    """
    Parse one MessageBatchResult from client.messages.batches.results().
    Returns the feature dict (keyed by article_id) or None if the result errored.
    """
    if result.result.type != "succeeded":
        return None
    tool_input = None
    for block in result.result.message.content:
        if block.type == "tool_use":
            tool_input = block.input
            break
    if tool_input is None:
        return None
    features = _parse_tool_input(tool_input)
    features["article_id"] = int(result.custom_id)
    return features


def extract_features(title: str, body: str | None, client: anthropic.Anthropic) -> dict:
    """Single-article synchronous extraction. Used for calibration runs."""
    response = client.messages.create(
        **build_batch_request(0, title, body)["params"]
    )

    tool_input = None
    for block in response.content:
        if block.type == "tool_use":
            tool_input = block.input
            break
    if tool_input is None:
        raise ValueError(f"No tool_use block in response for title: {title[:50]}")

    return _parse_tool_input(tool_input)
