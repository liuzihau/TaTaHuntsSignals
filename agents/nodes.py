"""LangGraph node functions for the quant research workflow."""

from __future__ import annotations

import logging
import re
from typing import Any

from data_providers.mock_provider import MockProvider
from models.state import AnalysisState, Intent
from rag.retriever import LocalKeywordRetriever
from scoring.engine import FactorScorer, confidence_from_state
from utils.llm import LLMClient

_PROVIDER = MockProvider()
_RETRIEVER = LocalKeywordRetriever()
_SCORER = FactorScorer()
LOGGER = logging.getLogger(__name__)

ANALYZE_TOOL_CHAIN = [
    "structured_data_collector",
    "fundamentals_collector",
    "text_retrieval_node",
    "factor_engine",
    "score_composer",
    "report_generator",
    "guardrail_critic",
]

try:
    _LLM = LLMClient()
except Exception as exc:  # pragma: no cover - depends on local env/package setup
    _LLM = None
    LOGGER.warning("LLM client unavailable at import time: %s", exc)


def _record_node(state: AnalysisState, node_name: str) -> None:
    path = state.tool_results.setdefault("graph_path", [])
    if isinstance(path, list):
        path.append(node_name)
    else:
        state.tool_results["graph_path"] = [node_name]


def _looks_like_ticker(value: str) -> bool:
    value = value.strip()
    return bool(value) and len(value) <= 5 and value.isalpha() and " " not in value


def _extract_tickers(text: str) -> list[str]:
    tokens = re.findall(r"\$?([A-Z]{1,5})\b", text)
    stopwords = {
        "A",
        "AN",
        "AND",
        "BUY",
        "FOR",
        "I",
        "IS",
        "OF",
        "ON",
        "OR",
        "SELL",
        "TERM",
        "THE",
        "TO",
        "WITH",
    }
    tickers = [token for token in tokens if token not in stopwords]
    return list(dict.fromkeys(tickers))


def _normalize_time_horizon(value: str) -> str:
    value = (value or "").strip().lower()
    if value in {"short_term", "short", "near_term", "this_quarter"}:
        return "short_term"
    if value in {"long_term", "long", "multi_year", "next_year"}:
        return "long_term"
    return "medium"


def _default_intent(command: str, fallback_ticker: str = "", confidence: float = 0.55) -> Intent:
    command_l = command.lower()
    action = "analyze"
    if "compare" in command_l:
        action = "compare"
    elif "screen" in command_l or "find stocks" in command_l:
        action = "screen"
    elif "backtest" in command_l:
        action = "backtest"

    tickers = _extract_tickers(command)
    if not tickers and fallback_ticker:
        tickers = [fallback_ticker]
    return Intent(
        primary_action=action,
        tickers=tickers,
        focus_areas=["fundamentals", "technicals", "sentiment"],
        time_horizon="medium",
        constraints={},
        confidence=confidence,
    )


def _intent_from_payload(payload: dict[str, Any], command: str, fallback_ticker: str = "") -> Intent:
    action = str(payload.get("primary_action", "analyze")).strip().lower()
    if action not in {"analyze", "compare", "screen", "backtest"}:
        action = "analyze"

    raw_tickers = payload.get("tickers", [])
    tickers: list[str] = []
    if isinstance(raw_tickers, list):
        for value in raw_tickers:
            token = str(value).strip().upper()
            if _looks_like_ticker(token):
                tickers.append(token)

    if not tickers:
        tickers = _extract_tickers(command)
    if not tickers and fallback_ticker:
        tickers = [fallback_ticker]

    raw_focus = payload.get("focus_areas", [])
    focus_areas: list[str] = []
    if isinstance(raw_focus, list):
        focus_areas = [str(value).strip().lower() for value in raw_focus if str(value).strip()]
    if not focus_areas:
        focus_areas = ["fundamentals", "technicals", "sentiment"]

    raw_constraints = payload.get("constraints", {})
    constraints = raw_constraints if isinstance(raw_constraints, dict) else {}

    raw_conf = payload.get("confidence", 0.6)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))

    return Intent(
        primary_action=action,
        tickers=list(dict.fromkeys(tickers)),
        focus_areas=focus_areas,
        time_horizon=_normalize_time_horizon(str(payload.get("time_horizon", "medium"))),
        constraints=constraints,
        confidence=confidence,
    )


def query_normalizer(state: AnalysisState) -> AnalysisState:
    _record_node(state, "query_normalizer")
    seed = (state.input_ticker or "").upper().strip()
    cmd = (state.user_command or "").strip()
    if not seed and _looks_like_ticker(cmd):
        seed = cmd.upper()

    if seed:
        state.normalized_ticker = seed
    elif not cmd:
        state.normalized_ticker = "NVDA"
        state.warnings.append("Empty command provided, defaulted to NVDA.")
    else:
        state.normalized_ticker = ""

    if state.normalized_ticker:
        state.asset_type = "etf" if state.normalized_ticker.endswith("Y") and len(state.normalized_ticker) <= 4 else "equity"
    else:
        state.asset_type = "unknown"
    return state


def intent_parser(state: AnalysisState) -> AnalysisState:
    """Parse user command into a structured intent."""
    _record_node(state, "intent_parser")
    command = (state.user_command or state.input_ticker or "").strip()
    fallback_ticker = state.normalized_ticker

    if _looks_like_ticker(command):
        ticker = command.upper()
        state.intent = Intent(
            primary_action="analyze",
            tickers=[ticker],
            focus_areas=["fundamentals", "technicals", "sentiment"],
            time_horizon="medium",
            constraints={},
            confidence=0.95,
        )
        return state

    if not command:
        state.intent = _default_intent(command="", fallback_ticker=fallback_ticker or "NVDA", confidence=0.4)
        return state

    prompt = (
        "You are a quantitative research assistant. Parse this user command into a structured intent.\n\n"
        f'User command: "{command}"\n\n'
        "Return JSON:\n"
        "{\n"
        '    "primary_action": "analyze" | "compare" | "screen" | "backtest",\n'
        '    "tickers": ["TICKER1", ...],\n'
        '    "focus_areas": ["fundamentals", "technicals", "sentiment", ...],\n'
        '    "time_horizon": "short_term" | "medium" | "long_term",\n'
        '    "constraints": {},\n'
        '    "confidence": 0.0-1.0\n'
        "}\n\n"
        "Rules:\n"
        "- If no tickers, return empty list\n"
        "- Infer action from verb (compare -> compare, find stocks -> screen)\n"
        "- Infer time_horizon from language (this quarter -> short_term, next year -> long_term)\n"
        "- List all mentioned domains in focus_areas\n"
        "- confidence: 0.8-1.0 if clear, 0.5-0.7 if somewhat ambiguous, 0.2-0.4 if unclear\n\n"
        "Return ONLY valid JSON, no markdown."
    )
    try:
        if _LLM is None:
            raise ValueError("LLM client is unavailable.")
        payload = _LLM.invoke_json(prompt)
        state.intent = _intent_from_payload(payload, command=command, fallback_ticker=fallback_ticker)
    except Exception as exc:
        warning = f"Intent parsing via LLM failed, using defaults: {exc}"
        LOGGER.warning(warning)
        state.warnings.append(warning)
        state.intent = _default_intent(command=command, fallback_ticker=fallback_ticker, confidence=0.5)

    return state


def tool_selector(state: AnalysisState) -> AnalysisState:
    """Select which tools/nodes should run for the parsed intent."""
    _record_node(state, "tool_selector")
    action = (state.intent.primary_action or "analyze").lower()
    if action != "analyze":
        warning = f"Unsupported action '{action}' in Phase 1. Defaulting to analyze."
        LOGGER.warning(warning)
        state.warnings.append(warning)

    state.tools_to_run = list(ANALYZE_TOOL_CHAIN)
    state.tool_results["selected_tools"] = list(state.tools_to_run)
    if not state.intent.tickers:
        if state.normalized_ticker:
            state.intent.tickers = [state.normalized_ticker]
        else:
            state.intent.tickers = ["NVDA"]
            state.warnings.append("No ticker inferred from command, defaulted to NVDA.")

    state.normalized_ticker = state.intent.tickers[0].upper()
    state.asset_type = "etf" if state.normalized_ticker.endswith("Y") and len(state.normalized_ticker) <= 4 else "equity"
    state.text_query = (
        f"{state.normalized_ticker} catalysts risks guidance margin growth valuation"
    )
    return state


def structured_data_collector(state: AnalysisState) -> AnalysisState:
    _record_node(state, "structured_data_collector")
    snapshot = _PROVIDER.get_market_snapshot(state.normalized_ticker)
    state.market_data = snapshot
    state.technicals = snapshot.get("technical", {})
    return state


def fundamentals_collector(state: AnalysisState) -> AnalysisState:
    _record_node(state, "fundamentals_collector")
    state.fundamental_data = _PROVIDER.get_fundamentals(state.normalized_ticker)
    return state


def text_retrieval_node(state: AnalysisState) -> AnalysisState:
    _record_node(state, "text_retrieval_node")
    docs = _PROVIDER.get_documents(state.normalized_ticker)
    evidence = _RETRIEVER.retrieve(state.text_query, docs, k=6)
    state.evidence = evidence

    bullish: list[str] = []
    bearish: list[str] = []
    for snippet in evidence:
        text_l = snippet.snippet.lower()
        source_ref = f"[{snippet.source}]"
        if any(word in text_l for word in ["growth", "expansion", "beat", "demand", "tailwind"]):
            bullish.append(f"{source_ref} {snippet.snippet[:140]}...")
        if any(word in text_l for word in ["risk", "headwind", "decline", "pressure", "lawsuit"]):
            bearish.append(f"{source_ref} {snippet.snippet[:140]}...")

    state.text_summary = {
        "document_count": len(docs),
        "evidence_count": len(evidence),
        "bullish_catalysts": bullish[:3],
        "bearish_risks": bearish[:3],
    }
    if not evidence:
        state.warnings.append("No relevant text evidence retrieved.")
    return state


def factor_engine(state: AnalysisState) -> AnalysisState:
    _record_node(state, "factor_engine")
    factors, explanations, warnings = _SCORER.compute(
        market=state.market_data,
        fundamentals=state.fundamental_data,
        text_summary=state.text_summary,
    )
    state.factor_scores = factors
    state.score_explanations = explanations
    state.warnings.extend(warnings)
    return state


def score_composer(state: AnalysisState) -> AnalysisState:
    _record_node(state, "score_composer")
    state.total_score = state.factor_scores.total
    state.confidence = confidence_from_state(len(state.warnings))
    return state


def report_generator(state: AnalysisState) -> AnalysisState:
    _record_node(state, "report_generator")
    total = state.total_score
    bias = "bullish" if total >= 70 else "neutral" if total >= 45 else "bearish"

    state.strategy = {
        "bias": bias,
        "bull_case": "Trend continuation plus estimate stability and positive catalyst flow.",
        "base_case": "Mixed factor regime with selective upside and elevated two-way risk.",
        "bear_case": "Momentum fades while risk metrics and negative catalysts worsen.",
        "watchpoints": [
            "Next earnings or macro-sensitive update",
            "Volume confirmation and trend persistence",
            "Valuation compression/expansion versus growth",
        ],
        "style_fit": "Momentum swing + medium-term fundamental watch",
        "risk_notes": [
            "Research output only; not execution advice.",
            "Scenario ranges are qualitative, not return guarantees.",
        ],
    }

    state.report = {
        "ticker": state.normalized_ticker,
        "asset_type": state.asset_type,
        "generated_at": state.started_at,
        "market_summary": {
            "latest_price": state.market_data.get("latest_price"),
            "returns": state.market_data.get("returns", {}),
            "realized_volatility_pct": state.market_data.get("realized_volatility_pct"),
            "volume_trend": state.market_data.get("volume_trend"),
            "technical_summary": state.market_data.get("technical", {}),
        },
        "fundamental_summary": state.fundamental_data,
        "text_evidence_summary": state.text_summary,
        "factor_score": {
            "total": state.total_score,
            "breakdown": {
                "momentum": state.factor_scores.momentum,
                "quality": state.factor_scores.quality,
                "valuation": state.factor_scores.valuation,
                "risk": state.factor_scores.risk,
                "sentiment": state.factor_scores.sentiment,
            },
            "explanations": state.score_explanations,
            "confidence": state.confidence,
        },
        "strategy": state.strategy,
        "warnings": list(dict.fromkeys(state.warnings)),
    }
    return state


def guardrail_critic(state: AnalysisState) -> AnalysisState:
    """Basic post-check to reduce over-claim language."""
    _record_node(state, "guardrail_critic")
    risky_words = ["guaranteed", "certain", "risk-free", "sure win"]

    def sanitize(text: str) -> str:
        output = text
        for word in risky_words:
            output = output.replace(word, "uncertain")
        return output

    strategy = state.report.get("strategy", {})
    for key, value in list(strategy.items()):
        if isinstance(value, str):
            strategy[key] = sanitize(value)
        elif isinstance(value, list):
            strategy[key] = [sanitize(v) if isinstance(v, str) else v for v in value]

    state.report["strategy"] = strategy
    return state


def state_to_dict(state: AnalysisState) -> dict[str, Any]:
    return state.to_dict()
