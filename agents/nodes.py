"""LangGraph node functions for the quant research workflow."""

from __future__ import annotations

from typing import Any

from data_providers.mock_provider import MockProvider
from models.state import AnalysisState
from rag.retriever import LocalKeywordRetriever
from scoring.engine import FactorScorer, confidence_from_state

_PROVIDER = MockProvider()
_RETRIEVER = LocalKeywordRetriever()
_SCORER = FactorScorer()


def query_normalizer(state: AnalysisState) -> AnalysisState:
    ticker = (state.input_ticker or "").upper().strip()
    state.normalized_ticker = ticker
    state.asset_type = "etf" if ticker.endswith("Y") and len(ticker) <= 4 else "equity"
    if not ticker:
        state.normalized_ticker = "NVDA"
        state.warnings.append("Empty ticker provided, defaulted to NVDA.")
    return state


def data_router(state: AnalysisState) -> AnalysisState:
    state.text_query = (
        f"{state.normalized_ticker} catalysts risks guidance margin growth valuation"
    )
    return state


def structured_data_collector(state: AnalysisState) -> AnalysisState:
    snapshot = _PROVIDER.get_market_snapshot(state.normalized_ticker)
    state.market_data = snapshot
    state.technicals = snapshot.get("technical", {})
    return state


def fundamentals_collector(state: AnalysisState) -> AnalysisState:
    state.fundamental_data = _PROVIDER.get_fundamentals(state.normalized_ticker)
    return state


def text_retrieval_node(state: AnalysisState) -> AnalysisState:
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
    state.total_score = state.factor_scores.total
    state.confidence = confidence_from_state(len(state.warnings))
    return state


def report_generator(state: AnalysisState) -> AnalysisState:
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
