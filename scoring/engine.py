"""Deterministic, explainable factor scoring engine (0-100 total)."""

from __future__ import annotations

from models.state import FactorBreakdown


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


class FactorScorer:
    """Computes rule-based factors from structured data + text summary."""

    def compute(
        self,
        market: dict,
        fundamentals: dict,
        text_summary: dict,
    ) -> tuple[FactorBreakdown, dict[str, str], list[str]]:
        warnings: list[str] = []
        explanations: dict[str, str] = {}

        momentum = self._momentum_score(market, explanations, warnings)
        quality = self._quality_score(fundamentals, explanations, warnings)
        valuation = self._valuation_score(fundamentals, explanations, warnings)
        risk = self._risk_score(market, fundamentals, explanations, warnings)
        sentiment = self._sentiment_score(text_summary, explanations, warnings)

        factors = FactorBreakdown(
            momentum=momentum,
            quality=quality,
            valuation=valuation,
            risk=risk,
            sentiment=sentiment,
        )
        return factors, explanations, warnings

    def _momentum_score(self, market: dict, explanations: dict[str, str], warnings: list[str]) -> float:
        if not market:
            warnings.append("Missing market data: momentum score reduced.")
            explanations["momentum"] = "No market data available."
            return 5.0

        r1m = market.get("returns", {}).get("1m", 0)
        r3m = market.get("returns", {}).get("3m", 0)
        r6m = market.get("returns", {}).get("6m", 0)
        ma_signal = market.get("technical", {}).get("ma_alignment", "neutral")
        rsi = market.get("technical", {}).get("rsi_14", 50)

        raw = 10.0 + (r1m * 0.20) + (r3m * 0.18) + (r6m * 0.12)
        if ma_signal == "bullish":
            raw += 3
        elif ma_signal == "bearish":
            raw -= 3

        if 45 <= rsi <= 65:
            raw += 2
        elif rsi > 75 or rsi < 30:
            raw -= 2

        score = round(clamp(raw, 0, 20), 2)
        explanations["momentum"] = (
            f"Returns (1M/3M/6M) = {r1m}/{r3m}/{r6m}, MA alignment={ma_signal}, RSI={rsi}."
        )
        return score

    def _quality_score(self, fundamentals: dict, explanations: dict[str, str], warnings: list[str]) -> float:
        if not fundamentals:
            warnings.append("Missing fundamentals: quality score reduced.")
            explanations["quality"] = "No fundamental data available."
            return 6.0

        rev_growth = fundamentals.get("revenue_growth_pct", 0)
        op_margin = fundamentals.get("operating_margin_pct", 0)
        roe = fundamentals.get("roe_pct", 0)
        debt_to_equity = fundamentals.get("debt_to_equity", 1.2)

        raw = 8.0 + (rev_growth * 0.15) + (op_margin * 0.15) + (roe * 0.08)
        raw += 2 if debt_to_equity < 0.8 else -2 if debt_to_equity > 2 else 0

        score = round(clamp(raw, 0, 20), 2)
        explanations["quality"] = (
            f"Revenue growth={rev_growth}%, operating margin={op_margin}%, ROE={roe}%, debt/equity={debt_to_equity}."
        )
        return score

    def _valuation_score(self, fundamentals: dict, explanations: dict[str, str], warnings: list[str]) -> float:
        pe = fundamentals.get("pe_ratio")
        ev_ebitda = fundamentals.get("ev_ebitda")
        pb = fundamentals.get("pb_ratio")

        if pe is None and ev_ebitda is None and pb is None:
            warnings.append("Valuation metrics unavailable: valuation score reduced.")
            explanations["valuation"] = "No valuation metrics available."
            return 8.0

        score = 10.0
        if pe is not None:
            score += 3 if pe < 18 else -3 if pe > 35 else 0
        if ev_ebitda is not None:
            score += 3 if ev_ebitda < 14 else -2 if ev_ebitda > 28 else 0
        if pb is not None:
            score += 2 if pb < 4 else -2 if pb > 10 else 0

        final = round(clamp(score, 0, 20), 2)
        explanations["valuation"] = f"P/E={pe}, EV/EBITDA={ev_ebitda}, P/B={pb}."
        return final

    def _risk_score(self, market: dict, fundamentals: dict, explanations: dict[str, str], warnings: list[str]) -> float:
        vol = market.get("realized_volatility_pct", 35)
        max_dd = market.get("max_drawdown_pct", 25)
        debt_to_equity = fundamentals.get("debt_to_equity", 1.5)
        event_risk = fundamentals.get("event_risk", "medium")

        score = 14.0
        score += 2 if vol < 22 else -3 if vol > 45 else 0
        score += 2 if max_dd < 15 else -2 if max_dd > 30 else 0
        score += 1 if debt_to_equity < 1 else -2 if debt_to_equity > 2 else 0
        score += 1 if event_risk == "low" else -2 if event_risk == "high" else 0

        final = round(clamp(score, 0, 20), 2)
        explanations["risk"] = (
            f"Realized vol={vol}%, max drawdown={max_dd}%, debt/equity={debt_to_equity}, event risk={event_risk}."
        )
        return final

    def _sentiment_score(
        self,
        text_summary: dict,
        explanations: dict[str, str],
        warnings: list[str],
    ) -> float:
        if not text_summary:
            warnings.append("No text evidence summary: sentiment score reduced.")
            explanations["sentiment"] = "No news/filing summary available."
            return 7.0

        bullish = len(text_summary.get("bullish_catalysts", []))
        bearish = len(text_summary.get("bearish_risks", []))

        score = 10 + (bullish * 2.5) - (bearish * 2.0)
        final = round(clamp(score, 0, 20), 2)
        explanations["sentiment"] = f"Bullish catalysts={bullish}, bearish risks={bearish}."
        return final


def confidence_from_state(warnings_count: int) -> str:
    if warnings_count <= 1:
        return "high"
    if warnings_count <= 3:
        return "medium"
    return "low"
