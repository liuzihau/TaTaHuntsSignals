from scoring.engine import FactorScorer


def test_scoring_bounds() -> None:
    scorer = FactorScorer()
    factors, _, _ = scorer.compute(
        market={"returns": {"1m": 6, "3m": 12, "6m": 22}, "technical": {"ma_alignment": "bullish", "rsi_14": 58}, "realized_volatility_pct": 25, "max_drawdown_pct": 18},
        fundamentals={"revenue_growth_pct": 18, "operating_margin_pct": 24, "roe_pct": 30, "debt_to_equity": 0.6, "pe_ratio": 30, "ev_ebitda": 19, "pb_ratio": 12, "event_risk": "medium"},
        text_summary={"bullish_catalysts": ["a", "b"], "bearish_risks": ["x"]},
    )

    assert 0 <= factors.total <= 100
    assert 0 <= factors.momentum <= 20
    assert 0 <= factors.quality <= 20
    assert 0 <= factors.valuation <= 20
    assert 0 <= factors.risk <= 20
    assert 0 <= factors.sentiment <= 20
