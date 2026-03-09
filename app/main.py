"""Streamlit UI for TaTaHuntsSignals MVP."""

from __future__ import annotations

import json

import streamlit as st

from app.service import run_analysis
from utils.config import get_settings

settings = get_settings()

st.set_page_config(page_title="TaTaHuntsSignals", layout="wide")
st.title("TaTaHuntsSignals")
st.caption("Explainable research assistant for single-ticker US equity/ETF analysis.")

with st.sidebar:
    st.subheader("Mode")
    st.write(f"Runtime mode: `{settings.app_mode}`")
    st.write("Demo mode works with local sample data only.")

col_a, col_b = st.columns([3, 1])
with col_a:
    ticker = st.text_input("Ticker", value=settings.default_ticker, placeholder="NVDA")
with col_b:
    run = st.button("Run Analysis", use_container_width=True)

if run:
    with st.spinner("Running LangGraph workflow..."):
        report = run_analysis(ticker)

    factor = report.get("factor_score", {})
    breakdown = factor.get("breakdown", {})

    st.subheader(f"{report.get('ticker')} | Total Score: {factor.get('total', 0):.1f}/100")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", factor.get("confidence", "n/a").upper())
    c2.metric("Bias", report.get("strategy", {}).get("bias", "n/a").upper())
    c3.metric("Price", report.get("market_summary", {}).get("latest_price", "n/a"))
    c4.metric(
        "Realized Vol",
        f"{report.get('market_summary', {}).get('realized_volatility_pct', 'n/a')}%",
    )

    st.markdown("### Factor Breakdown")
    st.bar_chart(
        {
            "Score": {
                "Momentum": breakdown.get("momentum", 0),
                "Quality": breakdown.get("quality", 0),
                "Valuation": breakdown.get("valuation", 0),
                "Risk": breakdown.get("risk", 0),
                "Sentiment": breakdown.get("sentiment", 0),
            }
        }
    )

    st.markdown("### Market Summary")
    st.json(report.get("market_summary", {}), expanded=False)

    st.markdown("### Fundamental Summary")
    st.json(report.get("fundamental_summary", {}), expanded=False)

    st.markdown("### Evidence")
    text_summary = report.get("text_evidence_summary", {})
    st.write("Bullish Catalysts")
    for item in text_summary.get("bullish_catalysts", []):
        st.write(f"- {item}")
    st.write("Bearish Risks")
    for item in text_summary.get("bearish_risks", []):
        st.write(f"- {item}")

    st.markdown("### Strategy and Risk Notes")
    st.json(report.get("strategy", {}), expanded=True)

    if report.get("warnings"):
        st.warning("Data/coverage warnings")
        for warning in report["warnings"]:
            st.write(f"- {warning}")

    with st.expander("Raw JSON Report"):
        st.code(json.dumps(report, indent=2), language="json")
