"""Backend service entrypoint used by UI and tests."""

from __future__ import annotations

from graph.workflow import build_workflow
from models.state import AnalysisState


def run_analysis(ticker: str) -> dict:
    """Execute LangGraph workflow and return final structured report."""
    app = build_workflow()
    initial = AnalysisState(input_ticker=ticker)
    final_state = app.invoke(initial)
    return final_state.report
