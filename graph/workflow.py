"""Workflow composition for the MVP research pipeline.

Uses LangGraph when installed; falls back to a deterministic local runner.
"""

from __future__ import annotations

from agents import nodes
from models.state import AnalysisState


class _LocalCompiledWorkflow:
    """Minimal fallback interface compatible with `.invoke(state)`."""

    def __init__(self) -> None:
        self._chain = [
            nodes.query_normalizer,
            nodes.data_router,
            nodes.structured_data_collector,
            nodes.fundamentals_collector,
            nodes.text_retrieval_node,
            nodes.factor_engine,
            nodes.score_composer,
            nodes.report_generator,
            nodes.guardrail_critic,
        ]

    def invoke(self, state: AnalysisState) -> AnalysisState:
        current = state
        for fn in self._chain:
            current = fn(current)
        return current


def build_workflow():
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return _LocalCompiledWorkflow()

    graph = StateGraph(AnalysisState)

    graph.add_node("query_normalizer", nodes.query_normalizer)
    graph.add_node("data_router", nodes.data_router)
    graph.add_node("structured_data_collector", nodes.structured_data_collector)
    graph.add_node("fundamentals_collector", nodes.fundamentals_collector)
    graph.add_node("text_retrieval_node", nodes.text_retrieval_node)
    graph.add_node("factor_engine", nodes.factor_engine)
    graph.add_node("score_composer", nodes.score_composer)
    graph.add_node("report_generator", nodes.report_generator)
    graph.add_node("guardrail_critic", nodes.guardrail_critic)

    graph.set_entry_point("query_normalizer")
    graph.add_edge("query_normalizer", "data_router")
    graph.add_edge("data_router", "structured_data_collector")
    graph.add_edge("structured_data_collector", "fundamentals_collector")
    graph.add_edge("fundamentals_collector", "text_retrieval_node")
    graph.add_edge("text_retrieval_node", "factor_engine")
    graph.add_edge("factor_engine", "score_composer")
    graph.add_edge("score_composer", "report_generator")
    graph.add_edge("report_generator", "guardrail_critic")
    graph.add_edge("guardrail_critic", END)

    return graph.compile()
