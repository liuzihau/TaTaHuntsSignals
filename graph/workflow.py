"""Workflow composition for the MVP research pipeline.

Uses LangGraph when installed; falls back to a deterministic local runner.
"""

from __future__ import annotations

from agents import nodes
from models.state import AnalysisState


class _LocalCompiledWorkflowV2:
    """Minimal fallback interface compatible with `.invoke(state)`."""

    def __init__(self) -> None:
        self._prefix_chain = [
            nodes.query_normalizer,
            nodes.intent_parser,
            nodes.tool_selector,
        ]
        self._tool_nodes = {
            "structured_data_collector": nodes.structured_data_collector,
            "fundamentals_collector": nodes.fundamentals_collector,
            "text_retrieval_node": nodes.text_retrieval_node,
            "factor_engine": nodes.factor_engine,
            "score_composer": nodes.score_composer,
            "report_generator": nodes.report_generator,
            "guardrail_critic": nodes.guardrail_critic,
        }

    def invoke(self, state: AnalysisState) -> AnalysisState:
        current = state
        for fn in self._prefix_chain:
            current = fn(current)

        for name in current.tools_to_run:
            fn = self._tool_nodes.get(name)
            if fn is None:
                current.warnings.append(f"Unknown tool '{name}' requested; skipped.")
                continue
            current = fn(current)
        return current


def build_workflow():
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return _LocalCompiledWorkflowV2()

    graph = StateGraph(AnalysisState)

    graph.add_node("query_normalizer", nodes.query_normalizer)
    graph.add_node("intent_parser", nodes.intent_parser)
    graph.add_node("tool_selector", nodes.tool_selector)
    graph.add_node("structured_data_collector", nodes.structured_data_collector)
    graph.add_node("fundamentals_collector", nodes.fundamentals_collector)
    graph.add_node("text_retrieval_node", nodes.text_retrieval_node)
    graph.add_node("factor_engine", nodes.factor_engine)
    graph.add_node("score_composer", nodes.score_composer)
    graph.add_node("report_generator", nodes.report_generator)
    graph.add_node("guardrail_critic", nodes.guardrail_critic)

    def route_from_tool_selector(state: AnalysisState) -> str:
        if not state.tools_to_run:
            return "end"
        first_tool = state.tools_to_run[0]
        if first_tool in nodes.ANALYZE_TOOL_CHAIN:
            return first_tool
        return "end"

    graph.set_entry_point("query_normalizer")
    graph.add_edge("query_normalizer", "intent_parser")
    graph.add_edge("intent_parser", "tool_selector")
    graph.add_conditional_edges(
        "tool_selector",
        route_from_tool_selector,
        {
            "structured_data_collector": "structured_data_collector",
            "fundamentals_collector": "fundamentals_collector",
            "text_retrieval_node": "text_retrieval_node",
            "factor_engine": "factor_engine",
            "score_composer": "score_composer",
            "report_generator": "report_generator",
            "guardrail_critic": "guardrail_critic",
            "end": END,
        },
    )
    graph.add_edge("structured_data_collector", "fundamentals_collector")
    graph.add_edge("fundamentals_collector", "text_retrieval_node")
    graph.add_edge("text_retrieval_node", "factor_engine")
    graph.add_edge("factor_engine", "score_composer")
    graph.add_edge("score_composer", "report_generator")
    graph.add_edge("report_generator", "guardrail_critic")
    graph.add_edge("guardrail_critic", END)

    return graph.compile()
