"""Backend service entrypoint used by UI and tests."""

from __future__ import annotations

from graph.workflow import build_workflow
from models.state import AnalysisState, Intent
from utils.trace_logger import log_workflow_run


def _intent_to_dict(intent: Intent | dict) -> dict:
    if isinstance(intent, Intent):
        return {
            "primary_action": intent.primary_action,
            "tickers": list(intent.tickers),
            "focus_areas": list(intent.focus_areas),
            "time_horizon": intent.time_horizon,
            "confidence": float(intent.confidence),
        }
    if isinstance(intent, dict):
        try:
            confidence = float(intent.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return {
            "primary_action": str(intent.get("primary_action", "analyze")),
            "tickers": list(intent.get("tickers", [])),
            "focus_areas": list(intent.get("focus_areas", [])),
            "time_horizon": str(intent.get("time_horizon", "medium")),
            "confidence": confidence,
        }
    return {
        "primary_action": "analyze",
        "tickers": [],
        "focus_areas": [],
        "time_horizon": "medium",
        "confidence": 0.0,
    }


def run_analysis(user_command: str) -> dict:
    """Execute workflow from a natural-language command and return report payload."""
    app = build_workflow()
    initial = AnalysisState(input_ticker="", user_command=user_command)
    result = app.invoke(initial)

    if isinstance(result, AnalysisState):
        report = dict(result.report)
        report["intent"] = _intent_to_dict(result.intent)
        report["tools_run"] = list(result.tools_to_run)
        graph_path = result.tool_results.get("graph_path", [])
        report["graph_path"] = list(graph_path) if isinstance(graph_path, list) else []
        log_workflow_run(
            {
                "user_command": user_command,
                "intent": report["intent"],
                "tools_run": report["tools_run"],
                "graph_path": report["graph_path"],
                "ticker": report.get("ticker"),
                "warnings": report.get("warnings", []),
                "total_score": report.get("factor_score", {}).get("total"),
            }
        )
        return report

    if isinstance(result, dict):
        report = result.get("report", {})
        output = dict(report) if isinstance(report, dict) else {}
        output["intent"] = _intent_to_dict(result.get("intent", {}))
        tools = result.get("tools_to_run", [])
        output["tools_run"] = list(tools) if isinstance(tools, list) else []
        tool_results = result.get("tool_results", {})
        graph_path = []
        if isinstance(tool_results, dict):
            path = tool_results.get("graph_path", [])
            graph_path = list(path) if isinstance(path, list) else []
        output["graph_path"] = graph_path
        log_workflow_run(
            {
                "user_command": user_command,
                "intent": output["intent"],
                "tools_run": output["tools_run"],
                "graph_path": output["graph_path"],
                "ticker": output.get("ticker"),
                "warnings": output.get("warnings", []),
                "total_score": output.get("factor_score", {}).get("total"),
            }
        )
        return output

    log_workflow_run(
        {
            "user_command": user_command,
            "intent": {},
            "tools_run": [],
            "graph_path": [],
            "ticker": None,
            "warnings": ["Workflow returned unsupported result type."],
            "total_score": None,
        }
    )
    return {}
