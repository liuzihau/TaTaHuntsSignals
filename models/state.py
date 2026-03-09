"""Core state and schema models for the agentic quant copilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FactorBreakdown:
    """Deterministic factor scores. Each field is 0-20."""

    momentum: float = 0.0
    quality: float = 0.0
    valuation: float = 0.0
    risk: float = 0.0
    sentiment: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.momentum + self.quality + self.valuation + self.risk + self.sentiment,
            2,
        )


@dataclass
class EvidenceSnippet:
    """One retrieved text snippet used in report generation."""

    source: str
    snippet: str
    score: float


@dataclass
class Intent:
    """Parsed user intent for deciding workflow actions."""

    primary_action: str = "analyze"  # "analyze" | "compare" | "screen" | "backtest"
    tickers: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=lambda: ["fundamentals", "technicals", "sentiment"])
    time_horizon: str = "medium"  # "short_term" | "medium" | "long_term"
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.7  # 0.0-1.0


@dataclass
class AnalysisState:
    """Shared graph state used by all nodes."""

    input_ticker: str = ""
    user_command: str = ""
    intent: Intent = field(default_factory=Intent)
    tools_to_run: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)

    normalized_ticker: str = ""
    asset_type: str = "unknown"
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    market_data: dict[str, Any] = field(default_factory=dict)
    fundamental_data: dict[str, Any] = field(default_factory=dict)
    technicals: dict[str, Any] = field(default_factory=dict)

    text_query: str = ""
    evidence: list[EvidenceSnippet] = field(default_factory=list)
    text_summary: dict[str, Any] = field(default_factory=dict)

    factor_scores: FactorBreakdown = field(default_factory=FactorBreakdown)
    score_explanations: dict[str, str] = field(default_factory=dict)

    total_score: float = 0.0
    confidence: str = "low"
    warnings: list[str] = field(default_factory=list)

    strategy: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable form for graph outputs/UI."""
        data = asdict(self)
        data["total_score"] = round(float(data.get("total_score", 0.0)), 2)
        return data
