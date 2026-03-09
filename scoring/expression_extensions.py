"""Extension points for a future mini quant expression engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


Operator = Callable[..., float]


@dataclass
class ExpressionRegistry:
    """Registry placeholders for future time-series/cross-sectional operators."""

    time_series_ops: dict[str, Operator] = field(default_factory=dict)
    cross_sectional_ops: dict[str, Operator] = field(default_factory=dict)
    group_ops: dict[str, Operator] = field(default_factory=dict)
    features: dict[str, Operator] = field(default_factory=dict)

    def register_time_series(self, name: str, fn: Operator) -> None:
        self.time_series_ops[name] = fn

    def register_cross_sectional(self, name: str, fn: Operator) -> None:
        self.cross_sectional_ops[name] = fn

    def register_group(self, name: str, fn: Operator) -> None:
        self.group_ops[name] = fn

    def register_feature(self, name: str, fn: Operator) -> None:
        self.features[name] = fn
