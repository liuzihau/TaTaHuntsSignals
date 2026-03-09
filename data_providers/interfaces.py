"""Provider interfaces for data abstraction and future provider swapping."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MarketDataProvider(ABC):
    @abstractmethod
    def get_market_snapshot(self, ticker: str) -> dict[str, Any]:
        """Return market and technical-ready time-window snapshot."""


class FundamentalsProvider(ABC):
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        """Return basic company/ETF fundamentals."""


class TextDataProvider(ABC):
    @abstractmethod
    def get_documents(self, ticker: str) -> list[dict[str, str]]:
        """Return local/news/filing-like documents for retrieval."""
