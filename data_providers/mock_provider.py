"""Demo provider for local-only MVP execution without API keys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_providers.interfaces import FundamentalsProvider, MarketDataProvider, TextDataProvider

_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


class MockProvider(MarketDataProvider, FundamentalsProvider, TextDataProvider):
    """Loads deterministic sample snapshots for a handful of tickers."""

    def __init__(self) -> None:
        self.market_payload = self._load_json("market_data.json")
        self.fund_payload = self._load_json("fundamentals.json")

    @staticmethod
    def _load_json(name: str) -> dict[str, Any]:
        with (_SAMPLE_DIR / name).open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_market_snapshot(self, ticker: str) -> dict[str, Any]:
        t = ticker.upper()
        return self.market_payload.get(t, self.market_payload["DEFAULT"])

    def get_fundamentals(self, ticker: str) -> dict[str, Any]:
        t = ticker.upper()
        return self.fund_payload.get(t, self.fund_payload["DEFAULT"])

    def get_documents(self, ticker: str) -> list[dict[str, str]]:
        t = ticker.upper()
        docs: list[dict[str, str]] = []
        text_root = _SAMPLE_DIR / "texts"

        specific = sorted(text_root.glob(f"{t}_*.txt"))
        generic = sorted(text_root.glob("GENERIC_*.txt"))
        source_files = specific if specific else generic

        for path in source_files:
            docs.append(
                {
                    "id": path.stem,
                    "source": path.name,
                    "content": path.read_text(encoding="utf-8").strip(),
                }
            )
        return docs
