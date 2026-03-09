"""Simple local retrieval for MVP RAG over sample filings/news text."""

from __future__ import annotations

import re
from collections import Counter

from models.state import EvidenceSnippet


class LocalKeywordRetriever:
    """Lightweight chunking + overlap scoring retrieval for demo mode."""

    def __init__(self, chunk_size: int = 380) -> None:
        self.chunk_size = chunk_size

    def retrieve(
        self,
        query: str,
        documents: list[dict[str, str]],
        k: int = 5,
    ) -> list[EvidenceSnippet]:
        query_terms = self._tokenize(query)
        query_counts = Counter(query_terms)
        results: list[EvidenceSnippet] = []

        for doc in documents:
            chunks = self._chunk_text(doc.get("content", ""))
            for chunk in chunks:
                score = self._score(query_counts, Counter(self._tokenize(chunk)))
                if score <= 0:
                    continue
                results.append(
                    EvidenceSnippet(source=doc.get("source", "unknown"), snippet=chunk, score=score)
                )

        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        return ranked[:k]

    def _chunk_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @staticmethod
    def _score(q: Counter[str], d: Counter[str]) -> float:
        overlap = sum(min(q[t], d[t]) for t in q)
        return float(overlap)
