"""PostgreSQL-based vector similarity retrieval for news articles."""

from __future__ import annotations

import logging
from typing import Any

from models.state import EvidenceSnippet
from storage.postgres_client import PostgreSQLClient

LOGGER = logging.getLogger(__name__)


class PostgresVectorRetriever:
    """Retrieve relevant news articles using PostgreSQL pgvector similarity search."""

    def __init__(self, postgres_client: PostgreSQLClient | None = None) -> None:
        """Initialize retriever with PostgreSQL client."""
        self.postgres = postgres_client or PostgreSQLClient()

    def retrieve(
        self,
        query: str,
        ticker: str,
        limit: int = 10,
        days_back: int = 30,
    ) -> list[EvidenceSnippet]:
        """
        Retrieve articles similar to query using vector similarity.
        
        Args:
            query: Query text (will be encoded with FinBERT)
            ticker: Stock ticker (e.g., 'AAPL')
            limit: Number of results to return
            days_back: Only consider articles from past N days
            
        Returns:
            List of EvidenceSnippet objects with relevant articles
        """
        try:
            # Import here to avoid circular dependency
            from embedding.finbert_encoder import FinBertEncoder
            
            encoder = FinBertEncoder()
            query_embedding = encoder.encode(query)
            
            # Search PostgreSQL with vector similarity
            articles = self.postgres.search_articles_by_vector(
                query_embedding=query_embedding,
                ticker=ticker,
                limit=limit,
                days_back=days_back,
            )
            
            # Convert to EvidenceSnippet format
            snippets = [
                EvidenceSnippet(
                    source=article.get("source", "unknown"),
                    snippet=article.get("body", "")[:300],  # Truncate for brevity
                    score=float(article.get("similarity", 0.0))
                )
                for article in articles
            ]
            
            LOGGER.info(f"Retrieved {len(snippets)} articles for {ticker}")
            return snippets
            
        except Exception as exc:
            LOGGER.error(f"Retrieval failed: {exc}")
            return []

    def close(self) -> None:
        """Close database connection."""
        if self.postgres:
            self.postgres.close()


# Fallback: Simple keyword-based retriever for compatibility
class LocalKeywordRetriever:
    """Lightweight chunking + keyword overlap scoring (fallback/demo mode)."""

    def __init__(self, chunk_size: int = 380) -> None:
        self.chunk_size = chunk_size

    def retrieve(
        self,
        query: str,
        documents: list[dict[str, str]],
        k: int = 5,
    ) -> list[EvidenceSnippet]:
        """Retrieve documents using keyword overlap scoring."""
        import re
        from collections import Counter
        
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
                    EvidenceSnippet(
                        source=doc.get("source", "unknown"),
                        snippet=chunk,
                        score=score
                    )
                )

        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        return ranked[:k]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
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
        """Simple tokenization."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @staticmethod
    def _score(q: Counter, d: Counter) -> float:
        """Calculate keyword overlap score."""
        overlap = sum(min(q[t], d[t]) for t in q)
        return float(overlap)
