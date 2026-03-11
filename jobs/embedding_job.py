"""Batch embedding job for articles using FinBERT encoder."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


class EmbeddingJob:
    """Fetch unembedded articles, vectorize them with FinBERT, and persist to PostgreSQL."""

    def __init__(self, postgres_client: Any, embedder: Any) -> None:
        """
        Initialize embedding job.
        
        Args:
            postgres_client: PostgreSQL client instance
            embedder: FinBERT encoder instance (from embedding.finbert_encoder.FinBertEncoder)
        """
        self.postgres = postgres_client
        self.embedder = embedder

    def embed_new_articles(self, limit: int = 10000) -> dict[str, Any]:
        """
        Execute embedding pipeline for new articles using FinBERT.
        
        Embeds articles and stores 768-dim vectors directly to PostgreSQL pgvector column.

        Args:
            limit: Maximum number of articles to embed in this run

        Returns:
            {"embedded": int, "failed": int, "errors": list[str]}
        """
        LOGGER.info("Starting FinBERT embedding job (limit=%s).", limit)
        articles = self.postgres.get_articles_needing_embedding(limit=limit)
        if not articles:
            LOGGER.info("No articles needing embedding.")
            return {"embedded": 0, "failed": 0, "errors": []}

        LOGGER.info("Preparing %d texts for embedding.", len(articles))
        texts: list[str] = []
        for item in articles:
            title = str(item.get("title", "")).strip()
            body = str(item.get("body", "")).strip()
            # Combine title + body for encoding (FinBERT optimized for financial text)
            merged = f"{title} {body}".strip()
            texts.append(merged)

        try:
            LOGGER.info("Encoding batch of %d articles...", len(texts))
            embeddings = self.embedder.encode_batch(texts, batch_size=32)
        except Exception as exc:
            message = f"Embedding generation failed: {exc}"
            LOGGER.error(message)
            return {"embedded": 0, "failed": len(articles), "errors": [message]}

        if not embeddings or len(embeddings) != len(articles):
            message = f"Embedding mismatch: got={len(embeddings)} expected={len(articles)}"
            LOGGER.error(message)
            return {"embedded": 0, "failed": len(articles), "errors": [message]}

        # Store embeddings directly to PostgreSQL
        article_ids = [str(item["id"]) for item in articles if item.get("id")]
        try:
            LOGGER.info("Storing %d embeddings to PostgreSQL...", len(embeddings))
            success = self.postgres.store_embeddings(
                article_ids=article_ids,
                embeddings=embeddings,
                embedding_model="finbert",
                embedding_dimension=768,
            )
        except Exception as exc:
            message = f"PostgreSQL insert failed: {exc}"
            LOGGER.error(message)
            return {"embedded": 0, "failed": len(articles), "errors": [message]}

        if not success:
            message = "PostgreSQL store_embeddings returned False."
            LOGGER.error(message)
            return {"embedded": 0, "failed": len(articles), "errors": [message]}

        LOGGER.info("✅ Embedding job completed: embedded=%d articles", len(articles))
        return {"embedded": len(articles), "failed": 0, "errors": []}

