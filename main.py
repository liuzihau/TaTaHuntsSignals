"""TaTaHunts RAG Pipeline - Crawl + Embedding runtime."""

from __future__ import annotations

import logging
import os
import time

from agents import nodes as agent_nodes
from data_sources.news_detail_crawler import NewsDetailCrawler
from data_sources.yahoo_rss_crawler import YahooRSSCrawler
from embedding.embedder import NewsEmbedder
from jobs.embedding_job import EmbeddingJob
from schedulers.task_scheduler import setup_scheduler
from storage.milvus_client import MilvusClient
from storage.postgres_client import PostgreSQLClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_pipeline.log"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)


class AppContext:
    """Global runtime context for crawl + embedding pipeline."""

    def __init__(self, top_tickers: list[str]) -> None:
        LOGGER.info("Initializing pipeline context...")

        self.postgres = PostgreSQLClient(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "tatahunts_user"),
            password=os.getenv("POSTGRES_PASSWORD", "tatahunts_password"),
            database=os.getenv("POSTGRES_DB", "tatahunts_rag"),
        )

        self.milvus = MilvusClient(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=int(os.getenv("MILVUS_PORT", "19530")),
        )
        self.milvus.initialize_collection()

        self.rss_crawler = YahooRSSCrawler(self.postgres)
        self.detail_crawler = NewsDetailCrawler(parallel_workers=5)

        self.embedder = NewsEmbedder(
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en"),
            device=os.getenv("EMBEDDING_DEVICE", "cuda"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )
        self.embedding_job = EmbeddingJob(self.postgres, self.milvus, self.embedder)

        # Wire vector backend into analysis retrieval node.
        agent_nodes.configure_text_retrieval_backend(
            milvus_client=self.milvus,
            postgres_client=self.postgres,
            embedder=self.embedder,
        )

        self.tickers = top_tickers
        self.scheduler = setup_scheduler(self)
        LOGGER.info("Pipeline context initialized.")

    def shutdown(self) -> None:
        """Graceful shutdown for scheduler and clients."""
        LOGGER.info("Shutting down pipeline...")
        try:
            if self.scheduler and self.scheduler.running:
                self.scheduler.shutdown()
        except Exception as exc:  # pragma: no cover - scheduler runtime
            LOGGER.warning("Scheduler shutdown warning: %s", exc)

        try:
            self.postgres.close()
        except Exception as exc:  # pragma: no cover - connection runtime
            LOGGER.warning("PostgreSQL close warning: %s", exc)

        try:
            self.milvus.close()
        except Exception as exc:  # pragma: no cover - connection runtime
            LOGGER.warning("Milvus close warning: %s", exc)


def main() -> None:
    """Entry point for persistent pipeline runtime."""
    top_tickers = [
        "NVDA",
        "AAPL",
        "MSFT",
        "TSLA",
        "AMZN",
        "GOOGL",
        "META",
        "NFLX",
    ]

    app = AppContext(top_tickers)
    try:
        LOGGER.info("TaTaHunts RAG pipeline started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Shutdown signal received.")
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()

