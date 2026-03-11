"""PostgreSQL client for crawler and staging CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

LOGGER = logging.getLogger(__name__)


class PostgreSQLClient:
    """Low-level PostgreSQL CRUD client for raw/staging/article layers."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "tatahunts_user",
        password: str = "tatahunts_password",
        database: str = "tatahunts_rag",
    ) -> None:
        self.dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.conn: psycopg2.extensions.connection | None = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        if self.conn and not self.conn.closed:
            return
        try:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = False
            LOGGER.info("Connected to PostgreSQL")
        except psycopg2.Error as exc:
            LOGGER.error("Failed to connect to PostgreSQL: %s", exc)
            raise

    def close(self) -> None:
        """Close PostgreSQL connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            LOGGER.info("PostgreSQL connection closed")

    def _require_conn(self) -> psycopg2.extensions.connection:
        if self.conn is None or self.conn.closed:
            self.connect()
        if self.conn is None:
            raise RuntimeError("PostgreSQL connection is unavailable.")
        return self.conn

    # ========= RAW LAYER =========

    def insert_raw_articles(self, articles: list[dict[str, Any]]) -> list[int]:
        """Insert raw articles and return inserted raw IDs."""
        if not articles:
            return []

        conn = self._require_conn()
        rows = [
            (
                article.get("url"),
                article.get("url_hash"),
                article.get("title"),
                self._normalize_datetime(article.get("published_date")),
                article.get("source", "yahoo_finance"),
                self._normalize_datetime(article.get("crawled_at")) or datetime.now(timezone.utc),
                article.get("crawler_version", "rss_v1"),
                float(article.get("extraction_confidence", 1.0)),
            )
            for article in articles
            if article.get("url") and article.get("title")
        ]

        if not rows:
            return []

        query = """
        INSERT INTO raw_articles
            (url, url_hash, title, published_date, source, crawled_at, crawler_version, extraction_confidence)
        VALUES %s
        ON CONFLICT (url) DO NOTHING
        RETURNING id
        """

        try:
            with conn.cursor() as cur:
                execute_values(cur, query, rows, page_size=1000)
                inserted = cur.fetchall()
            conn.commit()
            inserted_ids = [int(row[0]) for row in inserted]
            LOGGER.info(
                "Inserted %d raw articles (%d attempted).",
                len(inserted_ids),
                len(rows),
            )
            return inserted_ids
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to insert raw articles: %s", exc)
            return []

    def update_raw_article_body(self, raw_article_id: int, body: str, metadata: dict[str, Any]) -> bool:
        """Update raw article with crawled body and metadata from detail crawl."""
        conn = self._require_conn()
        query = """
        UPDATE raw_articles
        SET raw_body = %s,
            author = %s,
            views = %s,
            shares = %s,
            comments = %s,
            crawled_at = NOW()
        WHERE id = %s
        """
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        body,
                        metadata.get("author"),
                        int(metadata.get("views", 0) or 0),
                        int(metadata.get("shares", 0) or 0),
                        int(metadata.get("comments", 0) or 0),
                        raw_article_id,
                    ),
                )
            conn.commit()
            return True
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to update raw article %s: %s", raw_article_id, exc)
            return False

    def get_unprocessed_raw_articles(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Fetch raw articles that have not yet been detail-crawled."""
        conn = self._require_conn()
        query = """
        SELECT id, url, url_hash, title, published_date, source, crawled_at
        FROM raw_articles
        WHERE raw_body IS NULL
        ORDER BY crawled_at ASC
        LIMIT %s
        """
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
            return [dict(row) for row in rows]
        except psycopg2.Error as exc:
            LOGGER.error("Failed to fetch unprocessed raw articles: %s", exc)
            return []

    def url_exists(self, url: str) -> bool:
        """Check whether a URL already exists in raw_articles."""
        conn = self._require_conn()
        query = "SELECT EXISTS(SELECT 1 FROM raw_articles WHERE url = %s)"
        try:
            with conn.cursor() as cur:
                cur.execute(query, (url,))
                result = cur.fetchone()
            return bool(result[0]) if result else False
        except psycopg2.Error as exc:
            LOGGER.error("Failed URL existence check: %s", exc)
            return False

    # ========= STAGING LAYER =========

    def insert_staging_articles(self, articles: list[dict[str, Any]]) -> list[str]:
        """Insert standardized records into staging_articles."""
        if not articles:
            return []

        conn = self._require_conn()
        rows = [
            (
                article.get("id"),
                article.get("raw_article_id"),
                article.get("url"),
                article.get("title"),
                article.get("body"),
                self._normalize_datetime(article.get("published_date")) or datetime.now(timezone.utc),
                article.get("source", "yahoo_finance"),
                article.get("author"),
                int(article.get("views", 0) or 0),
                int(article.get("shares", 0) or 0),
                int(article.get("comments", 0) or 0),
                bool(article.get("ready_for_ml", bool(article.get("body")))),
            )
            for article in articles
            if article.get("id") and article.get("url")
        ]

        if not rows:
            return []

        query = """
        INSERT INTO staging_articles
            (id, raw_article_id, url, title, body, published_date, source, author, views, shares, comments, ready_for_ml)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            body = EXCLUDED.body,
            author = EXCLUDED.author,
            views = EXCLUDED.views,
            shares = EXCLUDED.shares,
            comments = EXCLUDED.comments,
            ready_for_ml = EXCLUDED.ready_for_ml,
            updated_at = NOW()
        RETURNING id
        """

        try:
            with conn.cursor() as cur:
                execute_values(cur, query, rows, page_size=1000)
                inserted = cur.fetchall()
            conn.commit()
            inserted_ids = [str(row[0]) for row in inserted]
            LOGGER.info(
                "Upserted %d staging articles (%d attempted).",
                len(inserted_ids),
                len(rows),
            )
            return inserted_ids
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to insert staging articles: %s", exc)
            return []

    def get_unprocessed_staging_articles(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Fetch staging records that are ready for transform."""
        conn = self._require_conn()
        query = """
        SELECT id, raw_article_id, url, title, body, published_date, source, author, views, shares, comments
        FROM staging_articles
        WHERE processing_status = 'pending'
          AND ready_for_ml = TRUE
        ORDER BY published_date DESC
        LIMIT %s
        """
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
            return [dict(row) for row in rows]
        except psycopg2.Error as exc:
            LOGGER.error("Failed to fetch unprocessed staging articles: %s", exc)
            return []

    # ========= PRODUCTION ARTICLES =========

    def get_articles_needing_embedding(self, limit: int = 10000) -> list[dict[str, Any]]:
        """Fetch production articles that have not been embedded yet."""
        conn = self._require_conn()
        query = """
        SELECT
            id,
            title,
            body,
            summary,
            ticker,
            sentiment,
            category,
            credibility_tier,
            engagement_score,
            published_date
        FROM articles
        WHERE COALESCE(embedding_version, 0) = 0
           OR COALESCE(has_embedding, FALSE) = FALSE
        ORDER BY published_date DESC
        LIMIT %s
        """
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
            records = [dict(row) for row in rows]
            LOGGER.info("Fetched %d articles needing embedding.", len(records))
            return records
        except psycopg2.Error as exc:
            LOGGER.error("Failed to fetch articles needing embedding: %s", exc)
            return []

    def store_embeddings(
        self,
        article_ids: list[str],
        embeddings: list[list[float]],
        embedding_model: str = "finbert",
        embedding_dimension: int = 768,
    ) -> bool:
        """
        Store FinBERT embeddings directly to PostgreSQL pgvector column.
        
        Args:
            article_ids: List of article IDs
            embeddings: List of embedding vectors (each is a list of floats)
            embedding_model: Name of the embedding model (e.g., "finbert")
            embedding_dimension: Dimension of the embeddings (e.g., 768)
            
        Returns:
            True if successful, False otherwise
        """
        if not article_ids or not embeddings:
            return True

        if len(article_ids) != len(embeddings):
            LOGGER.error(
                "Mismatch: %d article_ids but %d embeddings",
                len(article_ids),
                len(embeddings),
            )
            return False

        conn = self._require_conn()
        
        # Prepare rows for batch insert
        rows = list(zip(article_ids, embeddings))
        
        query = """
        UPDATE articles
        SET embedding = %s,
            embedding_model = %s,
            embedding_dimension = %s,
            embedding_version = 1,
            has_embedding = TRUE,
            embedding_updated = NOW(),
            updated_at = NOW()
        WHERE id = %s
        """
        
        try:
            with conn.cursor() as cur:
                for article_id, embedding in rows:
                    # Convert embedding list to PostgreSQL vector format
                    # Format: [val1, val2, ..., valN]
                    embedding_str = "[" + ",".join(str(float(v)) for v in embedding) + "]"
                    cur.execute(
                        query,
                        (embedding_str, embedding_model, embedding_dimension, article_id),
                    )
            conn.commit()
            LOGGER.info(
                "Stored %d embeddings (%s, %d-dim) to PostgreSQL.",
                len(embeddings),
                embedding_model,
                embedding_dimension,
            )
            return True
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to store embeddings: %s", exc)
            return False

    def mark_articles_embedded(self, article_ids: list[str], embedding_version: int = 1) -> bool:
        """Mark articles as embedded after Milvus insertion."""
        if not article_ids:
            return True

        conn = self._require_conn()
        query = """
        UPDATE articles
        SET embedding_version = %s,
            has_embedding = TRUE,
            embedding_updated = NOW(),
            updated_at = NOW()
        WHERE id = ANY(%s::text[])
        """
        try:
            with conn.cursor() as cur:
                cur.execute(query, (int(embedding_version), article_ids))
            conn.commit()
            LOGGER.info("Marked %d articles as embedded.", len(article_ids))
            return True
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to mark articles embedded: %s", exc)
            return False

    def get_articles_by_ids(self, article_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full article rows by IDs for retrieval nodes."""
        if not article_ids:
            return []

        conn = self._require_conn()
        query = """
        SELECT
            id,
            url,
            title,
            body,
            summary,
            published_date,
            source,
            author,
            ticker,
            related_tickers,
            category,
            sentiment,
            credibility_tier,
            engagement_score
        FROM articles
        WHERE id = ANY(%s::text[])
        """
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (article_ids,))
                rows = [dict(row) for row in cur.fetchall()]

            # Preserve request order for downstream deterministic behavior.
            rank = {value: idx for idx, value in enumerate(article_ids)}
            rows.sort(key=lambda row: rank.get(str(row.get("id")), len(rank)))
            return rows
        except psycopg2.Error as exc:
            LOGGER.error("Failed to fetch articles by ids: %s", exc)
            return []

    # ========= CRAWLER STATE =========

    def upsert_crawler_state(
        self,
        ticker: str,
        *,
        last_rss_crawl: datetime | None = None,
        last_detail_crawl: datetime | None = None,
        last_article_date: datetime | None = None,
        total_articles_scraped: int | None = None,
        duplicates_found: int | None = None,
        failed_attempts: int | None = None,
        crawl_status: str | None = None,
        last_error: str | None = None,
    ) -> bool:
        """Insert or update crawler state for a ticker."""
        conn = self._require_conn()
        query = """
        INSERT INTO crawler_state
            (ticker, last_rss_crawl, last_detail_crawl, last_article_date,
             total_articles_scraped, duplicates_found, failed_attempts, crawl_status, last_error, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s, 'pending'), %s, NOW())
        ON CONFLICT (ticker) DO UPDATE SET
            last_rss_crawl = COALESCE(EXCLUDED.last_rss_crawl, crawler_state.last_rss_crawl),
            last_detail_crawl = COALESCE(EXCLUDED.last_detail_crawl, crawler_state.last_detail_crawl),
            last_article_date = COALESCE(EXCLUDED.last_article_date, crawler_state.last_article_date),
            total_articles_scraped = COALESCE(EXCLUDED.total_articles_scraped, crawler_state.total_articles_scraped),
            duplicates_found = COALESCE(EXCLUDED.duplicates_found, crawler_state.duplicates_found),
            failed_attempts = COALESCE(EXCLUDED.failed_attempts, crawler_state.failed_attempts),
            crawl_status = COALESCE(EXCLUDED.crawl_status, crawler_state.crawl_status),
            last_error = COALESCE(EXCLUDED.last_error, crawler_state.last_error),
            updated_at = NOW()
        """
        try:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        ticker.upper().strip(),
                        self._normalize_datetime(last_rss_crawl),
                        self._normalize_datetime(last_detail_crawl),
                        self._normalize_datetime(last_article_date),
                        total_articles_scraped,
                        duplicates_found,
                        failed_attempts,
                        crawl_status,
                        last_error,
                    ),
                )
            conn.commit()
            return True
        except psycopg2.Error as exc:
            conn.rollback()
            LOGGER.error("Failed to upsert crawler state for %s: %s", ticker, exc)
            return False

    @staticmethod
    def _normalize_datetime(value: Any) -> datetime | None:
        """Best-effort datetime parsing for DB writes."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None
