"""Lightweight Yahoo Finance RSS crawler."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import logging
import time
from typing import Any
import xml.etree.ElementTree as ET

import requests

LOGGER = logging.getLogger(__name__)

try:
    import feedparser
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    feedparser = None


@dataclass(slots=True)
class RawArticle:
    """Raw article payload extracted from RSS."""

    url: str
    title: str
    published_date: datetime
    source: str = "yahoo_finance"

    @property
    def url_hash(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "url_hash": self.url_hash,
            "title": self.title,
            "published_date": self.published_date,
            "source": self.source,
        }


class YahooRSSCrawler:
    """Fetch Yahoo Finance RSS feeds for a set of tickers."""

    RSS_URL_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    TIMEOUT = 10

    def __init__(self, postgres_client: Any) -> None:
        self.postgres = postgres_client
        self.session = self._create_session()

    @staticmethod
    def _create_session() -> requests.Session:
        """Create a persistent HTTP session."""
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (TaTaHunts News Crawler 1.0)"})
        return session

    def crawl_ticker(self, ticker: str) -> list[RawArticle]:
        """
        Crawl RSS feed for a single ticker.

        Returns only new URLs not present in raw_articles.
        """
        normalized = ticker.upper().strip()
        if not normalized:
            return []

        url = self.RSS_URL_TEMPLATE.format(ticker=normalized)
        try:
            response = self.session.get(url, timeout=self.TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.error("Failed to fetch RSS for %s: %s", normalized, exc)
            self._safe_update_state(
                normalized,
                crawl_status="failed",
                failed_attempts=1,
                last_error=str(exc),
            )
            return []

        if feedparser is not None:
            feed = feedparser.parse(response.content)
            entries = getattr(feed, "entries", [])
        else:
            entries = self._parse_rss_fallback(response.content)
        if not entries:
            LOGGER.warning("No RSS entries found for %s", normalized)
            self._safe_update_state(
                normalized,
                crawl_status="success",
                total_articles_scraped=0,
                duplicates_found=0,
            )
            return []

        articles: list[RawArticle] = []
        duplicates = 0
        newest: datetime | None = None

        for entry in entries:
            article_url = str(entry.get("link", "")).strip()
            title = str(entry.get("title", "Untitled")).strip()
            if not article_url:
                continue

            if self.postgres.url_exists(article_url):
                duplicates += 1
                continue

            published = self._parse_date(entry) or datetime.now(timezone.utc)
            newest = max(newest, published) if newest else published
            articles.append(
                RawArticle(
                    url=article_url,
                    title=title or "Untitled",
                    published_date=published,
                    source="yahoo_finance",
                )
            )

        self._safe_update_state(
            normalized,
            crawl_status="success",
            total_articles_scraped=len(articles),
            duplicates_found=duplicates,
            last_article_date=newest,
        )
        LOGGER.info(
            "RSS crawl %s -> %d new, %d duplicate.",
            normalized,
            len(articles),
            duplicates,
        )
        return articles

    def crawl_batch(self, tickers: list[str], sleep_ms_between: int = 0) -> dict[str, Any]:
        """
        Crawl multiple tickers.

        Returns:
            {"success": int, "articles": list[RawArticle], "errors": list[str]}
        """
        all_articles: list[RawArticle] = []
        errors: list[str] = []
        success_count = 0

        for idx, ticker in enumerate(tickers):
            try:
                batch = self.crawl_ticker(ticker)
                all_articles.extend(batch)
                success_count += 1
            except Exception as exc:  # pragma: no cover - safety net
                message = f"Error crawling {ticker}: {exc}"
                LOGGER.error(message)
                errors.append(message)
            if sleep_ms_between > 0 and idx < len(tickers) - 1:
                time.sleep(sleep_ms_between / 1000)

        return {
            "success": success_count,
            "articles": all_articles,
            "errors": errors,
        }

    @staticmethod
    def _parse_date(entry: Any) -> datetime | None:
        """Parse entry date from feedparser fields."""
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            try:
                return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass

        raw = entry.get("published") or entry.get("updated")
        if isinstance(raw, str) and raw.strip():
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    dt = parsedate_to_datetime(raw)
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _parse_rss_fallback(content: bytes) -> list[dict[str, Any]]:
        """Minimal RSS parser when feedparser dependency is unavailable."""
        entries: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return entries

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if not link:
                continue
            entries.append({"title": title, "link": link, "published": pub_date})
        return entries

    def _safe_update_state(
        self,
        ticker: str,
        *,
        crawl_status: str,
        total_articles_scraped: int | None = None,
        duplicates_found: int | None = None,
        failed_attempts: int | None = None,
        last_article_date: datetime | None = None,
        last_error: str | None = None,
    ) -> None:
        """Update crawler state if client supports the method."""
        if not hasattr(self.postgres, "upsert_crawler_state"):
            return
        try:
            self.postgres.upsert_crawler_state(
                ticker=ticker,
                last_rss_crawl=datetime.now(timezone.utc),
                last_article_date=last_article_date,
                total_articles_scraped=total_articles_scraped,
                duplicates_found=duplicates_found,
                failed_attempts=failed_attempts,
                crawl_status=crawl_status,
                last_error=last_error,
            )
        except Exception as exc:  # pragma: no cover - best-effort logging only
            LOGGER.warning("Failed to update crawler_state for %s: %s", ticker, exc)
