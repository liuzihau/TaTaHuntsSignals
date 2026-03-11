"""Extract full article body and engagement metadata from article URLs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOGGER = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    BeautifulSoup = Any  # type: ignore[assignment,misc]
    _BS4_AVAILABLE = False
else:
    _BS4_AVAILABLE = True


class NewsDetailCrawler:
    """Fetch article body, author, and lightweight engagement metadata."""

    def __init__(self, parallel_workers: int = 5, backoff_factor: float = 1.5) -> None:
        self.workers = max(1, int(parallel_workers))
        self.backoff_factor = backoff_factor
        self.session = self._create_session_with_backoff()

    def _create_session_with_backoff(self) -> requests.Session:
        """Build an HTTP session with retry strategy."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=self.workers, pool_maxsize=self.workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "Mozilla/5.0 (TaTaHunts News Crawler 1.0)"})
        return session

    def crawl_url(self, url: str, timeout: int = 10) -> dict[str, Any] | None:
        """
        Crawl one URL.

        Returns:
            {"url", "body", "author", "views", "shares", "comments"} or None.
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            LOGGER.warning("Failed to crawl %s: %s", url, exc)
            return None

        if not _BS4_AVAILABLE:
            LOGGER.error("beautifulsoup4 is not installed. Install dependencies to run detail crawl.")
            return None

        try:
            soup = BeautifulSoup(response.content, "html.parser")
            body = self._extract_body(soup)
            if not body:
                LOGGER.warning("No article body extracted for %s", url)
                return None

            return {
                "url": url,
                "body": body,
                "author": self._extract_author(soup),
                "views": self._extract_metric(soup, "view"),
                "shares": self._extract_metric(soup, "share"),
                "comments": self._extract_metric(soup, "comment"),
            }
        except Exception as exc:  # pragma: no cover - parser safety net
            LOGGER.error("Failed to parse %s: %s", url, exc)
            return None

    @staticmethod
    def _extract_body(soup: BeautifulSoup) -> str:
        """Extract main text from common article containers."""
        selectors = (
            "article",
            "div[class*='article-body']",
            "div[class*='story-body']",
            "div[class*='caas-body']",
            "div[class*='content']",
            "main",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if not node:
                continue
            text = node.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 100:
                return text[:10000]
        return ""

    @staticmethod
    def _extract_author(soup: BeautifulSoup) -> str | None:
        """Extract likely author value from common selectors."""
        selectors = (
            "meta[name='author']",
            "span[class*='author']",
            "div[class*='author']",
            "a[class*='author']",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if not node:
                continue
            if node.name == "meta":
                value = (node.get("content") or "").strip()
            else:
                value = node.get_text(strip=True)
            if value:
                return value
        return None

    @staticmethod
    def _extract_metric(soup: BeautifulSoup, metric_type: str) -> int:
        """Extract one engagement metric based on class/aria patterns."""
        selectors = (
            f"span[class*='{metric_type}']",
            f"div[class*='{metric_type}']",
            f"[aria-label*='{metric_type}']",
        )
        for selector in selectors:
            node = soup.select_one(selector)
            if not node:
                continue
            text = node.get_text(strip=True)
            match = re.search(r"\d[\d,]*", text)
            if match:
                return int(match.group().replace(",", ""))
        return 0

    def crawl_batch(self, urls: list[str], *, use_threads: bool = True) -> list[dict[str, Any]]:
        """
        Crawl a list of URLs and keep input order for successful results.

        When `use_threads=True`, crawls in parallel with `self.workers`.
        """
        if not urls:
            return []

        if not use_threads or self.workers == 1:
            return self._crawl_batch_sequential(urls)

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.crawl_url, url): idx for idx, url in enumerate(urls)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    payload = future.result()
                    if payload:
                        results[idx] = payload
                except Exception as exc:  # pragma: no cover - safety net
                    LOGGER.warning("Detail crawl worker failed for index=%s: %s", idx, exc)

        return [results[i] for i in sorted(results)]

    def _crawl_batch_sequential(self, urls: list[str]) -> list[dict[str, Any]]:
        """Sequential crawl fallback with mild rate limiting."""
        output: list[dict[str, Any]] = []
        for idx, url in enumerate(urls):
            payload = self.crawl_url(url)
            if payload:
                output.append(payload)
            if (idx + 1) % 10 == 0:
                time.sleep(1)
        return output
