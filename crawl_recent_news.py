#!/usr/bin/env python3
"""
Crawl recent news for top US stocks and populate PostgreSQL.

Usage:
    python3 crawl_recent_news.py [--tickers AAPL,MSFT,NVDA] [--limit 100]

Output:
    - raw_articles table: RSS entries
    - staging_articles table: Detail-crawled articles (body + metadata)
    - articles table: Production-ready articles
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
import sys
import time
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s"
)
LOGGER = logging.getLogger(__name__)

from data_sources.yahoo_rss_crawler import YahooRSSCrawler
from data_sources.news_detail_crawler import NewsDetailCrawler
from storage.postgres_client import PostgreSQLClient

# Top 100 US stocks by market cap (for default crawling)
TOP_100_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "JNJ", "XOM",
    "UNH", "JPM", "V", "PG", "MA", "AVGO", "HD", "PFE", "COST", "MCD",
    "ACN", "CRM", "KO", "ADBE", "PEP", "DIS", "INTC", "NKE", "TXN", "CAT",
    "CSCO", "VZ", "CMCSA", "ABT", "QCOM", "MMM", "AMD", "HON", "INTU", "ORCL",
    "LLY", "WMT", "T", "OXY", "IBM", "BA", "RTX", "AXP", "NEE", "GE",
    "CVX", "SQ", "NFLX", "WBA", "NOW", "PLD", "MU", "PM", "RSG", "COP",
    "SO", "EOG", "AEP", "LOW", "F", "EXC", "ZTS", "HUM", "GOLD", "DASH",
    "WEC", "DHI", "EL", "PAYC", "AMAT", "MSI", "FANG", "WY", "EW", "LRCX",
    "FDX", "TMO", "MSTR", "ROP", "SLB", "URI", "RELX", "MRK", "CL", "CPB",
    "WELL", "LVS", "BBY", "BX", "FI", "FITB", "APA", "COIN", "HRB", "GS"
]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Crawl recent news for stocks")
    parser.add_argument(
        "--tickers",
        type=str,
        default=",".join(TOP_100_TICKERS[:50]),  # Default: top 50
        help="Comma-separated tickers to crawl (default: top 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to crawl per ticker (default: no limit)",
    )
    parser.add_argument(
        "--skip-detail",
        action="store_true",
        help="Skip detail crawl (body extraction); RSS metadata only",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers for detail crawl",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=500,
        help="Sleep milliseconds between RSS crawls",
    )
    return parser.parse_args()


def crawl_rss(pg: PostgreSQLClient, tickers: list[str], sleep_ms: int) -> dict[str, Any]:
    """Phase 1: Crawl RSS feeds for all tickers."""
    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 1: RSS Crawl")
    LOGGER.info("=" * 70)
    
    crawler = YahooRSSCrawler(pg)
    result = crawler.crawl_batch(tickers, sleep_ms_between=sleep_ms)
    
    all_articles = result["articles"]
    success = result["success"]
    errors = result["errors"]
    
    LOGGER.info(f"✅ RSS crawl complete:")
    LOGGER.info(f"   - Successful tickers: {success}/{len(tickers)}")
    LOGGER.info(f"   - New articles found: {len(all_articles)}")
    if errors:
        for err in errors:
            LOGGER.warning(f"   - {err}")
    
    # Insert into raw_articles
    if all_articles:
        raw_ids = pg.insert_raw_articles([a.to_dict() for a in all_articles])
        LOGGER.info(f"   - Inserted into raw_articles: {len(raw_ids)}")
    else:
        LOGGER.warning("   ⚠️ No articles found!")
    
    return {"articles": all_articles, "success": success, "errors": errors}


def crawl_detail(pg: PostgreSQLClient, articles: list[Any], workers: int, skip: bool) -> int:
    """Phase 2: Extract body + metadata from article URLs."""
    if skip or not articles:
        LOGGER.info("Skipping detail crawl (--skip-detail set)")
        return 0
    
    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 2: Detail Crawl (Extract Body + Metadata)")
    LOGGER.info("=" * 70)
    
    urls = [a.url for a in articles]
    LOGGER.info(f"Crawling {len(urls)} article URLs (workers={workers})...")
    
    crawler = NewsDetailCrawler(parallel_workers=workers)
    results = crawler.crawl_batch(urls, use_threads=True)
    
    LOGGER.info(f"✅ Detail crawl complete: {len(results)}/{len(urls)} successful")
    
    # Map back to raw articles and update
    for detail_result in results:
        article_url = detail_result["url"]
        # Find raw article with this URL
        with pg._require_conn().cursor() as cur:
            cur.execute("SELECT id FROM raw_articles WHERE url = %s", (article_url,))
            row = cur.fetchone()
            if row:
                raw_id = row[0]
                pg.update_raw_article_body(raw_id, detail_result["body"], {
                    "author": detail_result.get("author"),
                    "views": detail_result.get("views", 0),
                    "shares": detail_result.get("shares", 0),
                    "comments": detail_result.get("comments", 0),
                })
    
    LOGGER.info(f"Updated raw_articles with body and metadata")
    return len(results)


def transform_to_production(pg: PostgreSQLClient) -> int:
    """Phase 3: Transform raw → staging → production articles."""
    LOGGER.info("=" * 70)
    LOGGER.info("PHASE 3: Transform to Production")
    LOGGER.info("=" * 70)
    
    with pg._require_conn().cursor() as cur:
        # Get raw articles with body
        cur.execute("""
            SELECT id, url, title, raw_body, published_date, source, author, 
                   views, shares, comments
            FROM raw_articles
            WHERE raw_body IS NOT NULL
            ORDER BY published_date DESC
        """)
        raw_articles = cur.fetchall()
    
    LOGGER.info(f"Found {len(raw_articles)} raw articles with body")
    
    # Extract ticker from title/body (simple heuristic)
    def extract_ticker(title: str, body: str | None = None) -> str:
        """Simple ticker extraction from title."""
        text = (title or "") + " " + (body or "")
        # Look for uppercase 1-4 letter words that look like tickers
        import re
        matches = re.findall(r'\b([A-Z]{1,5})\b', text[:500])
        for match in matches:
            if len(match) <= 5 and match not in ["THE", "AND", "FOR", "WITH", "FROM"]:
                return match
        return "UNKNOWN"
    
    staging_articles = []
    for raw_id, url, title, body, pub_date, source, author, views, shares, comments in raw_articles:
        ticker = extract_ticker(title, body)
        staging_articles.append({
            "id": f"article_{raw_id:08d}",  # Simple ID
            "raw_article_id": raw_id,
            "url": url,
            "title": title,
            "body": body,
            "published_date": pub_date,
            "source": source,
            "author": author,
            "views": views,
            "shares": shares,
            "comments": comments,
            "ready_for_ml": True,
        })
    
    if staging_articles:
        staging_ids = pg.insert_staging_articles(staging_articles)
        LOGGER.info(f"✅ Inserted {len(staging_ids)} articles into staging")
    
    # Move staging → production (simplified)
    with pg._require_conn().cursor() as cur:
        cur.execute("""
            INSERT INTO articles
                (id, raw_article_id, staging_article_id, url, title, body, 
                 published_date, crawled_date, source, ticker, 
                 views, shares, comments, created_at)
            SELECT 
                id, raw_article_id, id, url, title, body,
                published_date, NOW(), source,
                CASE 
                    WHEN title LIKE '%AAPL%' THEN 'AAPL'
                    WHEN title LIKE '%MSFT%' THEN 'MSFT'
                    WHEN title LIKE '%NVDA%' THEN 'NVDA'
                    WHEN title LIKE '%GOOGL%' THEN 'GOOGL'
                    WHEN title LIKE '%AMZN%' THEN 'AMZN'
                    ELSE 'UNKNOWN'
                END,
                views, shares, comments, NOW()
            FROM staging_articles
            WHERE ready_for_ml = TRUE
            ON CONFLICT (id) DO NOTHING
        """)
        pg._require_conn().commit()
    
    with pg._require_conn().cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM articles")
        count = cur.fetchone()[0]
    
    LOGGER.info(f"✅ Transformed {count} total articles to production")
    return count


def main() -> None:
    """Main crawl pipeline."""
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    
    LOGGER.info(f"🚀 TaTaHunts News Crawler v1")
    LOGGER.info(f"   Tickers: {len(tickers)} ({', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''})")
    LOGGER.info(f"   Skip detail: {args.skip_detail}")
    LOGGER.info(f"   Workers: {args.workers}")
    
    pg = PostgreSQLClient()
    
    try:
        # Phase 1: RSS crawl
        rss_result = crawl_rss(pg, tickers, args.sleep_ms)
        
        # Phase 2: Detail crawl
        detail_count = crawl_detail(pg, rss_result["articles"], args.workers, args.skip_detail)
        
        # Phase 3: Transform to production
        prod_count = transform_to_production(pg)
        
        LOGGER.info("=" * 70)
        LOGGER.info("✅ CRAWL COMPLETE")
        LOGGER.info("=" * 70)
        LOGGER.info(f"Total articles in DB: {prod_count}")
        LOGGER.info(f"Next: Run FinBERT encoding with: python3 run_finbert.py")
        
    except Exception as e:
        LOGGER.error(f"❌ Crawl failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pg.close()


if __name__ == "__main__":
    main()
