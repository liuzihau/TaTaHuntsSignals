"""APScheduler task scheduler for crawl and embedding pipelines."""

from __future__ import annotations

import logging
from typing import Any

from jobs.crawl_jobs import run_detail_crawl_to_staging, run_rss_ingest

LOGGER = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    BackgroundScheduler = None  # type: ignore[assignment]
    CronTrigger = None  # type: ignore[assignment]


def setup_scheduler(app_context: Any):
    """Initialize and start background scheduler with crawl + embedding tasks."""
    if BackgroundScheduler is None or CronTrigger is None:
        raise RuntimeError("APScheduler is not installed. Install requirements to use scheduler.")

    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=rss_scan_task,
        trigger=CronTrigger(hour="*/6"),  # 0, 6, 12, 18 UTC
        args=[app_context],
        id="rss_scan",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        func=detail_crawl_task,
        trigger=CronTrigger(hour="*/4"),
        args=[app_context],
        id="detail_crawl",
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        func=embedding_task,
        trigger=CronTrigger(hour=2, minute=0),  # daily 2:00 UTC
        args=[app_context],
        id="embedding",
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    LOGGER.info("Scheduler started with rss_scan, detail_crawl, embedding tasks.")
    return scheduler


def rss_scan_task(app_context: Any) -> None:
    """Fast RSS scan and raw ingest for configured tickers."""
    try:
        result = run_rss_ingest(app_context.rss_crawler, app_context.tickers)
        LOGGER.info("RSS scan result: %s", result)
    except Exception as exc:
        LOGGER.error("RSS task failed: %s", exc, exc_info=True)


def detail_crawl_task(app_context: Any) -> None:
    """Detail crawl and staging upsert task."""
    try:
        result = run_detail_crawl_to_staging(
            postgres_client=app_context.postgres,
            crawler=app_context.detail_crawler,
            limit=5000,
        )
        LOGGER.info("Detail crawl result: %s", result)
    except Exception as exc:
        LOGGER.error("Detail crawl task failed: %s", exc, exc_info=True)


def embedding_task(app_context: Any) -> None:
    """Embedding pipeline task."""
    try:
        result = app_context.embedding_job.embed_new_articles(limit=10000)
        if result.get("errors"):
            LOGGER.warning("Embedding task warnings: %s", result["errors"])
        LOGGER.info(
            "Embedding task complete: embedded=%s failed=%s",
            result.get("embedded", 0),
            result.get("failed", 0),
        )
    except Exception as exc:
        LOGGER.error("Embedding task failed: %s", exc, exc_info=True)

