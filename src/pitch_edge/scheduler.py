"""Local scheduler (APScheduler). Phase 3 replaces this with Cloud Scheduler + Cloud Run jobs.

Jobs: nightly data refresh + backtest, hourly news/prediction-market discovery. Every job
only *records* data or proposals; the approval gate is untouched.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from pitch_edge.config import get_settings
from pitch_edge.data.ingest import (
    ingest_news,
    ingest_pmxt_soccer_markets,
    ingest_referee_announcements,
    ingest_wikipedia_attention,
)
from pitch_edge.data.storage import Warehouse

logger = logging.getLogger(__name__)


def job_news_and_markets() -> None:
    """Hourly: news, the Polymarket soccer-market dim table (see `pmxt_archive.py` — the old
    live-snapshot `polymarket`/`kalshi` jobs were removed, replaced by the PMXT archive
    backfill, which runs separately via `scripts/stage_pmxt_backfill.py`), referee-appointment
    pages (announcement timestamps) and recent Wikipedia pageviews."""
    with Warehouse(get_settings().db_path) as wh:
        ingest_news(wh)
        ingest_pmxt_soccer_markets(wh)
        ingest_referee_announcements(wh)
        ingest_wikipedia_attention(wh, recent_days=14)


def job_full_refresh() -> None:
    from pitch_edge.pipeline import full_refresh

    full_refresh(fast=True)


def build_scheduler(blocking: bool = True):
    sched = BlockingScheduler() if blocking else BackgroundScheduler()
    sched.add_job(job_news_and_markets, "interval", hours=1, id="news_markets", replace_existing=True)
    sched.add_job(job_full_refresh, "cron", hour=3, minute=30, id="nightly_refresh", replace_existing=True)
    return sched


def run_forever() -> None:  # pragma: no cover - long-running process
    logging.basicConfig(level=logging.INFO)
    sched = build_scheduler(blocking=True)
    logger.info("Scheduler started: %s", [j.id for j in sched.get_jobs()])
    sched.start()
