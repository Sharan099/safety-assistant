import os
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from scheduler.tasks import run_crawlers_task

# Production default: real crawls. Set CRAWL_MOCK=true only for local offline dev.
CRAWL_MOCK = os.getenv("CRAWL_MOCK", "false").lower() == "true"


def start_scheduler():
    """APScheduler: weekly HEAD change-check + quarterly full crawl."""
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        func=lambda: run_crawlers_task.delay(mock=CRAWL_MOCK, full_fetch=False),
        trigger="cron",
        day_of_week=os.getenv("CRAWL_HEAD_DAY_OF_WEEK", "sun"),
        hour=int(os.getenv("CRAWL_HEAD_HOUR", "0")),
        minute=0,
        id="weekly_change_check_job",
    )

    scheduler.add_job(
        func=lambda: run_crawlers_task.delay(mock=CRAWL_MOCK, full_fetch=True),
        trigger="cron",
        month=os.getenv("CRAWL_FULL_MONTHS", "1,4,7,10"),
        day=int(os.getenv("CRAWL_FULL_DAY", "1")),
        hour=int(os.getenv("CRAWL_FULL_HOUR", "2")),
        minute=0,
        id="quarterly_full_crawl_job",
    )

    if os.getenv("CRAWL_ON_STARTUP", "false").lower() == "true":
        scheduler.add_job(
            func=lambda: run_crawlers_task.delay(mock=CRAWL_MOCK, full_fetch=False),
            trigger="date",
            id="startup_change_check_job",
        )

    try:
        scheduler.start()
        logger.info(
            "APScheduler started (mock=%s): weekly HEAD check + quarterly full crawl.",
            CRAWL_MOCK,
        )
    except Exception as e:
        logger.error(f"Failed to start APScheduler: {e}")
