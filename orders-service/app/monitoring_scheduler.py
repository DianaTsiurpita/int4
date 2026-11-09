import logging
import os
import time

from redis import Redis
from rq import Queue, Repeat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    redis_url = os.getenv(
        "REDIS_URL", "redis://redis:6379/0"
    )
    interval = int(os.getenv(
        "MONITOR_INTERVAL_SECONDS", "3600"
    ))

    if interval < 60:
        raise ValueError(
            "MONITOR_INTERVAL_SECONDS must be at least 60"
        )

    connection = Redis.from_url(redis_url)
    queue = Queue("monitoring", connection=connection)

    job_id = "periodic-data-drift-monitoring"

    existing = queue.fetch_job(job_id)

    if existing is None:
        job = queue.enqueue(
            "app.monitoring.run_monitoring",
            job_id=job_id,
            job_timeout=600,
            result_ttl=86400,
            repeat=Repeat(times=10000, interval=interval),
        )
        logger.info(
            "Monitoring scheduled: job=%s, interval=%ss",
            job.id, interval
        )
    else:
        logger.info(
            "Monitoring job already exists: %s",
            existing.id
        )

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

