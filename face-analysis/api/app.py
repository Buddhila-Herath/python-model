import logging
import threading
from time import sleep

from fastapi import FastAPI

from api.job_store import maintain_jobs
from api.routes import router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _cleanup_loop() -> None:
    while True:
        maintain_jobs()
        sleep(60)


_cleanup_started = False


app = FastAPI(title="Face Analysis API", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def start_cleanup() -> None:
    global _cleanup_started
    if _cleanup_started:
        return

    # NOTE: In multi-worker setups, each worker process may spawn its own cleanup thread.
    thread = threading.Thread(target=_cleanup_loop, daemon=True)
    thread.start()
    _cleanup_started = True
    logger.info("Started background cleanup thread")
