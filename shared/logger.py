import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys


def setup_logger():
    # Fetch environment variables
    loki_url = os.getenv("LOKI_URL")
    loki_username = os.getenv("LOKI_USERNAME")
    loki_password = os.getenv("LOKI_PASSWORD")
    app_tag = os.getenv("WORKER_NAME", str(uuid.uuid4()))

    # Validate environment variables
    if not loki_url:
        raise ValueError("LOKI_URL environment variable is not set")
    if not loki_username:
        raise ValueError("LOKI_USERNAME environment variable is not set")
    if not loki_password:
        raise ValueError("LOKI_PASSWORD environment variable is not set")

    # Set up the logging queue and handlers
    queue = Queue(-1)
    handler = logging.handlers.QueueHandler(queue)

    # Set up the Loki handler
    handler_loki = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"application": app_tag},
        auth=(loki_username, loki_password),
        version="1",
    )

    # Set up the listener to handle log entries from the queue
    listener = logging.handlers.QueueListener(queue, handler_loki)

    # Start the listener
    listener.start()

    # Set up the stdout handler for console logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)

    # Set up the logger
    logger = logging.getLogger("sc-worker-logger")
    logger.addHandler(handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.INFO)

    return logger, listener


logger, listener = setup_logger()


def stop_listener():
    listener.stop()


if __name__ == "__main__":
    try:
        logger.info("Starting worker...")
    finally:
        stop_listener()
