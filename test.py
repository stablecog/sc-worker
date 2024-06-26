import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys

# Singleton instance for logger and listener
_logger = None
_listener = None


def setup_logger():
    global _logger, _listener
    if _logger and _listener:
        return _logger, _listener

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
    _listener = logging.handlers.QueueListener(queue, handler_loki)

    # Start the listener
    _listener.start()

    # Set up the stdout handler for console logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)

    # Set up the logger
    _logger = logging.getLogger("sc-worker-logger")
    _logger.addHandler(handler)
    _logger.addHandler(stdout_handler)
    _logger.setLevel(logging.INFO)

    return _logger, _listener


def stop_listener():
    global _listener
    if _listener:
        _listener.stop()


if __name__ == "__main__":
    logger, listener = setup_logger()
    try:
        logger.info("Starting worker...")
        # Your main worker code goes here
    finally:
        stop_listener()
