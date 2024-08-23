import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys
from dotenv import load_dotenv
import io

load_dotenv()


class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = io.StringIO()

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def setup_logger():
    # Fetch environment variables
    loki_url = os.getenv("LOKI_URL")
    loki_username = os.getenv("LOKI_USERNAME")
    loki_password = os.getenv("LOKI_PASSWORD")
    worker_name = os.getenv("WORKER_NAME", str(uuid.uuid4()))

    # Validate environment variables
    if not loki_url:
        raise ValueError("LOKI_URL environment variable is not set")
    if not loki_username:
        raise ValueError("LOKI_USERNAME environment variable is not set")
    if not loki_password:
        raise ValueError("LOKI_PASSWORD environment variable is not set")

    # Set up the logging queue and handler
    queue = Queue(-1)
    queue_handler = logging.handlers.QueueHandler(queue)

    # Set up the Loki handler
    handler_loki = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"worker_name": worker_name, "application": "sc-worker"},
        auth=(loki_username, loki_password),
        version="1",
    )

    # Set up the stdout handler for console logging
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stdout_handler.setFormatter(formatter)

    # Set up the listener to handle log entries from the queue
    listener = logging.handlers.QueueListener(queue, handler_loki, stdout_handler)

    # Start the listener
    listener.start()

    # Set up the root logger
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid double logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    # Redirect stdout and stderr to the logger
    sys.stdout = LoggerWriter(root_logger, logging.INFO)
    sys.stderr = LoggerWriter(root_logger, logging.ERROR)

    return listener


# Usage example
logger = logging.getLogger(__name__)
listener = setup_logger()

try:
    # Your main code here
    logger.info("This is a log message")
    print("This is a print statement")
finally:
    # Stop the listener when your program is done
    listener.stop()
