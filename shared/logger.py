import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()


class LokiHandler(logging_loki.LokiHandler):
    def emit(self, record):
        if getattr(record, "raw_print", False):
            # For print statements, send the raw message
            record.msg = record.getMessage()
            record.levelname = "PRINT"
        super().emit(record)


class PrintCapturer:
    def __init__(self, logger):
        self.logger = logger
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, message):
        if message.strip():
            # Log the raw print message
            self.logger.log(logging.INFO, message.rstrip(), extra={"raw_print": True})
        # Always write to the original stdout
        self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()


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

    # Set up the custom Loki handler
    handler_loki = LokiHandler(
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

    # Redirect stdout to the PrintCapturer
    sys.stdout = PrintCapturer(root_logger)
    sys.stderr = PrintCapturer(root_logger)

    return listener
