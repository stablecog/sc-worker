import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()


class StdoutCapture:
    def __init__(self, loki_handler):
        self.stdout = sys.stdout
        self.loki_handler = loki_handler

    def write(self, message):
        self.stdout.write(message)
        if message.strip():
            record = logging.LogRecord(
                name="stdout",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message.strip(),
                args=(),
                exc_info=None,
            )
            self.loki_handler.emit(record)

    def flush(self):
        self.stdout.flush()


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

    # Set up the root logger
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid double logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler_loki)
    root_logger.addHandler(stdout_handler)
    root_logger.setLevel(logging.INFO)

    # Capture stdout and send to Loki without formatting
    sys.stdout = StdoutCapture(handler_loki)

    return root_logger
