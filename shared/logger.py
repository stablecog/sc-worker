import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()


class TerminalCapture:
    def __init__(self, original_stream, loki_handler):
        self.original_stream = original_stream
        self.loki_handler = loki_handler

    def write(self, message):
        self.original_stream.write(message)
        self.original_stream.flush()
        if message.strip():
            # Send raw message to Loki without formatting
            record = logging.LogRecord(
                name="terminal",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message.strip(),
                args=(),
                exc_info=None,
            )
            self.loki_handler.emit(record)

    def flush(self):
        self.original_stream.flush()


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
    loki_handler = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"worker_name": worker_name, "application": "sc-worker"},
        auth=(loki_username, loki_password),
        version="1",
    )

    # Set up formatter for logger calls
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    loki_handler.setFormatter(formatter)

    # Set up the root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(loki_handler)
    root_logger.setLevel(logging.INFO)

    # Capture terminal output
    sys.stdout = TerminalCapture(sys.stdout, loki_handler)
    sys.stderr = TerminalCapture(sys.stderr, loki_handler)

    return root_logger


# Function to restore original stdout and stderr
def restore_streams():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
