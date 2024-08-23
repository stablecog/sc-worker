import logging
import logging.handlers
import logging_loki
from multiprocessing import Queue
import os
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()


class PrintCapturer:
    def __init__(self, queue, original_stream):
        self.queue = queue
        self.original_stream = original_stream

    def write(self, message):
        if message.strip():
            # Send the raw message to the queue
            record = logging.LogRecord(
                name="print",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None,
            )
            self.queue.put_nowait(record)
        # Always write to the original stream
        self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()


class LokiHandler(logging_loki.LokiHandler):
    def emit(self, record):
        if record.name == "print":
            # For print statements, send the raw message
            self.send_message(record.msg)
        else:
            # For other log records, use the standard formatting
            super().emit(record)

    def send_message(self, message):
        self._write_message(message)


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

    # Set up the logging queue
    queue = Queue(-1)

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
    root_logger.addHandler(logging.handlers.QueueHandler(queue))
    root_logger.setLevel(logging.INFO)

    # Capture stdout and stderr
    sys.stdout = PrintCapturer(queue, sys.stdout)
    sys.stderr = PrintCapturer(queue, sys.stderr)

    return listener
