import logging
import uuid
from tabulate import tabulate
from logging_loki import LokiQueueHandler
from multiprocessing import Queue
import os
import base64


class CustomLogger:
    _instance = None
    _worker_uuid = str(uuid.uuid4())

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CustomLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, loki_url, loki_username, loki_password):
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Generate a unique identifier (single use)
        self.app_tag = f"sc-worker-{self._worker_uuid}"

        # Create a queue for the Loki handler
        self.log_queue = Queue(-1)

        # Configure the LokiQueueHandler with basic auth
        auth_header = base64.b64encode(
            f"{loki_username}:{loki_password}".encode()
        ).decode()
        headers = {"Authorization": f"Basic {auth_header}"}
        self.loki_handler = LokiQueueHandler(
            self.log_queue,
            url=loki_url,  # Replace with your Loki URL
            tags={"application": self.app_tag},  # Single tag with unique UUID
            version="1",
            headers=headers,
        )

        # Set up the logging configuration
        self.logger = logging.getLogger("sc-worker-logger")

        # Ensure logger has no handlers (to avoid adding multiple handlers)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(self.loki_handler)

        self._initialized = True

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def info_tuple(self, a, b):
        self.info(tabulate([[a, b]], tablefmt="simple_grid"))


# Example usage
loki_url = os.getenv("LOKI_URL")
loki_username = os.getenv("LOKI_USERNAME")
loki_password = os.getenv("LOKI_PASSWORD")

if not loki_url:
    raise ValueError("LOKI_URL environment variable is not set")
if not loki_username:
    raise ValueError("LOKI_USERNAME environment variable is not set")
if not loki_password:
    raise ValueError("LOKI_PASSWORD environment variable is not set")

custom_logger = CustomLogger(loki_url, loki_username, loki_password)
