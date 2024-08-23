import logging
import logging.handlers
import logging_loki
import os
import uuid
import sys
from dotenv import load_dotenv

load_dotenv()


class PrintCapture:
    def __init__(self, loki_handler):
        self.stdout = sys.stdout
        self.loki_handler = loki_handler

    def write(self, message):
        self.stdout.write(message)
        if message.strip():
            self.loki_handler.emit(
                logging.LogRecord(
                    name="print",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=message,
                    args=None,
                    exc_info=None,
                )
            )

    def flush(self):
        self.stdout.flush()


def setup_logger():
    loki_url = os.getenv("LOKI_URL")
    loki_username = os.getenv("LOKI_USERNAME")
    loki_password = os.getenv("LOKI_PASSWORD")
    worker_name = os.getenv("WORKER_NAME", str(uuid.uuid4()))

    if not all([loki_url, loki_username, loki_password]):
        raise ValueError("LOKI_URL, LOKI_USERNAME, and LOKI_PASSWORD must be set")

    handler_loki = logging_loki.LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"worker_name": worker_name, "application": "sc-worker"},
        auth=(loki_username, loki_password),
        version="1",
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(handler_loki)

    sys.stdout = PrintCapture(handler_loki)

    return logger
