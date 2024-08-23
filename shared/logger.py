import logging
import logging_loki
import os
from dotenv import load_dotenv

load_dotenv()


class LokiHandler(logging_loki.LokiHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def emit(self, record):
        if record.name == "root" and record.levelno == logging.INFO:
            # For print statements, send the raw message
            self.formatter = None
        else:
            # For logger calls, use the formatted message
            self.formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        super().emit(record)


def setup_logger():
    # Fetch environment variables
    loki_url = os.getenv("LOKI_URL")
    loki_username = os.getenv("LOKI_USERNAME")
    loki_password = os.getenv("LOKI_PASSWORD")
    worker_name = os.getenv("WORKER_NAME", "default_worker")

    # Validate environment variables
    if not all([loki_url, loki_username, loki_password]):
        raise ValueError("Loki environment variables are not set properly")

    # Set up the Loki handler
    loki_handler = LokiHandler(
        url=f"{loki_url}/loki/api/v1/push",
        tags={"worker_name": worker_name, "application": "sc-worker"},
        auth=(loki_username, loki_password),
        version="1",
    )

    # Set up the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(loki_handler)
    root_logger.addHandler(console_handler)

    # Capture print statements
    print_logger = logging.getLogger("print")
    print_logger.setLevel(logging.INFO)

    def print_override(*args, sep=" ", end="\n", file=None):
        message = sep.join(map(str, args)) + end
        print_logger.info(message)
        if file:
            print(*args, sep=sep, end=end, file=file)
        else:
            __builtins__["print"](*args, sep=sep, end=end)

    __builtins__["print"] = print_override

    return root_logger


# To be called at the end of your application
def teardown_logger():
    __builtins__["print"] = __builtins__["__print__"]
