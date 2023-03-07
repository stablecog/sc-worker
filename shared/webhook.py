import os
import time
import traceback
from typing import Any, Callable, Set, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore

from rdqueue.events import Status, Event


class ResponseThrottler:
    def __init__(self, response_interval: float) -> None:
        self.last_sent_response_time = 0.0
        self.response_interval = response_interval

    def should_send_response(self, response: dict) -> bool:
        if Status.is_terminal(response["status"]):
            return True

        return self.seconds_since_last_response() >= self.response_interval

    def update_last_sent_response_time(self) -> None:
        self.last_sent_response_time = time.time()

    def seconds_since_last_response(self) -> float:
        return time.time() - self.last_sent_response_time


def post_webhook(url: str, data: Dict[str, Any]) -> Callable:
    # TODO: we probably don't need to create new sessions and new throttlers
    # for every prediction.
    throttler = ResponseThrottler(response_interval=0.5)

    default_session = requests_session()
    retry_session = requests_session_with_retries()

    if throttler.should_send_response(data):
        if Status.is_terminal(data["status"]):
            # For terminal updates, retry persistently
            retry_session.post(url, json=data)
        else:
            # For other requests, don't retry, and ignore any errors
            try:
                default_session.post(url, json=data)
            except requests.exceptions.RequestException:
                tb = traceback.format_exc()
                print(f"caught exception while sending webhook {tb}\n")
        throttler.update_last_sent_response_time()


def requests_session() -> requests.Session:
    session = requests.Session()
    webhook_sig = os.environ.get("WEBHOOK_SIGNATURE")
    if webhook_sig:
        session.headers["signature"] = webhook_sig

    return session


def requests_session_with_retries() -> requests.Session:
    # This session will retry requests up to 12 times, with exponential
    # backoff. In total it'll try for up to roughly 320 seconds, providing
    # resilience through temporary networking and availability issues.
    session = requests_session()
    adapter = HTTPAdapter(
        max_retries=Retry(
            total=12,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
