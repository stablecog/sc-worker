import os
from typing import Any, Callable, Set, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore


def post_webhook(url: str, data: Dict[str, Any]) -> Callable:
    retry_session = requests_session_with_retries()
    retry_session.post(url, json=data)


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
            status_forcelist=[
                x for x in requests.status_codes._codes if x not in [400, 401]
            ],
            allowed_methods=["POST"],
        )
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
