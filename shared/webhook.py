import os
from typing import Any, Dict, Any

import requests
import time

MAX_RETRIES = 3


def post_webhook(url: str, data: Dict[str, Any], retries: int = 0) -> int:
    """Retry a POST request to a webhook URL, return status code"""
    ret = requests.post(
        url, json=data, headers={"signature": os.environ.get("WEBHOOK_SIGNATURE")}
    )
    if ret.status_code not in [200, 400, 401] and retries < MAX_RETRIES:
        print(f"Webhook failed with status code {ret.status_code}")
        # Sleep 150ms * retries
        sleep_time = 0.15 * retries
        print(f"Sleeping {sleep_time} seconds before retrying webhook")
        time.sleep(sleep_time)
        return post_webhook(url, data, retries + 1)
    return ret.status_code
