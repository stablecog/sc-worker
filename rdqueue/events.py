from typing import Optional, Set
from enum import Enum


class Status(str, Enum):
    STARTING = "starting"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"
    FAILED = "failed"

    @staticmethod
    def is_terminal(status: Optional["Status"]) -> bool:
        return status in {Status.SUCCEEDED, Status.CANCELED, Status.FAILED}


class Event(str, Enum):
    START = "start"
    OUTPUT = "output"
    LOGS = "logs"
    COMPLETED = "completed"

    @classmethod
    def default_events(cls) -> Set["Event"]:
        return {cls.START, cls.OUTPUT, cls.LOGS, cls.COMPLETED}
