from typing import Any
from io import BytesIO


class PredictOutput:
    def __init__(
        self,
        audio_bytes: BytesIO,
        target_extension: str,
        sample_rate: int,
    ):
        self.audio_bytes = audio_bytes
        self.target_extension = target_extension
        self.sample_rate = sample_rate


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
    ):
        self.outputs = outputs
