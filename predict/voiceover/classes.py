from typing import Any
from io import BytesIO


class PredictOutput:
    def __init__(
        self,
        audio_bytes: BytesIO,
        audio_duration: float,
        target_extension: str,
        sample_rate: int,
        remove_silence: bool,
    ):
        self.audio_bytes = audio_bytes
        self.audio_duration = audio_duration
        self.target_extension = target_extension
        self.sample_rate = sample_rate
        self.remove_silence = remove_silence


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
    ):
        self.outputs = outputs
