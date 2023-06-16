from typing import Any
from io import BytesIO


class RemoveSilenceParams:
    def __init__(
        self,
        should_remove: bool,
        min_silence_len: int,
        silence_thresh: float,
        keep_silence_len: int,
    ):
        self.should_remove = should_remove
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence_len = keep_silence_len


class PredictOutput:
    def __init__(
        self,
        audio_bytes: BytesIO,
        target_extension: str,
        sample_rate: int,
        remove_silence_params: RemoveSilenceParams,
    ):
        self.audio_bytes = audio_bytes
        self.target_extension = target_extension
        self.sample_rate = sample_rate
        self.remove_silence_params = remove_silence_params


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
    ):
        self.outputs = outputs
