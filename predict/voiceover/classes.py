from typing import Any
from io import BytesIO


class PredictOutput:
    def __init__(self, audio_file: BytesIO):
        self.audio_file = audio_file


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
    ):
        self.outputs = outputs
