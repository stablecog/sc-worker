from typing import Any


class PredictOutput:
    def __init__(self, audio_file: Any):
        self.audio_file = audio_file


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
    ):
        self.outputs = outputs
