import time

from typing import List
from .classes import PredictOutput, PredictResult
from .constants import models, modelsSpeakers
from pydantic import BaseModel, Field, validator
from shared.helpers import return_value_if_in_list
from models.bark.generate import (
    generate_voiceover as generate_voiceover_with_bark,
)
import os


class PredictInput(BaseModel):
    prompt: str = Field(description="Prompt for the voiceover.", default="")
    temp: float = Field(
        description="Temperature for the speech.",
        ge=0,
        le=1,
        default=0.7,
    )
    speaker: str = Field(
        description="Speaker for the voiceover.",
        default=models[0],
    )
    model: str = Field(
        description="Model for the voiceover.",
        default=modelsSpeakers[models[0]][0],
    )
    seed: int = Field(description="Seed for the voiceover.", default=None)
    output_audio_extension: str = Field(
        description="Audio extention for the output. Can be 'mp3' or 'wav'.",
        default="mp3",
    )

    @validator("model")
    def validate_model(cls, v):
        return return_value_if_in_list(v, models)

    @validator("output_audio_extension")
    def validate_audio_image_extension(cls, v):
        return return_value_if_in_list(v, ["wav", "mp3"])


def predict(
    input: PredictInput,
) -> PredictResult:
    process_start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳ Voiceover - Process started ⏳")

    if input.seed is None:
        input.seed = int.from_bytes(os.urandom(2), "big")

    voiceovers = generate_voiceover_with_bark(
        prompt=input.prompt,
        speaker=input.speaker,
        temp=input.temp,
        seed=input.seed,
    )

    outputs: List[PredictOutput] = [None] * len(voiceovers)

    for i, voiceover in enumerate(voiceovers):
        outputs[i] = PredictOutput(
            audio_bytes=voiceover.wav_bytes,
            sample_rate=voiceover.sample_rate,
            target_extension=input.output_audio_extension,
        )

    result = PredictResult(
        outputs=outputs,
    )

    process_end = time.time()
    print(
        f"✅ Voiceover - Process completed in: {round(process_end - process_start, 2)} sec. ✅"
    )
    print("//////////////////////////////////////////////////////////////////")

    return result
