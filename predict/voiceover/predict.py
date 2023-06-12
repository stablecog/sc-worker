import time

from typing import List

from predict.voiceover.setup import ModelsPack
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
    temperature: float = Field(
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
    denoise_audio: bool = Field(description="Denoise the audio.", default=True)
    remove_silence: bool = Field(
        description="Remove silence from the audio.", default=True
    )

    @validator("model")
    def validate_model(cls, v):
        return return_value_if_in_list(v, models)

    @validator("output_audio_extension")
    def validate_audio_image_extension(cls, v):
        return return_value_if_in_list(v, ["wav", "mp3"])


def predict(
    input: PredictInput,
    models_pack: ModelsPack,
) -> PredictResult:
    process_start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳ Voiceover - Process started ⏳")

    if input.seed is None:
        input.seed = int.from_bytes(os.urandom(2), "big")

    settings_log_str = f"Model: {input.model} - Speaker: {input.speaker} - Temperature: {input.temperature} - Seed: {input.seed} - Extension: {input.output_audio_extension}"
    print(f"{settings_log_str}")

    voiceovers = generate_voiceover_with_bark(
        prompt=input.prompt,
        speaker=input.speaker,
        temperature=input.temperature,
        seed=input.seed,
        denoiser_model=models_pack.denoiser_model,
        denoise_audio=input.denoise_audio,
        remove_silence=input.remove_silence,
    )

    outputs: List[PredictOutput] = [None] * len(voiceovers)

    for i, voiceover in enumerate(voiceovers):
        outputs[i] = PredictOutput(
            audio_bytes=voiceover.wav_bytes,
            audio_duration=voiceover.audio_duration,
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
