import time

from typing import List

from predict.voiceover.setup import ModelsPack
from .classes import PredictOutput, PredictResult, RemoveSilenceParams
from .constants import models, models_speakers
from pydantic import BaseModel, Field, validator
from shared.helpers import return_value_if_in_list
from models.bark.generate import (
    generate_voiceover as generate_voiceover_with_bark,
)
import os
from tabulate import tabulate


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
        default=models_speakers[models[0]][0],
    )
    seed: int = Field(description="Seed for the voiceover.", default=None)
    output_audio_extension: str = Field(
        description="Audio extention for the output. Can be 'mp3' or 'wav'.",
        default="mp3",
    )
    denoise_audio: bool = Field(description="Denoise the audio.", default=True)
    normalize_audio_loudness: bool = Field(
        description="Normalize the loudness of the audio.", default=True
    )
    remove_silence: bool = Field(
        description="Remove silence from the audio.", default=True
    )
    remove_silence_min_silence_len: int = Field(
        description="Minimum silence length in milliseconds.",
        default=500,
    )
    remove_silence_silence_thresh: int = Field(
        description="Silence threshold in dB.",
        default=-45,
    )
    remove_silence_keep_silence_len: int = Field(
        description="Add back silence length in milliseconds.",
        default=250,
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

    log_table = [
        ["Prompt", input.prompt],
        ["Model", input.model],
        ["Speaker", input.speaker],
        ["Temperature", input.temperature],
        ["Seed", input.seed],
        ["Output audio extension", input.output_audio_extension],
        ["Denoise audio", input.denoise_audio],
        ["Normalize audio loudness", input.normalize_audio_loudness],
        ["Remove silence", input.remove_silence],
        ["RS min silence len", input.remove_silence_min_silence_len],
        ["RS silence thresh", input.remove_silence_silence_thresh],
        ["RS keep silence len", input.remove_silence_keep_silence_len],
    ]

    print(tabulate([["🎤 Generation 🟡", "Started"]] + log_table, tablefmt="double_grid"))

    voiceover_start = time.time()
    voiceovers = generate_voiceover_with_bark(
        prompt=input.prompt,
        speaker=input.speaker,
        temperature=input.temperature,
        seed=input.seed,
        denoiser_model=models_pack.denoiser_model,
        should_denoise=input.denoise_audio,
        normalize_audio_loudness=input.normalize_audio_loudness,
    )
    voiceover_end = time.time()

    print(
        tabulate(
            [
                [
                    "🎤 Geneneration 🟢",
                    f"{round(voiceover_end - voiceover_start, 2)} sec.",
                ]
            ]
            + log_table,
            tablefmt="double_grid",
        )
    )

    outputs: List[PredictOutput] = [None] * len(voiceovers)

    for i, voiceover in enumerate(voiceovers):
        outputs[i] = PredictOutput(
            audio_bytes=voiceover.wav_bytes,
            sample_rate=voiceover.sample_rate,
            target_extension=input.output_audio_extension,
            remove_silence_params=RemoveSilenceParams(
                should_remove=input.remove_silence,
                min_silence_len=input.remove_silence_min_silence_len,
                silence_thresh=input.remove_silence_silence_thresh,
                keep_silence_len=input.remove_silence_keep_silence_len,
            ),
            normalize_audio_loudness=input.normalize_audio_loudness,
            speaker=input.speaker,
            prompt=input.prompt,
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
