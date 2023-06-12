import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio


def denoise_audio(audio: np.array, sample_rate: int, model: Any) -> np.ndarray:
    wav = convert_audio(
        wav=audio.cuda(),
        from_samplerate=sample_rate,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    return denoised_audio
