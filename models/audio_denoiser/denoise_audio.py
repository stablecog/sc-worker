import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio


def denoise_audio(audio: np.ndarray, sample_rate: int, model: Any) -> np.ndarray:
    wav = convert_audio(audio, sample_rate, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    return denoised_audio
