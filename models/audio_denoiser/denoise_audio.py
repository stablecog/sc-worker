import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio


def denoise_audio(audio: np.ndarray, sample_rate: int, model: Any) -> np.ndarray:
    print(audio, sample_rate, model.sample_rate, model.chin)
    wav = convert_audio(
        wav=audio,
        from_samplerate=sample_rate,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    print(wav)
    with torch.no_grad():
        res = model(wav)
        print(res.shape)
        denoised_audio = model(wav[None])[0]
    return denoised_audio
