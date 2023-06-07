from denoiser.dsp import convert_audio
from numpy import ndarray
import torch
from typing import Any
import time


def denoise_audio(wav: ndarray, sample_rate: int, model: Any):
    start = time.time()
    print("Denoising audio...")
    converted_wav = convert_audio(wav, sample_rate, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(converted_wav[None])[0]
    end = time.time()
    print(f"Denoised audio in: {round(end - start, 2)} sec.")
    return denoised
