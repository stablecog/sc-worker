import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio
from io import BytesIO
import torchaudio


def denoise_audio(audio: BytesIO, model: Any) -> np.ndarray:
    wav, sr = torchaudio.load(audio)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    print(wav)
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    return denoised_audio
