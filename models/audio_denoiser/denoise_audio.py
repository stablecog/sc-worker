import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio
from scipy.io import wavfile
import torchaudio


def denoise_audio(audio: np.array, sample_rate: int, model: Any) -> np.ndarray:
    wavfile.write("temp.wav", sample_rate, audio)
    wav, sr = torchaudio.load("temp.wav")
    wav = convert_audio(
        wav=wav.cuda(),
        from_samplerate=sr,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    return denoised_audio.data.cpu().numpy(), model.sample_rate
