import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio
from io import BytesIO
import torchaudio

from shared.helpers import numpy_to_wav_bytes


def denoise_audio(audio: BytesIO, model: Any) -> np.ndarray:
    wav, sr = torchaudio.load(audio)
    wav = convert_audio(
        wav=wav.cuda(),
        from_samplerate=sr,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    denoised_audio = numpy_to_wav_bytes(denoise_audio, model.sample_rate)
    return denoised_audio
