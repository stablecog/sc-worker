import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio
from scipy.io import wavfile
import torchaudio


def denoise_audio(audio: np.array, sample_rate: int, model: Any) -> np.ndarray:
    reshaped_data = audio.reshape(-1, 2)
    transposed_data = reshaped_data.T
    tensor_data = torch.from_numpy(transposed_data).cuda()
    wav = convert_audio(
        wav=tensor_data,
        from_samplerate=sample_rate,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    return denoised_audio.data.cpu().numpy(), model.sample_rate
