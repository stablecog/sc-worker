import numpy as np
from typing import Any
import torch
from denoiser.dsp import convert_audio
import time
from typing import Tuple


def denoise_audio(
    audio: np.array, sample_rate: int, model: Any
) -> Tuple[np.ndarray, int]:
    s = time.time()
    tensor = torch.from_numpy(audio).cuda()
    tensor = tensor.view(1, -1)
    wav = convert_audio(
        wav=tensor,
        from_samplerate=sample_rate,
        to_samplerate=model.sample_rate,
        channels=model.chin,
    )
    with torch.no_grad():
        denoised_audio = model(wav[None])[0]
    arr = denoised_audio.data.cpu().numpy()
    arr = arr.reshape(-1)
    e = time.time()
    print(f"🔊 Denoised audio in: {round((e - s) * 1000)} ms 🔊")
    return arr, model.sample_rate
