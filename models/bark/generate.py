from models.audio_denoiser.denoise_audio import denoise_audio
import nltk
import pytorch_seed
from bark.generation import generate_text_semantic
from bark import SAMPLE_RATE
from bark.api import semantic_to_waveform
import time
import numpy as np
from shared.helpers import do_normalize_audio_loudness, numpy_to_wav_bytes
from io import BytesIO
from typing import List, Any
import os


class GenerateVoiceoverOutputBark:
    def __init__(
        self,
        wav_bytes: BytesIO,
        sample_rate: int,
    ):
        self.wav_bytes = wav_bytes
        self.sample_rate = sample_rate


def generate_voiceover(
    prompt: str,
    speaker: str,
    temperature: float,
    seed: int,
    denoiser_model: Any,
    should_denoise: bool,
    normalize_audio_loudness: bool,
) -> List[GenerateVoiceoverOutputBark]:
    start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print("⏳ Generating voiceover ⏳")

    script = prompt.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(script)

    if speaker.startswith("c_"):
        path = os.path.dirname(os.path.abspath(__file__))
        custom_speakers_dir = os.path.join(path, "custom_speakers")
        speaker = os.path.join(custom_speakers_dir, speaker + ".npz")

    pieces = []
    stc_len = len(sentences)
    for i, sentence in enumerate(sentences):
        print(f"-- Generating: {i+1}/{stc_len} --")
        with pytorch_seed.SavedRNG(seed):
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt=speaker,
                temp=temperature,
                min_eos_p=0.05,  # this controls how likely the generation is to end
                use_kv_caching=True,
            )
            audio_array = semantic_to_waveform(
                semantic_tokens, history_prompt=speaker, temp=temperature
            )
        pieces += [audio_array]

    end = time.time()
    print(f"🎤 Generated voiceover in: {round(end - start, 2)} sec. 🎤")
    print("//////////////////////////////////////////////////////////////////")

    np_array = np.concatenate(pieces)
    sample_rate = SAMPLE_RATE

    if normalize_audio_loudness:
        np_array = do_normalize_audio_loudness(
            audio_arr=np_array, sample_rate=sample_rate
        )

    if should_denoise:
        arr, sr = denoise_audio(
            audio=np_array,
            sample_rate=SAMPLE_RATE,
            model=denoiser_model,
        )
        np_array = arr
        sample_rate = sr

    wav = numpy_to_wav_bytes(np_array, sample_rate)

    result = GenerateVoiceoverOutputBark(
        wav_bytes=wav,
        sample_rate=SAMPLE_RATE,
    )

    return [result]
