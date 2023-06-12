from models.audio_denoiser.denoise_audio import denoise_audio
import nltk
import pytorch_seed
from bark.generation import generate_text_semantic
from bark import SAMPLE_RATE
from bark.api import semantic_to_waveform
import time
import numpy as np
from shared.helpers import numpy_to_wav_bytes
from io import BytesIO
from typing import List, Any


class GenerateVoiceoverOutputBark:
    def __init__(
        self,
        wav_bytes: BytesIO,
        sample_rate: int,
        audio_duration: float,
    ):
        self.wav_bytes = wav_bytes
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration


def generate_voiceover(
    prompt: str,
    speaker: str,
    temperature: float,
    seed: int,
    denoiser_model: Any,
    should_denoise: bool,
    should_remove_silence: bool,
) -> List[GenerateVoiceoverOutputBark]:
    start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print("⏳ Generating voiceover ⏳")

    script = prompt.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(script)

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
    audio_duration = len(np_array) / SAMPLE_RATE
    if should_denoise:
        np_array = denoise_audio(
            audio=np_array,
            sample_rate=SAMPLE_RATE,
            model=denoiser_model,
        )
    wav = numpy_to_wav_bytes(np_array, SAMPLE_RATE)
    result = GenerateVoiceoverOutputBark(
        wav_bytes=wav,
        sample_rate=SAMPLE_RATE,
        audio_duration=audio_duration,
    )

    # get duration of audio
    return [result]
