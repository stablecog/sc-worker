import nltk
import pytorch_seed
from bark.generation import (
    generate_text_semantic,
)
from bark.api import semantic_to_waveform
import time
import numpy as np


def generate_voiceover(
    prompt: str,
    speaker: str,
    temp: float,
    seed: int,
):
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
                temp=temp,
                min_eos_p=0.05,  # this controls how likely the generation is to end
                use_kv_caching=True,
            )
            audio_array = semantic_to_waveform(
                semantic_tokens, history_prompt=speaker, temp=temp
            )
        pieces += [audio_array]

    end = time.time()
    print(f"✅🎤 Generated voiceover in: {round((end - start))} sec. 🎤")
    print("//////////////////////////////////////////////////////////////////")

    result = np.concatenate(pieces)
    return result
