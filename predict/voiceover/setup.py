import time
from shared.constants import WORKER_VERSION
from bark.generation import (
    preload_models,
)
import nltk
from denoiser import pretrained


class ModelsPackVoiceover:
    def __init__(self, denoiser_model: Any):
        self.denoiser_model = denoiser_model


def setup():
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    nltk.download("punkt")
    preload_models()

    denoiser_model = pretrained.dns64().cuda()

    end = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"✅ Predict setup is done in: {round(end - start)} sec.")
    print("//////////////////////////////////////////////////////////////////")

    return ModelsPackVoiceover(
        denoiser_model=denoiser_model,
    )
