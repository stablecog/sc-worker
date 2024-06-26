import time
from shared.constants import WORKER_VERSION
from bark.generation import (
    preload_models,
)
import nltk
from typing import Any
from denoiser import pretrained
from shared.log import custom_log


class ModelsPack:
    def __init__(
        self,
        denoiser_model: Any,
    ):
        self.denoiser_model = denoiser_model


def setup() -> ModelsPack:
    start = time.time()
    custom_log(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    nltk.download("punkt")
    preload_models()

    denoiser_model = pretrained.dns64().cuda()

    pack = ModelsPack(
        denoiser_model=denoiser_model,
    )

    end = time.time()
    custom_log("//////////////////////////////////////////////////////////////////")
    custom_log(f"✅ Predict setup is done in: {round(end - start)} sec.")
    custom_log("//////////////////////////////////////////////////////////////////")

    return pack
