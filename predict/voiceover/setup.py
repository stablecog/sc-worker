import time
from shared.constants import WORKER_VERSION
from bark.generation import (
    preload_models,
)
import nltk
from typing import Any
from denoiser import pretrained
from shared.logger import logger
from tabulate import tabulate


class ModelsPack:
    def __init__(
        self,
        denoiser_model: Any,
    ):
        self.denoiser_model = denoiser_model


def setup() -> ModelsPack:
    start = time.time()
    version_str = f"Version: {WORKER_VERSION}"
    logger.info(
        tabulate([["⏳ Setup has started", version_str]], tablefmt="double_grid")
    )

    nltk.download("punkt")
    preload_models()

    denoiser_model = pretrained.dns64().cuda()

    pack = ModelsPack(
        denoiser_model=denoiser_model,
    )

    end = time.time()
    logger.info("//////////////////////////////////////////////////////////////////")
    logger.info(f"✅ Predict setup is done in: {round(end - start)} sec.")
    logger.info("//////////////////////////////////////////////////////////////////")

    return pack
