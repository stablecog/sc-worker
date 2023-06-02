import os
import time
from shared.constants import WORKER_VERSION
from bark.generation import (
    preload_models,
)
import nltk


def setup():
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    nltk.download("punkt")
    preload_models()

    end = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"✅ Predict setup is done in: {round((end - start))} sec.")
    print("//////////////////////////////////////////////////////////////////")

    return
