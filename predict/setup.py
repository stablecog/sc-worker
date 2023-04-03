from typing import Any

from lingua import LanguageDetectorBuilder
from boto3_type_annotations.s3 import ServiceResource

from shared.constants import WORKER_VERSION
from models.stable_diffusion.constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    DiffusionPipeline,
)
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download_from_bucket import download_all_models_from_bucket
from models.download.download_from_hf import download_models_from_hf
import time
from models.constants import DEVICE
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from models.open_clip.constants import OPEN_CLIP_MODEL_ID
import os
from huggingface_hub import _login
from models.nllb.constants import TRANSLATOR_MODEL_ID, TRANSLATOR_CACHE


class ModelsPack:
    def __init__(
        self,
        sd_pipes: dict[str, DiffusionPipeline],
        upscaler: Any,
        translator: Any,
        open_clip: Any,
    ):
        self.sd_pipes = sd_pipes
        self.upscaler = upscaler
        self.translator = translator
        self.open_clip = open_clip


def setup(s3: ServiceResource, bucket_name: str) -> ModelsPack:
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        print(f"⏳ Logging in to HuggingFace")
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    if os.environ.get("USE_HF", "0") == "1":
        download_models_from_hf(downloadAll=False)
    else:
        download_all_models_from_bucket(s3, bucket_name)

    sd_pipes: dict[
        str,
        DiffusionPipeline,
    ] = {}

    for key in SD_MODELS:
        print(f"⏳ Loading SD model: {key}")
        pipe = DiffusionPipeline.from_pretrained(
            SD_MODELS[key]["id"],
            custom_pipeline="stable_diffusion_mega",
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
        sd_pipes[key] = pipe.to(DEVICE)
        sd_pipes[key].enable_xformers_memory_efficient_attention()
        print(f"✅ Loaded SD model: {key}")

    # For upscaler
    upscaler_args = get_args_swinir()
    upscaler_args.task = TASKS_SWINIR["Real-World Image Super-Resolution-Large"]
    upscaler_args.scale = 4
    upscaler_args.model_path = MODELS_SWINIR["real_sr"]["large"]
    upscaler_args.large_model = True
    upscaler_pipe = define_model_swinir(upscaler_args)
    upscaler_pipe.eval()
    upscaler_pipe = upscaler_pipe.to(DEVICE_SWINIR)
    upscaler = {
        "pipe": upscaler_pipe,
        "args": upscaler_args,
    }
    print("✅ Loaded upscaler")

    # For translator
    translator = {
        "tokenizer": AutoTokenizer.from_pretrained(
            TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        ),
        "model": AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        ),
        "detector": (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        ),
    }
    print("✅ Loaded translator")

    # For Open CLIP
    open_clip = {
        "model": AutoModel.from_pretrained(OPEN_CLIP_MODEL_ID).to(DEVICE),
        "processor": AutoProcessor.from_pretrained(OPEN_CLIP_MODEL_ID),
        "tokenizer": AutoTokenizer.from_pretrained(OPEN_CLIP_MODEL_ID),
    }
    print("✅ Loaded Open CLIP")

    end = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"✅ Predict setup is done in: {round((end - start))} sec.")
    print("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        sd_pipes=sd_pipes,
        upscaler=upscaler,
        translator=translator,
        open_clip=open_clip,
    )
