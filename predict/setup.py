from typing import Any

from lingua import LanguageDetectorBuilder, LanguageDetector
from boto3_type_annotations.s3 import ServiceResource

from shared.constants import WORKER_VERSION
from models.stable_diffusion.constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
)
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download_from_bucket import download_all_models_from_bucket
from models.download.download_from_hf import download_all_models_from_hf
import time
from models.constants import DEVICE
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
)
from models.clip.constants import CLIP_MODEL_ID
import os
from huggingface_hub import _login


class ModelsPack:
    def __init__(
        self,
        txt2img_pipes: dict[str, StableDiffusionPipeline],
        upscaler: Any,
        language_detector_pipe: LanguageDetector,
        clip: Any,
    ):
        self.txt2img_pipes = txt2img_pipes
        self.upscaler = upscaler
        self.language_detector_pipe = language_detector_pipe
        self.clip = clip


def setup(s3: ServiceResource, bucket_name: str) -> ModelsPack:
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        print(f"⏳ Logging in to HuggingFace")
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    if os.environ.get("USE_HF", "0") == "1":
        download_all_models_from_hf()
    else:
        download_all_models_from_bucket(s3, bucket_name)

    txt2img_pipes: dict[
        str,
        StableDiffusionPipeline,
    ] = {}

    for key in SD_MODELS:
        print(f"⏳ Loading SD model: {key}")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODELS[key]["id"],
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
        txt2img_pipes[key] = pipe.to(DEVICE)
        txt2img_pipes[key].enable_xformers_memory_efficient_attention()
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
    language_detector_pipe = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    print("✅ Loaded language detector")

    # For CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_ID)
    clip = {
        "model": clip_model,
        "processor": clip_processor,
        "tokenizer": clip_tokenizer,
    }
    print("✅ Loaded CLIP model")

    end = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"✅ Predict setup is done in: {round((end - start))} sec.")
    print("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        txt2img_pipes=txt2img_pipes,
        upscaler=upscaler,
        language_detector_pipe=language_detector_pipe,
        clip=clip,
    )
