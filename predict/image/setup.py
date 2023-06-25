from typing import Any

from lingua import LanguageDetectorBuilder
from boto3_type_annotations.s3 import ServiceResource
from models.nllb.constants import TRANSLATOR_CACHE
from shared.constants import WORKER_VERSION
from models.stable_diffusion.constants import (
    SD_MODEL_FOR_SAFETY_CHECKER,
    SD_MODELS,
    SD_MODEL_CACHE,
)
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download_from_bucket import download_all_models_from_bucket
from models.download.download_from_hf import download_models_from_hf
import time
from models.constants import DEVICE
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from models.open_clip.constants import OPEN_CLIP_MODEL_ID
import os
from huggingface_hub import _login
from kandinsky2 import get_kandinsky2
from functools import partial
from models.stable_diffusion.filter import forward_inspect
from diffusers import (
    KandinskyImg2ImgPipeline,
    KandinskyInpaintPipeline,
    KandinskyPriorPipeline,
    DiffusionPipeline,
)
import torch


class ModelsPack:
    def __init__(
        self,
        sd_pipes: dict[str, DiffusionPipeline],
        upscaler: Any,
        translator: Any,
        open_clip: Any,
        kandinsky: Any,
        safety_checker: Any,
    ):
        self.sd_pipes = sd_pipes
        self.upscaler = upscaler
        self.translator = translator
        self.open_clip = open_clip
        self.kandinsky = kandinsky
        self.safety_checker = safety_checker


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

    safety_checker = None

    for key in SD_MODELS:
        print(f"⏳ Loading SD model: {key}")
        pipe = DiffusionPipeline.from_pretrained(
            SD_MODELS[key]["id"],
            custom_pipeline="stable_diffusion_mega",
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
        pipe = pipe.to(DEVICE)
        sd_pipes[key] = pipe
        print(f"✅ Loaded SD model: {key}")

    # Safety checker
    print("⏳ Loading safety checker")
    safety_pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODELS[SD_MODEL_FOR_SAFETY_CHECKER]["id"],
        torch_dtype=SD_MODELS[key]["torch_dtype"],
        cache_dir=SD_MODEL_CACHE,
    )
    safety_pipe = safety_pipe.to(DEVICE)
    safety_pipe.safety_checker.forward = partial(
        forward_inspect, self=safety_pipe.safety_checker
    )
    safety_checker = {
        "checker": safety_pipe.safety_checker,
        "feature_extractor": safety_pipe.feature_extractor,
    }
    print("✅ Loaded safety checker")

    # Kandinsky
    print("⏳ Loading Kandinsky")
    kandinsky_prior = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
    )
    kandinsky_prior.to("cuda")
    kandinsky_t2i = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1",
        torch_dtype=torch.float16,
    )
    kandinsky_i2i = KandinskyImg2ImgPipeline(**kandinsky_t2i.components)
    kandinsky_inp = KandinskyInpaintPipeline(**kandinsky_t2i.components)
    kandinsky_t2i = kandinsky_t2i.to(DEVICE)

    """ kandinsky_t2i.unet.to(memory_format=torch.channels_last)
    kandinsky_t2i.unet = torch.compile(
        kandinsky_t2i.unet, mode="reduce-overhead", fullgraph=True
    ) """

    kandinsky = {
        "prior": kandinsky_prior,
        "text2img": kandinsky_t2i,
        "img2img": kandinsky_i2i,
        "inpaint": kandinsky_inp,
    }
    print("✅ Loaded Kandinsky")

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
        "detector": (
            LanguageDetectorBuilder.from_all_languages()
            .with_preloaded_language_models()
            .build()
        ),
    }
    print("✅ Loaded translator")

    # For OpenCLIP
    print("⏳ Loading OpenCLIP")
    open_clip = {
        "model": AutoModel.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        ).to(DEVICE),
        "processor": AutoProcessor.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        ),
        "tokenizer": AutoTokenizer.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=TRANSLATOR_CACHE
        ),
    }
    print("✅ Loaded OpenCLIP")

    end = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"✅ Predict setup is done in: {round((end - start))} sec.")
    print("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        sd_pipes=sd_pipes,
        upscaler=upscaler,
        translator=translator,
        open_clip=open_clip,
        kandinsky=kandinsky,
        safety_checker=safety_checker,
    )
