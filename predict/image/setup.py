from typing import Any

from lingua import LanguageDetectorBuilder
from boto3_type_annotations.s3 import ServiceResource
from models.kandinsky.constants import (
    KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID,
    KANDINSKY_2_2_DECODER_MODEL_ID,
    KANDINSKY_2_2_PRIOR_MODEL_ID,
)
from models.nllb.constants import TRANSLATOR_CACHE
from shared.constants import (
    SHOULD_LOAD_KANDINSKY_2_1,
    SHOULD_LOAD_KANDINSKY_2_2,
    SHOULD_LOAD_KANDINSKY_SAFETY_CHECKER,
    SKIP_SAFETY_CHECKER,
    WORKER_VERSION,
)
from models.stable_diffusion.constants import (
    SD_MODEL_FOR_SAFETY_CHECKER,
    SD_MODELS,
    SD_MODEL_CACHE,
)
from diffusers import StableDiffusionPipeline
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download_from_bucket import download_all_models_from_bucket
from models.download.download_from_hf import download_models_from_hf
import time
from models.constants import DEVICE
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
)
from models.open_clip.constants import OPEN_CLIP_MODEL_ID
import os
from huggingface_hub import _login
from kandinsky2 import get_kandinsky2
from functools import partial
from models.stable_diffusion.filter import forward_inspect
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintPipeline,
)
from diffusers.models import AutoencoderKL
import torch

from shared.helpers import print_tuple


class SDPipe:
    def __init__(
        self,
        text2img: StableDiffusionPipeline | StableDiffusionXLPipeline,
        img2img: StableDiffusionImg2ImgPipeline | StableDiffusionImg2ImgPipeline,
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
        refiner_inpaint: StableDiffusionXLInpaintPipeline | None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner
        self.refiner_inpaint = refiner_inpaint


class KandinskyPipe:
    def __init__(
        self,
        text2img: Any,
        img2img: Any,
        inpaint: Any,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint


class KandinskyPipe_2_2:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        text2img: KandinskyV22Pipeline,
        img2img: KandinskyV22Img2ImgPipeline,
        inpaint: KandinskyV22InpaintPipeline,
    ):
        self.prior = prior
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint


class ModelsPack:
    def __init__(
        self,
        sd_pipes: dict[
            str, SDPipe | StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
        ],
        upscaler: Any,
        translator: Any,
        open_clip: Any,
        kandinsky: KandinskyPipe,
        kandinsky_2_2: KandinskyPipe_2_2,
        safety_checker: Any,
    ):
        self.sd_pipes = sd_pipes
        self.upscaler = upscaler
        self.translator = translator
        self.open_clip = open_clip
        self.kandinsky = kandinsky
        self.kandinsky_2_2 = kandinsky_2_2
        self.safety_checker = safety_checker


def setup(s3: ServiceResource, bucket_name: str) -> ModelsPack:
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    sd_pipes: dict[
        str,
        SDPipe | StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
    ] = {}

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
        open_clip=open_clip,
    )
