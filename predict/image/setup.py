from typing import Any

from lingua import LanguageDetectorBuilder
from boto3_type_annotations.s3 import ServiceResource
from models.kandinsky.constants import (
    KANDINSKY_2_2_DECODER_MODEL_ID,
    KANDINSKY_2_2_PRIOR_MODEL_ID,
    KANDINSKY_INPAINT_MODEL_ID,
    KANDINSKY_MODEL_ID,
    KANDINSKY_PRIOR_MODEL_ID,
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
    CLIPVisionModelWithProjection,
)
from diffusers.models import UNet2DConditionModel
from models.open_clip.constants import OPEN_CLIP_MODEL_ID
import os
from huggingface_hub import _login
from kandinsky2 import get_kandinsky2
from functools import partial
from models.stable_diffusion.filter import forward_inspect
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
    KandinskyPriorPipeline,
    KandinskyPipeline,
    KandinskyImg2ImgPipeline,
    KandinskyInpaintPipeline,
)
import torch


class SDPipe:
    def __init__(
        self,
        text2img: StableDiffusionPipeline | StableDiffusionXLPipeline,
        img2img: StableDiffusionImg2ImgPipeline | StableDiffusionImg2ImgPipeline,
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner


class KandinskyPipe:
    def __init__(
        self,
        prior: KandinskyPriorPipeline,
        text2img: KandinskyPipeline,
        img2img: KandinskyImg2ImgPipeline,
        inpaint: KandinskyInpaintPipeline,
    ):
        self.prior = prior
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint


class KandinskyPipe_2_2:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        decoder: KandinskyV22Pipeline,
    ):
        self.prior = prior
        self.decoder = decoder


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
        print(f"⏳ Logging in to HuggingFace")
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    if os.environ.get("USE_HF", "0") == "1":
        download_models_from_hf(downloadAll=False)
    else:
        download_all_models_from_bucket(s3, bucket_name)

    sd_pipes: dict[
        str,
        SDPipe | StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
    ] = {}

    for key in SD_MODELS:
        s = time.time()
        print(f"⏳ Loading SD model: {key}")

        if key == "SDXL":
            text2img = StableDiffusionXLPipeline.from_pretrained(
                SD_MODELS[key]["id"],
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                variant=SD_MODELS[key]["variant"],
                use_safetensors=True,
            )
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                SD_MODELS[key]["refiner_id"],
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                variant=SD_MODELS[key]["variant"],
                use_safetensors=True,
            )
            text2img = text2img.to(DEVICE)
            refiner = refiner.to(DEVICE)
            img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)
            pipe = SDPipe(
                text2img=text2img,
                img2img=img2img,
                inpaint=None,
                refiner=refiner,
            )
        else:
            extra_args = {}
            if SKIP_SAFETY_CHECKER == "1":
                extra_args["safety_checker"] = None
            text2img = StableDiffusionPipeline.from_pretrained(
                SD_MODELS[key]["id"],
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                **extra_args,
            )
            if (
                hasattr(SD_MODELS[key], "enable_model_cpu_offload")
                and SD_MODELS[key]["enable_model_cpu_offload"] == True
            ):
                print(f"Enabling CPU offload for: {key}")
                text2img.enable_model_cpu_offload()
            else:
                text2img = text2img.to(DEVICE)
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            inpaint = StableDiffusionInpaintPipeline(**text2img.components)
            pipe = SDPipe(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=None,
            )

        sd_pipes[key] = pipe
        print(
            f"✅ Loaded SD model: {key} | Duration: {round(time.time() - s, 1)} seconds"
        )

    # Safety checker for Kandinsky
    safety_checker = None
    if SHOULD_LOAD_KANDINSKY_SAFETY_CHECKER == "1":
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

    # Kandinsky 2.1
    kandinsky = None
    if SHOULD_LOAD_KANDINSKY_2_1 == "1":
        s = time.time()
        print("⏳ Loading Kandinsky 2.1")
        prior = KandinskyPriorPipeline.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        )
        prior = prior.to(DEVICE)
        text2img = KandinskyPipeline.from_pretrained(
            KANDINSKY_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        )
        text2img = text2img.to(DEVICE)
        img2img = KandinskyImg2ImgPipeline(**text2img.components)
        inpaint = KandinskyInpaintPipeline.from_pretrained(
            KANDINSKY_INPAINT_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        )
        inpaint = inpaint.to(DEVICE)
        kandinsky = KandinskyPipe(
            prior=prior,
            text2img=text2img,
            img2img=text2img,
            inpaint=inpaint,
        )
        print(f"✅ Loaded Kandinsky 2.1 | Duration: {round(time.time() - s, 1)} seconds")

    # Kandinsky 2.2
    kandinsky_2_2 = None
    if SHOULD_LOAD_KANDINSKY_2_2 == "1":
        s = time.time()
        print("⏳ Loading Kandinsky 2.2")
        image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(
                KANDINSKY_2_2_PRIOR_MODEL_ID,
                subfolder="image_encoder",
                cache_dir=SD_MODEL_CACHE,
            )
            .half()
            .to(DEVICE)
        )
        unet = (
            UNet2DConditionModel.from_pretrained(
                KANDINSKY_2_2_DECODER_MODEL_ID,
                subfolder="unet",
                cache_dir=SD_MODEL_CACHE,
            )
            .half()
            .to(DEVICE)
        )
        extra_args = {}
        if SKIP_SAFETY_CHECKER == "1":
            extra_args["safety_checker"] = None
        prior = KandinskyV22PriorPipeline.from_pretrained(
            KANDINSKY_2_2_PRIOR_MODEL_ID,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
            **extra_args,
        ).to(DEVICE)
        decoder = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_2_2_DECODER_MODEL_ID,
            unet=unet,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
            **extra_args,
        ).to(DEVICE)
        kandinsky_2_2 = KandinskyPipe_2_2(
            prior=prior,
            decoder=decoder,
        )
        print(f"✅ Loaded Kandinsky 2.2 | Duration: {round(time.time() - s, 1)} seconds")

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
        kandinsky_2_2=kandinsky_2_2,
        safety_checker=safety_checker,
    )
