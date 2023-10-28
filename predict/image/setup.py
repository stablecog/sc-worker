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


def setup() -> ModelsPack:
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    download_models_from_hf(downloadAll=False)

    sd_pipes: dict[
        str,
        SDPipe | StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
    ] = {}

    for key in SD_MODELS:
        s = time.time()
        print(f"⏳ Loading SD model: {key}")

        if key == "SDXL" or key == "Waifu Diffusion XL" or key == "SSD 1B":
            refiner_vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae",
                torch_dtype=torch.float16,
                cache_dir=SD_MODEL_CACHE,
            )
            if key == "SSD 1B":
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16,
                    cache_dir=SD_MODEL_CACHE,
                )
            else:
                vae = refiner_vae

            text2img = StableDiffusionXLPipeline.from_pretrained(
                SD_MODELS[key]["id"],
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                variant=SD_MODELS[key]["variant"],
                use_safetensors=True,
                vae=vae,
                add_watermarker=False,
            )
            if "default_lora" in SD_MODELS[key]:
                lora = SD_MODELS[key]["default_lora"]
                text2img.load_lora_weights(
                    SD_MODELS[key]["id"],
                    weight_name=lora,
                )
                print(f"✅ Loaded LoRA weights: {lora}")
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                SD_MODELS[key]["refiner_id"],
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                variant=SD_MODELS[key]["variant"],
                use_safetensors=SD_MODELS[key]["use_safetensors"],
                vae=refiner_vae,
                add_watermarker=False,
            )
            refiner_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
                SD_MODELS[key]["refiner_id"],
                text_encoder_2=text2img.text_encoder_2,
                torch_dtype=SD_MODELS[key]["torch_dtype"],
                cache_dir=SD_MODEL_CACHE,
                variant=SD_MODELS[key]["variant"],
                use_safetensors=SD_MODELS[key]["use_safetensors"],
                vae=refiner_vae,
                add_watermarker=False,
            )
            text2img = text2img.to(DEVICE)
            refiner = refiner.to(DEVICE)
            refiner_inpaint = refiner_inpaint.to(DEVICE)
            img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)
            inpaint = StableDiffusionXLInpaintPipeline(**text2img.components)
            pipe = SDPipe(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=refiner,
                refiner_inpaint=refiner_inpaint,
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
            if "keep_in_cpu_when_idle" in SD_MODELS[key]:
                text2img = text2img.to("cpu", silence_dtype_warnings=True)
                print_tuple("🐌 Keep in CPU when idle", key)
            else:
                text2img = text2img.to(DEVICE)
                print_tuple("🚀 Keep in GPU", key)
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            inpaint = StableDiffusionInpaintPipeline(**text2img.components)
            pipe = SDPipe(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=None,
                refiner_inpaint=None,
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
        text2img = get_kandinsky2(
            "cuda",
            task_type="text2img",
            model_version="2.1",
            use_flash_attention=True,
            cache_dir="/app/data/kandinsky2",
        )
        inpaint = get_kandinsky2(
            "cuda",
            task_type="inpainting",
            model_version="2.1",
            use_flash_attention=True,
            cache_dir="/app/data/kandinsky2",
        )
        kandinsky = KandinskyPipe(
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
        prior = KandinskyV22PriorPipeline.from_pretrained(
            KANDINSKY_2_2_PRIOR_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(DEVICE)
        text2img = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_2_2_DECODER_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(DEVICE)
        img2img = KandinskyV22Pipeline(**text2img.components)
        inpaint = KandinskyV22InpaintPipeline.from_pretrained(
            KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(DEVICE)
        kandinsky_2_2 = KandinskyPipe_2_2(
            prior=prior,
            text2img=text2img,
            img2img=img2img,
            inpaint=inpaint,
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
