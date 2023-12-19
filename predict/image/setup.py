from typing import Any

from lingua import LanguageDetectorBuilder
from models.aesthetics_scorer.constants import (
    AESTHETICS_SCORER_CACHE_DIR,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
)
from models.aesthetics_scorer.model import load_model as load_aesthetics_scorer_model
from models.kandinsky.constants import (
    KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID,
    KANDINSKY_2_2_DECODER_MODEL_ID,
    KANDINSKY_2_2_PRIOR_MODEL_ID,
    LOAD_KANDINSKY_2_1,
    LOAD_KANDINSKY_2_2,
)
from models.nllb.constants import TRANSLATOR_CACHE
from shared.constants import (
    SKIP_SAFETY_CHECKER,
    WORKER_VERSION,
)
from models.stable_diffusion.constants import (
    SD_MODEL_FOR_SAFETY_CHECKER,
    SD_MODELS,
    SD_MODEL_CACHE,
)
from diffusers import StableDiffusionPipeline, AutoPipelineForInpainting
from models.swinir.helpers import get_args_swinir, define_model_swinir
from models.swinir.constants import TASKS_SWINIR, MODELS_SWINIR, DEVICE_SWINIR
from models.download.download_from_hf import (
    download_swinir_models,
)
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


class SDPipeSet:
    def __init__(
        self,
        text2img: StableDiffusionPipeline | StableDiffusionXLPipeline,
        img2img: StableDiffusionImg2ImgPipeline | StableDiffusionImg2ImgPipeline,
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
        vae: Any | None = None,
        inpaint_vae: Any | None = None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner
        self.vae = vae
        self.inpaint_vae = inpaint_vae


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
        sd_pipes: dict[str, SDPipeSet],
        upscaler: Any,
        translator: Any,
        open_clip: Any,
        kandinsky: KandinskyPipe,
        kandinsky_2_2: KandinskyPipe_2_2,
        safety_checker: Any,
        aesthetics_scorer: Any,
    ):
        self.sd_pipes = sd_pipes
        self.upscaler = upscaler
        self.translator = translator
        self.open_clip = open_clip
        self.kandinsky = kandinsky
        self.kandinsky_2_2 = kandinsky_2_2
        self.safety_checker = safety_checker
        self.aesthetics_scorer = aesthetics_scorer


def setup() -> ModelsPack:
    start = time.time()
    print(f"⏳ Setup has started - Version: {WORKER_VERSION}")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if hf_token is not None:
        _login.login(token=hf_token)
        print(f"✅ Logged in to HuggingFace")

    download_swinir_models()

    sd_pipes: dict[str, SDPipeSet] = {}

    def get_saved_sd_model(model_id_key: str, model_id: str, model_type_for_class: str):
        for key in sd_pipes:
            model_definition = SD_MODELS.get(key, None)
            if model_definition is None:
                continue
            relevant_model_id = model_definition.get(model_id_key, None)
            if relevant_model_id is None:
                continue
            if relevant_model_id == model_id:
                model = getattr(sd_pipes[key], model_type_for_class, None)
                if model:
                    return model
        return None

    for key in SD_MODELS:
        s = time.time()
        print(f"⏳ Loading SD model: {key}")

        base_model = SD_MODELS[key].get("base_model", None)

        if base_model == "SDXL":
            refiner_vae = None
            vae = None
            refiner_vae_id = SD_MODELS[key].get("refiner_vae", None)
            vae_id = SD_MODELS[key].get("vae", None)
            if refiner_vae_id is not None:
                refiner_vae = get_saved_sd_model(
                    model_id_key="refiner_vae",
                    model_id=refiner_vae_id,
                    model_type_for_class="refiner_vae",
                )
                if refiner_vae == None:
                    AutoencoderKL.from_pretrained(
                        refiner_vae_id,
                        torch_dtype=torch.float16,
                        cache_dir=SD_MODEL_CACHE,
                    )
            if key == "SDXL":
                vae = refiner_vae
            elif vae_id is not None:
                vae = get_saved_sd_model(
                    model_id_key="vae",
                    model_id=vae_id,
                    model_type_for_class="vae",
                )
                if vae == None:
                    AutoencoderKL.from_pretrained(
                        vae_id,
                        torch_dtype=torch.float16,
                        cache_dir=SD_MODEL_CACHE,
                    )
            args = {
                "pretrained_model_name_or_path": SD_MODELS[key]["id"],
                "torch_dtype": SD_MODELS[key]["torch_dtype"],
                "cache_dir": SD_MODEL_CACHE,
                "variant": SD_MODELS[key]["variant"],
                "use_safetensors": True,
                "add_watermarker": False,
            }
            if vae is not None:
                args["vae"] = vae
            text2img = StableDiffusionXLPipeline.from_pretrained(**args)

            if "default_lora" in SD_MODELS[key]:
                lora = SD_MODELS[key]["default_lora"]
                text2img.load_lora_weights(
                    SD_MODELS[key]["id"],
                    weight_name=lora,
                )
                print(f"✅ Loaded LoRA weights: {lora}")

            refiner = None
            if (
                "refiner_id" in SD_MODELS[key]
                and SD_MODELS[key]["refiner_id"] is not None
            ):
                refiner_args = {
                    "pretrained_model_name_or_path": SD_MODELS[key]["refiner_id"],
                    "torch_dtype": SD_MODELS[key]["torch_dtype"],
                    "cache_dir": SD_MODEL_CACHE,
                    "variant": SD_MODELS[key]["variant"],
                    "use_safetensors": True,
                    "add_watermarker": False,
                }
                if refiner_vae is not None:
                    refiner_args["vae"] = refiner_vae
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    SD_MODELS[key]["refiner_id"],
                    torch_dtype=SD_MODELS[key]["torch_dtype"],
                    cache_dir=SD_MODEL_CACHE,
                    variant=SD_MODELS[key]["variant"],
                    use_safetensors=True,
                    vae=refiner_vae,
                    add_watermarker=False,
                )
            text2img = text2img.to(DEVICE)
            if refiner is not None:
                refiner = refiner.to(DEVICE)
            img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)

            inpaint = get_saved_sd_model(
                model_id_key="inpaint_id",
                model_id=SD_MODELS[key]["inpaint_id"],
                model_type_for_class="inpaint",
            )
            if inpaint is None:
                inpaint = AutoPipelineForInpainting.from_pretrained(
                    SD_MODELS[key]["inpaint_id"],
                    torch_dtype=SD_MODELS[key]["torch_dtype"],
                    cache_dir=SD_MODEL_CACHE,
                    variant=SD_MODELS[key]["variant"],
                    use_safetensors=True,
                    add_watermarker=False,
                )
                inpaint.to(DEVICE)
            pipe = SDPipeSet(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=refiner,
                refiner_vae=refiner_vae,
                vae=vae,
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
            inpaint = get_saved_sd_model(
                model_id_key="inpaint_id",
                model_id=SD_MODELS[key]["inpaint_id"],
                model_type_for_class="inpaint",
            )
            if inpaint is None:
                inpaint = StableDiffusionInpaintPipeline(**text2img.components)
                pipe = SDPipeSet(
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
    if SKIP_SAFETY_CHECKER == "1":
        safety_checker = None
    else:
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
    if LOAD_KANDINSKY_2_1:
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
    if LOAD_KANDINSKY_2_2:
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

    # For asthetics scorer
    print("⏳ Loading Aesthetics Scorer")
    rating_model = load_aesthetics_scorer_model(
        weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
        cache_dir=AESTHETICS_SCORER_CACHE_DIR,
        config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
    ).to(DEVICE)
    artifact_model = load_aesthetics_scorer_model(
        weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
        cache_dir=AESTHETICS_SCORER_CACHE_DIR,
        config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
    ).to(DEVICE)
    aesthetics_scorer = {
        "rating_model": rating_model,
        "artifact_model": artifact_model,
    }
    print("✅ Loaded Aesthetics Scorer")

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
        aesthetics_scorer=aesthetics_scorer,
    )
