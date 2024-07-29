import os
import time
from typing import Any

import torch
from diffusers import (
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.models import AutoencoderKL
from huggingface_hub import login
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from models.aesthetics_scorer.constants import (
    AESTHETICS_SCORER_CACHE_DIR,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
)
from models.aesthetics_scorer.model import load_model as load_aesthetics_scorer_model
from models.constants import DEVICE_CPU, DEVICE_CUDA
from models.download.download_from_hf import download_swinir_models
from models.kandinsky.constants import (
    KANDINSKY_2_2_DECODER_MODEL_ID,
    KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE,
    KANDINSKY_2_2_PRIOR_MODEL_ID,
    LOAD_KANDINSKY_2_2,
)
from models.open_clip.constants import OPEN_CLIP_MODEL_CACHE, OPEN_CLIP_MODEL_ID
from models.stable_diffusion.constants import (
    SD_MODEL_CACHE,
    SD_MODELS,
)
from models.swinir.constants import DEVICE_SWINIR, MODELS_SWINIR, TASKS_SWINIR
from models.swinir.helpers import define_model_swinir, get_args_swinir
from shared.constants import WORKER_VERSION, TabulateLevels
import logging
from tabulate import tabulate
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline


class SDPipeSet:
    def __init__(
        self,
        text2img: (
            StableDiffusionPipeline
            | StableDiffusionXLPipeline
            | StableDiffusion3Pipeline
        ),
        img2img: StableDiffusionImg2ImgPipeline | None,
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
        vae: Any | None = None,
        refiner_vae: Any | None = None,
        inpaint_vae: Any | None = None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner
        self.vae = vae
        self.refiner_vae = refiner_vae
        self.inpaint_vae = inpaint_vae


class KandinskyPipeSet_2_2:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        text2img: KandinskyV22Pipeline,
        inpaint: KandinskyV22InpaintPipeline | None,
    ):
        self.prior = prior
        self.text2img = text2img
        self.inpaint = inpaint


class OpenCLIP:
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer


class AestheticsScorer:
    def __init__(self, rating_model, artifacts_model):
        self.rating_model = rating_model
        self.artifacts_model = artifacts_model


class ModelsPack:
    def __init__(
        self,
        sd_pipe_sets: dict[str, SDPipeSet],
        upscaler: Any,
        open_clip: OpenCLIP,
        kandinsky_2_2: KandinskyPipeSet_2_2,
        aesthetics_scorer: AestheticsScorer,
    ):
        self.sd_pipe_sets = sd_pipe_sets
        self.upscaler = upscaler
        self.open_clip = open_clip
        self.kandinsky_2_2 = kandinsky_2_2
        self.aesthetics_scorer = aesthetics_scorer


def auto_move_to_device(dict, key, pipe, description):
    if dict[key].get("keep_in_cpu_when_idle"):
        logging.info(f"🐌 Keep in {DEVICE_CPU} when idle: {description}")
        return pipe.to(DEVICE_CPU, silence_dtype_warnings=True)
    else:
        logging.info(f"🚀 Keep in {DEVICE_CUDA}: {description}")
        return pipe.to(DEVICE_CUDA)


def setup() -> ModelsPack:
    start = time.time()
    version_str = f"Version: {WORKER_VERSION}"
    logging.info(
        tabulate(
            [["🟡 Setup started", version_str]], tablefmt=TabulateLevels.PRIMARY.value
        )
    )

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is not None:
        login(token=hf_token)
        logging.info(f"✅ Logged in to HuggingFace")

    download_swinir_models()

    sd_pipe_sets: dict[str, SDPipeSet] = {}

    def get_saved_sd_model(model_id_key: str, model_id: str, model_type_for_class: str):
        for key in sd_pipe_sets:
            model_definition = SD_MODELS.get(key, None)
            if model_definition is None:
                continue
            relevant_model_id = model_definition.get(model_id_key, None)
            if relevant_model_id is None:
                continue
            if relevant_model_id == model_id:
                model = getattr(sd_pipe_sets[key], model_type_for_class, None)
                if model:
                    return model
        return None

    for key in SD_MODELS:
        s = time.time()
        logging.info(f"🟡 Loading SD model: {key}")

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
                    refiner_vae = AutoencoderKL.from_pretrained(
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
                    vae = AutoencoderKL.from_pretrained(
                        vae_id,
                        torch_dtype=torch.float16,
                        cache_dir=SD_MODEL_CACHE,
                    )
            args = {
                "pretrained_model_name_or_path": SD_MODELS[key]["id"],
                "torch_dtype": SD_MODELS[key]["torch_dtype"],
                "cache_dir": SD_MODEL_CACHE,
                "use_safetensors": True,
                "add_watermarker": False,
            }
            if "variant" in SD_MODELS[key]:
                args["variant"] = SD_MODELS[key]["variant"]
            if vae is not None:
                args["vae"] = vae
            text2img = StableDiffusionXLPipeline.from_pretrained(**args)

            if "default_lora" in SD_MODELS[key]:
                lora = SD_MODELS[key]["default_lora"]
                text2img.load_lora_weights(
                    SD_MODELS[key]["id"],
                    weight_name=lora,
                )
                logging.info(f"✅ Loaded LoRA weights: {lora}")

            refiner = None
            if SD_MODELS[key].get("refiner_id") is not None:
                refiner_args = {
                    "pretrained_model_name_or_path": SD_MODELS[key]["refiner_id"],
                    "torch_dtype": SD_MODELS[key]["torch_dtype"],
                    "cache_dir": SD_MODEL_CACHE,
                    "variant": SD_MODELS[key]["variant"],
                    "use_safetensors": True,
                    "add_watermarker": False,
                }
                if "variant" in SD_MODELS[key]:
                    refiner_args["variant"] = SD_MODELS[key]["variant"]
                if refiner_vae is not None:
                    refiner_args["vae"] = refiner_vae
                refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    **refiner_args
                )
                refiner = auto_move_to_device(SD_MODELS, key, refiner, f"{key} refiner")

            text2img = auto_move_to_device(SD_MODELS, key, text2img, f"{key} text2img")
            img2img = StableDiffusionXLImg2ImgPipeline(**text2img.components)

            inpaint = None
            """ inpaint = get_saved_sd_model(
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
                inpaint.to(DEVICE_CUDA) """
            pipe = SDPipeSet(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=refiner,
                refiner_vae=refiner_vae,
                vae=vae,
            )
        elif base_model == "Stable Diffusion 3":
            args = {
                "pretrained_model_name_or_path": SD_MODELS[key]["id"],
                "torch_dtype": SD_MODELS[key]["torch_dtype"],
                "cache_dir": SD_MODEL_CACHE,
                "text_encoder_3": None,
                "tokenizer_3": None,
                "safety_checker": None,
            }
            if "variant" in SD_MODELS[key]:
                args["variant"] = SD_MODELS[key]["variant"]

            text2img = StableDiffusion3Pipeline.from_pretrained(**args)
            text2img = auto_move_to_device(SD_MODELS, key, text2img, f"{key} text2img")
            img2img = StableDiffusion3Img2ImgPipeline(**text2img.components)
            inpaint = None
            pipe = SDPipeSet(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=None,
            )
        else:
            args = {
                "pretrained_model_name_or_path": SD_MODELS[key]["id"],
                "torch_dtype": SD_MODELS[key]["torch_dtype"],
                "cache_dir": SD_MODEL_CACHE,
                "safety_checker": None,
            }
            if "variant" in SD_MODELS[key]:
                args["variant"] = SD_MODELS[key]["variant"]
            text2img = StableDiffusionPipeline.from_pretrained(**args)
            text2img = auto_move_to_device(SD_MODELS, key, text2img, f"{key} text2img")
            img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
            inpaint = None

            pipe = SDPipeSet(
                text2img=text2img,
                img2img=img2img,
                inpaint=inpaint,
                refiner=None,
            )

        sd_pipe_sets[key] = pipe
        logging.info(
            f"✅ Loaded SD model: {key} | Duration: {round(time.time() - s, 1)} seconds"
        )

    # Kandinsky 2.2
    kandinsky_2_2 = None
    if LOAD_KANDINSKY_2_2:
        s = time.time()
        logging.info("🟡 Loading Kandinsky 2.2")
        kandinsky_device = DEVICE_CUDA
        if KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE:
            kandinsky_device = DEVICE_CPU
            logging.info(f"🐌 Keep in {DEVICE_CPU} when idle: Kandinsky 2.2")
        else:
            logging.info(f"🚀 Keep in {DEVICE_CUDA}: Kandinsky 2.2")
        prior = KandinskyV22PriorPipeline.from_pretrained(
            KANDINSKY_2_2_PRIOR_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(kandinsky_device)
        text2img = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_2_2_DECODER_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(kandinsky_device)
        inpaint = None
        """ KandinskyV22InpaintPipeline.from_pretrained(
            KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=SD_MODEL_CACHE,
        ).to(DEVICE_CUDA) """
        kandinsky_2_2 = KandinskyPipeSet_2_2(
            prior=prior,
            text2img=text2img,
            inpaint=inpaint,
        )
        logging.info(
            f"✅ Loaded Kandinsky 2.2 | Duration: {round(time.time() - s, 1)} seconds"
        )

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
    logging.info("✅ Loaded upscaler")

    # For OpenCLIP
    logging.info("🟡 Loading OpenCLIP")
    open_clip = OpenCLIP(
        model=AutoModel.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ).to(DEVICE_CUDA),
        processor=AutoProcessor.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ),
    )
    logging.info("✅ Loaded OpenCLIP")

    # For asthetics scorer
    logging.info("🟡 Loading Aesthetics Scorer")
    aesthetics_scorer = AestheticsScorer(
        rating_model=load_aesthetics_scorer_model(
            weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
            cache_dir=AESTHETICS_SCORER_CACHE_DIR,
            config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
        ).to(DEVICE_CUDA),
        artifacts_model=load_aesthetics_scorer_model(
            weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
            cache_dir=AESTHETICS_SCORER_CACHE_DIR,
            config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
        ).to(DEVICE_CUDA),
    )
    logging.info("✅ Loaded Aesthetics Scorer")

    end = time.time()
    logging.info("//////////////////////////////////////////////////////////////////")
    logging.info(f"✅ Predict setup is done in: {round((end - start))} sec.")
    logging.info("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        sd_pipe_sets=sd_pipe_sets,
        upscaler=upscaler,
        open_clip=open_clip,
        kandinsky_2_2=kandinsky_2_2,
        aesthetics_scorer=aesthetics_scorer,
    )
