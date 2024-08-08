import os
import time

import torch
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    FluxTransformer2DModel,
    FluxPipeline,
)
from transformers import T5EncoderModel, CLIPTextModel
from diffusers.models import AutoencoderKL
from huggingface_hub import login

from models.constants import DEVICE_CPU, DEVICE_CUDA
from models.download.download_from_hf import download_swinir_models
from models.flux1.constants import (
    FLUX1_DTYPE,
    FLUX1_KEEP_IN_CPU_WHEN_IDLE,
    FLUX1_MODEL_NAME,
    FLUX1_REPO,
    FLUX1_LOAD,
)
from models.kandinsky.constants import (
    KANDINSKY_2_2_DECODER_MODEL_ID,
    KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE,
    KANDINSKY_2_2_PRIOR_MODEL_ID,
    LOAD_KANDINSKY_2_2,
)
from optimum.quanto import freeze, qfloat8, quantize
from models.stable_diffusion.constants import (
    SD_MODEL_CACHE,
    SD_MODELS,
)
from models.swinir.constants import DEVICE_SWINIR, MODELS_SWINIR, TASKS_SWINIR
from models.swinir.helpers import define_model_swinir, get_args_swinir
from predict.image.classes import (
    Flux1PipeSet,
    KandinskyPipeSet_2_2,
    ModelsPack,
    SDPipeSet,
)
from shared.constants import WORKER_VERSION, TabulateLevels
import logging
from tabulate import tabulate


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

    flux1: Flux1PipeSet = None

    if FLUX1_LOAD:
        f1_s = time.time()
        logging.info(f"🟡 Loading {FLUX1_MODEL_NAME} models")
        f1_transformer = FluxTransformer2DModel.from_single_file(
            "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8.safetensors",
            torch_dtype=FLUX1_DTYPE,
        )
        logging.info(f"🟡 Quantizing {FLUX1_MODEL_NAME} transformer to {qfloat8.name}")
        quantize(f1_transformer, weights=qfloat8)
        logging.info(f"🟡 Freezing {FLUX1_MODEL_NAME} transformer")
        freeze(f1_transformer)

        logging.info(f"🟡 Loading {FLUX1_MODEL_NAME} text_encoder_2")
        f1_text_encoder_2 = T5EncoderModel.from_pretrained(
            FLUX1_REPO, subfolder="text_encoder_2", torch_dtype=FLUX1_DTYPE
        )
        logging.info(
            f"🟡 Quantizing {FLUX1_MODEL_NAME} text_encoder_2 to {qfloat8.name}"
        )
        quantize(f1_text_encoder_2, weights=qfloat8)
        logging.info(f"🟡 Freezing {FLUX1_MODEL_NAME} text_encoder_2")
        freeze(f1_text_encoder_2)

        f1_pipe = FluxPipeline.from_pretrained(
            FLUX1_REPO, transformer=None, text_encoder_2=None, torch_dtype=FLUX1_DTYPE
        )
        f1_pipe.transformer = f1_transformer
        f1_pipe.text_encoder_2 = f1_text_encoder_2
        if FLUX1_KEEP_IN_CPU_WHEN_IDLE:
            logging.info(f"🐌 Keep in {DEVICE_CPU} when idle: {FLUX1_MODEL_NAME}")
        else:
            pipe = pipe.to(DEVICE_CUDA)
            logging.info(f"🚀 Keep in {DEVICE_CUDA}: {FLUX1_MODEL_NAME}")
        flux1 = Flux1PipeSet(
            text2img=f1_pipe,
        )
        f1_e = time.time()
        logging.info(
            f"✅ Loaded {FLUX1_MODEL_NAME} | Duration: {round(f1_e - f1_s, 1)} sec."
        )

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

    end = time.time()
    logging.info("//////////////////////////////////////////////////////////////////")
    logging.info(f"✅ Predict setup is done in: {round((end - start))} sec.")
    logging.info("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        sd_pipe_sets=sd_pipe_sets,
        upscaler=upscaler,
        kandinsky_2_2=kandinsky_2_2,
        flux1=flux1,
    )
