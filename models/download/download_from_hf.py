from models.stable_diffusion.constants import SD_MODELS_ALL, SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
import concurrent.futures
import os
from models.swinir.constants import MODEL_DIR_SWINIR, MODEL_NAME_SWINIR
from huggingface_hub import login
import time
import logging


def download_models_from_hf(downloadAll=True):
    # Login to HuggingFace if there is a token
    token = os.environ.get("HF_TOKEN", None)
    if token is not None:
        logging.info(f"🟡 Logging in to HuggingFace")
        login(token=token)
        logging.info(f"✅ Logged in to HuggingFace")
    download_sd_models_from_hf(downloadAll=downloadAll)
    download_swinir_models()


def download_sd_model_from_hf(key):
    model_id = SD_MODELS_ALL[key]["id"]
    s = time.time()
    logging.info(f"🟡 Downloading model: {model_id}")
    if key == "SDXL" or key == "Waifu Diffusion XL":
        args = {
            "pretrained_model_name_or_path": SD_MODELS[key]["id"],
            "torch_dtype": SD_MODELS[key]["torch_dtype"],
            "cache_dir": SD_MODEL_CACHE,
            "variant": SD_MODELS[key]["variant"],
            "use_safetensors": True,
        }
        if "variant" in SD_MODELS[key]:
            args["variant"] = SD_MODELS[key]["variant"]
        pipe = StableDiffusionXLPipeline.from_pretrained(
            **args,
        )
        refiner_args = {
            "pretrained_model_name_or_path": SD_MODELS[key]["refiner_id"],
            "torch_dtype": SD_MODELS[key]["torch_dtype"],
            "cache_dir": SD_MODEL_CACHE,
            "use_safetensors": True,
        }
        if "variant" in SD_MODELS[key]:
            refiner_args["variant"] = SD_MODELS[key]["variant"]
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            **refiner_args,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=SD_MODELS_ALL[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
    logging.info(
        f"✅ Downloaded model: {key} | Duration: {round(time.time() - s, 1)} seconds"
    )
    return {"key": key}


def download_sd_models_from_hf(downloadAll=True):
    models = SD_MODELS_ALL if downloadAll else SD_MODELS
    for key in models:
        download_sd_model_from_hf(key)


def download_sd_models_concurrently_from_hf():
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [
            executor.submit(download_sd_model_from_hf, key) for key in SD_MODELS_ALL
        ]
        # Wait for all tasks to complete
        results = [
            task.result() for task in concurrent.futures.as_completed(download_tasks)
        ]
    executor.shutdown(wait=True)


def download_swinir_models():
    logging.info("🟡 Downloading SwinIR models...")
    if os.path.exists(os.path.join(MODEL_DIR_SWINIR, MODEL_NAME_SWINIR)):
        logging.info("✅ SwinIR models already downloaded")
    else:
        os.system(
            f"wget -q https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME_SWINIR} -P {MODEL_DIR_SWINIR}"
        )
        logging.info("✅ Downloaded SwinIR models")


if __name__ == "__main__":
    download_models_from_hf()
    logging.info("✅ Downloaded all models successfully")
