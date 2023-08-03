from models.stable_diffusion.constants import SD_MODELS_ALL, SD_MODELS, SD_MODEL_CACHE
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
import concurrent.futures
import os
from models.swinir.constants import MODEL_DIR_SWINIR, MODEL_NAME_SWINIR
from huggingface_hub import _login
import time


def download_models_from_hf(downloadAll=True):
    # Login to HuggingFace if there is a token
    if os.environ.get("HUGGINGFACE_TOKEN"):
        print(f"⏳ Logging in to HuggingFace")
        _login.login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        print(f"✅ Logged in to HuggingFace")
    download_sd_models_from_hf(downloadAll=downloadAll)
    download_swinir_models()


def download_sd_model_from_hf(key):
    model_id = SD_MODELS_ALL[key]["id"]
    s = time.time()
    print(f"⏳ Downloading model: {model_id}")
    if key == "SDXL" or key == "Waifu Diffusion XL":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SD_MODELS[key]["id"],
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
            variant=SD_MODELS[key]["variant"],
            use_safetensors=True,
        )
        pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            SD_MODELS[key]["refiner_id"],
            torch_dtype=SD_MODELS[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
            variant=SD_MODELS[key]["variant"],
            use_safetensors=True,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=SD_MODELS_ALL[key]["torch_dtype"],
            cache_dir=SD_MODEL_CACHE,
        )
    print(f"✅ Downloaded model: {key} | Duration: {round(time.time() - s, 1)} seconds")
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
    print("⏳ Downloading SwinIR models...")
    if os.path.exists(os.path.join(MODEL_DIR_SWINIR, MODEL_NAME_SWINIR)):
        print("✅ SwinIR models already downloaded")
    else:
        os.system(
            f"wget -q https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME_SWINIR} -P {MODEL_DIR_SWINIR}"
        )
        print("✅ Downloaded SwinIR models")


if __name__ == "__main__":
    download_models_from_hf()
    print("✅ Downloaded all models successfully")
