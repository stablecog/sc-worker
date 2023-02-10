from models.stable_diffusion.constants import SD_MODELS, SD_MODEL_CACHE
from diffusers import StableDiffusionPipeline
import concurrent.futures


def download_sd_model_from_hf(key):
    model_id = SD_MODELS[key]["id"]
    print(f"⏳ Downloading model: {model_id}")
    StableDiffusionPipeline.from_pretrained(
        SD_MODELS[key]["id"],
        torch_dtype=SD_MODELS[key]["torch_dtype"],
        cache_dir=SD_MODEL_CACHE,
    )
    print(f"✅ Downloaded model: {key}")
    return {"key": key}


def download_sd_models_concurrently_from_hf():
    for key in SD_MODELS:
        download_sd_model_from_hf(key)


if __name__ == "__main__":
    download_sd_models_concurrently_from_hf()
