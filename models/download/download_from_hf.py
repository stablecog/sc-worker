from models.stable_diffusion.constants import SD_MODELS_ALL, SD_MODEL_CACHE
from diffusers import StableDiffusionPipeline
import concurrent.futures
import os
from models.swinir.constants import MODEL_DIR_SWINIR, MODEL_NAME_SWINIR
from huggingface_hub import _login


def download_all_models():
    # Login to HuggingFace if there is a token
    if os.environ.get("HUGGINGFACE_TOKEN"):
        print(f"⏳ Logging in to HuggingFace")
        _login.login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        print(f"✅ Logged in to HuggingFace")
    download_sd_models_from_hf()
    download_swinir_models()


def download_sd_model_from_hf(key):
    model_id = SD_MODELS_ALL[key]["id"]
    print(f"⏳ Downloading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=SD_MODELS_ALL[key]["torch_dtype"],
        cache_dir=SD_MODEL_CACHE,
    )
    print(f"✅ Downloaded model: {key}")
    return {"key": key}


def download_sd_models_from_hf():
    for key in SD_MODELS_ALL:
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
            f"wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME_SWINIR} -P {MODEL_DIR_SWINIR}"
        )
        print("✅ Downloaded SwinIR models")


if __name__ == "__main__":
    download_all_models()
    print("✅ Downloaded all models successfully")
