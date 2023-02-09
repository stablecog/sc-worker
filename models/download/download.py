import os
from models.stable_diffusion.helpers import download_sd_models_concurrently
from boto3_type_annotations.s3 import ServiceResource
from models.swinir.constants import MODEL_DIR_SWINIR, MODEL_NAME_SWINIR


def download_models(s3: ServiceResource, bucket_name: str):
    if os.environ.get("DOWNLOAD_SD_MODELS_ON_SETUP", "1") == "1":
        download_sd_models_concurrently(s3, bucket_name)
    # For the upscaler
    print("⏳ Downloading SwinIR models...")
    # Check if the model is already downloaded
    if os.path.exists(os.path.join(MODEL_DIR_SWINIR, MODEL_NAME_SWINIR)):
        print("✅ SwinIR models already downloaded")
    else:
        os.system(
            f"wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{MODEL_NAME_SWINIR} -P {MODEL_DIR_SWINIR}"
        )
        print("✅ Downloaded SwinIR models")
