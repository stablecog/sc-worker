import os
from models.stable_diffusion.helpers import download_sd_models_concurrently
from boto3_type_annotations.s3 import ServiceResource


def download_models(s3: ServiceResource, bucket_name: str):
    if os.environ.get("DOWNLOAD_SD_MODELS_ON_SETUP", "1") == "1":
        download_sd_models_concurrently(s3, bucket_name)
    # For the upscaler
    print("⏳ Downloading SwinIR models...")
    download_dir = "experiments/pretrained_models"
    model_name = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    # Check if the model is already downloaded
    if os.path.exists(os.path.join(download_dir, model_name)):
        print("✅ SwinIR models already downloaded")
    else:
        os.system(
            f"wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_name} -P {download_dir}"
        )
        print("✅ Downloaded SwinIR models")
