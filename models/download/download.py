import os
from models.stable_diffusion.helpers import download_sd_models_concurrently

def download_models():
  if os.environ.get("DOWNLOAD_SD_MODELS_ON_SETUP", "1") == "1":
    download_sd_models_concurrently()
  # For the upscaler
  print("⏳ Downloading SwinIR models...")
  download_dir = "experiments/pretrained_models"
  model_name = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
  # Check if the model is already downloaded
  if os.path.exists(os.path.join(download_dir, model_name)):
    print("✅ SwinIR models already downloaded")
  else: 
    os.system(f'wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{model_name} -P {download_dir}')
    print("✅ Downloaded SwinIR models")
