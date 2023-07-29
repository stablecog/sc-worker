import os
import torch
from models.constants import DEVICE

TASKS_SWINIR = {"Real-World Image Super-Resolution-Large": "real_sr"}

MODEL_DIR_SWINIR = "/app/data/upscalers-cache/swinir"
MODEL_NAME_SWINIR = "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"

MODELS_SWINIR = {
    "real_sr": {"large": os.path.join(MODEL_DIR_SWINIR, MODEL_NAME_SWINIR)}
}


DEVICE_SWINIR = torch.device(DEVICE)
