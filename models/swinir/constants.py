import os
import torch

TASKS_SWINIR = {"Real-World Image Super-Resolution-Large": "real_sr"}

MODEL_DIR_SWINIR = "experiments/pretrained_models"

MODELS_SWINIR = {
    "real_sr": {
        "large": os.path.join(
            MODEL_DIR_SWINIR, "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
        )
    }
}


DEVICE_SWINIR = torch.device("cuda")
