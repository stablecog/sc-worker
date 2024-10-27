from shared.constants import MODELS_FROM_ENV, MODELS_FROM_ENV_LIST
from shared.vram import device_vram_gb
import torch

FLUX1_MODEL_NAME = "FLUX.1"
FLUX1_REPO = "black-forest-labs/FLUX.1-schnell"
FLUX1_TRANSFORMER_REPO = "stablecog-hf-1/FLUX.1-schnell-8bit-transformer"
FLUX1_KEEP_IN_CPU_WHEN_IDLE = device_vram_gb < 40
FLUX1_DTYPE = torch.bfloat16

FLUX1_FP8_TRANSFORMER_FILE = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors"
FLUX1_LOAD = MODELS_FROM_ENV == "all" or FLUX1_MODEL_NAME in MODELS_FROM_ENV_LIST
