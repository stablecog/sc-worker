import torch
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    UniPCMultistepScheduler,
    DDPMScheduler,
)
from dotenv import load_dotenv

from shared.constants import MODELS_FROM_ENV, MODELS_FROM_ENV_LIST


load_dotenv()


SD_ENV_KEY_TO_KEY = {
    "OJ": "Openjourney",
    "RSD": "Redshift Diffusion",
    "GD": "Ghibli Diffusion",
    "WD": "Waifu Diffusion",
    "22D": "22h Diffusion",
    "LD": "Luna Diffusion",
    "SX": "SDXL",
    "WDX": "Waifu Diffusion XL",
    "AD": "Arcane Diffusion",
    "SSD": "SSD-1B",
}
SD_MODEL_CACHE = "/app/data/diffusers-cache"
SD_MODELS_ALL = {
    "SDXL": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "refiner_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "default_lora": "sd_xl_offset_example-lora_1.0.safetensors",
        "vae": "stabilityai/sdxl-vae",
        "refiner_vae": "stabilityai/sdxl-vae",
        "base_model": "SDXL",
    },
    "SSD-1B": {
        "id": "segmind/SSD-1B",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "refiner_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "refiner_vae": "stabilityai/sdxl-vae",
        "base_model": "SDXL",
    },
    "Segmind Vega": {
        "id": "segmind/Segmind-Vega",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "torch_dtype": torch.float16,
        "base_model": "SDXL",
    },
    "Luna Diffusion": {
        "id": "proximasanfinetuning/luna-diffusion",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "torch_dtype": torch.float16,
        "keep_in_cpu_when_idle": True,
    },
    "Waifu Diffusion": {
        "id": "hakurei/waifu-diffusion",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt_prefix": "masterpiece, best quality, high quality",
        "negative_prompt_prefix": "worst quality, low quality, deleted, nsfw, blurry",
        "torch_dtype": torch.float16,
        "branch": "fp16",
        "keep_in_cpu_when_idle": True,
    },
    "Openjourney": {
        "id": "prompthero/openjourney",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt_prefix": "mdjrny-v4 style",
        "torch_dtype": torch.float16,
        "keep_in_cpu_when_idle": True,
    },
    "22h Diffusion": {
        "id": "22h/vintedois-diffusion-v0-1",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt_prefix": "estilovintedois",
        "torch_dtype": torch.float16,
        "keep_in_cpu_when_idle": True,
    },
    "Redshift Diffusion": {
        "id": "nitrosocke/redshift-diffusion",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt_prefix": "redshift style",
        "torch_dtype": torch.float16,
        "keep_in_cpu_when_idle": True,
    },
    "Arcane Diffusion": {
        "id": "nitrosocke/Arcane-Diffusion",
        "inpaint_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "prompt_prefix": "arcane style",
        "torch_dtype": torch.float16,
        "keep_in_cpu_when_idle": True,
    },
}

SD_MODEL_FOR_SAFETY_CHECKER = "Luna Diffusion"
SD_MODELS = {}
if MODELS_FROM_ENV == "all":
    SD_MODELS = SD_MODELS_ALL
else:
    for model_env in MODELS_FROM_ENV_LIST:
        if model_env in SD_MODELS_ALL:
            SD_MODELS[model_env] = SD_MODELS_ALL[model_env]
        elif model_env in SD_ENV_KEY_TO_KEY:
            key = SD_ENV_KEY_TO_KEY[model_env]
            SD_MODELS[key] = SD_MODELS_ALL[key]

SD_MODEL_CHOICES = list(SD_MODELS.keys())
SD_MODEL_DEFAULT_KEY = SD_MODEL_CHOICES[0]
SD_MODEL_DEFAULT = SD_MODELS[SD_MODEL_DEFAULT_KEY]
SD_MODEL_DEFAULT_ID = SD_MODEL_DEFAULT["id"]

SD_SCHEDULERS = {
    "K_LMS": {"from_config": LMSDiscreteScheduler.from_config},
    "PNDM": {"from_config": PNDMScheduler.from_config},
    "DDIM": {"from_config": DDIMScheduler.from_config},
    "K_EULER": {"from_config": EulerDiscreteScheduler.from_config},
    "K_EULER_ANCESTRAL": {"from_config": EulerAncestralDiscreteScheduler.from_config},
    "HEUN": {"from_config": HeunDiscreteScheduler.from_config},
    "DPM++_2M": {"from_config": DPMSolverMultistepScheduler.from_config},
    "DPM++_2S": {"from_config": DPMSolverSinglestepScheduler.from_config},
    "DEIS": {"from_config": DEISMultistepScheduler.from_config},
    "UNI_PC": {"from_config": UniPCMultistepScheduler.from_config},
    "DDPM": {"from_config": DDPMScheduler.from_config},
}

SD_SCHEDULER_CHOICES = [*SD_SCHEDULERS.keys()]
SD_SCHEDULER_DEFAULT = SD_SCHEDULER_CHOICES[0]
