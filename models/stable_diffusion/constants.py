import os
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
)


def clean_prefix_or_suffix_space(text: str):
    if text.startswith(" "):
        text = text[1:]
    if text.endswith(" "):
        text = text[:-1]
    return text


SD_MODEL_CACHE = "diffusers-cache"
SD_MODELS_ALL = {
    "Stable Diffusion v1.5": {
        "id": "runwayml/stable-diffusion-v1-5",
        "torch_dtype": torch.float16,
        "branch": "fp16",
    },
    "Openjourney": {
        "id": "prompthero/openjourney",
        "prompt_prefix": "mdjrny-v4 style",
        "torch_dtype": torch.float16,
    },
    "Redshift Diffusion": {
        "id": "nitrosocke/redshift-diffusion",
        "prompt_prefix": "redshift style",
        "torch_dtype": torch.float16,
    },
    "Arcane Diffusion": {
        "id": "nitrosocke/Arcane-Diffusion",
        "prompt_prefix": "arcane style",
        "torch_dtype": torch.float16,
    },
    "Ghibli Diffusion": {
        "id": "nitrosocke/Ghibli-Diffusion",
        "prompt_prefix": "ghibli style",
        "torch_dtype": torch.float16,
    },
    "Waifu Diffusion v1.4": {
        "id": "hakurei/waifu-diffusion",
        "prompt_prefix": "masterpiece, best quality, high quality",
        "negative_prompt_prefix": "worst quality, low quality, deleted, nsfw",
        "torch_dtype": torch.float16,
        "branch": "fp16",
    },
    "22h Diffusion v0.1": {
        "id": "22h/vintedois-diffusion-v0-1",
        "prompt_prefix": "estilovintedois",
        "torch_dtype": torch.float16,
    },
}

SD_MODELS = {}
models_from_env = os.environ.get("MODELS", "Stable Diffusion v1.5, Openjourney")
if models_from_env == "all":
    SD_MODELS = SD_MODELS_ALL
else:
    models_from_env_list = map(
        lambda x: clean_prefix_or_suffix_space(x), models_from_env.split(",")
    )
    for model_env in models_from_env_list:
        if model_env in SD_MODELS_ALL:
            SD_MODELS[model_env] = SD_MODELS_ALL[model_env]

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
}

SD_SCHEDULER_CHOICES = [*SD_SCHEDULERS.keys()]
SD_SCHEDULER_DEFAULT = SD_SCHEDULER_CHOICES[0]
