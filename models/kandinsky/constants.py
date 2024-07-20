from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)

from shared.constants import MODELS_FROM_ENV, MODELS_FROM_ENV_LIST
from shared.vram import device_vram_gb


KANDINSKY_2_2_DECODER_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
KANDINSKY_2_2_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID = (
    "kandinsky-community/kandinsky-2-2-decoder-inpaint"
)

KANDINKSY_2_2_MODEL_NAME = "Kandinsky 2.2"
KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE = device_vram_gb < 45


KANDINSKY_2_2_SCHEDULERS = {
    "DDPM": {
        "scheduler": DDPMScheduler,
        "from_config": True,
    },
    "DDIM": {"scheduler": DDIMScheduler},
    "DPM++_2M": {"scheduler": DPMSolverMultistepScheduler},
}

LOAD_KANDINSKY_2_2 = (
    MODELS_FROM_ENV == "all" or KANDINKSY_2_2_MODEL_NAME in MODELS_FROM_ENV_LIST
)
