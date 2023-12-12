from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)

from shared.constants import MODELS_FROM_ENV, MODELS_FROM_ENV_LIST

KANDIKSKY_2_1_SCHEDULERS = {
    "P_SAMPLER": "p_sampler",
    "DDIM": "ddim_sampler",
}

KANDINSKY_2_1_SCHEDULER_CHOICES = [*KANDIKSKY_2_1_SCHEDULERS.keys()]

KANDINSKY_2_1_MODEL_ID = "kandinsky-community/kandinsky-2-1"
KANDINSKY_2_1_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-1-prior"

KANDINSKY_2_2_DECODER_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
KANDINSKY_2_2_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID = (
    "kandinsky-community/kandinsky-2-2-decoder-inpaint"
)

KANDINSKY_2_1_MODEL_NAME = "Kandinsky"
KANDINKSY_2_2_MODEL_NAME = "Kandinsky 2.2"


KANDINSKY_2_2_SCHEDULERS = {
    "DDPM": {
        "scheduler": DDPMScheduler,
        "from_config": True,
    },
    "DDIM": {"scheduler": DDIMScheduler},
    "DPM++_2M": {"scheduler": DPMSolverMultistepScheduler},
}

LOAD_KANDINSKY_2_1 = (
    MODELS_FROM_ENV == "all" or KANDINSKY_2_1_MODEL_NAME in MODELS_FROM_ENV_LIST
)
LOAD_KANDINSKY_2_2 = (
    MODELS_FROM_ENV == "all" or KANDINKSY_2_2_MODEL_NAME in MODELS_FROM_ENV_LIST
)
