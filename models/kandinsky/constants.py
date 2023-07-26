from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)


KANDIKSKY_SCHEDULERS = {
    "P_SAMPLER": "p_sampler",
    "DDIM": "ddim_sampler",
}

KANDINSKY_SCHEDULER_CHOICES = [*KANDIKSKY_SCHEDULERS.keys()]

KANDINSKY_MODEL_NAME = "Kandinsky"
KANDINKSY_2_2_MODEL_NAME = "Kandinsky 2.2"

KANDINSKY_MODEL_ID = "kandinsky-community/kandinsky-2-1"
KANDINSKY_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-1-prior"

KANDINSKY_2_2_DECODER_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
KANDINSKY_2_2_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_2_2_DECODER_INPAINT_MODEL_ID = (
    "kandinsky-community/kandinsky-2-2-decoder-inpaint"
)


KANDINSKY_2_2_SCHEDULERS = {
    "DDPM": {
        "scheduler": DDPMScheduler,
        "from_config": True,
    },
    "DDIM": {"scheduler": DDIMScheduler},
    "DPM++_2M": {"scheduler": DPMSolverMultistepScheduler},
}
