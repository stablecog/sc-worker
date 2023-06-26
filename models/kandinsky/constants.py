from diffusers import DDPMScheduler, DDIMScheduler

KANDIKSKY_SCHEDULERS = {
    "P_SAMPLER": DDPMScheduler.from_pretrained(
        "kandinsky-community/kandinsky-2-1", subfolder="ddpm_scheduler"
    ),
    "DDIM": DDIMScheduler.from_pretrained(
        "kandinsky-community/kandinsky-2-1", subfolder="ddim_scheduler"
    ),
}

KANDINSKY_SCHEDULER_CHOICES = [*KANDIKSKY_SCHEDULERS.keys()]

KANDINSKY_MODEL_NAME = "Kandinsky"
