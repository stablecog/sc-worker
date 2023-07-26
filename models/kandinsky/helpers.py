from .constants import KANDINSKY_2_2_SCHEDULERS
from diffusers import (
    KandinskyV22Pipeline,
    KandinskyV22Img2ImgPipeline,
    KandinskyV22InpaintPipeline,
)


def get_scheduler(
    name: str,
    pipeline: KandinskyV22Pipeline
    | KandinskyV22InpaintPipeline
    | KandinskyV22Img2ImgPipeline,
):
    if "from_config" in KANDINSKY_2_2_SCHEDULERS[name]:
        return KANDINSKY_2_2_SCHEDULERS[name]["scheduler"].from_config(
            pipeline.scheduler.config
        )
    else:
        return KANDINSKY_2_2_SCHEDULERS[name]["scheduler"]()
