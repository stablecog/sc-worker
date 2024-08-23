from diffusers import (
    KandinskyV22InpaintPipeline,
    KandinskyV22Pipeline,
    KandinskyV22PriorPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
)
from typing import Any
from PIL import Image


class PredictOutput:
    def __init__(
        self,
        pil_image: Image.Image,
        target_extension: str,
        target_quality: int,
    ):
        self.pil_image = pil_image
        self.target_extension = target_extension
        self.target_quality = target_quality


class PredictResult:
    def __init__(
        self,
        outputs: list[PredictOutput],
        signed_urls: list[str],
        nsfw_count: int,
    ):
        self.outputs = outputs
        self.signed_urls = signed_urls
        self.nsfw_count = nsfw_count


class SDPipeSet:
    def __init__(
        self,
        text2img: (
            StableDiffusionPipeline
            | StableDiffusionXLPipeline
            | StableDiffusion3Pipeline
        ),
        img2img: StableDiffusionImg2ImgPipeline | None,
        inpaint: StableDiffusionInpaintPipeline | None,
        refiner: StableDiffusionXLImg2ImgPipeline | None,
    ):
        self.text2img = text2img
        self.img2img = img2img
        self.inpaint = inpaint
        self.refiner = refiner


class KandinskyPipeSet_2_2:
    def __init__(
        self,
        prior: KandinskyV22PriorPipeline,
        text2img: KandinskyV22Pipeline,
        inpaint: KandinskyV22InpaintPipeline | None,
    ):
        self.prior = prior
        self.text2img = text2img
        self.inpaint = inpaint


class Flux1PipeSet:
    def __init__(
        self,
        text2img: FluxPipeline,
    ):
        self.text2img = text2img


class ModelsPack:
    def __init__(
        self,
        sd_pipe_sets: dict[str, SDPipeSet],
        upscaler: Any,
        kandinsky_2_2: KandinskyPipeSet_2_2,
        flux1: Flux1PipeSet | None,
    ):
        self.sd_pipe_sets = sd_pipe_sets
        self.upscaler = upscaler
        self.kandinsky_2_2 = kandinsky_2_2
        self.flux1 = flux1
