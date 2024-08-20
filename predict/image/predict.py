import time
from models.flux1.constants import FLUX1_LOAD, FLUX1_MODEL_NAME
from models.kandinsky.constants import (
    KANDINSKY_2_2_MODEL_NAME,
    LOAD_KANDINSKY_2_2,
)
from models.flux1.generate import generate as generate_with_flux1
from models.kandinsky.generate import generate_2_2 as generate_with_kandinsky_2_2
from models.stable_diffusion.constants import (
    SD_MODEL_CHOICES,
    SD_MODEL_DEFAULT_KEY,
    SD_SCHEDULER_CHOICES,
    SD_SCHEDULER_DEFAULT,
)

from models.stable_diffusion.generate import generate as generate_with_sd
from models.swinir.upscale import upscale

from typing import List

from shared.constants import TabulateLevels
from shared.oom import with_oom_protection

from .classes import PredictOutput, PredictResult, ModelsPack
from .constants import SIZE_LIST
from pydantic import BaseModel, Field, validator
from shared.helpers import log_gpu_memory, return_value_if_in_list, wrap_text
from tabulate import tabulate
import logging


class PredictInput(BaseModel):
    prompt: str = Field(description="Input prompt.", default="")
    negative_prompt: str = Field(description="Input negative prompt.", default="")
    num_outputs: int = Field(
        description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
        ge=1,
        le=10,
        default=1,
    )
    init_image_url: str = Field(
        description="Init image url to be used with img2img.",
        default=None,
    )
    mask_image_url: str = Field(
        description="Mask image url to be used with img2img.",
        default=None,
    )
    prompt_strength: float = Field(
        description="The strength of the prompt when using img2img, between 0-1. When 1, it'll essentially ignore the image.",
        ge=0,
        le=1,
        default=None,
    )
    num_inference_steps: int = Field(
        description="Number of denoising steps", ge=1, le=500, default=30
    )
    guidance_scale: float = Field(
        description="Scale for classifier-free guidance.", ge=1, le=20, default=7.5
    )
    model: str = Field(
        default=SD_MODEL_DEFAULT_KEY,
        description=f'Choose a model. Defaults to "{SD_MODEL_DEFAULT_KEY}".',
    )
    scheduler: str = Field(
        default=SD_SCHEDULER_DEFAULT,
        description=f'Choose a scheduler. Defaults to "{SD_SCHEDULER_DEFAULT}".',
    )
    seed: int = Field(
        description="Random seed. Leave blank to randomize the seed.", default=None
    )
    prompt_flores_200_code: str = Field(
        description="Prompt language code (FLORES-200). It overrides the language auto-detection.",
        default=None,
    )
    negative_prompt_flores_200_code: str = Field(
        description="Negative prompt language code (FLORES-200). It overrides the language auto-detection.",
        default=None,
    )
    prompt_prefix: str = Field(description="Prompt prefix.", default=None)
    negative_prompt_prefix: str = Field(
        description="Negative prompt prefix.", default=None
    )
    output_image_extension: str = Field(
        description="Output type of the image. Can be 'png' or 'jpeg' or 'webp'.",
        default="jpeg",
    )
    output_image_quality: int = Field(
        description="Output quality of the image. Can be 1-100.", default=90
    )
    image_to_upscale: str = Field(
        description="Input image for the upscaler (SwinIR).", default=None
    )
    process_type: str = Field(
        description="Choose a process type. Can be 'generate', 'upscale' or 'generate_and_upscale'. Defaults to 'generate'",
        default="generate",
    )
    width: int = Field(
        description="Width of output image.",
        default=512,
    )
    height: int = Field(
        description="Height of output image.",
        default=512,
    )
    signed_urls: List[str] = Field(
        description="List of signed URLs for images to be uploaded to.", default=None
    )

    @validator("model")
    def validate_model(cls, v):
        rest = []
        if LOAD_KANDINSKY_2_2:
            rest += [KANDINSKY_2_2_MODEL_NAME]
        if FLUX1_LOAD:
            rest += [FLUX1_MODEL_NAME]
        choices = SD_MODEL_CHOICES + rest
        return return_value_if_in_list(v, choices)

    @validator("scheduler")
    def validate_scheduler(cls, v):
        choices = SD_SCHEDULER_CHOICES
        return return_value_if_in_list(v, choices)

    @validator("height")
    def validate_height(cls, v: int, values):
        if values["process_type"] == "upscale":
            return v
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("width")
    def validate_width(cls, v: int, values):
        if values["process_type"] == "upscale":
            return v
        return return_value_if_in_list(
            v,
            SIZE_LIST,
        )

    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        return return_value_if_in_list(v, ["png", "jpeg", "webp"])

    @validator("process_type")
    def validate_process_type(cls, v):
        return return_value_if_in_list(
            v, ["generate", "upscale", "generate_and_upscale"]
        )


@with_oom_protection(max_retries=1)
def predict(
    input: PredictInput,
    models_pack: ModelsPack,
) -> PredictResult:
    process_start = time.time()
    logging.info("//////////////////////////////////////////////////////////////////")
    logging.info(
        tabulate(
            [
                [
                    f"🔧 Process: {input.process_type}",
                    f"🟡 Started",
                ],
            ],
            tablefmt=TabulateLevels.PRIMARY.value,
        )
    )
    log_gpu_memory(message="Before inference")
    output_images = []
    nsfw_count = 0

    if input.process_type == "generate" or input.process_type == "generate_and_upscale":
        if input.signed_urls is None or len(input.signed_urls) < input.num_outputs:
            raise ValueError(
                f"🔴 Signed URLs are required for {input.num_outputs} outputs. Got {len(input.signed_urls) if input.signed_urls is not None else 0}."
            )
    elif input.process_type == "upscale":
        if input.signed_urls is None or len(input.signed_urls) < 1:
            raise ValueError("🔴 A signed URL is required for the image to upscale.")

    if input.process_type == "generate" or input.process_type == "generate_and_upscale":
        generator_pipe = None
        if input.model == KANDINSKY_2_2_MODEL_NAME:
            generator_pipe = models_pack.kandinsky_2_2
        elif input.model == FLUX1_MODEL_NAME:
            generator_pipe = models_pack.flux1
        else:
            generator_pipe = models_pack.sd_pipe_sets[input.model]

        if hasattr(generator_pipe, "safety_checker"):
            generator_pipe.safety_checker = None

        prompt_final = input.prompt
        negative_prompt_final = input.negative_prompt

        log_table = [
            ["Model", input.model],
            ["Width", input.width],
            ["Height", input.height],
            ["Steps", input.num_inference_steps],
            ["Guidance Scale", input.guidance_scale],
            ["Outputs", input.num_outputs],
            ["Scheduler", input.scheduler],
            ["Seed", input.seed],
            [
                "Init Image URL",
                (
                    wrap_text(input.init_image_url)
                    if input.init_image_url is not None
                    else None
                ),
            ],
            [
                "Mask Image URL",
                (
                    wrap_text(input.mask_image_url)
                    if input.mask_image_url is not None
                    else None
                ),
            ],
            ["Prompt Strength", input.prompt_strength],
            ["Prompt", wrap_text(prompt_final)],
            ["Negative Prompt", wrap_text(negative_prompt_final)],
        ]
        logging.info(
            tabulate(
                [["🖼️ Generation", "🟡 Started"]] + log_table,
                tablefmt=TabulateLevels.PRIMARY.value,
            )
        )

        startTime = time.time()
        args = {
            "prompt": prompt_final,
            "negative_prompt": negative_prompt_final,
            "prompt_prefix": input.prompt_prefix,
            "negative_prompt_prefix": input.negative_prompt_prefix,
            "width": input.width,
            "height": input.height,
            "num_outputs": input.num_outputs,
            "num_inference_steps": input.num_inference_steps,
            "guidance_scale": input.guidance_scale,
            "init_image_url": input.init_image_url,
            "mask_image_url": input.mask_image_url,
            "prompt_strength": input.prompt_strength,
            "scheduler": input.scheduler,
            "seed": input.seed,
            "model": input.model,
            "pipe": generator_pipe,
        }

        if input.model == KANDINSKY_2_2_MODEL_NAME:
            generate_output_images, generate_nsfw_count = generate_with_kandinsky_2_2(
                **args, safety_checker=None, models_pack=models_pack
            )
        elif input.model == FLUX1_MODEL_NAME:
            generate_output_images, generate_nsfw_count = generate_with_flux1(
                **args, models_pack=models_pack
            )
        else:
            generate_output_images, generate_nsfw_count = generate_with_sd(
                **args, models_pack=models_pack
            )

        output_images = generate_output_images
        nsfw_count = generate_nsfw_count

        endTime = time.time()
        logging.info(
            tabulate(
                [["🖼️ Generate", f"🟢 {round((endTime - startTime) * 1000)} ms"]]
                + log_table,
                tablefmt=TabulateLevels.PRIMARY.value,
            ),
        )

    if input.process_type == "upscale" or input.process_type == "generate_and_upscale":
        logging.info(
            tabulate(
                [["⭐️ Upscale", f"🟡 Started"]],
                tablefmt=TabulateLevels.PRIMARY.value,
            ),
        )
        u_start = time.time()
        if input.process_type == "upscale":
            upscale_output_image = upscale(
                image=input.image_to_upscale,
                upscaler=models_pack.upscaler,
                models_pack=models_pack,
            )
            output_images = [upscale_output_image]
        else:
            upscale_output_images = []
            for image in output_images:
                upscale_output_image = upscale(
                    image=image,
                    upscaler=models_pack.upscaler,
                    models_pack=models_pack,
                )
                upscale_output_images.append(upscale_output_image)
            output_images = upscale_output_images
        u_end = time.time()
        logging.info(
            tabulate(
                [["⭐️ Upscale", f"🟢 {round((u_end - u_start) * 1000)} ms"]],
                tablefmt=TabulateLevels.PRIMARY.value,
            ),
        )

    # Prepare output objects
    output_objects: List[PredictOutput] = []
    for i, image in enumerate(output_images):
        obj = PredictOutput(
            pil_image=image,
            target_quality=input.output_image_quality,
            target_extension=input.output_image_extension,
        )
        output_objects.append(obj)

    result = PredictResult(
        outputs=output_objects,
        nsfw_count=nsfw_count,
        signed_urls=input.signed_urls,
    )
    process_end = time.time()

    logging.info(
        tabulate(
            [
                [
                    f"🔧 Process: {input.process_type}",
                    f"🟢 {round((process_end - process_start) * 1000)} ms",
                ]
            ],
            tablefmt=TabulateLevels.PRIMARY.value,
        )
    )
    logging.info("//////////////////////////////////////////////////////////////////")

    return result
