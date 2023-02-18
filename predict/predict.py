import time

import torch
from models.nllb.constants import TRANSLATOR_COG_URL
from models.stable_diffusion.constants import SD_MODEL_CHOICES, SD_MODEL_DEFAULT_KEY, SD_SCHEDULER_CHOICES, SD_SCHEDULER_DEFAULT

from models.stable_diffusion.generate import generate
from models.nllb.translate import translate_text
from models.swinir.upscale import upscale

from typing import List
from .classes import PredictOutput, PredictResult
from .setup import ModelsPack
from models.clip.main import get_embeds_of_images, get_embeds_of_texts
from pydantic import BaseModel, Field, validator
from .helpers import get_value_if_in_list

class PredictInput(BaseModel):
    prompt: str = Field(description="Input prompt.", default="")
    negative_prompt: str = Field(description="Input negative prompt.", default="")
    width: int = Field(
        description="Width of output image.",
        default=512,
    )
    @validator("width")
    def validate_width(cls, v: int):
        return get_value_if_in_list(v, [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024])
    height: int = Field(
        description="Height of output image.",
        default=512,
    )
    @validator("height")
    def validate_height(cls, v: int):
        return get_value_if_in_list(v, [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024])
    num_outputs: int = Field(
        description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
        ge=1,
        le=10,
        default=1,
    )
    num_inference_steps: int = Field(
        description="Number of denoising steps", ge=1, le=500, default=30
    )
    guidance_scale: float = Field(
        description="Scale for classifier-free guidance.", ge=1, le=20, default=7.5
    )
    scheduler: str = Field(
        default=SD_SCHEDULER_DEFAULT,
        description=f'Choose a scheduler. Defaults to "{SD_SCHEDULER_DEFAULT}".',
    )
    @validator("scheduler")
    def validate_scheduler(cls, v):
        return get_value_if_in_list(v, SD_SCHEDULER_CHOICES)
    model: str = Field(
        default=SD_MODEL_DEFAULT_KEY,
        description=f'Choose a model. Defaults to "{SD_MODEL_DEFAULT_KEY}".',
    )
    @validator("model")
    def validate_model(cls, v):
        return get_value_if_in_list(v, SD_MODEL_CHOICES)
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
    @validator("output_image_extension")
    def validate_output_image_extension(cls, v):
        return get_value_if_in_list(v, ["png", "jpeg", "webp"])
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
    @validator("process_type")
    def validate_process_type(cls, v):
        return get_value_if_in_list(v, ["generate", "upscale", "generate_and_upscale"])
    translator_cog_url: str = Field(
        description="URL of the translator cog. If it's blank, TRANSLATOR_COG_URL environment variable will be used (if it exists).",
        default=TRANSLATOR_COG_URL,
    )

@torch.inference_mode()
def predict(
    input: PredictInput,
    models_pack: ModelsPack,
) -> PredictResult:
    print(input.model)
    process_start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳ Process started: {input.process_type} ⏳")
    output_images = []
    nsfw_count = 0
    embeds_of_images = None
    embed_of_prompt = None

    if input.process_type == "generate" or input.process_type == "generate_and_upscale":
        t_prompt = input.prompt
        t_negative_prompt = input.negative_prompt
        if input.translator_cog_url is not None:
            [t_prompt, t_negative_prompt] = translate_text(
                input.prompt,
                input.prompt_flores_200_code,
                input.negative_prompt,
                input.negative_prompt_flores_200_code,
                input.translator_cog_url,
                models_pack.language_detector_pipe,
                "Prompt & Negative Prompt",
            )
        else:
            print("-- Translator cog URL is not set. Skipping translation. --")

        txt2img_pipe = models_pack.txt2img_pipes[input.model]
        print(
            f"🖥️ Generating - Model: {input.model} - Width: {input.width} - Height: {input.height} - Steps: {input.num_inference_steps} - Outputs: {input.num_outputs} 🖥️"
        )
        startTime = time.time()
        generate_output_images, generate_nsfw_count = generate(
            t_prompt,
            t_negative_prompt,
            input.prompt_prefix,
            input.negative_prompt_prefix,
            input.width,
            input.height,
            input.num_outputs,
            input.num_inference_steps,
            input.guidance_scale,
            input.scheduler,
            input.seed,
            input.model,
            txt2img_pipe,
        )
        output_images = generate_output_images
        nsfw_count = generate_nsfw_count
        endTime = time.time()
        print(
            f"🖥️ Generated in {round((endTime - startTime) * 1000)} ms - Model: {input.model} - Width: {input.width} - Height: {input.height} - Steps: {input.num_inference_steps} - Outputs: {input.num_outputs} 🖥️"
        )

        start_clip_image = time.time()
        embeds_of_images = get_embeds_of_images(
            output_images, models_pack.clip["model"], models_pack.clip["processor"]
        )
        end_clip_image = time.time()
        print(
            f"🖼️ CLIP image embeddings in: {round((end_clip_image - start_clip_image) * 1000)} ms - {len(output_images)} images 🖼️"
        )

        start_clip_prompt = time.time()
        embed_of_prompt = get_embeds_of_texts(
            [input.prompt], models_pack.clip["model"], models_pack.clip["tokenizer"]
        )
        end_clip_prompt = time.time()
        print(
            f"📜 CLIP prompt embedding in: {round((end_clip_prompt - start_clip_prompt) * 1000)} ms 📜"
        )

    if input.process_type == "upscale" or input.process_type == "generate_and_upscale":
        startTime = time.time()
        if input.process_type == "upscale":
            upscale_output_image = upscale(input.image_to_upscale, models_pack.upscaler)
            output_images = [upscale_output_image]
        else:
            upscale_output_images = []
            for image in output_images:
                upscale_output_image = upscale(image, models_pack.upscaler)
                upscale_output_images.append(upscale_output_image)
            output_images = upscale_output_images
        endTime = time.time()
        print(f"⭐️ Upscaled in: {round((endTime - startTime) * 1000)} ms ⭐️")

    # Prepare output objects
    output_objects: List[PredictOutput] = []
    for i, image in enumerate(output_images):
        obj = PredictOutput(
            pil_image=image,
            target_quality=input.output_image_quality,
            target_extension=input.output_image_extension,
            image_embed=embeds_of_images[i] if embeds_of_images is not None else None,
            prompt_embed=embed_of_prompt if embed_of_prompt is not None else None,
        )
        output_objects.append(obj)

    result = PredictResult(
        outputs=output_objects,
        nsfw_count=nsfw_count,
    )
    process_end = time.time()
    print(f"✅ Process completed in: {round((process_end - process_start) * 1000)} ms ✅")
    print("//////////////////////////////////////////////////////////////////")

    return result
