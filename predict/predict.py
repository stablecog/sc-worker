import time
import os

import torch

from models.stable_diffusion.generate import generate
from models.nllb.translate import translate_text
from models.swinir.upscale import upscale

from typing import List, Optional
from .classes import PredictOutput, PredictResult
from .setup import ModelPack
from models.clip.main import get_features_of_images


@torch.inference_mode()
def predict(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_outputs: int,
    num_inference_steps: int,
    guidance_scale: float,
    scheduler: str,
    model: str,
    seed: int,
    prompt_flores_200_code: str,
    negative_prompt_flores_200_code: str,
    output_image_extension: str,
    output_image_quality: int,
    process_type: str,
    prompt_prefix: str,
    negative_prompt_prefix: str,
    image_to_upscale: Optional[str],
    translator_cog_url: Optional[str],
    model_pack: ModelPack,
) -> PredictResult:
    process_start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳ Process started: {process_type} ⏳")
    output_images = []
    nsfw_count = 0

    if process_type == "generate" or process_type == "generate_and_upscale":
        if translator_cog_url is None:
            translator_cog_url = os.environ.get("TRANSLATOR_COG_URL", None)

        t_prompt = prompt
        t_negative_prompt = negative_prompt
        if translator_cog_url is not None:
            [t_prompt, t_negative_prompt] = translate_text(
                prompt,
                prompt_flores_200_code,
                negative_prompt,
                negative_prompt_flores_200_code,
                translator_cog_url,
                model_pack.language_detector_pipe,
                "Prompt & Negative Prompt",
            )
        else:
            print("-- Translator cog URL is not set. Skipping translation. --")

        txt2img_pipe = model_pack.txt2img_pipes[model]
        print(
            f"🖥️ Generating - Model: {model} - Width: {width} - Height: {height} - Steps: {num_inference_steps} - Outputs: {num_outputs} 🖥️"
        )
        startTime = time.time()
        generate_output_images, generate_nsfw_count = generate(
            t_prompt,
            t_negative_prompt,
            prompt_prefix,
            negative_prompt_prefix,
            width,
            height,
            num_outputs,
            num_inference_steps,
            guidance_scale,
            scheduler,
            seed,
            model,
            txt2img_pipe,
        )
        output_images = generate_output_images
        nsfw_count = generate_nsfw_count
        endTime = time.time()
        print(
            f"🖥️ Generated in {round((endTime - startTime) * 1000)} ms - Model: {model} - Width: {width} - Height: {height} - Steps: {num_inference_steps} - Outputs: {num_outputs} 🖥️"
        )

        start_clip = time.time()
        features_of_images = get_features_of_images(
            output_images, model_pack.clip["model"], model_pack.clip["processor"]
        )
        end_clip = time.time()
        print(
            f"🖼️ CLIP embeddings in: {round((end_clip - start_clip) * 1000)} ms 🖼️"
        )

    if process_type == "upscale" or process_type == "generate_and_upscale":
        startTime = time.time()
        if process_type == "upscale":
            upscale_output_image = upscale(image_to_upscale, model_pack.upscaler)
            output_images = [upscale_output_image]
        else:
            upscale_output_images = []
            for image in output_images:
                upscale_output_image = upscale(image, model_pack.upscaler)
                upscale_output_images.append(upscale_output_image)
            output_images = upscale_output_images
        endTime = time.time()
        print(f"⭐️ Upscaled in: {round((endTime - startTime) * 1000)} ms ⭐️")

    # Prepare output objects
    output_objects: List[PredictOutput] = []
    for i, image in enumerate(output_images):
        obj = PredictOutput(
            pil_image=image,
            target_quality=output_image_quality,
            target_extension=output_image_extension,
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
