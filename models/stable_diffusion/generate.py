import os
import torch
from .helpers import get_scheduler
from .constants import SD_MODELS
from models.constants import DEVICE
import time
from shared.helpers import download_image, fit_image


def generate(
    prompt,
    negative_prompt,
    prompt_prefix,
    negative_prompt_prefix,
    width,
    height,
    num_outputs,
    num_inference_steps,
    guidance_scale,
    init_image_url,
    prompt_strength,
    scheduler,
    seed,
    model,
    pipe,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"
    else:
        default_prompt_prefix = SD_MODELS[model].get("prompt_prefix", None)
        if default_prompt_prefix is not None:
            prompt = f"{default_prompt_prefix} {prompt}"

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"
    else:
        default_negative_prompt_prefix = SD_MODELS[model].get(
            "negative_prompt_prefix", None
        )
        if default_negative_prompt_prefix is not None:
            if negative_prompt is None or negative_prompt == "":
                negative_prompt = default_negative_prompt_prefix
            else:
                negative_prompt = f"{default_negative_prompt_prefix} {negative_prompt}"

    print(f"-- Prompt: {prompt} --")
    print(f"-- Negative Prompt: {negative_prompt} --")

    extra_kwargs = {}
    pipe.scheduler = get_scheduler(scheduler, pipe.scheduler.config)
    pipe_selected = None
    if init_image_url is not None:
        pipe_selected = pipe.img2img
        start_i = time.time()
        init_image = download_image(init_image_url)
        init_image = fit_image(init_image, width, height)
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )
        extra_kwargs["image"] = [init_image] * num_outputs
        extra_kwargs["strength"] = prompt_strength
    else:
        pipe_selected = pipe.text2img

    generator = torch.Generator(DEVICE).manual_seed(seed)
    output = pipe_selected(
        prompt=[prompt] * num_outputs if prompt is not None else None,
        negative_prompt=[negative_prompt] * num_outputs
        if negative_prompt is not None
        else None,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
        **extra_kwargs,
    )

    output_images = []
    nsfw_count = 0

    if output.nsfw_content_detected is None:
        for i, nsfw_flag in enumerate(output.nsfw_content_detected):
            if nsfw_flag:
                nsfw_count += 1
            else:
                output_images.append(output.images[i])
    else:
        output_images = output.images

    if nsfw_count > 0:
        print(f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs.")

    return output_images, nsfw_count
