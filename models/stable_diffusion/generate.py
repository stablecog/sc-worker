import os
import torch
from .helpers import get_scheduler, download_image, fit_image
from .constants import SD_MODELS
from models.constants import DEVICE
import time

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
    sd_pipe,
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
    sd_pipe.scheduler = get_scheduler(scheduler, sd_pipe.scheduler.config)
    pipe = None
    if init_image_url is not None:
        pipe = sd_pipe.img2img
        start_i = time.time()
        init_image_url = download_image(init_image_url)
        init_image_url = fit_image(init_image_url, width, height)
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )
        extra_kwargs["image"] = [init_image_url] * num_outputs
        extra_kwargs["strength"] = prompt_strength
    else:
        pipe = sd_pipe.text2img
    

    generator = torch.Generator(DEVICE).manual_seed(seed)
    output = pipe(
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

    for i, nsfw_flag in enumerate(output.nsfw_content_detected):
        if nsfw_flag:
            nsfw_count += 1
        else:
            output_images.append(output.images[i])

    if nsfw_count > 0:
        print(f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs.")

    return output_images, nsfw_count
