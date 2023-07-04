import os
import time
from models.constants import DEVICE
from models.kandinsky.constants import KANDIKSKY_SCHEDULERS
from shared.helpers import (
    download_and_fit_image,
    download_and_fit_image_mask,
)
import torch
from torch.cuda.amp import autocast
import numpy as np


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
    mask_image_url,
    prompt_strength,
    scheduler,
    seed,
    model,
    pipe,
    safety_checker,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    generator = [
        torch.Generator(device=DEVICE).manual_seed(seed + i)
        for i in range(seed, seed + num_outputs)
    ]

    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"
    if negative_prompt is None:
        negative_prompt = ""

    extra_kwargs = {}

    output_images = None

    pipe_prior = pipe["prior"]
    pipe_main = None
    if init_image_url is not None and mask_image_url is not None:
        pipe_main = pipe["inpaint"]
    elif init_image_url is not None:
        pipe_main = pipe["img2img"]
    else:
        pipe_main = pipe["text2img"]

    prior_output = pipe_prior(
        [prompt] * num_outputs,
        guidance_scale=4,
        num_inference_steps=5,
        generator=generator,
    )

    if init_image_url is not None and mask_image_url is not None:
        start = time.time()
        init_image = download_and_fit_image(init_image_url, width, height)
        end = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)} ms"
        )
        start = time.time()
        mask_image = download_and_fit_image_mask(
            url=mask_image_url,
            width=width,
            height=height,
            inverted=True,
        )
        end = time.time()
        print(
            f"-- Downloaded and cropped mask image in: {round((end - start) * 1000)} ms"
        )
        extra_kwargs["mask_image"] = mask_image
        extra_kwargs["strength"] = prompt_strength
        extra_kwargs["image"] = init_image
    elif init_image_url is not None:
        start_i = time.time()
        init_image = download_and_fit_image(init_image_url, width, height)
        extra_kwargs["image"] = init_image
        extra_kwargs["strength"] = prompt_strength
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )

    output_images = pipe_main(
        prompt=[prompt] * 4,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        **prior_output,
        **extra_kwargs,
    ).images

    output_images_nsfw_results = []
    with autocast():
        for image in output_images:
            has_nsfw_concepts = False
            result = None
            if safety_checker is not None:
                safety_checker_input = safety_checker["feature_extractor"](
                    images=image, return_tensors="pt"
                ).to("cuda")
                result, has_nsfw_concepts = safety_checker["checker"].forward(
                    clip_input=safety_checker_input.pixel_values, images=image
                )
            res = {
                "result": result,
                "has_nsfw_concepts": has_nsfw_concepts,
            }
            output_images_nsfw_results.append(res)

    nsfw_count = 0
    filtered_output_images = []

    for i, res in enumerate(output_images_nsfw_results):
        if res["has_nsfw_concepts"]:
            nsfw_count += 1
        else:
            filtered_output_images.append(output_images[i])

    return filtered_output_images, nsfw_count
