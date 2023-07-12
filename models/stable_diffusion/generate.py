import os
import torch
from .helpers import get_scheduler
from .constants import SD_MODELS
import time
from shared.helpers import download_and_fit_image


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
    pipe_refiner=None,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    generator = torch.Generator(device="cuda").manual_seed(seed)

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
        if mask_image_url is not None:
            pipe_selected = pipe.inpaint
            start_i = time.time()
            extra_kwargs["mask_image"] = download_and_fit_image(
                url=mask_image_url,
                width=width,
                height=height,
            )
            end_i = time.time()
            print(
                f"-- Downloaded and cropped mask image in: {round((end_i - start_i) * 1000)} ms"
            )
        else:
            pipe_selected = pipe.img2img
            extra_kwargs["width"] = width
            extra_kwargs["height"] = height
        start_i = time.time()
        init_image = download_and_fit_image(
            url=init_image_url,
            width=width,
            height=height,
        )
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )
        extra_kwargs["image"] = init_image
        extra_kwargs["strength"] = prompt_strength
    else:
        if model == "SDXL":
            pipe_selected = pipe
        else:
            pipe_selected = pipe.text2img
        extra_kwargs["width"] = width
        extra_kwargs["height"] = height
        if pipe_refiner is not None:
            extra_kwargs["output_type"] = "latent"

    output = pipe_selected(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_inference_steps,
        **extra_kwargs,
    )

    output_images = []
    nsfw_count = 0

    if (
        hasattr(output, "nsfw_content_detected")
        and output.nsfw_content_detected is not None
    ):
        for i, nsfw_flag in enumerate(output.nsfw_content_detected):
            if nsfw_flag:
                nsfw_count += 1
            else:
                output_images.append(output.images[i])
    else:
        output_images = output.images

    if pipe_refiner is not None:
        output_images = pipe_refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_outputs,
            num_inference_steps=num_inference_steps,
            image=output_images,
        ).images

    if nsfw_count > 0:
        print(f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs.")

    return output_images, nsfw_count
