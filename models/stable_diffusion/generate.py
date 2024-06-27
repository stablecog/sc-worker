import os
import torch

from models.constants import DEVICE
from .helpers import get_scheduler
from .constants import SD_MODELS
import time
from shared.helpers import (
    download_and_fit_image,
    log_gpu_memory,
    move_pipe_to_device,
)
from shared.logger import logger


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
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    logger.info(f"Using seed: {seed}")
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

    logger.info(f"-- Prompt: {prompt} --")
    logger.info(f"-- Negative Prompt: {negative_prompt} --")

    extra_kwargs = {}
    pipe_selected = None

    if pipe.refiner is not None:
        extra_kwargs["output_type"] = "latent"
    if init_image_url is not None:
        # The process is: img2img or inpainting
        start_i = time.time()
        extra_kwargs["image"] = download_and_fit_image(
            url=init_image_url,
            width=width,
            height=height,
        )
        extra_kwargs["strength"] = prompt_strength
        end_i = time.time()
        logger.info(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )

        if mask_image_url is not None and pipe.inpaint is not None:
            # The process is: inpainting
            pipe_selected = pipe.inpaint
            start_i = time.time()
            extra_kwargs["mask_image"] = download_and_fit_image(
                url=mask_image_url,
                width=width,
                height=height,
            )
            extra_kwargs["strength"] = 0.99
            end_i = time.time()
            logger.info(
                f"-- Downloaded and cropped mask image in: {round((end_i - start_i) * 1000)} ms"
            )
        else:
            # The process is: img2img
            pipe_selected = pipe.img2img
    else:
        # The process is: text2img
        pipe_selected = pipe.text2img
        extra_kwargs["width"] = width
        extra_kwargs["height"] = height

    if "keep_in_cpu_when_idle" in SD_MODELS[model]:
        pipe_selected = move_pipe_to_device(
            pipe=pipe_selected, model_name=model, device=DEVICE
        )

    pipe_selected.scheduler = get_scheduler(scheduler, pipe_selected.scheduler.config)
    output = pipe_selected(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_inference_steps,
        **extra_kwargs,
    )
    log_gpu_memory(message="GPU status after inference")

    if "keep_in_cpu_when_idle" in SD_MODELS[model]:
        pipe_selected = move_pipe_to_device(
            pipe=pipe_selected, model_name=model, device="cpu"
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

    if pipe.refiner is not None:
        args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": num_outputs,
            "num_inference_steps": num_inference_steps,
            "image": output_images,
        }

        if "keep_in_cpu_when_idle" in SD_MODELS[model]:
            pipe.refiner = move_pipe_to_device(
                pipe=pipe.refiner, model_name=f"{model} refiner", device=DEVICE
            )

        s = time.time()
        output_images = pipe.refiner(**args).images
        e = time.time()
        logger.info(
            f"🖌️ Refined {len(output_images)} images in: {round((e - s) * 1000)} ms"
        )

        if "keep_in_cpu_when_idle" in SD_MODELS[model]:
            pipe.refiner = move_pipe_to_device(
                pipe=pipe.refiner, model_name=f"{model} refiner", device="cpu"
            )

    if nsfw_count > 0:
        logger.info(
            f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs."
        )

    return output_images, nsfw_count
