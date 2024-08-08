import os
from typing import List
import torch

from models.constants import DEVICE_CUDA
from predict.image.classes import ModelsPack
from shared.move_to_cpu import move_other_models_to_cpu
from .helpers import get_scheduler
from .constants import SD_MODELS
import time
from shared.helpers import (
    download_and_fit_image,
    log_gpu_memory,
    move_pipe_to_device,
)
import logging


class SDOutput:
    def __init__(self, images: List[str], nsfw_content_detected: List[bool]):
        self.images = images
        self.nsfw_content_detected = nsfw_content_detected


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
    models_pack: ModelsPack,
):
    #### Move other models to CPU if needed
    main_model_pipe = "text2img"
    if (
        init_image_url is not None
        and mask_image_url is not None
        and pipe.inpaint is not None
    ):
        main_model_pipe = "inpaint"
    elif init_image_url is not None and pipe.img2img is not None:
        main_model_pipe = "img2img"
    move_other_models_to_cpu(
        main_model_name=model, main_model_pipe=main_model_pipe, models_pack=models_pack
    )
    #####################################

    if seed is None:
        seed = int.from_bytes(os.urandom(3), "big")
        logging.info(f"Using seed: {seed}")
    generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed)

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

    extra_kwargs = {}
    pipe_selected = None

    if pipe.refiner is not None:
        extra_kwargs["output_type"] = "latent"

    if init_image_url is not None and (
        pipe.img2img is not None or pipe.inpaint is not None
    ):
        start_i = time.time()
        extra_kwargs["image"] = download_and_fit_image(
            url=init_image_url,
            width=width,
            height=height,
        )
        extra_kwargs["strength"] = prompt_strength
        end_i = time.time()
        logging.info(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )

        if pipe.inpaint is not None and mask_image_url is not None:
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
            logging.info(
                f"-- Downloaded and cropped mask image in: {round((end_i - start_i) * 1000)} ms"
            )
        elif pipe.img2img is not None:
            # The process is: img2img
            pipe_selected = pipe.img2img
    else:
        # The process is: text2img
        pipe_selected = pipe.text2img
        extra_kwargs["width"] = width
        extra_kwargs["height"] = height

    if pipe_selected.device.type != DEVICE_CUDA:
        pipe_selected = move_pipe_to_device(
            pipe=pipe_selected,
            model_name=f"{model} {main_model_pipe}",
            device=DEVICE_CUDA,
        )

    if SD_MODELS[model].get("base_model", None) != "Stable Diffusion 3":
        pipe_selected.scheduler = get_scheduler(
            scheduler, pipe_selected.scheduler.config
        )

    output: SDOutput = SDOutput(images=[], nsfw_content_detected=[])

    for i in range(num_outputs):
        generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
        out = pipe_selected(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )
        for img in out.images:
            output.images.append(img)
        if (
            hasattr(out, "nsfw_content_detected")
            and out.nsfw_content_detected is not None
        ):
            for nsfw_flag in out.nsfw_content_detected:
                output.nsfw_content_detected.append(nsfw_flag)
        else:
            output.nsfw_content_detected.append(False)

    log_gpu_memory(message="After inference")

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
            "num_inference_steps": num_inference_steps,
            "image": output_images,
        }

        if pipe.refiner.device.type != DEVICE_CUDA:
            pipe.refiner = move_pipe_to_device(
                pipe=pipe.refiner, model_name=f"{model} refiner", device=DEVICE_CUDA
            )

        s = time.time()
        for i in range(num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            out_image = pipe.refiner(
                **args,
                generator=generator,
                num_images_per_prompt=1,
            ).images[0]
            output_images[i] = out_image
        e = time.time()
        logging.info(
            f"🖌️ Refined {len(output_images)} image(s) in: {round((e - s) * 1000)} ms"
        )

    if nsfw_count > 0:
        logging.info(
            f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs."
        )

    return output_images, nsfw_count
