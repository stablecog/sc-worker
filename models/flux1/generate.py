import os
from typing import List
import torch

from models.constants import DEVICE_CUDA, is_not_cuda
from predict.image.classes import ModelsPack
from shared.move_to_cpu import move_other_models_to_cpu
from shared.helpers import (
    log_gpu_memory,
    move_pipe_to_device,
)
import logging
import time
from PIL import Image


class Flex1Output:
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
    move_other_models_to_cpu(
        main_model_name=model, main_model_pipe=main_model_pipe, models_pack=models_pack
    )
    #####################################

    inference_start = time.time()

    if seed is None:
        seed = int.from_bytes(os.urandom(3), "big")
        logging.info(f"Using seed: {seed}")
    generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed)

    extra_kwargs = {}

    # The process is: text2img
    pipe_selected = pipe.text2img
    extra_kwargs["width"] = width
    extra_kwargs["height"] = height

    if is_not_cuda(pipe_selected.device.type):
        pipe_selected = move_pipe_to_device(
            pipe=pipe_selected,
            model_name=f"{model} {main_model_pipe}",
            device=DEVICE_CUDA,
        )

    output: Flex1Output = Flex1Output(images=[], nsfw_content_detected=[])

    for i in range(num_outputs):
        generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
        out = pipe_selected(
            prompt=prompt,
            generator=generator,
            guidance_scale=0,
            num_images_per_prompt=1,
            num_inference_steps=4,
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

    output_images: List[Image.Image] = []
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

    if nsfw_count > 0:
        logging.info(
            f"NSFW content detected in {nsfw_count}/{num_outputs} of the outputs."
        )

    inference_end = time.time()
    logging.info(
        f"🔮 🟢 Inference | {model} | {num_outputs} image(s) | {round((inference_end - inference_start) * 1000)}ms"
    )

    return output_images, nsfw_count
