import os
import time

from models.constants import DEVICE
from models.kandinsky.constants import KANDINSKY_2_2_IN_CPU_WHEN_IDLE
from .helpers import get_scheduler
from predict.image.setup import KandinskyPipe_2_2
from shared.helpers import (
    crop_images,
    download_and_fit_image,
    download_and_fit_image_mask,
    move_pipe_to_device,
    pad_image_mask_nd,
    pad_image_pil,
)
import torch
from torch.cuda.amp import autocast
from shared.log import custom_logger

PRIOR_STEPS = 25
PRIOR_GUIDANCE_SCALE = 4.0


kandinsky_2_2_negative_prompt_prefix = "overexposed"

kandinsky_2_2_model_name = "Kandinsky 2.2"


def generate_2_2(
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
    pipe: KandinskyPipe_2_2,
    safety_checker,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    custom_logger.info(f"Using seed: {seed}")

    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"

    if negative_prompt is None or negative_prompt == "":
        negative_prompt = kandinsky_2_2_negative_prompt_prefix
    else:
        negative_prompt = f"{kandinsky_2_2_negative_prompt_prefix}, {negative_prompt}"

    custom_logger.info(f"Negative prompt for Kandinsky 2.2: {negative_prompt}")

    output_images = None

    if KANDINSKY_2_2_IN_CPU_WHEN_IDLE:
        pipe.prior = move_pipe_to_device(
            pipe=pipe.prior,
            model_name=f"{kandinsky_2_2_model_name} Prior",
            device=DEVICE,
        )

    if (
        init_image_url is not None
        and mask_image_url is not None
        and pipe.inpaint is not None
    ):
        pipe.inpaint.scheduler = get_scheduler(scheduler, pipe.inpaint)
        start = time.time()
        init_image = download_and_fit_image(init_image_url, width, height)
        init_image = pad_image_pil(init_image, 64)
        end = time.time()
        custom_logger.info(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)} ms"
        )
        start = time.time()
        mask_image = download_and_fit_image_mask(
            url=mask_image_url,
            width=width,
            height=height,
        )
        mask_image = pad_image_mask_nd(mask_image, 64, 0)
        end = time.time()
        custom_logger.info(
            f"-- Downloaded and cropped mask image in: {round((end - start) * 1000)} ms"
        )
        if KANDINSKY_2_2_IN_CPU_WHEN_IDLE:
            pipe.inpaint = move_pipe_to_device(
                pipe=pipe.inpaint,
                name=f"{kandinsky_2_2_model_name} Inpaint",
                device=DEVICE,
            )
        img_emb = pipe.prior(
            prompt=prompt,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE_SCALE,
            num_images_per_prompt=num_outputs,
            generator=generator,
        )
        neg_emb = pipe.prior(
            prompt=negative_prompt,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE_SCALE,
            num_images_per_prompt=num_outputs,
            generator=generator,
        )
        output_images = pipe.inpaint(
            image=[init_image] * num_outputs,
            mask_image=[mask_image] * num_outputs,
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=neg_emb.image_embeds,
            width=init_image.width,
            height=init_image.height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
    elif init_image_url is not None:
        pipe.text2img.scheduler = get_scheduler(scheduler, pipe.text2img)
        start = time.time()
        init_image = download_and_fit_image(init_image_url, width, height)
        end = time.time()
        custom_logger.info(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)} ms"
        )
        start = time.time()
        images_and_texts = [prompt, init_image]
        weights = [prompt_strength, 1 - prompt_strength]
        if KANDINSKY_2_2_IN_CPU_WHEN_IDLE:
            pipe.text2img = move_pipe_to_device(
                pipe=pipe.text2img, model_name=kandinsky_2_2_model_name, device=DEVICE
            )
        prior_out = pipe.prior.interpolate(
            images_and_texts,
            weights,
            negative_prompt=negative_prompt,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE_SCALE,
            num_images_per_prompt=num_outputs,
            generator=generator,
        )
        output_images = pipe.text2img(
            **prior_out,
            width=width,
            height=height,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
    else:
        pipe.text2img.scheduler = get_scheduler(scheduler, pipe.text2img)
        if KANDINSKY_2_2_IN_CPU_WHEN_IDLE:
            pipe.text2img = move_pipe_to_device(
                pipe=pipe.text2img, model_name=kandinsky_2_2_model_name, device=DEVICE
            )
        img_emb = pipe.prior(
            prompt=prompt,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE_SCALE,
            num_images_per_prompt=num_outputs,
            generator=generator,
        )
        neg_emb = pipe.prior(
            prompt=negative_prompt,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE_SCALE,
            num_images_per_prompt=num_outputs,
            generator=generator,
        )
        output_images = pipe.text2img(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=neg_emb.image_embeds,
        ).images

    output_images = crop_images(image_array=output_images, width=width, height=height)

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

    if KANDINSKY_2_2_IN_CPU_WHEN_IDLE:
        pipe.prior = move_pipe_to_device(
            pipe=pipe.prior,
            model_name=f"{kandinsky_2_2_model_name} Prior",
            device="cpu",
        )
        if (
            init_image_url is not None
            and mask_image_url is not None
            and pipe.inpaint is not None
        ):
            pipe.inpaint = move_pipe_to_device(
                pipe=pipe.inpaint,
                model_name=f"{kandinsky_2_2_model_name} Inpaint",
                device="cpu",
            )
        else:
            pipe.text2img = move_pipe_to_device(
                pipe=pipe.text2img, model_name=kandinsky_2_2_model_name, device="cpu"
            )

    return filtered_output_images, nsfw_count
