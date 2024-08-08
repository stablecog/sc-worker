import os
import time

from models.constants import DEVICE_CUDA, is_not_cuda
from models.kandinsky.constants import (
    KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE,
    KANDINSKY_2_2_MODEL_NAME,
)
from shared.move_to_cpu import move_other_models_to_cpu
from .helpers import get_scheduler
from predict.image.classes import KandinskyPipeSet_2_2, ModelsPack
from shared.helpers import (
    crop_images,
    download_and_fit_image,
    download_and_fit_image_mask,
    move_pipe_to_device,
    pad_image_mask_nd,
    pad_image_pil,
)
import torch
from torch.amp import autocast
import logging

PRIOR_STEPS = 25
PRIOR_GUIDANCE_SCALE = 4.0


kandinsky_2_2_negative_prompt_prefix = "overexposed"


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
    pipe: KandinskyPipeSet_2_2,
    safety_checker,
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

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"

    if negative_prompt is None or negative_prompt == "":
        negative_prompt = kandinsky_2_2_negative_prompt_prefix
    else:
        negative_prompt = f"{kandinsky_2_2_negative_prompt_prefix}, {negative_prompt}"

    logging.info(f"Negative prompt for Kandinsky 2.2: {negative_prompt}")

    output_images = []

    if is_not_cuda(pipe.prior.device.type):
        pipe.prior = move_pipe_to_device(
            pipe=pipe.prior,
            model_name=f"{KANDINSKY_2_2_MODEL_NAME} prior",
            device=DEVICE_CUDA,
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
        logging.info(
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
        logging.info(
            f"-- Downloaded and cropped mask image in: {round((end - start) * 1000)} ms"
        )
        if is_not_cuda(pipe.inpaint.device.type):
            pipe.inpaint = move_pipe_to_device(
                pipe=pipe.inpaint,
                name=f"{KANDINSKY_2_2_MODEL_NAME} {main_model_pipe}",
                device=DEVICE_CUDA,
            )
        for i in range(num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            img_emb = pipe.prior(
                prompt=prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            neg_emb = pipe.prior(
                prompt=negative_prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            out = pipe.inpaint(
                image=[init_image] * 1,
                mask_image=[mask_image] * 1,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=neg_emb.image_embeds,
                width=init_image.width,
                height=init_image.height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            output_images.append(out)
    elif init_image_url is not None:
        pipe.text2img.scheduler = get_scheduler(scheduler, pipe.text2img)
        start = time.time()
        init_image = download_and_fit_image(init_image_url, width, height)
        end = time.time()
        logging.info(
            f"-- Downloaded and cropped init image in: {round((end - start) * 1000)} ms"
        )
        start = time.time()
        images_and_texts = [prompt, init_image]
        weights = [prompt_strength, 1 - prompt_strength]
        if KANDINSKY_2_2_KEEP_IN_CPU_WHEN_IDLE:
            pipe.text2img = move_pipe_to_device(
                pipe=pipe.text2img,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} {main_model_pipe}",
                device=DEVICE_CUDA,
            )
        for i in range(num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            prior_out = pipe.prior.interpolate(
                images_and_texts,
                weights,
                negative_prompt=negative_prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            out = pipe.text2img(
                **prior_out,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            output_images.append(out)
    else:
        pipe.text2img.scheduler = get_scheduler(scheduler, pipe.text2img)
        if is_not_cuda(pipe.text2img.device.type):
            pipe.text2img = move_pipe_to_device(
                pipe=pipe.text2img,
                model_name=f"{KANDINSKY_2_2_MODEL_NAME} {main_model_pipe}",
                device=DEVICE_CUDA,
            )
        for i in range(num_outputs):
            generator = torch.Generator(device=DEVICE_CUDA).manual_seed(seed + i)
            img_emb = pipe.prior(
                prompt=prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            neg_emb = pipe.prior(
                prompt=negative_prompt,
                num_inference_steps=PRIOR_STEPS,
                guidance_scale=PRIOR_GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=generator,
            )
            out = pipe.text2img(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
                image_embeds=img_emb.image_embeds,
                negative_image_embeds=neg_emb.image_embeds,
            ).images[0]
            output_images.append(out)

    output_images = crop_images(image_array=output_images, width=width, height=height)

    output_images_nsfw_results = []
    with autocast("cuda"):
        for image in output_images:
            has_nsfw_concepts = False
            result = None
            if safety_checker is not None:
                safety_checker_input = safety_checker["feature_extractor"](
                    images=image, return_tensors="pt"
                ).to(DEVICE_CUDA)
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
