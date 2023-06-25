import os
import time
from models.constants import DEVICE
from models.kandinsky.constants import KANDIKSKY_SCHEDULERS
from shared.helpers import download_image, fit_image
import torch
from torch.cuda.amp import autocast


def generate_with_kandinsky(
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
    pipe,
    model,
    safety_checker,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")
    torch.manual_seed(seed)
    print(f"Using seed: {seed}")

    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"
    image_embeds, negative_image_embeds = pipe["prior"](
        prompt, guidance_scale=1.0
    ).to_tuple()
    args = {
        "image_embeds": [image_embeds] * num_outputs,
        "negative_image_embeds": [negative_image_embeds] * num_outputs,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
    }
    output_images = None
    if init_image_url is not None:
        start_i = time.time()
        init_image = download_image(init_image_url)
        init_image = fit_image(init_image, width, height)
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )
        images_and_texts = [prompt, init_image]
        weights = [prompt_strength, 1 - prompt_strength]
        output_images = pipe.mix_images(
            images_and_texts,
            weights,
            **args,
        )
    else:
        output_images = pipe["text2img"](
            prompt=[prompt] * num_outputs,
            negative_prompt=[negative_prompt] * num_outputs,
            **args,
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
