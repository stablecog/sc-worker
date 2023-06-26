import os
import time
from models.constants import DEVICE
from shared.helpers import download_image, fit_image
import torch


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
    print(f"Using seed: {seed}")

    if prompt_prefix is not None:
        prompt = f"{prompt_prefix} {prompt}"

    if negative_prompt_prefix is not None:
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = negative_prompt_prefix
        else:
            negative_prompt = f"{negative_prompt_prefix} {negative_prompt}"

    image_embeds, negative_image_embeds = pipe["prior"](
        prompt, negative_prompt
    ).to_tuple()

    generator = torch.Generator(DEVICE).manual_seed(seed)
    args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_embeds": image_embeds,
        "negative_image_embeds": negative_image_embeds,
        "num_images_per_prompt": num_outputs,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "generator": generator,
    }

    output = None
    if init_image_url is not None:
        start_i = time.time()
        init_image = download_image(init_image_url)
        init_image = fit_image(init_image, width, height)
        end_i = time.time()
        print(
            f"-- Downloaded and cropped init image in: {round((end_i - start_i) * 1000)} ms"
        )
        output = pipe["img2img"](
            image=init_image,
            strength=prompt_strength,
            **args,
        )
    else:
        output = pipe["text2img"](
            **args,
        )

    output_images = []
    nsfw_count = 0

    if output.nsfw_content_detected is not None:
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
