import os
import time
from models.constants import DEVICE
from models.kandinsky.constants import KANDIKSKY_SCHEDULERS
from shared.helpers import download_image, fit_image
import torch
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image


def create_scaled_mask(width, height, scale_factor):
    # First, create an initial mask filled with zeros
    mask = np.zeros((height, width), dtype=np.float32)

    # Calculate the dimensions of the scaled region
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)

    # Calculate the top left position of the scaled region
    start_x = (width - scaled_width) // 2
    start_y = (height - scaled_height) // 2

    # Set the pixels within the scaled region to one (white)
    mask[start_y : start_y + scaled_height, start_x : start_x + scaled_width] = 1.0

    return mask


def resize_to_mask(img, mask):
    # Identify the "white" region in the mask
    where_white = np.where(mask == 1.0)

    # Calculate the dimensions of the "white" region
    min_y, max_y = np.min(where_white[0]), np.max(where_white[0])
    min_x, max_x = np.min(where_white[1]), np.max(where_white[1])

    # Get the width and height of the "white" region
    region_width = max_x - min_x
    region_height = max_y - min_y

    # Resize the image to match the dimensions of the "white" region
    resized_img = img.resize((region_width, region_height))

    # Create a new image filled with transparent pixels
    new_img = Image.new("RGBA", img.size, (0, 0, 0, 0))

    # Paste the resized image onto the new image at the appropriate location
    new_img.paste(resized_img, (min_x, min_y))

    return new_img


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
    model,
    pipe,
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
    args = {
        "num_steps": num_inference_steps,
        "batch_size": num_outputs,
        "guidance_scale": guidance_scale,
        "h": height,
        "w": width,
        "sampler": KANDIKSKY_SCHEDULERS[scheduler],
        "prior_cf_scale": 4,
        "prior_steps": "5",
        "negative_prior_prompt": negative_prompt,
        "negative_decoder_prompt": "",
    }

    pipe_text2img = pipe["text2img"]
    pipe_inpainting = pipe["inpainting"]

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
        output_images = pipe_text2img.mix_images(
            images_and_texts,
            weights,
            **args,
        )
    else:
        output_images = pipe_text2img.generate_text2img(
            prompt,
            **args,
        )
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
    for i, image in enumerate(filtered_output_images):
        mask = create_scaled_mask(width, height, 0.5)
        init_image = resize_to_mask(image, mask)
        filtered_output_images[i] = pipe_inpainting.generate_inpainting(
            prompt=prompt,
            pil_img=init_image,
            img_mask=mask,
            **args,
        )[0]
    return filtered_output_images, nsfw_count
