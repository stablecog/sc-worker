import time
from PIL import Image
from typing import Any
from urllib.parse import urlparse
import requests
from io import BytesIO
import logging


from predict.image.classes import ModelsPack, Upscaler
from shared.move_to_cpu import move_other_models_to_cpu


def upscale(
    image: Image.Image | str, upscaler: Upscaler, models_pack: ModelsPack
) -> Image.Image:
    #### Move other models to CPU if needed
    move_other_models_to_cpu(
        main_model_name="upscaler", main_model_pipe="upscaler", models_pack=models_pack
    )
    #####################################
    if is_url(image):
        s = time.time()
        logging.info("🟡 Upscale | Image is a URL, downloading...")
        image = load_image_from_url(image)
        e = time.time()
        logging.info(f"🟢 Upscale | Image downloaded | {round((e - s) * 1000)}ms")

    inf_start_time = time.time()
    upscaled_image = upscaler.pipe.upscale_4x_overlapped(image)
    inf_end_time = time.time()
    logging.info(
        f"🔮 Upscale | Inference | {round((inf_end_time - inf_start_time) * 1000)}ms"
    )

    return upscaled_image


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def load_image_from_url(url, timeout=10, max_size=8 * 1024 * 1024):
    # Validate URL
    if not url or not urlparse(url).scheme:
        raise ValueError("Invalid URL")

    try:
        # Send request with timeout
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(
                f"URL does not point to an image (Content-Type: {content_type})"
            )

        # Check file size
        if len(response.content) > max_size:
            raise ValueError(
                f"Image size ({len(response.content)} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )

        # Load image
        image_data = BytesIO(response.content)
        image = Image.open(image_data)

        return image

    except requests.RequestException as e:
        logging.error(f"Error fetching image from URL: {e}")
        raise
    except (IOError, ValueError) as e:
        logging.error(f"Error processing image: {e}")
        raise
