import os
import shutil
from typing import Optional
import datetime
import time
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageOps
from typing import TypeVar, List
from io import BytesIO
import numpy as np
import textwrap
import torch

from io import BytesIO
import logging
from tabulate import tabulate

from models.constants import DEVICE_CUDA
from .constants import TabulateLevels
from contextlib import contextmanager


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.info("Failed to delete %s. Reason: %s" % (file_path, e))


def ensure_trailing_slash(url: str) -> str:
    """
    Adds a trailing slash to `url` if not already present, and then returns it.
    """
    if url.endswith("/"):
        return url
    else:
        return url + "/"


def parse_content_type(extension: str) -> Optional[str]:
    if extension == "jpeg" or extension == "jpg":
        return "image/jpeg"
    elif extension == "png":
        return "image/png"
    elif extension == "webp":
        return "image/webp"

    return None


def format_datetime(timestamp: datetime.datetime) -> str:
    """
    Formats a datetime in ISO8601 with a trailing Z, so it's also RFC3339 for
    easier parsing by things like Golang.
    """
    return timestamp.isoformat() + "Z"


@contextmanager
def time_log(job_name: str, ms=False, start_log=True, prefix=True):
    start_prefix = "🟡 Started | "
    end_prefix = "🟢 Finished | "

    if prefix is False:
        start_prefix = ""
        end_prefix = ""

    if start_log is True:
        logging.info(f"{start_prefix}{job_name}")

    start_time = time.time()

    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        unit = "s"
        if ms:
            execution_time *= 1000
            unit = "ms"
        logging.info(f"{end_prefix}{job_name} | {execution_time:.0f}{unit}")


def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    return Image.open(BytesIO(response.content)).convert("RGB")


def fit_image(image, width, height):
    resized_image = ImageOps.fit(image, (width, height))
    return resized_image


def download_and_fit_image(url, width, height):
    image = download_image(url=url)
    if image.width == width and image.height == height:
        return image
    return fit_image(image, width, height)


def download_and_fit_image_mask(url, width, height, inverted=False):
    image = download_and_fit_image(url, width, height)
    image = image.convert("L")
    mask = 1 - np.array(image) / 255.0 if inverted else np.array(image) / 255.0
    return mask


def download_images(urls, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, url) for url in urls]
        images = [future.result() for future in futures]
    return images


T = TypeVar("T")


def return_value_if_in_list(value: T, list_of_values: List[T]) -> bool:
    if value not in list_of_values:
        raise ValueError(f'"{value}" is not in the list of choices')
    return value


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


def wrap_text(text, width=50):
    # This function wraps the text to a certain width
    return "\n".join(textwrap.wrap(text, width=width))


def pad_image_mask_nd(
    img: np.ndarray, multiple: int, pad_value: int = 255
) -> np.ndarray:
    # Calculate the number of rows and columns to be padded
    pad_rows = (multiple - img.shape[0] % multiple) % multiple
    pad_cols = (multiple - img.shape[1] % multiple) % multiple

    # Pad the image
    # 'constant_values=pad_value' pads with the given value
    padded_img = np.pad(
        img,
        ((pad_rows // 2, (pad_rows + 1) // 2), (pad_cols // 2, (pad_cols + 1) // 2)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def pad_image_nd(img: np.ndarray, multiple: int, pad_value: int = 255) -> np.ndarray:
    # Calculate the number of rows and columns to be padded
    pad_rows = (multiple - img.shape[0] % multiple) % multiple
    pad_cols = (multiple - img.shape[1] % multiple) % multiple

    # Pad the image
    # 'constant_values=255' pads with white for an 8-bit image
    padded_img = np.pad(
        img,
        (
            (pad_rows // 2, (pad_rows + 1) // 2),
            (pad_cols // 2, (pad_cols + 1) // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def pad_image_pil(img: Image.Image, multiple: int, pad_value: int = 255) -> Image.Image:
    # Calculate the number of rows and columns to be padded
    width, height = img.size
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    # Pad the image
    # 'fill=255' pads with white for an 8-bit image
    padded_img = ImageOps.expand(
        img,
        (pad_width // 2, pad_height // 2, (pad_width + 1) // 2, (pad_height + 1) // 2),
        fill=pad_value,
    )

    return padded_img


def pad_dim(width: int, height: int, multiple: int):
    # Calculate the number of rows and columns to be padded
    pad_width = (multiple - width % multiple) % multiple
    pad_height = (multiple - height % multiple) % multiple

    # Add the padding to the original dimensions
    new_width = width + pad_width
    new_height = height + pad_height

    return new_width, new_height


def crop_images(image_array, width, height):
    cropped_images = []
    for image in image_array:
        old_width, old_height = image.size
        if old_width < width or old_height < height:
            cropped_images.append(image)
        else:
            left = (old_width - width) / 2
            top = (old_height - height) / 2
            right = (old_width + width) / 2
            bottom = (old_height + height) / 2

            cropped_image = image.crop((left, top, right, bottom))
            cropped_images.append(cropped_image)
    return cropped_images


def log_gpu_memory(device_id=0, message="Value"):
    try:
        device_properties = torch.cuda.get_device_properties(device_id)

        total_memory = device_properties.total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)

        total = total_memory / (1024**3)
        allocated = allocated_memory / (1024**3)
        total_and_allocated_str = f"{allocated:.1f} / {total:.1f} GB"
        cached = cached_memory / (1024**3)
        cached_str = f"{cached:.1f} GB"
        log_table = [
            ["GPU Memory Log", message],
            ["Allocated / Total (GB)", total_and_allocated_str],
            ["Cached Memory (GB)", cached_str],
        ]
        logging.info(tabulate(log_table, tablefmt=TabulateLevels.PRIMARY.value))
    except Exception as e:
        logging.info(f"Failed to log GPU memory")


def move_pipe_to_device(pipe, model_name, device):
    if pipe is None:
        return None
    s = time.time()
    pipe = pipe.to(device, silence_dtype_warnings=True)
    e = time.time()
    emoji = "🚀" if device == DEVICE_CUDA else "🐌"
    logging.info(
        f"{emoji} Moved {model_name} to {device} in: {round((e - s) * 1000)} ms"
    )
    return pipe
