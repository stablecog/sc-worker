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
from scipy.io.wavfile import write
from pydub import AudioSegment
from io import BytesIO
from pydub.silence import split_on_silence
import numpy as np
import textwrap
import torch

from pydub import AudioSegment
from pyloudnorm import Meter, normalize
from io import BytesIO


from predict.voiceover.classes import RemoveSilenceParams
from tabulate import tabulate


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


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


def time_it(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"Function {func.__name__!r} executed in {((t2-t1)*1000):.0f}ms")
        return result

    return wrap_func


class time_code_block:
    def __init__(self, prefix=None):
        self.prefix = prefix

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = (self.end_time - self.start_time) * 1000
        statement = f"Executed in: {self.elapsed_time:.2f} ms"
        if self.prefix:
            statement = f"{self.prefix} - {statement}"
        print(statement)


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


def download_image_from_s3(key, bucket):
    try:
        image_object = bucket.Object(key)
        image_data = image_object.get().get("Body").read()
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        return None


def download_images_from_s3(keys, bucket, max_workers=25):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(download_image_from_s3, keys, [bucket] * len(keys)))

    return images


T = TypeVar("T")


def return_value_if_in_list(value: T, list_of_values: List[T]) -> bool:
    if value not in list_of_values:
        raise ValueError(f'"{value}" is not in the list of choices')
    return value


def numpy_to_wav_bytes(numpy_array, sample_rate):
    wav_io = BytesIO()
    write(wav_io, sample_rate, numpy_array)
    wav_io.seek(0)
    return wav_io


def remove_silence_from_wav(
    wav_bytes: BytesIO,
    remove_silence_params: RemoveSilenceParams,
) -> BytesIO:
    audio_segment = AudioSegment.from_wav(wav_bytes)
    audio_chunks = split_on_silence(
        audio_segment,
        min_silence_len=remove_silence_params.min_silence_len,
        silence_thresh=remove_silence_params.silence_thresh,
        keep_silence=remove_silence_params.keep_silence_len,
    )
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    wav_io = BytesIO()
    combined.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io


def convert_wav_to_mp3(wav_bytes: BytesIO):
    audio_segment = AudioSegment.from_wav(wav_bytes)
    mp3_io = BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")
    mp3_io.seek(0)
    return mp3_io


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


def do_normalize_audio_loudness(audio_arr, sample_rate, target_lufs=-16):
    s = time.time()
    # Create a meter instance
    meter = Meter(sample_rate)
    # Measure the loudness of the audio
    loudness = meter.integrated_loudness(audio_arr)
    # Normalize the audio to the target LUFS
    normalized_audio_arr = normalize.loudness(audio_arr, loudness, target_lufs)
    e = time.time()
    print(f"🔊 Normalized audio loudness in: {round((e - s) * 1000)} ms 🔊")
    return normalized_audio_arr


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


def print_tuple(a, b):
    print(tabulate([[a, b]], tablefmt="simple_grid"))


def clean_prefix_or_suffix_space(text: str):
    if text.startswith(" "):
        text = text[1:]
    if text.endswith(" "):
        text = text[:-1]
    return text


def log_gpu_memory(device_id=0, message=None):
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
            ["Allocated / Total (GB)", total_and_allocated_str],
            ["Cached Memory (GB)", cached_str],
        ]
        if message is not None:
            print(message)
        print(
            tabulate([["GPU Memory Log", "Value"]] + log_table, tablefmt="double_grid")
        )
    except Exception as e:
        print(f"Failed to log GPU memory")
