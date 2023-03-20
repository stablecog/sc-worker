from .constants import SD_SCHEDULERS
import requests
from PIL import Image, ImageOps
from io import BytesIO

def get_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def fit_image(image, width, height):
    resized_image = ImageOps.fit(image, (width, height))
    return resized_image