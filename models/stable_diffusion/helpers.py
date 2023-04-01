from .constants import SD_SCHEDULERS
from PIL import ImageOps


def get_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def fit_image(image, width, height):
    resized_image = ImageOps.fit(image, (width, height))
    return resized_image
