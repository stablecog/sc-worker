from .constants import SD_SCHEDULERS


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)
