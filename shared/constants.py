import os

from .helpers import clean_prefix_or_suffix_space

WORKER_VERSION = "v2.51"
SKIP_SAFETY_CHECKER = os.environ.get("SKIP_SAFETY_CHECKER", "1")
MODELS_FROM_ENV = os.environ.get("MODELS", "Luna Diffusion")
MODELS_FROM_ENV_LIST = map(
    lambda x: clean_prefix_or_suffix_space(x), MODELS_FROM_ENV.split(",")
)
