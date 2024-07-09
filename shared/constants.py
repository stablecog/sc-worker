import os
from .helpers import clean_prefix_or_suffix_space
from dotenv import load_dotenv

load_dotenv()

WORKER_VERSION = "v2.74"
MODELS_FROM_ENV = os.environ.get("MODELS", "all")
MODELS_FROM_ENV_LIST = map(
    lambda x: clean_prefix_or_suffix_space(x), MODELS_FROM_ENV.split(",")
)
