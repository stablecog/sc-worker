import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()

WORKER_VERSION = "v2.84"
MODELS_FROM_ENV = os.environ.get("MODELS", "all")
MODELS_FROM_ENV_LIST = map(lambda x: x.lstrip().rstrip(), MODELS_FROM_ENV.split(","))


class TabulateLevels(Enum):
    PRIMARY = "simple_grid"
    SECONDARY = "simple"
