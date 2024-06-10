import os
from dotenv import load_dotenv

load_dotenv()

WORKER_VERSION = "v2.51"
SKIP_SAFETY_CHECKER = os.environ.get("SKIP_SAFETY_CHECKER", "1")
MODELS_FROM_ENV = os.environ.get("MODELS", "Luna Diffusion")
MODELS_FROM_ENV_LIST = [ x.strip() for x in MODELS_FROM_ENV.split(",") ]