import os
import boto3
from boto3.s3.transfer import TransferConfig
from .constants import SD_SCHEDULERS
from .constants import SD_MODELS, SD_MODEL_CACHE
import concurrent.futures
from io import BytesIO
from PIL import Image


s3 = boto3.resource(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)
bucket_name = os.environ.get("S3_BUCKET_NAME")


def download_sd_model(key):
    model_id = SD_MODELS[key]["id"]
    model_dir = SD_MODEL_CACHE + "/" + "models--" + model_id.replace("/", "--")
    print(f"⏳ Downloading model: {model_id}")
    bucket = s3.Bucket(bucket_name)
    # Loop through all files in the S3 directory
    for object in bucket.objects.filter(Prefix=model_dir):
        # Get the file key and local file path
        key = object.key
        local_file_path = key
        # Skip if the local file already exists and is the same size
        if (
            os.path.exists(local_file_path)
            and os.path.getsize(local_file_path) == object.size
        ):
            continue
        # Create the local directory if it doesn't exist
        local_directory_path = os.path.dirname(local_file_path)
        if not os.path.exists(local_directory_path):
            os.makedirs(local_directory_path)
        print(f"Downloading: {key}")
        bucket.download_file(key, local_file_path)
    print(f"✅ Downloaded model: {key}")
    return {"key": key}


def download_sd_models_concurrently():
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [executor.submit(download_sd_model, key) for key in SD_MODELS]
        # Wait for all tasks to complete
        results = [
            task.result() for task in concurrent.futures.as_completed(download_tasks)
        ]
    executor.shutdown(wait=True)


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def png_image_to_bytes(image: Image.Image) -> bytes:
    with BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()
