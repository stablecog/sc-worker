import os
from boto3_type_annotations.s3 import ServiceResource
from models.swinir.constants import MODEL_DIR_SWINIR, MODEL_NAME_SWINIR
from models.stable_diffusion.constants import SD_MODELS, SD_MODEL_CACHE
import concurrent.futures


def download_all_models_from_bucket(s3: ServiceResource, bucket_name: str):
    if os.environ.get("DOWNLOAD_MODELS_ON_SETUP", "1") == "1":
        download_sd_models_concurrently_from_bucket(s3, bucket_name)
        download_swinir_models_from_bucket(s3, bucket_name)


def download_sd_model_from_bucket(key: str, s3: ServiceResource, bucket_name: str):
    model_id = SD_MODELS[key]["id"]
    model_dir = SD_MODEL_CACHE + "/" + "models--" + model_id.replace("/", "--")
    download_model_from_bucket(model_id, model_dir, s3, bucket_name)


def download_swinir_model_from_bucket(
    model_id: str, s3: ServiceResource, bucket_name: str
):
    model_dir = MODEL_DIR_SWINIR
    if os.path.exists(os.path.join(model_dir, model_id)):
        print("✅ SwinIR models already downloaded")
    else:
        download_model_from_bucket(model_id, model_dir, s3, bucket_name)


def download_model_from_bucket(
    model_id: str, model_dir: str, s3: ServiceResource, bucket_name: str
):
    print(f"⏳ Downloading model: {model_id}")
    bucket = s3.Bucket(bucket_name)
    key = model_id
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


def download_swinir_models_from_bucket(s3: ServiceResource, bucket_name: str):
    download_swinir_model_from_bucket(MODEL_NAME_SWINIR, s3, bucket_name)


def download_sd_models_concurrently_from_bucket(s3: ServiceResource, bucket_name: str):
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        # Start the download tasks
        download_tasks = [
            executor.submit(download_sd_model_from_bucket, key, s3, bucket_name)
            for key in SD_MODELS
        ]
        # Wait for all tasks to complete
        results = [
            task.result() for task in concurrent.futures.as_completed(download_tasks)
        ]
    executor.shutdown(wait=True)
