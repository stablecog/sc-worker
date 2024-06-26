from PIL import Image
from shared.helpers import (
    parse_content_type,
)
from typing import Any, Dict, Iterable, List
from predict.image.predict import (
    PredictOutput as PredictOutputForImage,
)
import requests
import time
from io import BytesIO
from concurrent.futures import Future, ThreadPoolExecutor
from urllib.parse import urlparse
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from shared.log import custom_log


def extract_key_from_signed_url(signed_url: str) -> str:
    """Helper function to extract the key from the signed URL."""
    parsed_url = urlparse(signed_url)
    # Extract the bucket name and object key from the URL
    bucket_name = parsed_url.netloc.split(".")[0]
    object_key = parsed_url.path.lstrip("/")
    final_key = f"s3://{bucket_name}/{object_key}"
    return final_key


def convert_and_upload_image_to_signed_url(
    pil_image: Image.Image,
    signed_url: str,
    target_quality: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    """Convert an individual image to a target format and upload to the provided signed URL."""
    start_conv = time.time()

    _pil_image = pil_image
    if target_extension == "jpeg":
        custom_log(f"-- Upload: Converting to JPEG")
        _pil_image = _pil_image.convert("RGB")
    img_format = target_extension.upper()
    img_bytes = BytesIO()
    _pil_image.save(img_bytes, format=img_format, quality=target_quality)
    file_bytes = img_bytes.getvalue()
    end_conv = time.time()
    custom_log(
        f"Converted image in: {round((end_conv - start_conv) * 1000)} ms - {img_format} - {target_quality}"
    )

    # Define the retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504]
        + list(range(400, 429))
        + list(range(431, 500))
        + list(range(501, 600)),  # List of status codes to retry on
        allowed_methods=["PUT"],  # HTTP methods to retry
        backoff_factor=0.5,  # A backoff factor to apply between attempts
    )

    # Create an HTTPAdapter with the retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create a session and mount the adapter
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    start_upload = time.time()
    custom_log(f"-- Upload: Uploading to signed URL")
    response = session.put(
        signed_url,
        data=file_bytes,
        headers={"Content-Type": parse_content_type(target_extension)},
    )
    end_upload = time.time()

    if response.status_code == 200:
        custom_log(f"Uploaded image in: {round((end_upload - start_upload) * 1000)} ms")
        final_key = extract_key_from_signed_url(signed_url)
        custom_log(f"Final key for image is: {final_key}")
        return final_key
    else:
        custom_log(f"Failed to upload image. Status code: {response.status_code}")
        response.raise_for_status()


def upload_files_for_image(
    upload_objects: List[PredictOutputForImage],
    signed_urls: List[str],
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    custom_log("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(upload_objects)) as executor:
        custom_log(f"-- Upload: Submitting to thread")
        for i, uo in enumerate(upload_objects):
            signed_url = signed_urls[i]
            tasks.append(
                executor.submit(
                    convert_and_upload_image_to_signed_url,
                    uo.pil_image,
                    signed_url,
                    uo.target_quality,
                    uo.target_extension,
                    upload_path_prefix,
                )
            )

    # Get results
    results = []
    for i, task in enumerate(tasks):
        custom_log(f"-- Upload: Got result")
        uploadObject = upload_objects[i]
        results.append(
            {
                "image": task.result(),
                "image_embed": uploadObject.open_clip_image_embed,
                "aesthetic_rating_score": uploadObject.aesthetic_rating_score,
                "aesthetic_artifact_score": uploadObject.aesthetic_artifact_score,
            }
        )

    end = time.time()
    custom_log(f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms 📤")

    return results
