from PIL import Image
from shared.helpers import (
    parse_content_type,
    time_log,
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
import logging


def extract_s3_url_from_signed_url(signed_url: str) -> str:
    """Helper function to extract the key from the signed URL."""
    parsed_url = urlparse(signed_url)
    path = parsed_url.path.lstrip("/")
    return f"s3://{path}"


def convert_and_upload_image_to_signed_url(
    pil_image: Image.Image,
    signed_url: str,
    target_quality: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    """Convert an individual image to a target format and upload to the provided signed URL."""

    with time_log(
        f"📨 Converted image to {target_extension}",
        ms=True,
        start_log=False,
        prefix=False,
    ):
        _pil_image = pil_image
        if target_extension == "jpeg":
            _pil_image = _pil_image.convert("RGB")
        img_format = target_extension.upper()
        img_bytes = BytesIO()
        _pil_image.save(img_bytes, format=img_format, quality=target_quality)
        file_bytes = img_bytes.getvalue()

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

    with time_log(
        f"📨 Uploaded image to S3",
        ms=True,
        start_log=False,
        prefix=False,
    ):
        response = session.put(
            signed_url,
            data=file_bytes,
            headers={"Content-Type": parse_content_type(target_extension)},
        )

    if response.status_code != 200:
        logging.info(f"^^ Failed to upload image. Status code: {response.status_code}")
        response.raise_for_status()

    s3_url = extract_s3_url_from_signed_url(signed_url)
    return s3_url


def upload_files_for_image(
    upload_objects: List[PredictOutputForImage],
    signed_urls: List[str],
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Upload all images to S3 in parallel and return the S3 URLs"""
    logging.info(
        f"^^ 📤 🟡 Started - Convert and upload {len(upload_objects)} image(s) to S3 in parallel"
    )
    start = time.time()

    logging.info(
        f"^^ Target extension: {upload_objects[0].target_extension} - Target quality: {upload_objects[0].target_quality}"
    )

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(upload_objects)) as executor:
        logging.info(f"^^ Submitting to thread")
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
        uploadObject = upload_objects[i]
        results.append(
            {
                "image": task.result(),
            }
        )

    end = time.time()
    logging.info(
        f"^^ 📤 🟢 {len(upload_objects)} image(s) converted and uploaded to S3 in: {round((end - start) *1000)}ms"
    )

    return results
