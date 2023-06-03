from boto3_type_annotations.s3 import ServiceResource
from PIL import Image
from shared.helpers import ensure_trailing_slash, parse_content_type
from typing import Any, Dict, Iterable, List
from predict.image.predict import (
    PredictOutput as PredictOutputForImage,
)
from predict.voiceover.predict import (
    PredictOutput as PredictOutputForVoiceover,
)
import time
from io import BytesIO
import uuid
from concurrent.futures import Future, ThreadPoolExecutor


def convert_and_upload_image_to_s3(
    s3: ServiceResource,
    s3_bucket: str,
    pil_image: Image.Image,
    target_quality: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    """Convert an individual image to a target format and upload to S3."""
    start_conv = time.time()

    _pil_image = pil_image
    if target_extension == "jpeg":
        print(f"-- Upload: Converting to JPEG")
        _pil_image = _pil_image.convert("RGB")
    img_format = target_extension.upper()
    img_bytes = BytesIO()
    _pil_image.save(img_bytes, format=img_format, quality=target_quality)
    file_bytes = img_bytes.getvalue()
    end_conv = time.time()
    print(
        f"Converted image in: {round((end_conv - start_conv) *1000)} ms - {img_format} - {target_quality}"
    )

    key = f"{str(uuid.uuid4())}.{target_extension}"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key = f"{ensure_trailing_slash(upload_path_prefix)}{key}"

    content_type = parse_content_type(target_extension)
    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(Body=file_bytes, Key=key, ContentType=content_type)
    end_upload = time.time()
    print(f"Uploaded image in: {round((end_upload - start_upload) *1000)} ms")

    return f"s3://{s3_bucket}/{key}"


def upload_files_for_image(
    uploadObjects: List[PredictOutputForImage],
    s3: ServiceResource,
    s3_bucket: str,
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    print("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(uploadObjects)) as executor:
        print(f"-- Upload: Submitting to thread")
        for uo in uploadObjects:
            tasks.append(
                executor.submit(
                    convert_and_upload_image_to_s3,
                    s3,
                    s3_bucket,
                    uo.pil_image,
                    uo.target_quality,
                    uo.target_extension,
                    upload_path_prefix,
                )
            )

    # Get results
    results = []
    for task in tasks:
        print(f"-- Upload: Got result")
        results.append(
            {"image": task.result(), "image_embed": uo.open_clip_image_embed}
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results


def convert_and_upload_audio_file_to_s3(
    s3: ServiceResource,
    s3_bucket: str,
    audio_file: BytesIO,
    upload_path_prefix: str,
) -> str:
    extension = "wav"
    content_type = "audio/wav"

    key = f"{str(uuid.uuid4())}.{extension}"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key = f"{ensure_trailing_slash(upload_path_prefix)}{key}"

    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(Body=audio_file, Key=key, ContentType=content_type)
    end_upload = time.time()
    print(f"Uploaded audio file in: {round((end_upload - start_upload) *1000)} ms")

    return f"s3://{s3_bucket}/{key}"


def upload_files_for_voiceover(
    uploadObjects: List[PredictOutputForVoiceover],
    s3: ServiceResource,
    s3_bucket: str,
    upload_path_prefix: str,
) -> Iterable[Dict[str, Any]]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    print("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(uploadObjects)) as executor:
        print(f"-- Upload: Submitting to thread")
        for uo in uploadObjects:
            tasks.append(
                executor.submit(
                    convert_and_upload_audio_file_to_s3,
                    s3,
                    s3_bucket,
                    uo.audio_file,
                    upload_path_prefix,
                )
            )

    # Get results
    results = []
    for task in tasks:
        print(f"-- Upload: Got result")
        results.append(
            {"image": task.result(), "image_embed": uo.open_clip_image_embed}
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results
