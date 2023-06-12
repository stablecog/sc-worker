from boto3_type_annotations.s3 import ServiceResource
from PIL import Image
from shared.helpers import (
    ensure_trailing_slash,
    parse_content_type,
    convert_wav_to_mp3,
    remove_silence_from_wav,
)
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
    audio_bytes: BytesIO,
    remove_silence: bool,
    sample_rate: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    if remove_silence:
        s = time.time()
        audio_bytes = remove_silence_from_wav(audio_bytes)
        e = time.time()
        print(f"🔊 Removed silence in: {round((e - s) *1000)} ms 🔊")
    s_conv = time.time()
    content_type = "audio/wav"
    if target_extension == "mp3":
        content_type = "audio/mpeg"
        audio_bytes = convert_wav_to_mp3(audio_bytes)
    e_conv = time.time()
    print(
        f"Converted audio in: {round((e_conv - s_conv) *1000)} ms - {target_extension}"
    )

    key = f"{str(uuid.uuid4())}.{target_extension}"
    if upload_path_prefix is not None and upload_path_prefix != "":
        key = f"{ensure_trailing_slash(upload_path_prefix)}{key}"

    start_upload = time.time()
    print(f"-- Upload: Uploading to S3")
    s3.Bucket(s3_bucket).put_object(Body=audio_bytes, Key=key, ContentType=content_type)
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
                    uo.audio_bytes,
                    uo.remove_silence,
                    uo.sample_rate,
                    uo.target_extension,
                    upload_path_prefix,
                )
            )

    # Get results
    results = []
    for task in tasks:
        print(f"-- Upload: Got result")
        results.append(
            {"audio_file": task.result(), "audio_duration": uo.audio_duration}
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results
