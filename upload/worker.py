import io
import json
import time
import traceback
import queue
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Iterable, Dict, Any, Optional

import redis
from boto3_type_annotations.s3 import ServiceResource
from PIL import Image

from predict.predict import PredictResult, PredictOutput
from rdqueue.events import Status
from shared.helpers import ensure_trailing_slash, parse_content_type


def convert_and_upload_to_s3(
    s3: ServiceResource,
    s3_bucket: str,
    pil_image: Image.Image,
    target_quality: int,
    target_extension: str,
    upload_path_prefix: str,
) -> str:
    """Convert an individual image to a target format and upload to S3."""
    start_conv = time.time()
    img_format = target_extension.upper()
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format=img_format, quality=target_quality)
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
    s3.Bucket(s3_bucket).put_object(Body=file_bytes, Key=key, ContentType=content_type)
    end_upload = time.time()
    print(f"Uploaded image in: {round((end_upload - start_upload) *1000)} ms")

    return f"s3://{s3_bucket}/{key}"


def upload_files(
    uploadObjects: List[PredictOutput],
    s3: ServiceResource,
    s3_bucket: str,
    upload_path_prefix: str,
) -> Iterable[str]:
    """Send all files to S3 in parallel and return the S3 URLs"""
    print("Started - Upload all files to S3 in parallel and return the S3 URLs")
    start = time.time()

    # Run all uploads at same time in threadpool
    tasks: List[Future] = []
    with ThreadPoolExecutor(max_workers=len(uploadObjects)) as executor:
        for uo in uploadObjects:
            tasks.append(
                executor.submit(
                    convert_and_upload_to_s3,
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
        results.append(task.result())

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results


def start_upload_worker(
    q: queue.Queue[Dict[str, Any]],
    s3: ServiceResource,
    bucket: str,
    redis: redis.Redis,
):
    """Starts a loop to read from the queue and upload files to S3, send responses to redis"""
    print("Starting upload thread...\n")
    # TODO - figure out how to exit this with SIGINT/SIGTERM
    while True:
        uploadMsg: List[Dict[str, Any]] = q.get()
        if "upload_output" in uploadMsg:
            predictresult: PredictResult = uploadMsg["upload_output"]
            try:
                uploadMsg["output"] = upload_files(
                    predictresult.outputs, s3, bucket, uploadMsg["upload_prefix"]
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error uploading files {tb}\n")
                uploadMsg["status"] = Status.FAILED
                uploadMsg["error"] = str(e)
            finally:
                if "upload_output" in uploadMsg:
                    del uploadMsg["upload_output"]
                if "upload_prefix" in uploadMsg:
                    del uploadMsg["upload_prefix"]
                redis.publish(uploadMsg["redis_pubsub_key"], json.dumps(uploadMsg))
