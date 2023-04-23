import io
import json
import time
import traceback
import queue
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Iterable, Dict, Any, Callable

import redis
from boto3_type_annotations.s3 import ServiceResource
from PIL import Image

from predict.predict import PredictResult, PredictOutput
from rdqueue.events import Status
from shared.helpers import ensure_trailing_slash, parse_content_type
from shared.webhook import post_webhook


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

    _pil_image = pil_image
    if target_extension == "jpeg":
        print(f"-- Upload: Converting to JPEG")
        _pil_image = _pil_image.convert("RGB")
    img_format = target_extension.upper()
    img_bytes = io.BytesIO()
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


def upload_files(
    uploadObjects: List[PredictOutput],
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
        print(f"-- Upload: Got result")
        results.append(
            {"image": task.result(), "image_embed": uo.open_clip_image_embed}
        )

    end = time.time()
    print(
        f"📤 All converted and uploaded to S3 in: {round((end - start) *1000)} ms - Bucket: {s3_bucket} 📤"
    )

    return results


def start_upload_worker(
    q: queue.Queue[Dict[str, Any]],
    s3: ServiceResource,
    s3_bucket: str,
):
    """Starts a loop to read from the queue and upload files to S3, send responses to redis"""
    print("Starting upload thread...\n")
    # TODO - figure out how to exit this with SIGINT/SIGTERM
    while True:
        try:
            print(f"-- Upload: Waiting for queue --\n")
            uploadMsg: List[Dict[str, Any]] = q.get()
            print(f"-- Upload: Got from queue --\n")
            if "upload_output" in uploadMsg:
                predict_result: PredictResult = uploadMsg["upload_output"]
                if len(predict_result.outputs) > 0:
                    print(
                        f"-- Upload: Uploading {len(predict_result.outputs)} files --\n"
                    )
                    try:
                        uploadMsg["output"] = {
                            "prompt_embed": predict_result.outputs[
                                0
                            ].open_clip_prompt_embed,
                            "images": upload_files(
                                predict_result.outputs,
                                s3,
                                s3_bucket,
                                uploadMsg["upload_prefix"],
                            ),
                        }
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Error uploading files {tb}\n")
                        uploadMsg["status"] = Status.FAILED
                        uploadMsg["error"] = str(e)
                    print(f"-- Upload: Finished uploading files --\n")

            if "upload_output" in uploadMsg:
                print(f"-- Upload: Deleting upload_output from message --\n")
                del uploadMsg["upload_output"]
            if "upload_prefix" in uploadMsg:
                print(f"-- Upload: Deleting upload_prefix from message --\n")
                del uploadMsg["upload_prefix"]

            print(f"-- Upload: Publishing to WEBHOOK --\n")
            post_webhook(uploadMsg["webhook_url"], uploadMsg)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Exception in upload process {tb}\n")
            print(f"Message was: {uploadMsg}\n")
