from threading import Thread
from typing import Any, Dict, Callable
import os
import queue

import redis
import boto3
from boto3_type_annotations.s3 import ServiceResource
from botocore.config import Config
from dotenv import load_dotenv
import torch

from predict.setup import setup
from rdqueue.worker import start_redis_queue_worker
from upload.constants import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME_MODELS,
    S3_BUCKET_NAME_UPLOAD,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)
from upload.worker import start_upload_worker
from clipapi.app import run_clipapi

if __name__ == "__main__":
    if torch.cuda.is_available() is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    load_dotenv()

    redisUrl = os.environ.get("REDIS_URL")
    redisInputQueue = os.environ.get("REDIS_INPUT_QUEUE")
    redisWorkerId = os.environ.get("WORKER_NAME", None)
    if redisWorkerId is None:
        raise ValueError("Missing WORKER_NAME environment variable.")

    # Configure S3 client
    s3: ServiceResource = boto3.resource(
        "s3",
        region_name=S3_REGION,
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=5,
        ),
    )

    # Setup predictor
    models_pack = setup(s3, S3_BUCKET_NAME_MODELS)

    # Setup redis
    redisConn = redis.BlockingConnectionPool.from_url(redisUrl)

    # Create queue for thread communication
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    # Create redis worker thread
    redis_worker_thread = Thread(
        target=lambda: start_redis_queue_worker(
            redis=redis.Redis(
                connection_pool=redisConn, socket_keepalive=True, socket_timeout=1000
            ),
            input_queue=redisInputQueue,
            s3_client=s3,
            s3_bucket=S3_BUCKET_NAME_UPLOAD,
            upload_queue=upload_queue,
            models_pack=models_pack,
        )
    )

    # Create upload thread
    upload_thread = Thread(
        target=lambda: start_upload_worker(
            q=upload_queue,
            s3=s3,
            s3_bucket=S3_BUCKET_NAME_UPLOAD,
        )
    )

    # Create clip API thread
    clipapi_thread = Thread(target=lambda: run_clipapi(models_pack=models_pack))

    redis_worker_thread.start()
    upload_thread.start()
    clipapi_thread.start()
    redis_worker_thread.join()
    upload_thread.join()
    clipapi_thread.join()
