from threading import Thread
from typing import Any, Dict
import os
import queue

import redis
import boto3
from boto3_type_annotations.s3 import ServiceResource
from botocore.config import Config
from dotenv import load_dotenv

from predict.setup import setup
from rdqueue.worker import start_redis_queue_worker
from upload.worker import start_upload_worker

if __name__ == "__main__":
    load_dotenv()

    s3Region = os.environ.get("S3_REGION")
    s3EndpointUrl = os.environ.get("S3_ENDPOINT_URL")
    s3BucketNameUpload = os.environ.get("S3_BUCKET_NAME_UPLOAD")
    s3BucketNameModels = os.environ.get("S3_BUCKET_NAME_MODELS")
    s3AccessKey = os.environ.get("AWS_ACCESS_KEY_ID")
    s3SecretKey = os.environ.get("AWS_SECRET_ACCESS_KEY")
    redisUrl = os.environ.get("REDIS_URL")
    redisInputQueue = os.environ.get("REDIS_INPUT_QUEUE")

    # Configure S3 client
    s3: ServiceResource = boto3.resource(
        "s3",
        region_name=s3Region,
        endpoint_url=s3EndpointUrl,
        aws_access_key_id=s3AccessKey,
        aws_secret_access_key=s3SecretKey,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )

    # Setup predictor
    model_pack = setup(s3, s3BucketNameModels)

    # Setup redis
    redis = redis.from_url(redisUrl)

    # Create queue for thread communication
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    # Create redis worker thread
    redis_worker_thread = Thread(
        target=lambda: start_redis_queue_worker(
            redis,
            input_queue=redisInputQueue,
            s3_client=s3,
            s3_bucket=s3BucketNameUpload,
            upload_queue=upload_queue,
            model_pack=model_pack,
        )
    )

    # Create upload thread
    upload_thread = Thread(
        target=lambda: start_upload_worker(
            q=upload_queue, s3=s3, s3_bucket=s3BucketNameUpload, redis=redis
        )
    )

    redis_worker_thread.start()
    upload_thread.start()
    redis_worker_thread.join()
    upload_thread.join()
