from threading import Thread, Event
from typing import Any, Dict
import logging
import os
import signal
import queue

import redis
import boto3
from boto3_type_annotations.s3 import ServiceResource
from botocore.config import Config
from dotenv import load_dotenv
import pika
import torch

from predict.image.setup import setup as image_setup
from predict.voiceover.setup import setup as voiceover_setup
from rabbitmq_consumer.worker import start_amqp_queue_worker
from upload.constants import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME_UPLOAD,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)
from upload.worker import start_upload_worker
from clipapi.app import run_clipapi

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(threadName)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Define an event to signal all threads to exit
shutdown_event = Event()

if __name__ == "__main__":
    if torch.cuda.is_available() is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    load_dotenv()

    WORKER_TYPE = os.environ.get("WORKER_TYPE", "image")

    redisUrl = os.environ.get("REDIS_URL")

    amqpUrl = os.environ.get("RABBITMQ_AMQP_URL", None)
    if amqpUrl is None:
        raise ValueError("Missing RABBITMQ_AMQP_URL environment variable.")
    amqpExchangeName = os.environ.get("RABBITMQ_EXCHANGE_NAME", None)
    if amqpExchangeName is None:
        raise ValueError("Missing RABBITMQ_EXCHANGE_NAME environment variable.")
    # Comma separated list of supported model UUIDs
    workerCapabilitiesStr = os.environ.get("WORKER_SUPPORTED_MODELS", None)
    if workerCapabilitiesStr is None:
        raise ValueError("Missing WORKER_SUPPORTED_MODELS environment variable.")
    workerCapabilities = workerCapabilitiesStr.split(",")

    # S3 client
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

    if WORKER_TYPE == "voiceover":
        models_pack = voiceover_setup()
    else:
        models_pack = image_setup()

    # Setup redis
    redisConn = redis.BlockingConnectionPool.from_url(redisUrl)

    # Create queue for thread communication
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    # Create rabbitmq connection
    # Parse the AMQP URL
    params = pika.URLParameters(amqpUrl)

    # Establish the connection using the parsed parameters
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    # Setup signal handler for exit
    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            print("Signal received, shutting down...")
            shutdown_event.set()
            channel.stop_consuming()
            channel.close()
            channel.connection.close()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create rabbitmq worker thread
    mq_worker_thread = Thread(
        target=lambda: start_amqp_queue_worker(
            worker_type=WORKER_TYPE,
            channel=channel,
            supported_models=workerCapabilities,
            exchange_name=amqpExchangeName,
            upload_queue=upload_queue,
            models_pack=models_pack,
            shutdown_event=shutdown_event,
            redisConn=redisConn,
        )
    )

    # Create upload thread
    upload_thread = Thread(
        target=lambda: start_upload_worker(
            worker_type=WORKER_TYPE,
            q=upload_queue,
            s3=s3,
            s3_bucket=S3_BUCKET_NAME_UPLOAD,
            shutdown_event=shutdown_event,
        )
    )

    try:
        mq_worker_thread.start()
        upload_thread.start()
        if WORKER_TYPE == "image":
            clipapi_thread = Thread(target=lambda: run_clipapi(models_pack=models_pack))
            clipapi_thread.start()
            clipapi_thread.join()
        mq_worker_thread.join()
        upload_thread.join()
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully. The signal handler already sets the shutdown_event.
    finally:
        # Any other cleanup in the main thread you want to perform.
        print("Main thread cleanup done!")
