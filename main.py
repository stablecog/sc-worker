from argparse import ArgumentParser
import queue

import redis
import boto3
from boto3_type_annotations.s3 import ServiceResource
from botocore.config import Config

from predict.setup import setup
from rdqueue.worker import start_redis_queue_worker

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--redis-url")
    parser.add_argument("--input-queue")
    parser.add_argument("--s3-access-key")
    parser.add_argument("--s3-secret-key")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-region")

    args = parser.parse_args()

    # Setup predictor
    txt2img_pipes, upscaler_pipe, upscaler_args, language_detector_pipe = setup()

    # Setup redis
    redis = redis.from_url(args.redis_url)

    # Configure S3 client
    s3: ServiceResource = boto3.resource(
        "s3",
        region_name=args.s3_region,
        endpoint_url=args.s3_endpoint_url,
        aws_access_key_id=args.s3_access_key,
        aws_secret_access_key=args.s3_secret_key,
        config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    )

    # Start worker
    start_redis_queue_worker(
        redis,
        input_queue=args.input_queue,
        s3_client=s3,
        s3_bucket=args.s3_bucket,
        upload_queue=queue.Queue(),
        txt2img_pipes=txt2img_pipes,
        upscaler_pipe=upscaler_pipe,
        upscaler_args=upscaler_args,
        language_detector_pipe=language_detector_pipe,
    )

    # # Configure boto3 client
    # s3: ServiceResource = boto3.resource(
    #     "s3",
    #     region_name=args.s3_region,
    #     endpoint_url=args.s3_endpoint_url,
    #     aws_access_key_id=args.s3_access_key,
    #     aws_secret_access_key=args.s3_secret_key,
    #     config=Config(retries={"max_attempts": 3, "mode": "standard"}),
    # )
    # worker = RedisQueueWorker(
    #     predictor_ref=predictor_ref,
    #     redis_url=args.redis_url,
    #     input_queue=args.input_queue,
    #     s3_client=s3,
    #     s3_bucket=args.s3_bucket,
    #     consumer_id=args.consumer_id,
    #     predict_timeout=args.predict_timeout,
    #     report_setup_run_url=args.report_setup_run_url,
    #     max_failure_count=args.max_failure_count,
    # )

    # workerThread = Thread(target=worker.start)
    # uploadThread = Thread(target=worker.start_upload_thread)
    # workerThread.start()
    # uploadThread.start()
    # workerThread.join()
    # uploadThread.join()
