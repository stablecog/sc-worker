from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--redis-url")
    parser.add_argument("--input-queue")
    parser.add_argument("--s3-access-key")
    parser.add_argument("--s3-secret-key")
    parser.add_argument("--s3-endpoint-url")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-region")
    parser.add_argument("--predict-timeout", type=int)

    args = parser.parse_args()

    print(args.s3_bucket)

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
