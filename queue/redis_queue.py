import datetime
import json
import queue
import sys
import traceback
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from boto3_type_annotations.s3 import ServiceResource
import redis
import numpy as np
import uuid
from PIL import Image
from lingua import LanguageDetector

from events import Status, Event
from predict.predict import predict


class RedisQueueWorker:
    def __init__(
        self,
        redis_url: str,
        input_queue: str,
        s3_client: ServiceResource,
        s3_bucket: str,
        upload_queue: queue.Queue[Dict[str, Any]],
        txt2img_pipes: Dict[str, Any],
        upscaler_pipe: Callable[[np.ndarray | Image.Image, Any, Any], Image.Image],
        upscaler_args: Any,
        language_detector_pipe: LanguageDetector,
        max_failure_count: Optional[int] = None,
    ):
        # We want to do upload on a separate thread so use a queue to
        self.redis_url = redis_url
        self.input_queue = input_queue
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.upload_queue = upload_queue
        self.txt2img_pipes = txt2img_pipes
        self.upscaler_pipe = upscaler_pipe
        self.upscaler_args = upscaler_args
        self.language_detector_pipe = language_detector_pipe
        self.consumer_id = f"cog-{uuid.uuid4()}"
        self.max_failure_count = max_failure_count
        # 1 minute
        self.autoclaim_messages_after = 1 * 60

        self.redis = redis.from_url(self.redis_url)
        self.should_exit = False
        self.stats_queue_length = 100

        sys.stderr.write(
            f"Connected to Redis: {self.redis.get_connection_kwargs().get('host')}\n"
        )

    def receive_message(self) -> Tuple[Optional[str], Optional[str]]:
        # first, try to autoclaim old messages from pending queue
        raw_messages = self.redis.execute_command(
            "XAUTOCLAIM",
            self.input_queue,
            self.input_queue,
            self.consumer_id,
            str(self.autoclaim_messages_after * 1000),
            "0-0",
            "COUNT",
            1,
        )
        # format: [[b'1619393873567-0', [b'mykey', b'myval']]]
        # since redis==4.3.4 an empty response from xautoclaim is indicated by [[b'0-0', []]]
        if raw_messages and raw_messages[0] is not None and len(raw_messages[0]) == 2:
            key, raw_message = raw_messages[0]
            assert raw_message[0] == b"value"
            return key.decode(), raw_message[1].decode()

        # if no old messages exist, get message from main queue
        raw_messages = self.redis.xreadgroup(
            groupname=self.input_queue,
            consumername=self.consumer_id,
            streams={self.input_queue: ">"},
            count=1,
            block=1000,
        )
        if not raw_messages:
            return None, None

        # format: [[b'mystream', [(b'1619395583065-0', {b'mykey': b'myval6'})]]]
        key, raw_message = raw_messages[0][1][0]
        return key.decode(), raw_message[b"value"].decode()

    def redis_publisher(self, redis_key: str) -> Callable:
        """Publishes the response to the redis key, alternative to webhook"""

        def setter(response: Any) -> None:
            self.redis.publish(redis_key, json.dumps(response))

        return setter

    def start_redis_queue_worker(self) -> None:
        sys.stderr.write("Starting redis queue worker\n")

        failure_count = 0
        sys.stderr.write(f"Waiting for message on {self.input_queue}\n")

        # TODO - figure out how to exit this with SIGINT/SIGTERM
        while True:
            try:
                message_id, message_json = self.receive_message()
                if message_json is None:
                    # tight loop in order to respect self.should_exit
                    continue

                message = json.loads(message_json)

                redis_key = message["redis_pubsub_key"]
                self.send_response = self.redis_publisher(redis_key)

                sys.stderr.write(
                    f"Received message {message_id} on {self.input_queue}\n"
                )

                if "webhook_events_filter" in message:
                    valid_events = {ev.value for ev in Event}

                    for event in message["webhook_events_filter"]:
                        if event not in valid_events:
                            raise ValueError(
                                f"Invalid webhook event {event}! Must be one of {valid_events}"
                            )

                    # We always send the completed event
                    events_filter = set(message["webhook_events_filter"]) | {
                        Event.COMPLETED
                    }
                else:
                    events_filter = Event.default_events()

                for response_event, response in self.run_prediction(message):
                    if (
                        "upload_outputs" in response
                        and len(response["upload_outputs"]) > 0
                    ):
                        self.upload_queue.put(response)
                    elif response_event in events_filter:
                        self.send_response(response)

                if self.max_failure_count is not None:
                    # Keep track of runs of failures to catch the situation
                    # where the worker has gotten into a bad state where it can
                    # only fail predictions, but isn't exiting.
                    if response["status"] == Status.FAILED:
                        failure_count += 1
                        if failure_count > self.max_failure_count:
                            self.should_exit = True
                            print(
                                f"Had {failure_count} failures in a row, exiting...",
                                file=sys.stderr,
                            )
                    else:
                        failure_count = 0

                self.redis.xack(self.input_queue, self.input_queue, message_id)
                self.redis.xdel(self.input_queue, message_id)

            except Exception as e:
                tb = traceback.format_exc()
                sys.stderr.write(f"Failed to handle message: {tb}\n")

    def run_prediction(
        self, message: Dict[str, Any], should_cancel: Callable
    ) -> Iterable[Tuple[Event, Dict[str, Any]]]:
        """Runs the prediction and yields events and responses."""

        # use the request message as the basis of our response so
        # that we echo back any additional fields sent to us
        response = message
        response["status"] = Status.PROCESSING
        response["output"] = None
        response["logs"] = ""

        started_at = datetime.datetime.now()

        try:
            input_obj: Dict[str, Any] = response["input"]
        except Exception as e:
            response["status"] = Status.FAILED
            response["error"] = str(e)
            yield (Event.COMPLETED, response)

            return

        response["started_at"] = format_datetime(started_at)
        response["logs"] = ""

        yield (Event.START, response)

        timed_out = False
        was_canceled = False
        done_event = None
        output_type = None
        had_error = False

        # If we have outputs that we need to upload
        response["upload_outputs"] = []

        try:
            predictResult = predict(
                prompt=input_obj.get("prompt"),
                negative_prompt=input_obj.get("negative_prompt"),
                width=input_obj.get("width"),
                height=input_obj.get("height"),
                num_outputs=input_obj.get("num_outputs"),
                num_inference_steps=input_obj.get("num_inference_steps"),
                guidance_scale=input_obj.get("guidance_scale"),
                scheduler=input_obj.get("scheduler"),
                model=input_obj.get("model"),
                seed=input_obj.get("seed"),
                prompt_flores_200_code=input_obj.get("prompt_flores_200_code"),
                negative_prompt_flores_200_code=input_obj.get(
                    "negative_prompt_flores_200_code"
                ),
                output_image_extension=input_obj.get("output_image_extension"),
                output_image_quality=input_obj.get("output_image_quality"),
                process_type=input_obj.get("process_type"),
                language_detector_pipe=self.language_detector_pipe,
                txt2img_pipes=self.txt2img_pipes,
                upscaler_pipe=self.upscaler_pipe,
                upscaler_args=self.upscaler_args,
            )

            try:
                response["upload_prefix"] = ""
                if "upload_path_prefix" in input_obj.dict():
                    response["upload_prefix"] = input_obj.dict()["upload_path_prefix"]

                if (predictResult.nsfw_count == 0) and (
                    len(predictResult.outputs) == 0
                ):
                    raise Exception("Missing outputs and nsfw_count")

                # Sometimes we could have, 0 outputs but a >0 nsfw_count
                response["output"] = []
                # if len(predictResult.outputs) > 0:
                #     # Copy files to memory
                #     for output in event.payload["outputs"]:
                #         response["upload_outputs"].append(
                #             UploadObject(
                #                 image_bytes=output["image_bytes"],
                #                 image_width=output["image_width"],
                #                 image_height=output["image_height"],
                #                 target_quality=output["target_quality"],
                #                 target_extension=output["target_extension"],
                #             )
                #         )
                # response["nsfw_count"] = event.payload["nsfw_count"]
            except Exception as e:
                sys.stderr.write(f"Error uploading files to S3: {e}\n")
                had_error = True

            completed_at = datetime.datetime.now()
            response["completed_at"] = format_datetime(completed_at)

            # It should only be possible to get here if we got a done event.
            assert done_event

            if had_error:
                response["status"] = Status.FAILED
                response["error"] = "Error uploading files"
            elif done_event.canceled and was_canceled:
                response["status"] = Status.CANCELED
            elif done_event.canceled and timed_out:
                response["status"] = Status.FAILED
                response["error"] = "Prediction timed out"
            elif done_event.error:
                response["status"] = Status.FAILED
                response["error"] = str(done_event.error_detail)
            else:
                response["status"] = Status.SUCCEEDED
                response["metrics"] = {
                    "predict_time": (completed_at - started_at).total_seconds()
                }
        except Exception as e:
            sys.stderr.write(f"Error in prediction: {e}\n")
            completed_at = datetime.datetime.now()
            response["completed_at"] = format_datetime(completed_at)
            response["status"] = Status.FAILED
            response["error"] = str(e)
        finally:
            yield (Event.COMPLETED, response)

            try:
                input_obj.cleanup()
            except Exception as e:
                print(f"Cleanup function caught error: {e}", file=sys.stderr)


def format_datetime(timestamp: datetime.datetime) -> str:
    """
    Formats a datetime in ISO8601 with a trailing Z, so it's also RFC3339 for
    easier parsing by things like Golang.
    """
    return timestamp.isoformat() + "Z"
