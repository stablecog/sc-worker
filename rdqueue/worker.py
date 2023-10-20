import datetime
import json
import queue
import os
import traceback
from typing import Any, Dict, Iterable, Tuple, Callable

from boto3_type_annotations.s3 import ServiceResource
import redis

from rdqueue.events import Status, Event
from predict.image.predict import (
    PredictInput as PredictInputForImage,
    predict as predict_for_image,
    PredictResult as PredictResultForImage,
)
from predict.voiceover.predict import (
    PredictInput as PredictInputForVoiceover,
    predict as predict_for_voiceover,
    PredictResult as PredictResultForVoiceover,
)
from shared.helpers import format_datetime
from predict.image.setup import ModelsPack as ModelsPackForImage
from predict.voiceover.setup import ModelsPack as ModelsPackForVoiceover
from shared.webhook import post_webhook


def start_redis_queue_worker(
    worker_type: str,
    redis: redis.Redis,
    input_queue: str,
    s3_client: ServiceResource,
    s3_bucket: str,
    upload_queue: queue.Queue[Dict[str, Any]],
    models_pack: ModelsPackForImage | ModelsPackForVoiceover,
) -> None:
    print(f"Starting redis queue worker, bucket is: {s3_bucket}\n")

    input_queue = input_queue
    s3_client = s3_client
    s3_bucket = s3_bucket
    upload_queue = upload_queue
    workerName = os.environ.get("WORKER_NAME")
    consumer_id = f"cog-{workerName}"
    # 1 minute
    autoclaim_messages_after = 1 * 60

    print(f"Waiting for message on {input_queue}\n")

    # TODO - figure out how to exit this with SIGINT/SIGTERM
    while True:
        try:
            # Receive message
            # first, try to autoclaim old messages from pending queue
            raw_messages = redis.execute_command(
                "XAUTOCLAIM",
                input_queue,
                input_queue,
                consumer_id,
                str(autoclaim_messages_after * 1000),
                "0-0",
                "COUNT",
                1,
            )
            # format: [[b'1619393873567-0', [b'mykey', b'myval']]]
            # since redis==4.3.4 an empty response from xautoclaim is indicated by [[b'0-0', []]]
            if (
                raw_messages
                and raw_messages[0] is not None
                and len(raw_messages[0]) == 2
            ):
                key, raw_message = raw_messages[0]
                assert raw_message[0] == b"value"
                return key.decode(), raw_message[1].decode()

            # if no old messages exist, get message from main queue
            raw_messages = redis.xreadgroup(
                groupname=input_queue,
                consumername=consumer_id,
                streams={input_queue: ">"},
                count=1,
                block=1000,
            )
            if not raw_messages:
                continue

            # format: [[b'mystream', [(b'1619395583065-0', {b'mykey': b'myval6'})]]]
            key, raw_message = raw_messages[0][1][0]
            message_id = key.decode()
            message_json = raw_message[b"value"].decode()

            if message_json is None:
                continue

            message = json.loads(message_json)

            webhook_url = message["webhook_url"]

            print(f"Received message {message_id} on {input_queue}\n")

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

            run_prediction = None
            args = {}
            if worker_type == "voiceover":
                run_prediction = run_prediction_for_voiceover
            else:
                run_prediction = run_prediction_for_image

            args["models_pack"] = models_pack

            for response_event, response in run_prediction(message, **args):
                if "upload_output" in response and isinstance(
                    response["upload_output"],
                    PredictResultForVoiceover
                    if worker_type == "voiceover"
                    else PredictResultForImage,
                ):
                    print(f"-- Upload: Putting to queue")
                    upload_queue.put(response)
                    print(f"-- Upload: Put to queue")
                elif response_event in events_filter:
                    status_code = post_webhook(webhook_url, response)
                    print(f"-- Webhook: {status_code}")

            redis.xack(input_queue, input_queue, message_id)
            redis.xdel(input_queue, message_id)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Failed to handle message: {tb}\n")
            try:
                redis.xack(input_queue, input_queue, message_id)
                redis.xdel(input_queue, message_id)
            except:
                pass


def run_prediction_for_image(
    message: Dict[str, Any],
    models_pack: ModelsPackForImage,
) -> Iterable[Tuple[Event, Dict[str, Any]]]:
    """Runs the prediction and yields events and responses."""

    # use the request message as the basis of our response so
    # that we echo back any additional fields sent to us
    response = message
    response["status"] = Status.PROCESSING
    response["outputs"] = None
    response["logs"] = ""

    started_at = datetime.datetime.now()

    try:
        input_obj: Dict[str, Any] = response["input"]
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to start prediction: {tb}\n")
        response["status"] = Status.FAILED
        response["error"] = str(e)
        yield (Event.COMPLETED, response)

        return

    response["started_at"] = format_datetime(started_at)
    response["logs"] = ""

    yield (Event.START, response)

    try:
        predictResult = predict_for_image(
            input=PredictInputForImage(**input_obj),
            models_pack=models_pack,
        )

        if (predictResult.nsfw_count == 0) and (len(predictResult.outputs) == 0):
            raise Exception("Missing outputs and nsfw_count")

        response["upload_prefix"] = input_obj.get("upload_path_prefix", "")
        response["upload_output"] = predictResult
        response["nsfw_count"] = predictResult.nsfw_count

        completed_at = datetime.datetime.now()
        response["completed_at"] = format_datetime(completed_at)

        response["status"] = Status.SUCCEEDED
        response["metrics"] = {
            "predict_time": (completed_at - started_at).total_seconds()
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to run prediction: {tb}\n")
        completed_at = datetime.datetime.now()
        response["completed_at"] = format_datetime(completed_at)
        response["status"] = Status.FAILED
        response["error"] = str(e)
    finally:
        yield (Event.COMPLETED, response)


def run_prediction_for_voiceover(
    message: Dict[str, Any],
    models_pack: ModelsPackForVoiceover,
) -> Iterable[Tuple[Event, Dict[str, Any]]]:
    """Runs the prediction and yields events and responses."""

    # use the request message as the basis of our response so
    # that we echo back any additional fields sent to us
    response = message
    response["status"] = Status.PROCESSING
    response["outputs"] = None
    response["logs"] = ""

    started_at = datetime.datetime.now()

    try:
        input_obj: Dict[str, Any] = response["input"]
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to start prediction: {tb}\n")
        response["status"] = Status.FAILED
        response["error"] = str(e)
        yield (Event.COMPLETED, response)

        return

    response["started_at"] = format_datetime(started_at)
    response["logs"] = ""

    yield (Event.START, response)

    try:
        predictResult = predict_for_voiceover(
            input=PredictInputForVoiceover(**input_obj),
            models_pack=models_pack,
        )

        if len(predictResult.outputs) == 0:
            raise Exception("Missing outputs")

        response["upload_prefix"] = input_obj.get("upload_path_prefix", "")
        response["upload_output"] = predictResult

        completed_at = datetime.datetime.now()
        response["completed_at"] = format_datetime(completed_at)

        response["status"] = Status.SUCCEEDED
        response["metrics"] = {
            "predict_time": (completed_at - started_at).total_seconds()
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to run prediction: {tb}\n")
        completed_at = datetime.datetime.now()
        response["completed_at"] = format_datetime(completed_at)
        response["status"] = Status.FAILED
        response["error"] = str(e)
    finally:
        yield (Event.COMPLETED, response)
