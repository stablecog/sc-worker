import datetime
import json
import queue
import hashlib
import os
import time
import traceback
from typing import Any, Dict, Iterable, Tuple, Callable
from threading import Event
import logging

from boto3_type_annotations.s3 import ServiceResource
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties
from pika.exceptions import ConnectionClosedByBroker, AMQPConnectionError, AMQPChannelError

from rabbitmq_consumer.events import Status, Event
from rabbitmq_consumer.connection import RabbitMQConnection
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
from tabulate import tabulate


def generate_queue_name_from_capabilities(
    exchange_name: str, capabilities: list[str]
) -> str:
    """Generate a unique queue name based on the provided capabilities."""

    # Sort capabilities for consistency
    sorted_capabilities = sorted(capabilities)

    # Create a string representation
    capabilities_str = "_".join(sorted_capabilities)

    # Generate SHA-1 hash
    hash_object = hashlib.sha1(capabilities_str.encode())
    hex_dig = hash_object.hexdigest()

    # Prefix with "queue_" for clarity
    queue_name = "q." + exchange_name + "." + hex_dig

    return queue_name


callback_in_progress = False


# def should_process(redisConn: redis.Redis, message_id):
#     """See if a message is being processed by another worker"""
#     # Check and set the message_id in Redis atomically
#     result = redisConn.set(message_id, 1, nx=True, ex=300)
#     logging.info(f"Redis set result for message {message_id}: {result}")
#     return bool(result)


def create_amqp_callback(
    queue_name: str,
    worker_type: str,
    upload_queue: queue.Queue[Dict[str, Any]],
    models_pack: ModelsPackForImage | ModelsPackForVoiceover,
):
    """Create the amqp callback to handle rabbitmq messages"""

    def amqp_callback(
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        # if not should_process(redisConn, properties.message_id):
        #     logging.info(f"Message {properties.message_id} is already being processed")
        #     channel.basic_ack(delivery_tag=method.delivery_tag)
        #     return

        global callback_in_progress
        try:
            callback_in_progress = True
            message = json.loads(body.decode("utf-8"))

            webhook_url = message["webhook_url"]

            log_table = [
                ["Queue Name", queue_name],
                ["Message ID", properties.message_id],
                ["Priority", properties.priority],
            ]
            logging.info("\n" + tabulate(log_table, tablefmt="double_grid"))

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
                    logging.info(f"-- Upload: Putting to queue")
                    upload_queue.put(response)
                    logging.info(f"-- Upload: Put to queue")
                elif response_event in events_filter:
                    status_code = post_webhook(webhook_url, response)
                    logging.info(f"-- Webhook: {status_code}")
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Failed to handle message: {tb}\n")
        finally:
            channel.basic_ack(delivery_tag=method.delivery_tag)
            callback_in_progress = False

    return amqp_callback


def start_amqp_queue_worker(
    worker_type: str,
    connection: RabbitMQConnection,
    queue_name: str,
    upload_queue: queue.Queue[Dict[str, Any]],
    models_pack: ModelsPackForImage | ModelsPackForVoiceover,
    shutdown_event: Event,
) -> None:
    logging.info(f"Listening for messages on {queue_name}\n")

    # ! Removed things that may be useful later, when using routing keys per model
    # Declare a queue with priority support
    # result = channel.queue_declare(
    #     queue=generate_queue_name_from_capabilities(exchange_name, supported_models),
    #     durable=True,
    #     arguments={
    #         "x-max-priority": 10,
    #         "x-message-ttl": 1800000,
    #     },
    # )
    # queue_name = result.method.queue

    # Bind the queue based on the worker's capabilities
    # for capability in supported_models:
    #     channel.queue_bind(
    #         exchange=exchange_name, queue=queue_name, routing_key=capability
    #     )

    # Create callback
    msg_callback = create_amqp_callback(
        queue_name, worker_type, upload_queue, models_pack
    )

    global callback_in_progress
    while not shutdown_event.is_set() or callback_in_progress:
        try:
            connection.channel.basic_qos(prefetch_count=1)
            connection.channel.basic_consume(queue=queue_name, on_message_callback=msg_callback)
            connection.channel.start_consuming()
        except ConnectionClosedByBroker as err:
            logging.error(f"ConnectionClosedByBroker {err}")
            connection.reconnect()
            continue
        except AMQPChannelError as err:
            logging.error(f"AMQPChannelError {err}")
            break
        except AMQPConnectionError as err:
            logging.error(f"AMQPConnectionError {err}")
            connection.reconnect()
            continue
    if callback_in_progress:
        logging.info(f"Waiting for callback to finish")
        # Give it a max of 30s to finish
        for i in range(60):
            if not callback_in_progress:
                break
            # Sleep for 500ms
            time.sleep(0.5)
            pass
    try:
        logging.info(f"Stopping rabbitmq queue channel")
        connection.channel.close()
        logging.info(f"Closing rabbitmq connection")
        connection.channel.connection.close()
        logging.info("rabbitmq worker terminated")
    except:
        pass
    # Shouldn't have gotten here, exit the whole program
    if not shutdown_event.is_set():
        logging.info("rabbitmq worker terminated unexpectedly")
        shutdown_event.set()


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
        logging.error(f"Failed to start prediction: {tb}\n")
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
        logging.error(f"Failed to run prediction: {tb}\n")
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
        logging.error(f"Failed to start prediction: {tb}\n")
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
        logging.error(f"Failed to run prediction: {tb}\n")
        completed_at = datetime.datetime.now()
        response["completed_at"] = format_datetime(completed_at)
        response["status"] = Status.FAILED
        response["error"] = str(e)
    finally:
        yield (Event.COMPLETED, response)
