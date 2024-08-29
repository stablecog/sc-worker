from shared.logger import setup_logger

logger_listener = setup_logger()

from threading import Thread, Event
from typing import Any, Dict
import os
import signal
import queue

from dotenv import load_dotenv
import torch

from predict.image.setup import setup as image_setup
from rabbitmq_consumer.worker import start_amqp_queue_worker
from rabbitmq_consumer.connection import RabbitMQConnection
from upload.worker import start_upload_worker
import logging

# Define an event to signal all threads to exit
shutdown_event = Event()

if __name__ == "__main__":
    if torch.cuda.is_available() is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    load_dotenv()

    WORKER_TYPE = os.environ.get("WORKER_TYPE", "image")

    amqpUrl = os.environ.get("RABBITMQ_AMQP_URL", None)
    if amqpUrl is None:
        raise ValueError("Missing RABBITMQ_AMQP_URL environment variable.")
    amqpQueueName = os.environ.get("RABBITMQ_QUEUE_NAME", None)
    if amqpQueueName is None:
        raise ValueError("Missing RABBITMQ_QUEUE_NAME environment variable.")

    models_pack = image_setup()

    # Create queue for thread communication
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    # Create rabbitmq connection
    connection = RabbitMQConnection(amqpUrl)

    # Setup signal handler for exit
    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            logging.info("Signal received, shutting down...")
            shutdown_event.set()
            connection.connection.close()
            logger_listener.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create rabbitmq worker thread
    mq_worker_thread = Thread(
        target=lambda: start_amqp_queue_worker(
            worker_type=WORKER_TYPE,
            connection=connection,
            queue_name=amqpQueueName,
            upload_queue=upload_queue,
            models_pack=models_pack,
            shutdown_event=shutdown_event,
        )
    )

    # Create upload thread
    upload_thread = Thread(
        target=lambda: start_upload_worker(
            worker_type=WORKER_TYPE,
            q=upload_queue,
            shutdown_event=shutdown_event,
        )
    )

    try:
        mq_worker_thread.start()
        upload_thread.start()
        mq_worker_thread.join()
        upload_thread.join()
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully. The signal handler already sets the shutdown_event.
    finally:
        # Any other cleanup in the main thread you want to perform.
        logging.info("Main thread cleanup done!")
