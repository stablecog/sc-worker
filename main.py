from threading import Thread, Event
from typing import Any, Dict
import os
import signal
import queue

from dotenv import load_dotenv
import torch

from models.nllb.constants import LAUNCH_NLLBAPI
from nllbapi.app import run_nllbapi
from predict.image.setup import setup as image_setup
from predict.voiceover.setup import setup as voiceover_setup
from rabbitmq_consumer.worker import start_amqp_queue_worker
from rabbitmq_consumer.connection import RabbitMQConnection
from upload.worker import start_upload_worker
from clipapi.app import run_clipapi
from shared.logger import logger

# Define an event to signal all threads to exit
shutdown_event = Event()

if __name__ == "__main__":
    if torch.cuda.is_available() is False:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    load_dotenv()

    """ package_name = "flash-attn==2.5.3"
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name]) """

    WORKER_TYPE = os.environ.get("WORKER_TYPE", "image")

    amqpUrl = os.environ.get("RABBITMQ_AMQP_URL", None)
    if amqpUrl is None:
        raise ValueError("Missing RABBITMQ_AMQP_URL environment variable.")
    amqpQueueName = os.environ.get("RABBITMQ_QUEUE_NAME", None)
    if amqpQueueName is None:
        raise ValueError("Missing RABBITMQ_QUEUE_NAME environment variable.")

    if WORKER_TYPE == "voiceover":
        models_pack = voiceover_setup()
    else:
        models_pack = image_setup()

    # Create queue for thread communication
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    # Create rabbitmq connection
    connection = RabbitMQConnection(amqpUrl)

    # Setup signal handler for exit
    def signal_handler(signum, frame):
        if not shutdown_event.is_set():
            logger.info("Signal received, shutting down...")
            shutdown_event.set()
            connection.connection.close()

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
        if WORKER_TYPE == "image":
            # CLIP
            clipapi_thread = Thread(target=lambda: run_clipapi(models_pack=models_pack))
            clipapi_thread.start()
            # NLLB
            if LAUNCH_NLLBAPI:
                nllbapi_thread = Thread(
                    target=lambda: run_nllbapi(models_pack=models_pack)
                )
                nllbapi_thread.start()

            # Join threads
            clipapi_thread.join()
            if LAUNCH_NLLBAPI:
                nllbapi_thread.join()
        mq_worker_thread.join()
        upload_thread.join()
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully. The signal handler already sets the shutdown_event.
    finally:
        # Any other cleanup in the main thread you want to perform.
        logger.info("Main thread cleanup done!")
