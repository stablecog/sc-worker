import os
import signal
import queue
import time
import logging
from threading import Thread, Event
from typing import Any, Dict
from contextlib import contextmanager

import torch
from dotenv import load_dotenv

from shared.logger import setup_logger
from predict.image.setup import setup as image_setup
from rabbitmq_consumer.worker import start_amqp_queue_worker
from rabbitmq_consumer.connection import RabbitMQConnection
from upload.worker import start_upload_worker

# Setup logger
logger_listener = setup_logger()
logger = logging.getLogger(__name__)


class GracefulKiller:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


@contextmanager
def managed_thread(target, *args, **kwargs):
    thread = Thread(target=target, *args, **kwargs)
    thread.start()
    try:
        yield thread
    finally:
        thread.join(timeout=60)  # Increased timeout to allow for queue draining
        if thread.is_alive():
            logger.warning(f"Thread {thread.name} did not shut down properly")


def main():
    if not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    load_dotenv()

    WORKER_TYPE = os.environ.get("WORKER_TYPE", "image")
    amqpUrl = os.environ.get("RABBITMQ_AMQP_URL")
    amqpQueueName = os.environ.get("RABBITMQ_QUEUE_NAME")

    if not amqpUrl or not amqpQueueName:
        raise ValueError(
            "Missing RABBITMQ_AMQP_URL or RABBITMQ_QUEUE_NAME environment variable."
        )

    models_pack = image_setup()
    upload_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    shutdown_event = Event()
    killer = GracefulKiller()

    with RabbitMQConnection(amqpUrl) as connection:
        with managed_thread(
            target=start_amqp_queue_worker,
            worker_type=WORKER_TYPE,
            connection=connection,
            queue_name=amqpQueueName,
            upload_queue=upload_queue,
            models_pack=models_pack,
            shutdown_event=shutdown_event,
        ) as mq_worker_thread, managed_thread(
            target=start_upload_worker,
            worker_type=WORKER_TYPE,
            q=upload_queue,
            shutdown_event=shutdown_event,
        ) as upload_thread:
            try:
                while not killer.kill_now:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}")
            finally:
                logger.info("Initiating shutdown...")
                shutdown_event.set()

                # Wait for threads to finish
                logger.info("Waiting for worker threads to finish...")
                start_time = time.time()
                while time.time() - start_time < 65:  # 65 seconds total wait time
                    if not mq_worker_thread.is_alive() and not upload_thread.is_alive():
                        break
                    if not upload_queue.empty():
                        logger.info("Waiting for upload queue to drain...")
                    time.sleep(1)

                if mq_worker_thread.is_alive() or upload_thread.is_alive():
                    logger.warning(
                        "Not all threads shut down properly within the timeout period."
                    )
                else:
                    logger.info("All threads shut down properly.")

    logger.info("All threads have been joined. Shutting down.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An unhandled exception occurred: {e}")
    finally:
        logger_listener.stop()
        logger.info("Shutdown complete")
