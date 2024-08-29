import traceback
import queue

from threading import Event
from typing import List, Dict, Any


from predict.image.predict import PredictResult as PredictResultForImage
from rabbitmq_consumer.events import Status
from shared.webhook import post_webhook
from upload.upload import upload_files_for_image
import logging
import time


LOG_INTERVAL = 15


def start_upload_worker(
    worker_type: str,
    q: queue.Queue[Dict[str, Any]],
    shutdown_event: Event,
):
    """Starts a loop to read from the queue and upload images to S3, send responses to webhook"""
    logging.info("Starting upload thread...")
    last_log_time = 0
    while not shutdown_event.is_set() or not q.empty():
        should_log = time.time() - last_log_time > LOG_INTERVAL
        try:
            if should_log:
                logging.info(f"^^ 🟡 Getting uploadMsg from queue")
                last_log_time = time.time()
            uploadMsg: List[Dict[str, Any]] = q.get(timeout=1)
            logging.info(f"^^ 🟢 Got uploadMsg from queue")
            if "upload_output" in uploadMsg:
                predict_result: PredictResultForImage = uploadMsg["upload_output"]
                if len(predict_result.outputs) > 0:
                    try:
                        logging.info(
                            f"^^ Got {len(predict_result.outputs)} outputs from uploadMsg, uploading..."
                        )
                        uploadMsg["output"] = {
                            "images": upload_files_for_image(
                                predict_result.outputs,
                                predict_result.signed_urls,
                                uploadMsg["upload_prefix"],
                            ),
                        }
                    except Exception as e:
                        tb = traceback.format_exc()
                        logging.error(f"^^ Error uploading files {tb}")
                        uploadMsg["status"] = Status.FAILED
                        uploadMsg["error"] = str(e)
                else:
                    logging.info(f"^^ No outputs to upload in uploadMsg")
            else:
                logging.info(f'^^ No "upload_outputs" object in uploadMsg')

            if "upload_output" in uploadMsg:
                logging.info(f"^^ Deleting upload_output from message")
                del uploadMsg["upload_output"]
            if "upload_prefix" in uploadMsg:
                logging.info(f"^^ Deleting upload_prefix from message")
                del uploadMsg["upload_prefix"]

            logging.info(f"^^ 🟡 Publishing to WEBHOOK")
            post_webhook(uploadMsg["webhook_url"], uploadMsg)
            logging.info(f"^^ 🟢 Published to WEBHOOK")
        except queue.Empty:
            if should_log:
                logging.info(f"^^ 🔵 Queue is empty, waiting...")
            continue
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"^^ 🔴 Exception in upload process {tb}")
            logging.error(f"^^ 🔴 Exception message was: {uploadMsg}")
    logging.info("^^ Upload thread exiting")
