import traceback
import queue

from threading import Event
from typing import List, Dict, Any


from predict.image.predict import PredictResult as PredictResultForImage
from predict.voiceover.predict import PredictResult as PredictResultForVoiceover
from rabbitmq_consumer.events import Status
from shared.webhook import post_webhook
from upload.upload import upload_files_for_image
from shared.logger import logger


def start_upload_worker(
    worker_type: str,
    q: queue.Queue[Dict[str, Any]],
    shutdown_event: Event,
):
    """Starts a loop to read from the queue and upload files to S3, send responses to webhook"""
    logger.info("Starting upload thread...")
    while not shutdown_event.is_set() or not q.empty():
        try:
            # logger.info(f"-- Upload: Waiting for queue --\n")
            uploadMsg: List[Dict[str, Any]] = q.get(timeout=1)
            # logger.info(f"-- Upload: Got from queue --\n")
            if "upload_output" in uploadMsg:
                predict_result: PredictResultForImage | PredictResultForVoiceover = (
                    uploadMsg["upload_output"]
                )
                if len(predict_result.outputs) > 0:
                    logger.info(
                        f"-- Upload: Uploading {len(predict_result.outputs)} files --"
                    )
                    try:
                        uploadMsg["output"] = {
                            "prompt_embed": predict_result.outputs[
                                0
                            ].open_clip_prompt_embed,
                            "images": upload_files_for_image(
                                predict_result.outputs,
                                predict_result.signed_urls,
                                uploadMsg["upload_prefix"],
                            ),
                        }
                    except Exception as e:
                        tb = traceback.format_exc()
                        logger.error(f"Error uploading files {tb}\n")
                        uploadMsg["status"] = Status.FAILED
                        uploadMsg["error"] = str(e)
                    logger.info(f"-- Upload: Finished uploading files --")

            if "upload_output" in uploadMsg:
                logger.info(f"-- Upload: Deleting upload_output from message --")
                del uploadMsg["upload_output"]
            if "upload_prefix" in uploadMsg:
                logger.info(f"-- Upload: Deleting upload_prefix from message --")
                del uploadMsg["upload_prefix"]

            logger.info(f"-- 🟡 Upload: Publishing to WEBHOOK --")
            post_webhook(uploadMsg["webhook_url"], uploadMsg)
            logger.info(f"-- 🟢 Upload: Published to WEBHOOK --")
        except queue.Empty:
            continue
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Exception in upload process {tb}\n")
            logger.error(f"Message was: {uploadMsg}\n")
    logger.info("Upload thread exiting")
