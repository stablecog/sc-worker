import os
import traceback

from flask import Flask, request, current_app, jsonify
from waitress import serve

from models.open_clip.main import (
    open_clip_get_embeds_of_texts,
    open_clip_get_embeds_of_images,
)
from predict.image.setup import ModelsPack
from shared.helpers import download_images
import time
from shared.helpers import time_code_block
from shared.logger import logger

clipapi = Flask(__name__)


@clipapi.route("/health", methods=["GET"])
def health():
    return "OK", 200


@clipapi.route("/clip/embed", methods=["POST"])
def clip_embed():
    s = time.time()
    with current_app.app_context():
        models_pack: ModelsPack = current_app.models_pack
    authheader = request.headers.get("Authorization")
    if authheader is None:
        return "Unauthorized", 401
    if authheader != os.environ["CLIPAPI_AUTH_TOKEN"]:
        return "Unauthorized", 401
    try:
        req_body = request.get_json()
    except Exception as e:
        tb = traceback.format_exc()
        logger.info(f"Error parsing request body: {tb}\n")
        return str(e), 400
    finally:
        if req_body is None:
            return "Missing request body", 400
        if isinstance(req_body, list) is not True:
            return "Body should be an array", 400

    embeds = [None for _ in range(len(req_body))]
    textObjects = []
    imageObjects = []
    for index, item in enumerate(req_body):
        if "text" in item:
            textObjects.append({"item": item, "index": index})
        if "image" in item:
            imageObjects.append({"item": item, "index": index})

    if len(textObjects) > 0:
        texts = [obj["item"]["text"] for obj in textObjects]
        text_embeds = open_clip_get_embeds_of_texts(
            texts,
            models_pack.open_clip.model,
            models_pack.open_clip.tokenizer,
        )
        for i, embed in enumerate(text_embeds):
            item = textObjects[i]["item"]
            index = textObjects[i]["index"]
            id = item.get("id", None)
            obj = {"input_text": item["text"], "embedding": embed}
            if id is not None:
                obj["id"] = id
            embeds[index] = obj

    if len(imageObjects) > 0:
        image_urls = []
        pil_images = []
        for obj in imageObjects:
            image_urls.append(obj["item"]["image"])
        try:
            with time_code_block(prefix=f"Downloaded {len(image_urls)} image(s)"):
                pil_images = download_images(urls=image_urls, max_workers=25)
        except Exception as e:
            tb = traceback.format_exc()
            logger.info(f"Failed to download images: {tb}\n")
            return str(e), 500
        image_embeds = open_clip_get_embeds_of_images(
            pil_images,
            models_pack.open_clip.model,
            models_pack.open_clip.processor,
        )
        for i, embed in enumerate(image_embeds):
            item = imageObjects[i]["item"]
            index = imageObjects[i]["index"]
            id = item.get("id", None)
            obj = {"image": image_urls[i], "embedding": embed}
            if id is not None:
                obj["id"] = id
            embeds[index] = obj

    e = time.time()
    logger.info(f"🖥️  Embedded {len(req_body)} items in: {e-s:.2f} seconds  🖥️\n")
    return jsonify({"embeddings": embeds})


def run_clipapi(models_pack: ModelsPack):
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    with clipapi.app_context():
        current_app.models_pack = models_pack
    logger.info("//////////////////////////////////////////////////////////////////")
    logger.info(f"🖥️🟢 Starting CLIP API on {host}:{port}")
    logger.info("//////////////////////////////////////////////////////////////////")
    serve(clipapi, host=host, port=port)
