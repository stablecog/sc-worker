import os
import traceback

from flask import Flask, request, current_app, jsonify
from waitress import serve

from models.open_clip.main import (
    open_clip_get_embeds_of_texts,
    open_clip_get_embeds_of_images,
)
from predict.image.setup import ModelsPack
from shared.helpers import download_images, download_images_from_s3
import time
import boto3
from boto3_type_annotations.s3 import ServiceResource
from botocore.config import Config
from upload.constants import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME_UPLOAD,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
)

clipapi = Flask(__name__)

s3: ServiceResource = boto3.resource(
    "s3",
    region_name=S3_REGION,
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    config=Config(
        retries={"max_attempts": 3, "mode": "standard"}, max_pool_connections=300
    ),
)
bucket = s3.Bucket(S3_BUCKET_NAME_UPLOAD)


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
        print(f"Error parsing request body: {tb}\n")
        return str(e), 400
    finally:
        if req_body is None:
            return "Missing request body", 400
        if isinstance(req_body, list) is not True:
            return "Body should be an array", 400

    embeds = [None for _ in range(len(req_body))]
    textObjects = []
    imageObjects = []
    imageIdObjects = []
    for index, item in enumerate(req_body):
        if "text" in item:
            textObjects.append({"item": item, "index": index})
        if "image" in item:
            imageObjects.append({"item": item, "index": index})
        if "image_id" in item:
            imageIdObjects.append({"item": item, "index": index})

    if len(textObjects) > 0:
        texts = [obj["item"]["text"] for obj in textObjects]
        text_embeds = open_clip_get_embeds_of_texts(
            texts,
            models_pack.open_clip["model"],
            models_pack.open_clip["tokenizer"],
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
            pil_images = download_images(urls=image_urls, max_workers=25)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Failed to download images: {tb}\n")
            return str(e), 500
        image_embeds = open_clip_get_embeds_of_images(
            pil_images,
            models_pack.open_clip["model"],
            models_pack.open_clip["processor"],
        )
        for i, embed in enumerate(image_embeds):
            item = imageObjects[i]["item"]
            index = imageObjects[i]["index"]
            id = item.get("id", None)
            obj = {"image": image_urls[i], "embedding": embed}
            if id is not None:
                obj["id"] = id
            embeds[index] = obj

    if len(imageIdObjects) > 0:
        image_ids = []
        pil_images = []
        for obj in imageIdObjects:
            image_ids.append(obj["item"]["image_id"])
        try:
            pil_images = download_images_from_s3(
                keys=image_ids, bucket=bucket, max_workers=100
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Failed to download images: {tb}\n")
            return str(e), 500

        filtered_pil_image_objects = []
        filtered_pil_images = []
        for i, pil_image in enumerate(pil_images):
            if pil_image is not None:
                filtered_pil_image_objects.append({"image": pil_image, "index": i})
                filtered_pil_images.append(pil_image)

        image_embeds = open_clip_get_embeds_of_images(
            filtered_pil_images,
            models_pack.open_clip["model"],
            models_pack.open_clip["processor"],
        )

        for i, embed in enumerate(image_embeds):
            index_f = filtered_pil_image_objects[i]["index"]
            item = imageIdObjects[index_f]["item"]
            index = imageIdObjects[index_f]["index"]
            id = item.get("id", None)
            obj = {"image_id": image_ids[index_f], "embedding": embed}
            if id is not None:
                obj["id"] = id
            embeds[index] = obj

        for i, pil_image in enumerate(pil_images):
            item = imageIdObjects[i]["item"]
            index = imageIdObjects[i]["index"]
            id = item.get("id", None)
            if pil_image is None:
                index = imageIdObjects[i]["index"]
                embeds[index] = {
                    "image_id": image_ids[i],
                    "error": "Image not found in S3",
                }
                if id is not None:
                    embeds[index]["id"] = id

    e = time.time()
    print(f"🖥️  Embedded {len(req_body)} items in: {e-s:.2f} seconds  🖥️\n")
    return jsonify({"embeddings": embeds})


def run_clipapi(models_pack: ModelsPack):
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    with clipapi.app_context():
        current_app.models_pack = models_pack
    # clipapi.run(host=host, port=port)
    serve(clipapi, host=host, port=port)
