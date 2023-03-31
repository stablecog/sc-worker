import os
import traceback

from flask import Flask, request, current_app, jsonify
from waitress import serve

from models.nllb.translate import translate_text
from models.nllb.constants import TRANSLATOR_COG_URL
from models.open_clip.main import (
    open_clip_get_embeds_of_texts,
)
from predict.setup import ModelsPack

clipapi = Flask(__name__)


@clipapi.route("/clip/embed", methods=["POST"])
def clip_embed():
    with current_app.app_context():
        models_pack: ModelsPack = current_app.models_pack
    authheader = request.headers.get("Authorization")
    if authheader is None:
        return "Unauthorized", 401
    if authheader != os.environ.get("CLIPAPI_AUTH_TOKEN"):
        return "Unauthorized", 401
    try:
        req_body = request.get_json()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error parsing request body: {tb}\n")
        return str(e), 400
    finally:
        print(req_body)
        if req_body is None:
            return "Missing request body", 400
        if isinstance(req_body, list) is not True:
            return "Body should be an array", 400

    embeds = []
    for item in req_body:
        if "text" in item:
            input_text = item["text"]
            if TRANSLATOR_COG_URL is not None:
                try:
                    [translated_text, _] = translate_text(
                        input_text,
                        "",
                        "",
                        "",
                        TRANSLATOR_COG_URL,
                        models_pack.language_detector_pipe,
                        "CLIP Query",
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"Failed to translate input: {tb}\n")
                    return str(e), 500
            try:
                text_embed = open_clip_get_embeds_of_texts(
                    [translated_text],
                    models_pack.open_clip["model"],
                    models_pack.open_clip["tokenizer"],
                )[0]
                obj = {"input_text": input_text, "embedding": text_embed}
                if translated_text != input_text:
                    obj["translated_text"] = translated_text
                embeds.append(obj)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Failed to get openCLIP embeds: {tb}\n")
                return str(e), 500

    return jsonify({"embeddings": embeds})


def run_clipapi(models_pack: ModelsPack):
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    with clipapi.app_context():
        current_app.models_pack = models_pack
    # clipapi.run(host=host, port=port)
    serve(clipapi, host=host, port=port)
