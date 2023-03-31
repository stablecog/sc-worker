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
    # authheader = request.headers.get("Authorization")
    # if authheader is None:
    #     return "Unauthorized", 401
    # if authheader != os.environ.get("CLIPAPI_AUTH_TOKEN"):
    #     return "Unauthorized", 401
    try:
        req_body = request.get_json()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error parsing request body: {tb}\n")
        return str(e), 400
    finally:
        if req_body is None or "text" not in req_body:
            return "Missing 'text' in request body", 400

    text = req_body["text"]
    if TRANSLATOR_COG_URL is not None:
        try:
            [text, _] = translate_text(
                text,
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
            [text],
            models_pack.open_clip["model"],
            models_pack.open_clip["tokenizer"],
        )[0]
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Failed to get openCLIP embeds: {tb}\n")
        return str(e), 500

    return jsonify({"embed": text_embed})


def run_clipapi(models_pack: ModelsPack):
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    with clipapi.app_context():
        current_app.models_pack = models_pack
    # clipapi.run(host=host, port=port)
    serve(clipapi, host=host, port=port)
