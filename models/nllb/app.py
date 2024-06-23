import os
import traceback

from flask import Flask, request, current_app, jsonify
from waitress import serve

from models.nllb.constants import (
    DETECTED_CONFIDENCE_SCORE_MIN,
    TARGET_LANG_FLORES,
    TARGET_LANG_SCORE_MAX,
)
from models.nllb.translate import translate_text
from predict.image.setup import ModelsPack
import time

nllbapi = Flask(__name__)


@nllbapi.route("/health", methods=["GET"])
def health():
    return "OK", 200


@nllbapi.route("/predictions", methods=["POST"])
def translate():
    start = time.time()
    print("//////////////////////////////////////////////////////////////////")
    print(f"⏳💬 Translation started 💬⏳")

    with current_app.app_context():
        models_pack: ModelsPack = current_app.models_pack
    authheader = request.headers.get("Authorization")
    if authheader is None:
        return "Unauthorized", 401
    if authheader != os.environ["NLLBAPI_AUTH_TOKEN"]:
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

    # Text 1
    text_1 = req_body.get("text_1", "")
    text_flores_1 = req_body.get("text_flores_1", None)
    target_flores_1 = req_body.get("target_flores_1", TARGET_LANG_FLORES)
    target_score_max_1 = req_body.get("target_score_max_1", TARGET_LANG_SCORE_MAX)
    detected_confidence_score_min_1 = req_body.get(
        "detected_confidence_score_min_1", DETECTED_CONFIDENCE_SCORE_MIN
    )
    label_1 = req_body.get("label_1", "Text")

    # Text 2
    text_2 = req_body.get("text_2", None)
    text_flores_2 = req_body.get("text_flores_2", None)
    target_flores_2 = req_body.get("target_flores_2", TARGET_LANG_FLORES)
    target_score_max_2 = req_body.get("target_score_max_2", TARGET_LANG_SCORE_MAX)
    detected_confidence_score_min_2 = req_body.get(
        "detected_confidence_score_min_2", DETECTED_CONFIDENCE_SCORE_MIN
    )
    label_2 = req_body.get("label_2", "Text")

    extra_kwargs = {
        "model": models_pack.translator.model,
        "tokenizer": models_pack.translator.tokenizer,
        "detector": models_pack.translator.detect_language,
    }

    output_strings = []
    translated_text = translate_text(
        text=text_1,
        text_flores=text_flores_1,
        target_flores=target_flores_1,
        detected_confidence_score_min=detected_confidence_score_min_1,
        target_score_max=target_score_max_1,
        label=label_1,
        **extra_kwargs,
    )
    output_strings.append(translated_text)
    if text_2 is not None:
        translated_text_2 = translate_text(
            text=text_2,
            text_flores=text_flores_2,
            target_flores=target_flores_2,
            detected_confidence_score_min=detected_confidence_score_min_2,
            target_score_max=target_score_max_2,
            label=label_2,
            **extra_kwargs,
        )
        output_strings.append(translated_text_2)

    end = time.time()
    print(f"✅💬 Translation completed in: {round((end - start) * 1000)} ms 💬✅")
    print("//////////////////////////////////////////////////////////////////")
    return jsonify({"output": output_strings})


def run_nllbapi(models_pack: ModelsPack):
    host = os.environ.get("NLLBAPI_HOST", "0.0.0.0")
    port = os.environ.get("NLLBAPI_PORT", 13350)
    with nllbapi.app_context():
        current_app.models_pack = models_pack
    serve(nllbapi, host=host, port=port)
