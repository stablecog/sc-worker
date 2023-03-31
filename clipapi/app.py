from flask import Flask
import os

clipapi = Flask(__name__)


@clipapi.route("/clip/transform", methods=["POST"])
def clip_transform():
    return "Hello, World!"


def run_clipapi():
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    clipapi.run(host=host, port=port)
