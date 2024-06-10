# FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04
ARG BASE_IMAGE=stablecog/stablecog-worker-base:latest
FROM $BASE_IMAGE

RUN mkdir -p /app/data
WORKDIR /app

ADD . .

HEALTHCHECK --interval=1s --timeout=2s --retries=300 \
  CMD curl -f http://localhost/health || exit 1

ENV CLIPAPI_PORT=80
EXPOSE $CLIPAPI_PORT

CMD . venv/bin/activate && exec python main.py
