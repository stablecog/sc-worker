FROM stablecog/cuda-torch:12.1.0-2.1.0-cudnn8-devel-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

ADD . .
RUN apt-get update && apt-get -y install git
RUN apt-get install -y ffmpeg
RUN python3 -m pip install virtualenv --no-cache-dir
RUN python3 -m virtualenv venv
RUN . venv/bin/activate && pip install -r requirements.txt --no-cache-dir && deactivate

ENV CLIPAPI_PORT=13337
EXPOSE $CLIPAPI_PORT

CMD . venv/bin/activate && exec python main.py