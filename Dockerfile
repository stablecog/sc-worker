FROM stablecog/cuda-torch:11.7.0-2.0.0-cudnn8-devel-ubuntu22.04-v4

ADD . .
RUN apt-get update && apt-get -y install git
RUN pip3 install -r requirements.txt --no-cache-dir

ENV CLIPAPI_PORT=13337
EXPOSE $CLIPAPI_PORT

CMD ["python3", "main.py"]