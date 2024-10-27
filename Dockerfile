FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

COPY requirements.txt .

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin

RUN apt-get update && apt-get install -qqy --no-install-recommends \
  python3 \ 
  python3-pip \
  python3-dev \
  ca-certificates \
  wget \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD python3 main.py