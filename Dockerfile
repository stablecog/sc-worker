FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
# Ensure PyTorch uses CUDA 12.8 for Blackwell (sm_120) support
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install -qqy --no-install-recommends \
  python3 \
  python3-pip \
  python3-dev \
  ca-certificates \
  wget \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD python3 main.py