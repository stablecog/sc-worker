# GB10 (NVIDIA DGX Spark / Grace Blackwell) build: linux/arm64, CUDA 13, sm_121.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Arch-neutral: the multiarch lib dir (aarch64-linux-gnu) is already on the default linker path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/usr/local/nvidia/bin
# GB10 is compute capability 12.1 (only used if a CUDA extension gets JIT-compiled)
ENV TORCH_CUDA_ARCH_LIST="12.1"
# GB10 has 128 GB of unified CPU/GPU memory: avoid allocator fragmentation across many resident models
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
