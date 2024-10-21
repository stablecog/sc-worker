FROM stablecog/cuda-torch:12.4.1-2.4.0-cudnn-devel-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD python3 main.py