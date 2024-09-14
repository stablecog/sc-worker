FROM stablecog/cuda-torch:12.1.0-2.5.0-cudnn8-devel-ubuntu22.04

RUN mkdir -p /app/data
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python main.py