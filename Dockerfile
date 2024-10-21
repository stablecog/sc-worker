FROM stablecog/cuda-torch:12.4.0-2.4.0

RUN mkdir -p /app/data
WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD python3 main.py