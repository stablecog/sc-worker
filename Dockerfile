FROM stablecog/cuda-torch:11.7.1-cudnn8-devel-1.13.1-ubuntu22.04

ADD . .
RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["python3", "main.py"]