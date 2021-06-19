FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt /tmp

RUN apt update && \
    apt install -y git ffmpeg libsm6 libxext6 wget git && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

RUN pip install -r /tmp/requirements.txt && \
    rm -r /root/.cache
