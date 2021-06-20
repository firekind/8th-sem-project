# nvcr.io/nvidia/deepstream:5.0.1-20.09-triton
FROM nvcr.io/nvidia/deepstream@sha256:15b1eaf3b85981cce5efb0fb0c8fd8f43a3695ed1f8f6f1a809b9c20ad777b4d

COPY requirements.txt /tmp

RUN apt update && \
    apt install -y git ffmpeg libsm6 libxext6 wget git python3-dev cmake && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

RUN pip3 install --upgrade pip && \
    pip3 install scikit-build && \
    pip3 install -r /tmp/requirements.txt && \
    rm -r /root/.cache
