# Override at build time if the server requires another CUDA/PyTorch pairing:
# docker build --build-arg BASE_IMAGE=<image> ...
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/multiencoder-transformer
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install '.[prepare]'

WORKDIR /workspace/project
CMD ["bash"]
