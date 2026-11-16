FROM python:3.12-slim

WORKDIR /workspace/ui

# uv installs the requirements; its cache mount reuses downloads across rebuilds.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/data/models/huggingface

COPY requirements.txt requirements-gpu.txt ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Her voice on the GPU. faster-whisper's CTranslate2 is built for CUDA 12 and cuDNN 9; ONNX
# Runtime 1.23 is the last GPU build for CUDA 12, so Kokoro shares the same libraries (1.24+ needs
# CUDA 13: a second copy of everything). It replaces the CPU onnxruntime that kokoro-onnx installs.
# The driver comes from the host (docker compose: gpus); about 2.7 GB of libraries.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-gpu.txt \
    && uv pip uninstall --system onnxruntime \
    && uv pip install --system --reinstall --no-deps "onnxruntime-gpu==1.23.2"
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
# CUDA compiles its kernels for this GPU on first use (about 30 s); kept on the data volume, the
# next start reuses them.
ENV CUDA_CACHE_PATH=/data/models/cuda-cache CUDA_CACHE_MAXSIZE=2147483648

COPY app ./app
# The release, passed in when GitHub Actions builds the image; blank for local builds.
ARG LUNA_VERSION=
ENV LUNA_VERSION=$LUNA_VERSION

EXPOSE 8501

CMD ["python", "-m", "app.main"]
