# Socia's own image with a Python kernel for VS Code: the notebooks run her code with her exact libraries.
ARG BASE=ghcr.io/pranjallk1995/luna-socia-ui:dev
FROM ${BASE}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ipykernel "nvidia-ml-py>=12" matplotlib
# Training (notebook 03): torch, PEFT and bitsandbytes in their own environment, which also sees Socia's
# libraries, so hers stay exactly as they are. Its kernel is "Luna training (torch)". cu128 for the RTX 50s.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python /usr/local/bin/python --system-site-packages /opt/train \
 && uv pip install --python /opt/train/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 \
 && uv pip install --python /opt/train/bin/python transformers==4.56.2 peft==0.17.1 bitsandbytes==0.47.0 \
      accelerate==1.10.1 sentencepiece==0.2.1 ipykernel \
 && /opt/train/bin/python -m ipykernel install --prefix /usr/local --name luna-train --display-name "Luna training (torch)"
# You, inside: files the notebooks write (results/) are yours on this machine.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" luna && useradd -m -u "$UID" -g "$GID" -s /bin/bash luna \
 && mkdir -p /training && chown luna:luna /training
USER luna
# The base model, adapters and llama.cpp's converters, in the luna_socia_training volume (about 17 GB).
ENV HF_HOME=/training/hf
ENV PYTHONPATH=/workspace/ui:/workspace/notebooks
WORKDIR /workspace/notebooks
CMD ["sleep", "infinity"]
