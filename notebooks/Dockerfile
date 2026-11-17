# Socia's own image with a Python kernel for VS Code: the notebooks run her code with her exact libraries.
ARG BASE=ghcr.io/pranjallk1995/luna-socia-ui:dev
FROM ${BASE}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ipykernel "nvidia-ml-py>=12" matplotlib
# You, inside: files the notebooks write (results/) are yours on this machine.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "$GID" luna && useradd -m -u "$UID" -g "$GID" -s /bin/bash luna
USER luna
ENV PYTHONPATH=/workspace/ui:/workspace/notebooks
WORKDIR /workspace/notebooks
CMD ["sleep", "infinity"]
