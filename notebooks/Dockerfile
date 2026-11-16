# Socia's own image with Jupyter added: the notebooks run her code with her exact libraries.
ARG BASE=ghcr.io/pranjallk1995/luna-socia-ui:dev
FROM ${BASE}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "jupyterlab>=4.4,<5" "jupytext>=1.17,<2" ipykernel "nvidia-ml-py>=12" matplotlib
ENV PYTHONPATH=/workspace/ui:/workspace/notebooks
WORKDIR /workspace/notebooks
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace/notebooks"]
