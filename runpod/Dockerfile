# syntax=docker/dockerfile:1.7
# RunPod single-container image: Heretic WebUI + a bundled Ollama server.
# RunPod pods run the image directly (no Docker-in-Docker, no compose), so
# everything docker-compose.yml provides locally must live in this image:
# the Ollama endpoint runs inside the pod and persistent paths point at the
# RunPod volume mounted at /workspace.
#
# Build from the repo root, after the base image exists:
#   docker compose build
#   docker build -f runpod/Dockerfile -t <dockerhub-user>/heretic-webui-runpod:latest .
ARG BASE_IMAGE=heretic-webui:local
FROM ${BASE_IMAGE}

ARG OLLAMA_VERSION=0.31.2
RUN curl -fL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tgz" \
        | tar -xz -C /usr/local \
    && /usr/local/bin/ollama --version

# /workspace is the RunPod persistent volume; every stateful path lives there
# so models, jobs, and Ollama blobs survive pod restarts.
ENV APP_DATA_DIR=/workspace/data \
    APP_OUTPUT_DIR=/workspace/outputs \
    APP_MODELS_DIR=/workspace/models \
    HF_HOME=/workspace/data/huggingface \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODELS=/workspace/ollama \
    OLLAMA_HOST=0.0.0.0:11434

COPY runpod/entrypoint-runpod.sh /usr/local/bin/entrypoint-runpod.sh
RUN chmod +x /usr/local/bin/entrypoint-runpod.sh

EXPOSE 8000 11434
ENTRYPOINT ["/usr/local/bin/entrypoint-runpod.sh"]
CMD []
