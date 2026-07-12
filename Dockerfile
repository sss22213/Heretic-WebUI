# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS llama-cpp-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_CPP_REF=e3546c7948e3af463d0b401e6421d5a4c2faf565
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth 1 origin "${LLAMA_CPP_REF}" \
    && git checkout --detach FETCH_HEAD \
    && cmake -S . -B build \
        -DGGML_CUDA=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release --target llama-quantize -j 4 \
    && cp build/bin/llama-quantize /tmp/llama-quantize \
    && rm -rf .git build \
    && mkdir -p build/bin \
    && mv /tmp/llama-quantize build/bin/llama-quantize

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl gosu libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY upstream-heretic /app/upstream-heretic
COPY requirements-web.txt /app/requirements-web.txt
COPY requirements-gguf.txt /app/requirements-gguf.txt
COPY --from=llama-cpp-builder /opt/llama.cpp /opt/llama.cpp
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e /app/upstream-heretic \
    && python -m pip install --no-cache-dir -r /app/requirements-gguf.txt \
    && python -m pip install --no-cache-dir -r /app/requirements-web.txt

COPY app /app/app
COPY patches /app/patches
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN mkdir -p /data/jobs /data/checkpoints /outputs /models \
    && groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid appuser appuser \
    && chown -R appuser:appuser /app /data /outputs /models \
    && chmod +x /usr/local/bin/entrypoint.sh

ENV APP_DATA_DIR=/data \
    APP_OUTPUT_DIR=/outputs \
    HF_HOME=/data/huggingface \
    LLAMA_CPP_DIR=/opt/llama.cpp \
    PUID=1000 \
    PGID=1000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/api/health || exit 1
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
