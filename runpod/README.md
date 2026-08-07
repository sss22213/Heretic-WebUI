# Heretic WebUI on RunPod

RunPod pods are themselves containers — there is no Docker daemon inside a pod, so
`docker-compose.yml` cannot be used there. Instead, this folder builds a single
self-contained image (WebUI **plus a bundled Ollama server**) that RunPod runs
directly. Nothing in the main project is modified; the RunPod image layers on top
of the normal `heretic-webui:local` build.

## 1. Build & push

From the **repo root** (the base image must exist first):

```bash
docker compose build                       # produces heretic-webui:local
docker build -f runpod/Dockerfile -t <dockerhub-user>/heretic-webui-runpod:latest .
docker login
docker push <dockerhub-user>/heretic-webui-runpod:latest
```

Pin a different Ollama release with `--build-arg OLLAMA_VERSION=<x.y.z>`
(default `0.31.2`, the minimum for Qwen3.5/3.6 renderer support).

## 2. Create the RunPod pod / template

| Setting | Value |
|---|---|
| Container Image | `<dockerhub-user>/heretic-webui-runpod:latest` |
| Expose HTTP Ports | `8000` (WebUI) |
| Expose TCP Ports | `11434` (optional — only if you want to reach Ollama from outside) |
| Volume Mount Path | `/workspace` |
| Volume Disk | 150 GB+ recommended for 27B work (full merge output alone is ~51 GB, plus HF cache, GGUFs, and Ollama blobs) |
| Container Disk | 30 GB+ |
| Environment | `HF_TOKEN` (optional, for gated/private Hugging Face repos) |

GPU sizing: abliterating or merging a 27B model wants an 80 GB card (A100/H100).
GGUF conversion, quantization, and Ollama-backed evaluation are fine on 24 GB.

Open the WebUI through RunPod's HTTP proxy for port 8000.

## What the image changes vs. the local setup

- **Ollama runs inside the pod** (started by `entrypoint-runpod.sh` before the
  WebUI). `OLLAMA_BASE_URL` is preset to `http://127.0.0.1:11434` — no
  `host.docker.internal`, no separate Ollama install.
- **All state lives under `/workspace`** (the RunPod persistent volume):
  `/workspace/data` (jobs, HF cache), `/workspace/outputs` (merged models),
  `/workspace/models` (local safetensors you upload), `/workspace/ollama`
  (Ollama blobs). Pod restarts and image updates keep your work.
- **Runs as root** — normal for RunPod; the PUID/PGID drop from the local
  entrypoint is skipped.
- `/dev/shm`: RunPod sizes shared memory from the pod's RAM allocation. If a
  heretic run hits shm limits, pick a pod type with more system RAM.

---

## 繁體中文說明

RunPod 的 pod 本身就是容器，裡面**沒有 Docker**，所以不能用 docker-compose。
做法是把 WebUI 和 **Ollama 一起打包成單一 image**，讓 RunPod 直接跑。
原本專案的檔案完全不動，這裡是在 `heretic-webui:local` 之上再疊一層。

**建置與推送**（在專案根目錄）：

```bash
docker compose build
docker build -f runpod/Dockerfile -t <dockerhub帳號>/heretic-webui-runpod:latest .
docker push <dockerhub帳號>/heretic-webui-runpod:latest
```

**RunPod 設定**：Container Image 填上面推的 tag；HTTP Port 開 `8000`（WebUI）、
TCP `11434` 可選（外部直連 Ollama 才需要）；Volume 掛 `/workspace`，建議 150 GB
以上（27B 合併輸出就 ~51 GB）；環境變數 `HF_TOKEN` 選填。GPU：27B 的消融/合併
建議 80 GB（A100/H100），GGUF 量化與跑分 24 GB 即可。

**與本機版的差異**：Ollama 由 entrypoint 在容器內自動啟動（WebUI 已預設連
`127.0.0.1:11434`）；所有資料（jobs、HF 快取、合併輸出、Ollama 模型）都放在
`/workspace` 持久卷，pod 重開不會遺失；以 root 執行（RunPod 慣例）。
