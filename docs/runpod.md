# RunPod 部署指南

把 Heretic WebUI 部署到 RunPod GPU Pod 上執行。整體流程：GitHub Actions 把 image 建置並推到 GHCR → RunPod 從 GHCR 拉 image 開 Pod → 資料放在掛載於 `/data` 的 volume → 透過 SSH tunnel（建議）或 RunPod HTTP proxy + Basic auth 使用 UI。

## 一次性準備

### 1. 讓 RunPod 拉得到 image

推上 GitHub 後，`.github/workflows/docker-image.yml` 會自動建置並推送
`ghcr.io/<你的帳號>/heretic-webui:latest`（master 每次 push 都會更新，另有 commit SHA tag 可鎖版本）。

GHCR 套件第一次發布預設是 **private**，二選一：

- **設為 public**（最簡單）：GitHub → 你的 Profile → Packages → `heretic-webui` → Package settings → Change visibility → Public。
- **保持 private**：RunPod 主控台 → Settings → Container Registry Auth 新增認證，Username 填 GitHub 帳號，Password 填有 `read:packages` 權限的 Personal Access Token；建 Pod 時選這組認證。

### 2. 帳號 SSH 金鑰

RunPod 主控台 → Settings → SSH Public Keys 貼上你的公鑰。Pod 啟動時 RunPod 會把它注入為 `PUBLIC_KEY` 環境變數，image 的 entrypoint 偵測到就會啟動 sshd（本機部署沒有這個變數，完全不受影響）。

## 選擇 Secure 或 Community

| | Secure Cloud | Community Cloud |
|---|---|---|
| 價格 | 較高 | 便宜三到五成 |
| Network Volume | ✔（跨 Pod 持久化） | ✘（只有綁定主機的 Pod volume） |
| 中斷後資料 | Volume 保留，換 Pod 接續 | 同一台主機回來才在；主機消失就沒了 |
| 適合 | 長時間跑、要接續 checkpoint | 拋棄式單次任務、成本優先 |

- **Secure**：先在目標資料中心建 Network Volume（建議 250GB 起：HF cache 的 27B 基底約 55GB、輸出與 GGUF 另計），建 Pod 時掛載。**Secure + Spot** 是省錢甜蜜點——價格接近 Community，但中斷時 volume 還在，重開 Pod 對失敗任務按「重試」即可從 checkpoint 接續。
- **Community**：把 Pod 當拋棄式——跑完立刻把成品推走（scp 抓回 GGUF、或直接推 Ollama/HF），別把唯一一份留在 Pod 上。挑列表裡 reliability 高、網速快（下載 ≥1Gbps，模型 55GB 要重抓時有差）的主機。

## 建立 Pod

Deploy → 選 GPU（27B BF16 全權重 ARA 建議 A100 80GB；4-bit ARA LoRA 用 48GB 級即可）→ **CUDA 版本 filter 選 12.8 以上**（image 基底是 PyTorch 2.8 cu128）→ Template 填：

| 欄位 | 值 |
|---|---|
| Container Image | `ghcr.io/<你的帳號>/heretic-webui:latest` |
| Container Disk | 40 GB（image 解壓與暫存） |
| Volume（Network 或 Pod volume）掛載路徑 | `/data` |
| Expose TCP Ports | `22`（SSH tunnel 用） |
| Expose HTTP Ports | `8000`（只在走 proxy 方案時填） |

環境變數：

| 變數 | 值 | 說明 |
|---|---|---|
| `APP_OUTPUT_DIR` | `/data/outputs` | 讓模型輸出也落在 volume，Pod 換掉不遺失 |
| `APP_BASIC_AUTH` | `帳號:密碼` | 走 HTTP proxy 時必填；瀏覽器會跳原生登入框。SSH tunnel 方案可不設 |

Docker command 留空（image 預設就是啟動 WebUI）。

## 連線使用

### 方式 A：SSH tunnel（建議）

Pod 頁面 → Connect 會顯示 TCP 22 對應的公網 IP 與 port：

```bash
ssh -p <PORT> -L 8000:localhost:8000 root@<IP>
```

瀏覽器開 `http://localhost:8000`，體感與本機相同，8000 不對外。同一條 SSH 也能取回成品：

```bash
scp -P <PORT> root@<IP>:/data/outputs/<模型>/... ./
```

### 方式 B：RunPod HTTP proxy

Template 有 expose 8000 的話，網址是 `https://<pod-id>-8000.proxy.runpod.net`。這個網址公開可達，**務必設 `APP_BASIC_AUTH`**——UI 會保存你的 HF token，不要裸奔。

## 使用注意

- **首次啟動**：開「Heretic 版本」頁讓 master/ara slot 下載建置（數十秒到數分鐘）；HF token 在建立任務表單填一次即保存於 `/data`。
- **中斷接續**：Pod 重啟後原任務會標記失敗，屬正常；只要 `/data` 還在，按「重試」即從 checkpoint 接續 trial。
- **成品出口**：在 Pod 上直接轉 GGUF 後 scp 回本地（16–22GB），或臨時在 Pod 裡跑一個 Ollama 把模型推到你的 Ollama 位址；55GB 的 safetensors 不建議搬運。
- **計費**：不用時 **Stop**（只收儲存費）；確定收工用 **Terminate**（Community 的 Pod volume 會一併消失，先確認成品已取回）。
