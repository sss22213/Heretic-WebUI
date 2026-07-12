# Heretic WebUI

[English](README.md) | 繁體中文

以 [p-e-w/heretic](https://github.com/p-e-w/heretic) 為核心的模型 abliteration WebUI。介面可建立任務、調整常用參數、串流執行日誌、取消處理，並將模型與 checkpoint 保存在主機上。

## 系統需求

- Linux 與 Docker Engine 24+
- Docker Compose v2
- NVIDIA GPU、驅動程式與 NVIDIA Container Toolkit
- 足夠的 VRAM、RAM 與磁碟空間。模型合併所需 RAM 可能顯著高於 4-bit 分析階段。

## 啟動

`upstream-heretic/` 直接保存 Heretic source snapshot，不使用 Git submodule；一般 `git clone` 即可取得完整專案。

```bash
cp .env.example .env
# 私有或 gated 模型才需要在 .env 填入 HF_TOKEN
# 若主機使用者不是 1000:1000，請一併修改 PUID/PGID
docker compose up --build -d
```

開啟 <http://localhost:8000>。查看服務日誌：

```bash
docker compose logs -f webui
```

## 匯入 Ollama

「匯入 Ollama」頁面會列出 `outputs` 中權重完整的合併模型。一般架構可透過 Ollama API 直接上傳 Safetensors；Gemma 4 Unified 等 Ollama 無法直接辨識的架構，會自動在 WebUI Container 內使用 llama.cpp 轉成 BF16 GGUF，再依選項量化為 2-bit `Q2_K`、3-bit `Q3_K_M`、4-bit `Q4_K_M`、6-bit `Q6_K` 或 8-bit `Q8_0`，最後以相同的 blob API 傳送。Ollama 可以位於另一個 Container，兩邊不需要共享模型 volume。

預設連線到 `http://host.docker.internal:11434`。請讓 Ollama Container 對主機發布 `11434` port，或在 `.env` 將 `OLLAMA_BASE_URL` 改成兩個 Container 都能解析的同一 Docker network 位址，例如：

```dotenv
OLLAMA_BASE_URL=http://ollama:11434
```

也可以在匯入頁面針對單次任務修改 API 位址。WebUI 會計算每個檔案的 SHA-256，略過 Ollama 已有的 blobs，並以串流方式傳輸大型權重；可選擇自動判斷、強制直接匯入 Safetensors，或強制轉換 GGUF。

GGUF 產物暫存在 `outputs/.gguf/<output-name>/`。轉換先寫入 `.partial`，成功後才原子改名；服務中斷後不會沿用不完整檔案。量化完成後預設刪除大型 BF16 中間檔，也可以在頁面勾選保留，方便日後產生不同量化。轉換尖峰空間約為原始模型大小加上量化輸出大小；開始前會先檢查可用磁碟空間。

Docker image 在建置時會從 llama.cpp 編譯 `llama-quantize` 並安裝轉換器依賴。預設固定在已確認支援 `Gemma4UnifiedForConditionalGeneration` 的 commit `e3546c7948e3af463d0b401e6421d5a4c2faf565`；也可用 build argument 指定其他 llama.cpp branch、tag 或 commit：

```bash
docker compose build --build-arg LLAMA_CPP_REF=master webui
docker compose up -d webui
```

Gemma 4 需要 Transformers 5。映像不會直接安裝 llama.cpp 的通用 requirements，因為其 CPU PyTorch、Transformers 4 與 NumPy 1 pins 會破壞 Heretic 的 CUDA 環境；`requirements-gguf.txt` 只補上 converter 尚缺的 tokenizer/protobuf 套件，其餘沿用 Heretic 環境。

`llama-quantize` 使用 CPU OpenMP 執行量化，因此 runtime image 也會安裝 `libgomp1`。若從未包含此套件的舊 image 更新，必須重新 build，單純 restart Container 不會安裝新的系統套件。

匯入時可直接編輯 Modelfile。支援 `FROM .`、`PARAMETER`、`TEMPLATE`、`SYSTEM`、`LICENSE` 與 `MESSAGE`；後端會先驗證語法，再轉成 Ollama 新版 `/api/create` 的對應欄位。完整 output 的來源已由模型選單決定，因此 `FROM` 必須保持為 `.`，且不支援 `ADAPTER`。

建立任務時也可以在 WebUI 的「HF Hub Token」欄位填入 read token。若有填入，該任務會用此 Token 下載 private 或 gated 模型，並以 `0600` 權限保存到 `/data/hf_token`；因為 `/data` 對應主機的 `./data`，容器重啟或重建後仍可沿用。Token 不會出現在任務紀錄、TOML 或日誌中。再次輸入可取代已保存的 Token；留空時依序使用已保存的 Token、容器的 `HF_TOKEN`，否則匿名下載。

## LoRA 管理

「LoRA 管理」頁面可以直接輸入 Hugging Face repository ID 下載 Safetensors 或 GGUF adapter。私有與 gated repository 會沿用上述 HF Token；下載先放在隱藏的 `.partial` 目錄，驗證含有完整 adapter 後才加入模型庫。若 GGUF repository 提供多個量化檔，建議指定單一檔名，避免全部下載。

匯入時需指定 Ollama 內已有、且與 adapter 訓練時相同的基底模型。若 `adapter_config.json` 含有 `base_model_name_or_path`，介面會自動帶入供確認。WebUI 會把 adapter 上傳至 Ollama blob API，再以 `/api/create` 的 `from` 與 `adapters` 建立新模型。Ollama 官方目前支援 Llama、Mistral 與部分 Gemma Safetensors adapter，也可使用 GGUF adapter；基底模型不符或不同量化方式的 QLoRA 可能產生不正常結果。

刪除 LoRA 只會移除 `/data/loras/<name>` 的本機 adapter，不會自動刪除已建立的 Ollama 模型。正在下載或匯入的 LoRA 會拒絕刪除。

## Heretic 版本管理

「Heretic 版本」頁面會顯示 Active Slot 與 commit，並可手動檢查官方 `origin/master`、更新至最新版，以及用「退回上一個版本」按鈕切回更新前的 Slot。每次成功更新只保留一個 rollback point。

版本存放在 `/data/heretic_slots/A` 與 `/data/heretic_slots/B`。更新時只會重建 Inactive Slot：checkout 官方 commit、套用 `patches/heretic/*.patch`、執行 Python compile 與 import smoke test；全部成功後才以原子寫入 `heretic_version.json` 的方式切換 Active Slot。任何步驟失敗都不會修改現行 Slot。

新任務會將建立當下的 slot 與 commit 寫入任務 metadata，執行期間固定從該 Slot 的 `src` 載入 Heretic。若 `pyproject.toml` 或 `uv.lock` 不同，介面會要求重新建置以安裝相符依賴：

```bash
docker compose up --build -d
```

版本切換會在有 Heretic 任務執行時拒絕操作。系統不會排程檢查或自動更新，只有使用者按下按鈕並確認後才會連線與切換版本。

本專案的 Gemma 4 Unified offload 儲存修正放在 `patches/heretic/0001-gemma4-offload-save.patch`。若新版已包含相同修正，會記錄為 `included_upstream`；若無法套用則候選 Slot 驗證失敗，Active Slot 保持不變。

停止服務不會刪除模型、checkpoint 或任務紀錄。中斷中的任務不會自動接回原程序；服務恢復後可在任務紀錄中重試：

```bash
docker compose down
```

若任務因模型載入或匯出錯誤而失敗，可在任務紀錄中按「重試任務」。重試會沿用同一份 config 與 checkpoint；已完成的最佳化 trials 不需重新計算。

## 儲存目錄

| 主機路徑 | 容器路徑 | 用途 |
|---|---|---|
| `./data` | `/data` | Hugging Face cache、下載的 LoRA、checkpoint、任務與日誌 |
| `./data/heretic_slots` | `/data/heretic_slots` | Heretic A/B source slots |
| `./data/heretic_upstream` | `/data/heretic_upstream` | 執行時從官方來源建立的 Git cache |
| `./outputs` | `/outputs` | Heretic 完成的模型或輸出 adapter |
| `./models` | `/models` | 唯讀本機來源模型 |
| `./upstream-heretic` | `/app/upstream-heretic` | 直接 vendored 的 Heretic source snapshot |

`./outputs/.gguf` 位於同一個持久化 outputs volume，存放已完成的 GGUF 與轉換中間檔。

WebUI 提供繁體中文、簡體中文、英文與日文。語言選擇同時保存於瀏覽器及 `/data/settings.json`，因此重新整理頁面或重啟 Container 後仍會沿用。

「模型與 Ollama」頁面的模型庫可以手動刪除已完成輸出。刪除前會顯示二次確認，並同時清理該 output 的 Safetensors 與 `outputs/.gguf/<output-name>`；正在執行 Heretic 或 Ollama 匯入的模型會拒絕刪除。刪除後原任務仍保留為完成狀態，並記錄其 output 已被移除。

若要處理本機模型，放入 `./models`，再於 WebUI 使用 `/models/<資料夾>`。單一 GPU 工作站一次只允許一個執行中任務。

## 選項說明

- `bnb_4bit`：降低載入與分析所需 VRAM；完整模型匯出仍可能需要大量系統 RAM。
- `merge`：輸出可直接由 Transformers 載入的完整模型。
- `adapter`：只輸出 abliteration LoRA，磁碟與合併 RAM 需求較低。
- `Trials`：Heretic 的 Optuna 最佳化次數。上游預設為 200。
- `Batch size = 0`：由 Heretic 自動測試硬體後決定。

## 開發與測試

後端是 FastAPI，正式執行固定使用單一 worker，因為任務狀態與子程序由該 worker 管理。

```bash
python -m pip install -r requirements-web.txt pytest
pytest -q
```

API 文件位於 <http://localhost:8000/api/docs>。

## 上游與授權

本專案以 [p-e-w/heretic](https://github.com/p-e-w/heretic) 為上游來源，`upstream-heretic/` 直接保存其 source snapshot，不是 Git submodule。目前整合的是上游 commit [`c8a254b8251fcd7eadd061242a725f7338d3296e`](https://github.com/p-e-w/heretic/commit/c8a254b8251fcd7eadd061242a725f7338d3296e)。手動 A/B 更新器會在執行時從同一官方 repository clone 至 `/data/heretic_upstream`。Heretic 與本 WebUI 依 AGPL-3.0-or-later 授權；上游著作權與完整授權文字見 `upstream-heretic/LICENSE`。

此工具會改變模型行為。請自行確認來源模型授權、資料集條款與實際用途符合所在地法律及部署政策。
