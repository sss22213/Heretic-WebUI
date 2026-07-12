# Heretic WebUI

English | [繁體中文](README.zh-TW.md)

A Docker-based web interface for [p-e-w/heretic](https://github.com/p-e-w/heretic). It provides a practical workspace for creating and monitoring model abliteration jobs, managing LoRA adapters, converting model outputs to GGUF, and publishing models to Ollama.

## Features

- Create and monitor Heretic abliteration jobs from a browser.
- Configure optimization, quantization, datasets, prompts, and export strategy.
- Stream job logs, cancel running jobs, and retry failed jobs from checkpoints.
- Store Hugging Face credentials securely outside job metadata and logs.
- Download and manage Safetensors or GGUF LoRA adapters from Hugging Face.
- Import full models and LoRA-based models into Ollama over its HTTP API.
- Convert unsupported Safetensors models to GGUF with llama.cpp.
- Manage completed model outputs and generated GGUF artifacts.
- Manually update Heretic through validated A/B source slots with one-click rollback.
- Use the interface in Traditional Chinese, Simplified Chinese, English, or Japanese.

## Requirements

- Linux
- Docker Engine 24 or later
- Docker Compose v2
- An NVIDIA GPU with a compatible driver
- NVIDIA Container Toolkit
- Enough VRAM, system RAM, and disk space for the selected model

Merging a model can require substantially more system RAM than the 4-bit analysis phase. GGUF conversion also temporarily requires space for both the BF16 intermediate and the quantized output.

## Quick Start

Clone the repository:

```bash
git clone <repository-url>
cd abliterated
```

```bash
cp .env.example .env

# Edit .env when you need an HF token, a custom Ollama URL,
# or host UID/GID values other than 1000:1000.

docker compose up --build -d
```

Open <http://localhost:8000>.

Follow the service logs with:

```bash
docker compose logs -f webui
```

Stop the service without deleting persistent data:

```bash
docker compose down
```

## Configuration

The main environment settings are available in `.env.example`:

```dotenv
HF_TOKEN=
OLLAMA_BASE_URL=http://host.docker.internal:11434
PUID=1000
PGID=1000
```

`HF_TOKEN` is only required for private or gated Hugging Face repositories. You can also enter a read token in the WebUI. A token entered through the UI is stored at `/data/hf_token` with `0600` permissions and is never written to job metadata, generated TOML files, or logs.

The token lookup order is:

1. A token supplied for the current job
2. The token saved by the WebUI
3. The container's `HF_TOKEN` environment variable
4. Anonymous Hugging Face access

## Creating Heretic Jobs

The **Create Job** page accepts a Hugging Face model ID or a model path inside the container. Local models can be placed in `./models` and selected with a path such as `/models/my-model`.

Only one GPU-intensive Heretic job is allowed at a time. Jobs continue running after you leave the page. Failed jobs can be retried with the same configuration and checkpoint, so completed Optuna trials do not need to be repeated.

### Export strategies

- `merge`: exports a complete model that can be loaded directly by Transformers.
- `adapter`: exports only the abliteration LoRA adapter, reducing disk and merge-time RAM requirements.

### Important options

- `bnb_4bit`: reduces VRAM usage during loading and analysis. A merged export can still require significant system RAM.
- `Trials`: the number of Optuna optimization trials. The upstream default is 200.
- `Batch size = 0`: lets Heretic determine an appropriate batch size automatically.

## Ollama Import

The **Models & Ollama** page lists complete merged outputs from `./outputs`. Supported architectures can be uploaded directly as Safetensors through the Ollama blob API. Architectures that cannot be imported reliably in that form, including the integrated Gemma 4 Unified path, are converted to GGUF inside the WebUI container first.

Available quantization options are:

- `Q2_K`
- `Q3_K_M`
- `Q4_K_M`
- `Q6_K`
- `Q8_0`

The default Ollama endpoint is:

```text
http://host.docker.internal:11434
```

Expose Ollama's port `11434` to the host, or configure an address reachable by both containers:

```dotenv
OLLAMA_BASE_URL=http://ollama:11434
```

The WebUI calculates a SHA-256 digest for every uploaded file and skips blobs already present in Ollama. Large files are streamed instead of being loaded into memory.

### GGUF conversion

Completed GGUF artifacts are stored under:

```text
outputs/.gguf/<output-name>/
```

Conversion writes to `.partial` files and renames them atomically after success. Incomplete files are never reused after an interruption. The large BF16 intermediate is deleted after quantization by default, but it can be retained when you plan to create additional quantization variants.

The Docker image builds `llama-quantize` from llama.cpp. The default llama.cpp revision is pinned to commit `e3546c7948e3af463d0b401e6421d5a4c2faf565`, which has been verified with `Gemma4UnifiedForConditionalGeneration`. Override it with:

```bash
docker compose build --build-arg LLAMA_CPP_REF=master webui
docker compose up -d webui
```

Gemma 4 requires Transformers 5. The image intentionally does not install llama.cpp's general requirements file because its CPU PyTorch, Transformers 4, and NumPy 1 pins conflict with the Heretic CUDA environment. `requirements-gguf.txt` only installs converter dependencies missing from the Heretic environment.

`llama-quantize` uses CPU OpenMP, so the runtime image includes `libgomp1`. Rebuild the image when upgrading from an older image that does not contain this package.

### Modelfile support

Full-model imports support these Modelfile directives:

- `FROM .`
- `PARAMETER`
- `TEMPLATE`
- `SYSTEM`
- `LICENSE`
- `MESSAGE`

The backend validates the Modelfile and converts it into fields accepted by Ollama's `/api/create` endpoint. The source model is selected separately in the UI, so `FROM` must remain `.` for full-output imports. `ADAPTER` is handled through the dedicated LoRA workflow instead.

## LoRA Management

The **LoRA Management** page downloads Safetensors or GGUF adapters directly from a Hugging Face repository. Downloads are written to a hidden `.partial` directory and only published to the local library after the adapter has passed validation.

For repositories containing multiple GGUF variants, specify a filename to avoid downloading every adapter artifact.

A valid Safetensors adapter must contain an `adapter_config.json` file and one or more `adapter_model` weight files. Sharded adapters are supported. GGUF adapters are detected by their `.gguf` extension.

When `adapter_config.json` contains `base_model_name_or_path`, the UI uses it as the suggested Ollama base model. Always verify that the selected Ollama model matches the model used to train the adapter. A mismatched base model can produce unpredictable results, and QLoRA adapters may be incompatible when quantization methods differ.

The adapter is uploaded through Ollama's blob API and a derived model is created with the `/api/create` `from` and `adapters` fields.

Deleting a LoRA removes the local adapter under `/data/loras/<name>`. It does not delete Ollama models that were previously created from that adapter. LoRAs currently being downloaded or imported cannot be deleted.

## Manual Heretic Updates with A/B Slots

Heretic updates are never scheduled or installed automatically. The system only checks or updates when a user explicitly presses the corresponding button and confirms the action.

The active Heretic source is stored in one of these persistent slots:

```text
/data/heretic_slots/A
/data/heretic_slots/B
```

An update follows this process:

1. Fetch the selected upstream branch into `/data/heretic_upstream`, a runtime Git cache cloned from the official repository.
2. Rebuild only the inactive slot at the new commit.
3. Apply every patch from `patches/heretic/*.patch`.
4. Compile the Heretic Python package.
5. Run an import smoke test against `heretic.main` from the candidate slot.
6. Atomically update `heretic_version.json` to activate the candidate slot.

The active slot is never modified while the candidate is being downloaded, patched, or validated. If any step fails, the existing active slot remains unchanged.

After a successful update, the previous slot becomes the rollback target. The **Roll Back to Previous Version** button atomically switches back to that slot. One rollback point is retained.

Each new job records the active slot and exact Heretic commit at creation time, then continues using that source for its entire run. Version switching is rejected while a Heretic job is queued or running.

If `pyproject.toml` or `uv.lock` differs between slots, the UI reports that the Docker image should be rebuilt before starting model jobs:

```bash
docker compose up --build -d
```

### Managed Gemma 4 patch

The Gemma 4 Unified offload save compatibility fix is maintained separately at:

```text
patches/heretic/0001-gemma4-offload-save.patch
```

An update is considered successful only after this patch has been applied or detected as already included upstream. An incompatible patch causes candidate-slot validation to fail without affecting the active slot.

## Persistent Storage

| Host path | Container path | Purpose |
|---|---|---|
| `./data` | `/data` | Hugging Face cache, downloaded LoRAs, checkpoints, jobs, and logs |
| `./data/heretic_slots` | `/data/heretic_slots` | Validated Heretic A/B source slots |
| `./data/heretic_upstream` | `/data/heretic_upstream` | Runtime clone of the official Heretic Git repository |
| `./outputs` | `/outputs` | Completed Heretic models and output adapters |
| `./models` | `/models` | Read-only local source models |
| `./upstream-heretic` | `/app/upstream-heretic` | Vendored Heretic source snapshot used when building the image |

Deleting a completed model from the model library removes both its Safetensors output and matching `outputs/.gguf/<output-name>` directory. Models used by active Heretic or Ollama operations cannot be deleted. Job history remains available and records that its output was removed.

## Development and Testing

The backend is implemented with FastAPI. Production runs with one Uvicorn worker because job, subprocess, and version-switching state is coordinated within that worker.

Install the web dependencies and run the test suite with:

```bash
python -m pip install -r requirements-web.txt pytest
pytest -q
```

Validate the frontend JavaScript with:

```bash
node --check app/static/app.js
```

Interactive API documentation is available at <http://localhost:8000/api/docs>.

## Upstream and License

This project is built on **Heretic**, created by Philipp Emanuel Weidmann and maintained in the official [`p-e-w/heretic`](https://github.com/p-e-w/heretic) repository. The source snapshot under `upstream-heretic/` is vendored directly into this repository and is not a Git submodule. The initial integrated upstream revision is [`c8a254b8251fcd7eadd061242a725f7338d3296e`](https://github.com/p-e-w/heretic/commit/c8a254b8251fcd7eadd061242a725f7338d3296e).

At runtime, the manual A/B updater clones the same official repository into `/data/heretic_upstream`. No scheduled or automatic update is performed.

Heretic and this WebUI are licensed under the GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE), [upstream-heretic/LICENSE](upstream-heretic/LICENSE), and the official [Heretic repository](https://github.com/p-e-w/heretic) for complete license terms and upstream copyright notices.

## Responsible Use

This software changes model behavior. You are responsible for ensuring that the source model license, dataset terms, deployment policy, and intended use comply with applicable laws and organizational requirements.
