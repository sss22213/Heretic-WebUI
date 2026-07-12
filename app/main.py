# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .heretic_version import HereticVersionManager
from .lora_manager import LoRAManager
from .ollama_import import OllamaImportManager

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "app" / "static"
DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.getenv("APP_OUTPUT_DIR", ROOT / "outputs")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
HF_TOKEN_FILE = DATA_DIR / "hf_token"
SETTINGS_FILE = DATA_DIR / "settings.json"
HERETIC_VERSION_FILE = DATA_DIR / "heretic_version.json"
HERETIC_SOURCE_DIR = Path(
    os.getenv("HERETIC_SOURCE_DIR", DATA_DIR / "heretic_upstream")
).resolve()
HERETIC_PATCH_DIR = Path(os.getenv("HERETIC_PATCH_DIR", ROOT / "patches" / "heretic")).resolve()
HERETIC_UPSTREAM_URL = os.getenv(
    "HERETIC_UPSTREAM_URL", "https://github.com/p-e-w/heretic.git"
)
HERETIC_INITIAL_REF = os.getenv(
    "HERETIC_INITIAL_REF", "c8a254b8251fcd7eadd061242a725f7338d3296e"
)
HERETIC_BIN = os.getenv("HERETIC_BIN", "heretic")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")

for directory in (JOBS_DIR, CHECKPOINT_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return slug[:72] or "model"


class JobRequest(BaseModel):
    model: str = Field(min_length=1, max_length=300)
    hf_token: str | None = Field(default=None, max_length=2048, exclude=True, repr=False)
    output_name: str | None = Field(default=None, max_length=100)
    quantization: Literal["none", "bnb_4bit"] = "none"
    export_strategy: Literal["merge", "adapter"] = "merge"
    n_trials: int = Field(default=200, ge=1, le=5000)
    n_startup_trials: int = Field(default=60, ge=0, le=5000)
    batch_size: int = Field(default=0, ge=0, le=2048)
    max_batch_size: int = Field(default=128, ge=1, le=4096)
    max_response_length: int = Field(default=100, ge=1, le=4096)
    max_shard_size: str = Field(default="5GB", pattern=r"^[1-9][0-9]*(MB|GB)$")
    offload_outputs_to_cpu: bool = True
    orthogonalize_direction: bool = True
    row_normalization: Literal["none", "pre", "full"] = "full"
    lora_rank: int = Field(default=3, ge=1, le=512)
    system_prompt: str = Field(default="You are a helpful assistant.", max_length=4000)
    good_dataset: str = Field(default="mlabonne/harmless_alpaca", min_length=1, max_length=300)
    good_split: str = Field(default="train[:400]", min_length=1, max_length=100)
    good_column: str = Field(default="text", min_length=1, max_length=100)
    bad_dataset: str = Field(default="mlabonne/harmful_behaviors", min_length=1, max_length=300)
    bad_split: str = Field(default="train[:400]", min_length=1, max_length=100)
    bad_column: str = Field(default="text", min_length=1, max_length=100)

    @field_validator("model", "good_dataset", "bad_dataset")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("不可包含控制字元")
        return value

    @field_validator("hf_token")
    @classmethod
    def normalize_hf_token(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if any(ord(char) < 32 for char in value):
            raise ValueError("HF Hub Token 不可包含控制字元")
        return value

    @field_validator("n_startup_trials")
    @classmethod
    def startup_not_excessive(cls, value: int, info):
        trials = info.data.get("n_trials")
        if trials is not None and value > trials:
            raise ValueError("n_startup_trials 不可大於 n_trials")
        return value


class OllamaImportRequest(BaseModel):
    output_name: str = Field(min_length=1, max_length=160, pattern=r"^[a-zA-Z0-9._-]+$")
    model_name: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*(?::[a-zA-Z0-9._-]+)?$",
    )
    base_url: str = Field(min_length=8, max_length=500)
    quantize: Literal["q2_K", "q3_K_M", "q4_K_M", "q6_K", "q8_0"] | None = None
    import_format: Literal["auto", "safetensors", "gguf"] = "auto"
    keep_intermediate: bool = False
    modelfile: str = Field(default="FROM .", max_length=65536)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("Ollama API 位址必須以 http:// 或 https:// 開頭")
        return value


class LoRADownloadRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=300)
    revision: str = Field(default="main", min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9._-]+$")
    filename: str | None = Field(default=None, max_length=500)
    hf_token: str | None = Field(default=None, max_length=2048, exclude=True, repr=False)

    @field_validator("repo_id", "revision", "filename", "hf_token")
    @classmethod
    def normalize_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if any(ord(char) < 32 for char in value):
            raise ValueError("不可包含控制字元")
        return value or None


class LoRAImportRequest(BaseModel):
    lora_name: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9._-]+$")
    model_name: str = Field(
        min_length=1, max_length=200,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*(?::[a-zA-Z0-9._-]+)?$",
    )
    base_model: str = Field(
        min_length=1, max_length=200,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._/-]*(?::[a-zA-Z0-9._-]+)?$",
    )
    base_url: str = Field(min_length=8, max_length=500)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("Ollama API 位址必須以 http:// 或 https:// 開頭")
        return value


class UISettingsRequest(BaseModel):
    language: Literal["zh-TW", "zh-CN", "en", "ja"] = "zh-TW"


class HereticVersionActionRequest(BaseModel):
    confirmation: Literal["UPDATE", "ROLLBACK"]


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()

    def get(self) -> dict[str, str]:
        with self.lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                return UISettingsRequest.model_validate(raw).model_dump()
            except (FileNotFoundError, OSError, ValueError, TypeError):
                return UISettingsRequest().model_dump()

    def save(self, settings: UISettingsRequest) -> dict[str, str]:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(
                json.dumps(settings.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path)
            return settings.model_dump()

@dataclass
class Job:
    id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: dict
    output_directory: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    pid: int | None = None
    log_size: int = 0
    command: list[str] = field(default_factory=list)
    output_deleted: bool = False
    heretic_slot: str | None = None
    heretic_commit: str | None = None


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_config(request: JobRequest, output_directory: Path, job_id: str) -> str:
    values = {
        "model": request.model,
        "quantization": request.quantization,
        "device_map": "auto",
        "offload_outputs_to_cpu": request.offload_outputs_to_cpu,
        "batch_size": request.batch_size,
        "max_batch_size": request.max_batch_size,
        "max_response_length": request.max_response_length,
        "orthogonalize_direction": request.orthogonalize_direction,
        "row_normalization": request.row_normalization,
        "full_normalization_lora_rank": request.lora_rank,
        "n_trials": request.n_trials,
        "n_startup_trials": request.n_startup_trials,
        "study_checkpoint_dir": str(CHECKPOINT_DIR / job_id),
        "max_shard_size": request.max_shard_size,
        "export_strategy": request.export_strategy,
        "checkpoint_action": "continue",
        "trial_index": 0,
        "model_action": "save",
        "save_directory": str(output_directory),
        "system_prompt": request.system_prompt,
    }
    lines: list[str] = ["# Generated by Heretic WebUI. Do not edit while the job is running."]
    for key, value in values.items():
        if isinstance(value, bool):
            encoded = str(value).lower()
        elif isinstance(value, int):
            encoded = str(value)
        else:
            encoded = toml_string(value)
        lines.append(f"{key} = {encoded}")
    lines.extend(
        [
            "",
            "[good_prompts]",
            f"dataset = {toml_string(request.good_dataset)}",
            f"split = {toml_string(request.good_split)}",
            f"column = {toml_string(request.good_column)}",
            "",
            "[bad_prompts]",
            f"dataset = {toml_string(request.bad_dataset)}",
            f"split = {toml_string(request.bad_split)}",
            f"column = {toml_string(request.bad_column)}",
            "",
        ]
    )
    return "\n".join(lines)


def job_environment(hf_token: str | None, heretic_source: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "TERM": "dumb", "NO_COLOR": "1"})
    if heretic_source is not None:
        slot_source = str(heretic_source / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = slot_source + (os.pathsep + existing if existing else "")
    if hf_token:
        # huggingface_hub reads HF_TOKEN automatically. Keep the legacy name for
        # compatibility with dependencies that still expect it.
        env.update({"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token})
    return env


def output_artifacts_complete(job: Job, output_directory: Path | None = None) -> bool:
    directory = output_directory or Path(job.output_directory)
    if not directory.is_dir():
        return False

    strategy = job.request.get("export_strategy", "merge")
    if strategy == "adapter":
        return (directory / "adapter_config.json").is_file() and any(
            (directory / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return bool(shards) and all((directory / shard).is_file() for shard in shards)

    return any(
        (directory / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


class HFTokenStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()

    def get(self) -> str | None:
        with self.lock:
            try:
                token = self.path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                return None
            except OSError:
                return None
            return token or None

    def save(self, token: str) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(token)
                    handle.write("\n")
                temporary.replace(self.path)
                self.path.chmod(0o600)
            finally:
                temporary.unlink(missing_ok=True)


hf_token_store = HFTokenStore(HF_TOKEN_FILE)
settings_store = SettingsStore(SETTINGS_FILE)


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        # Per-job copies stay in memory and are removed as soon as the worker
        # starts. Persistence is handled only by the private HFTokenStore file.
        self.job_tokens: dict[str, str] = {}
        self._load_jobs()

    def _job_dir(self, job_id: str) -> Path:
        return JOBS_DIR / job_id

    def _output_dir(self, job: Job) -> Path:
        path = Path(job.output_directory)
        if path.exists() or path.parent == OUTPUT_DIR:
            return path
        # Job metadata created in Docker stores /outputs/...; map it back to the
        # configured output root when running tests or development on the host.
        if path.parent == Path("/outputs"):
            return OUTPUT_DIR / path.name
        return path

    def _load_jobs(self) -> None:
        for metadata in JOBS_DIR.glob("*/job.json"):
            try:
                raw = json.loads(metadata.read_text(encoding="utf-8"))
                job = Job(**raw)
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.finished_at = utc_now()
                    job.error = "Web 服務重啟，原執行程序已失聯。請建立新任務重新執行。"
                    self._persist(job)
                elif job.status == "completed" and not job.output_deleted and not output_artifacts_complete(
                    job, self._output_dir(job)
                ):
                    job.status = "failed"
                    job.error = "Heretic 雖正常結束，但未產生完整的模型權重，請重試任務。"
                    self._persist(job)
                self.jobs[job.id] = job
            except (OSError, ValueError, TypeError):
                continue

    def _persist(self, job: Job) -> None:
        directory = self._job_dir(job.id)
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / "job.json.tmp"
        temp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory / "job.json")

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    def output_in_use(self, output_name: str) -> bool:
        with self.lock:
            return any(
                job.status in ("queued", "running")
                and self._output_dir(job).name == output_name
                for job in self.jobs.values()
            )

    def mark_output_deleted(self, output_name: str) -> None:
        with self.lock:
            for job in self.jobs.values():
                if self._output_dir(job).name == output_name:
                    job.output_deleted = True
                    self._persist(job)

    def create(self, request: JobRequest) -> Job:
        with self.lock:
            if any(job.status in ("queued", "running") for job in self.jobs.values()):
                raise RuntimeError("GPU 正由另一個任務使用中")
            if request.hf_token:
                hf_token_store.save(request.hf_token)
            job_id = uuid.uuid4().hex[:12]
            runtime = heretic_version_manager.runtime_info()
            output_name = safe_slug(request.output_name or f"{request.model.split('/')[-1]}-heretic")
            output_directory = OUTPUT_DIR / f"{output_name}-{job_id[:6]}"
            job = Job(
                id=job_id,
                status="queued",
                request=request.model_dump(),
                output_directory=str(output_directory),
                created_at=utc_now(),
                command=[HERETIC_BIN],
                heretic_slot=runtime["slot"],
                heretic_commit=runtime["commit"],
            )
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True)
            (job_dir / "config.toml").write_text(
                render_config(request, output_directory, job_id), encoding="utf-8"
            )
            self.jobs[job.id] = job
            if request.hf_token:
                self.job_tokens[job.id] = request.hf_token
            self._persist(job)
            threading.Thread(target=self._run, args=(job.id,), daemon=True).start()
            return job

    def _run(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        log_path = job_dir / "run.log"
        with self.lock:
            job = self.jobs[job_id]
            hf_token = self.job_tokens.pop(job_id, None) or hf_token_store.get()
            if job.status == "cancelled":
                job.finished_at = utc_now()
                self._persist(job)
                return
            job.status = "running"
            job.started_at = utc_now()
            self._persist(job)
        try:
            runtime = heretic_version_manager.runtime_info(job.heretic_slot)
            if job.heretic_commit and runtime["commit"] != job.heretic_commit:
                raise RuntimeError(
                    f"任務指定的 Heretic {job.heretic_commit[:7]} 已不在 slot {job.heretic_slot}"
                )
            env = job_environment(hf_token, Path(runtime["path"]))
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [HERETIC_BIN],
                    cwd=job_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
                with self.lock:
                    self.processes[job_id] = process
                    job.pid = process.pid
                    self._persist(job)
                exit_code = process.wait()
            with self.lock:
                if job.status != "cancelled":
                    artifacts_complete = output_artifacts_complete(job, self._output_dir(job))
                    job.status = "completed" if exit_code == 0 and artifacts_complete else "failed"
                    if exit_code == 0 and not artifacts_complete:
                        job.error = "Heretic 雖正常結束，但未產生完整的模型權重，請重試任務。"
                    elif exit_code != 0:
                        job.error = f"Heretic 結束碼：{exit_code}"
                job.exit_code = exit_code
        except Exception as exc:
            with self.lock:
                job.status = "failed"
                job.error = str(exc)
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                job.pid = None
                job.finished_at = utc_now()
                job.log_size = log_path.stat().st_size if log_path.exists() else 0
                self._persist(job)

    def cancel(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            if job.status not in ("queued", "running"):
                raise RuntimeError("此任務目前無法取消")
            job.status = "cancelled"
            job.error = "使用者取消任務"
            process = self.processes.get(job_id)
            self._persist(job)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return job

    def retry(self, job_id: str) -> Job:
        with self.lock:
            job = self.get(job_id)
            if job.status != "failed":
                raise RuntimeError("只有失敗的任務可以重試")
            if any(
                other.status in ("queued", "running")
                for other in self.jobs.values()
                if other.id != job_id
            ):
                raise RuntimeError("GPU 正由另一個任務使用中")
            if job.heretic_slot and job.heretic_commit:
                runtime = heretic_version_manager.runtime_info(job.heretic_slot)
                if runtime["commit"] != job.heretic_commit:
                    raise RuntimeError(
                        "此任務原本使用的 Heretic commit 已被 A/B Slot 更新覆蓋，"
                        "請建立新任務。"
                    )

            job.status = "queued"
            job.started_at = None
            job.finished_at = None
            job.exit_code = None
            job.error = None
            job.pid = None
            self._persist(job)

            log_path = self._job_dir(job_id) / "run.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n--- {utc_now()}：重試任務（沿用現有 checkpoint）---\n")

            threading.Thread(target=self._run, args=(job.id,), daemon=True).start()
            return job

    def log(self, job_id: str, offset: int) -> tuple[str, int]:
        self.get(job_id)
        path = self._job_dir(job_id) / "run.log"
        if not path.exists():
            return "", 0
        size = path.stat().st_size
        offset = min(max(offset, 0), size)
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read(256 * 1024)
        return chunk.decode("utf-8", errors="replace"), offset + len(chunk)


manager = JobManager()
ollama_manager = OllamaImportManager(OUTPUT_DIR, DATA_DIR)
lora_manager = LoRAManager(DATA_DIR)
heretic_version_manager = HereticVersionManager(
    HERETIC_SOURCE_DIR,
    HERETIC_VERSION_FILE,
    HERETIC_PATCH_DIR,
    upstream_url=HERETIC_UPSTREAM_URL,
    initial_ref=HERETIC_INITIAL_REF,
)
app = FastAPI(title="Heretic WebUI", version="1.0.0", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def prevent_stale_frontend_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "heretic_available": shutil.which(HERETIC_BIN) is not None,
        "active_jobs": sum(job.status in ("queued", "running") for job in manager.list()),
    }


@app.get("/api/system")
def system_info():
    gpu = "未偵測到 NVIDIA GPU"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "gpu": gpu,
        "output_directory": str(OUTPUT_DIR),
        "hf_token_saved": hf_token_store.get() is not None,
        "ollama_base_url": OLLAMA_BASE_URL,
        "gguf_tools_available": ollama_manager.gguf_tools_available(),
    }


@app.get("/api/outputs")
def list_outputs():
    return ollama_manager.list_outputs()


@app.delete("/api/outputs/{output_name}")
def delete_output(output_name: str):
    if manager.output_in_use(output_name):
        raise HTTPException(status_code=409, detail="此模型正由 Heretic 任務使用，無法刪除")
    try:
        result = ollama_manager.delete_output(output_name)
        manager.mark_output_deleted(output_name)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/settings")
def get_settings():
    return settings_store.get()


@app.put("/api/settings")
def update_settings(request: UISettingsRequest):
    return settings_store.save(request)


def ensure_heretic_version_idle() -> None:
    if any(job.status in ("queued", "running") for job in manager.list()):
        raise HTTPException(status_code=409, detail="Heretic 任務執行中，無法切換版本")


@app.get("/api/heretic/version")
def get_heretic_version(check_remote: bool = False):
    try:
        return heretic_version_manager.status(check_remote=check_remote)
    except RuntimeError as exc:
        raise HTTPException(status_code=502 if check_remote else 409, detail=str(exc)) from exc


@app.post("/api/heretic/version/update")
def update_heretic_version(request: HereticVersionActionRequest):
    if request.confirmation != "UPDATE":
        raise HTTPException(status_code=400, detail="更新確認值不正確")
    ensure_heretic_version_idle()
    try:
        return heretic_version_manager.update()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/heretic/version/rollback")
def rollback_heretic_version(request: HereticVersionActionRequest):
    if request.confirmation != "ROLLBACK":
        raise HTTPException(status_code=400, detail="退版確認值不正確")
    ensure_heretic_version_idle()
    try:
        return heretic_version_manager.rollback()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/ollama/import")
def get_ollama_import():
    return ollama_manager.get()


@app.post("/api/ollama/import", status_code=202)
def create_ollama_import(request: OllamaImportRequest):
    try:
        return asdict(
            ollama_manager.start(
                request.output_name,
                request.model_name,
                request.base_url,
                request.quantize,
                request.modelfile,
                request.import_format,
                request.keep_intermediate,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/loras")
def list_loras():
    return lora_manager.list()


@app.post("/api/loras/download", status_code=202)
def download_lora(request: LoRADownloadRequest):
    token = request.hf_token or hf_token_store.get()
    if request.hf_token:
        hf_token_store.save(request.hf_token)
    try:
        return asdict(
            lora_manager.start_download(
                request.repo_id, request.revision or "main", request.name,
                token, request.filename,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/loras/task")
def get_lora_task():
    return lora_manager.get_task()


@app.post("/api/loras/import", status_code=202)
def import_lora(request: LoRAImportRequest):
    try:
        return asdict(
            lora_manager.start_import(
                request.lora_name, request.model_name, request.base_model, request.base_url
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/api/loras/{lora_name}")
def delete_lora(lora_name: str):
    try:
        return lora_manager.delete(lora_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs")
def list_jobs():
    return [asdict(job) for job in manager.list()]


@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest):
    try:
        return asdict(manager.create(request))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return asdict(manager.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str, offset: int = Query(default=0, ge=0)):
    try:
        content, next_offset = manager.log(job_id, offset)
        return {"content": content, "next_offset": next_offset}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    try:
        job = await asyncio.to_thread(manager.cancel, job_id)
        return asdict(job)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str):
    try:
        return asdict(manager.retry(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="找不到任務") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
