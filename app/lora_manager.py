# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import snapshot_download

from .ollama_import import OllamaClient, format_bytes


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def adapter_files(directory: Path) -> list[Path]:
    """Return only files understood by Ollama's adapter importer."""
    files = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(directory)
        is_adapter_model = path.name.startswith("adapter_model") and path.suffix.lower() in {
            ".safetensors", ".bin", ".json"
        }
        if path.suffix.lower() == ".gguf" or path.name == "adapter_config.json" or is_adapter_model:
            files.append(relative)
    return sorted(files)


def inspect_adapter(directory: Path) -> dict:
    files = adapter_files(directory)
    gguf = [path for path in files if path.suffix.lower() == ".gguf"]
    config_paths = [path for path in files if path.name == "adapter_config.json"]
    weights = [
        path for path in files
        if path.name.startswith("adapter_model") and path.suffix.lower() in (".safetensors", ".bin")
    ]
    if not gguf and not (config_paths and weights):
        raise ValueError("下載內容不含完整的 LoRA adapter（需要 GGUF，或 adapter_config.json + adapter_model）")
    base_model = None
    if config_paths:
        try:
            config = json.loads((directory / config_paths[0]).read_text(encoding="utf-8"))
            value = config.get("base_model_name_or_path")
            base_model = value if isinstance(value, str) and value.strip() else None
        except (OSError, ValueError, TypeError):
            pass
    return {
        "format": "gguf" if gguf else "safetensors",
        "base_model": base_model,
        "files": [path.as_posix() for path in files],
        "size": sum((directory / path).stat().st_size for path in files),
    }


@dataclass
class LoRATask:
    id: str
    operation: str
    status: str
    created_at: str
    lora_name: str
    repo_id: str | None = None
    revision: str | None = None
    model_name: str | None = None
    base_model: str | None = None
    base_url: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    phase: str = "queued"
    current_file: str | None = None
    bytes_completed: int = 0
    bytes_total: int = 0
    error: str | None = None


class LoRAManager:
    DOWNLOAD_PATTERNS = [
        "*.gguf", "**/*.gguf",
        "adapter_config.json", "**/adapter_config.json",
        "adapter_model*.safetensors", "**/adapter_model*.safetensors",
        "adapter_model*.bin", "**/adapter_model*.bin",
        "adapter_model*.json", "**/adapter_model*.json",
        "README*", "LICENSE*",
    ]

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "loras"
        self.tasks_dir = data_dir / "lora_tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.current: LoRATask | None = None
        self._load_latest()

    @staticmethod
    def validate_name(value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,120}", value):
            raise ValueError("LoRA 名稱只能包含英數字、點、底線與連字號")
        return value

    def _load_latest(self) -> None:
        paths = sorted(self.tasks_dir.glob("*/task.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not paths:
            return
        try:
            task = LoRATask(**json.loads(paths[0].read_text(encoding="utf-8")))
            if task.status in ("queued", "running"):
                task.status = "failed"
                task.phase = "failed"
                task.error = "Web 服務重啟，LoRA 作業已中斷，請重新執行。"
                task.finished_at = utc_now()
                self._persist(task)
            self.current = task
        except (OSError, ValueError, TypeError):
            pass

    def _persist(self, task: LoRATask) -> None:
        directory = self.tasks_dir / task.id
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "task.json.tmp"
        temporary.write_text(json.dumps(asdict(task), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "task.json")

    def _log(self, task: LoRATask, message: str) -> None:
        with self.lock:
            directory = self.tasks_dir / task.id
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(message.rstrip() + "\n")
            self._persist(task)

    def _new_task(self, operation: str, name: str, **values) -> LoRATask:
        with self.lock:
            if self.current and self.current.status in ("queued", "running"):
                raise RuntimeError("已有 LoRA 作業正在執行")
            task = LoRATask(
                id=uuid.uuid4().hex[:12], operation=operation, status="queued",
                created_at=utc_now(), lora_name=name, **values,
            )
            self.current = task
            self._persist(task)
            return task

    def list(self) -> list[dict]:
        results = []
        for directory in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            try:
                metadata = json.loads((directory / "lora.json").read_text(encoding="utf-8"))
                details = inspect_adapter(directory)
                results.append({**metadata, **details, "name": directory.name})
            except (OSError, ValueError, TypeError):
                continue
        return results

    def get_task(self) -> dict | None:
        with self.lock:
            if self.current is None:
                return None
            result = asdict(self.current)
            log = self.tasks_dir / self.current.id / "run.log"
            result["log"] = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            return result

    def start_download(
        self,
        repo_id: str,
        revision: str,
        name: str,
        token: str | None,
        filename: str | None,
    ) -> LoRATask:
        name = self.validate_name(name)
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id.strip()):
            raise ValueError("Hugging Face repo ID 格式應為 organization/repository")
        if (self.root / name).exists():
            raise RuntimeError("同名 LoRA 已存在，請先刪除或使用其他名稱")
        task = self._new_task(
            "download", name, repo_id=repo_id.strip(), revision=revision.strip() or "main"
        )
        threading.Thread(target=self._download, args=(task.id, token, filename), daemon=True).start()
        return task

    def _download(self, task_id: str, token: str | None, filename: str | None) -> None:
        with self.lock:
            task = self.current
            if task is None or task.id != task_id:
                return
            task.status = "running"
            task.phase = "downloading"
            task.started_at = utc_now()
            self._persist(task)
        staging = self.root / f".{task.lora_name}.{task.id}.partial"
        destination = self.root / task.lora_name
        try:
            patterns = list(self.DOWNLOAD_PATTERNS)
            if filename:
                filename = filename.strip().lstrip("/")
                if ".." in Path(filename).parts:
                    raise ValueError("無效的檔案名稱")
                patterns = [filename, "adapter_config.json", "**/adapter_config.json", "README*", "LICENSE*"]
            self._log(task, f"從 Hugging Face 下載：{task.repo_id}@{task.revision}")
            shutil.rmtree(staging, ignore_errors=True)
            snapshot_download(
                repo_id=task.repo_id,
                revision=task.revision,
                token=token,
                local_dir=staging,
                allow_patterns=patterns,
            )
            details = inspect_adapter(staging)
            metadata = {
                "repo_id": task.repo_id,
                "revision": task.revision,
                "downloaded_at": utc_now(),
            }
            (staging / "lora.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            staging.replace(destination)
            with self.lock:
                task.bytes_total = details["size"]
                task.bytes_completed = details["size"]
                task.status = "completed"
                task.phase = "completed"
                task.finished_at = utc_now()
                self._log(task, f"下載完成：{len(details['files'])} 個 adapter 檔案，共 {format_bytes(details['size'])}")
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            with self.lock:
                task.status = "failed"
                task.phase = "failed"
                task.error = str(exc)
                task.finished_at = utc_now()
                self._log(task, f"錯誤：{exc}")

    def delete(self, name: str) -> dict:
        name = self.validate_name(name)
        with self.lock:
            if self.current and self.current.status in ("queued", "running") and self.current.lora_name == name:
                raise RuntimeError("此 LoRA 正在使用中，無法刪除")
            directory = (self.root / name).resolve()
            if directory.parent != self.root.resolve() or not directory.is_dir():
                raise ValueError("找不到 LoRA")
            inspect_adapter(directory)
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory)
            return {"name": name, "deleted_bytes": size}

    def start_import(self, name: str, model_name: str, base_model: str, base_url: str) -> LoRATask:
        name = self.validate_name(name)
        directory = (self.root / name).resolve()
        if directory.parent != self.root.resolve() or not directory.is_dir():
            raise ValueError("找不到 LoRA")
        details = inspect_adapter(directory)
        task = self._new_task(
            "import", name, model_name=model_name, base_model=base_model,
            base_url=base_url.rstrip("/"), bytes_total=details["size"],
        )
        threading.Thread(target=self._import, args=(task.id,), daemon=True).start()
        return task

    def _import(self, task_id: str) -> None:
        with self.lock:
            task = self.current
            if task is None or task.id != task_id:
                return
            task.status = "running"
            task.phase = "uploading"
            task.started_at = utc_now()
            self._persist(task)
        try:
            directory = self.root / task.lora_name
            files = adapter_files(directory)
            client = OllamaClient(task.base_url or "")
            version = client.version().get("version", "未知")
            self._log(task, f"已連線至 Ollama {version}；基底模型：{task.base_model}")
            digests = {}
            completed = 0
            for relative in files:
                path = directory / relative
                task.current_file = relative.as_posix()
                digest_hash = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(8 * 1024 * 1024):
                        digest_hash.update(chunk)
                digest = f"sha256:{digest_hash.hexdigest()}"
                digests[relative.as_posix()] = digest
                if not client.blob_exists(digest):
                    self._log(task, f"上傳：{relative} ({format_bytes(path.stat().st_size)})")
                    client.upload_blob(
                        path,
                        digest,
                        lambda sent, _total, done=completed: self._progress(task, done + sent),
                    )
                completed += path.stat().st_size
                self._progress(task, completed)
            task.phase = "creating"
            task.current_file = None
            self._persist(task)
            self._log(task, f"建立 Ollama 模型：{task.model_name}")
            response = client.create(
                task.model_name or "",
                {},
                None,
                from_model=task.base_model,
                adapters=digests,
            )
            if response.get("status") != "success":
                raise RuntimeError(f"Ollama 未回報成功：{response}")
            with self.lock:
                task.status = "completed"
                task.phase = "completed"
                task.finished_at = utc_now()
                self._log(task, f"匯入完成：{task.model_name}")
        except Exception as exc:
            with self.lock:
                task.status = "failed"
                task.phase = "failed"
                task.error = str(exc)
                task.finished_at = utc_now()
                self._log(task, f"錯誤：{exc}")

    def _progress(self, task: LoRATask, completed: int) -> None:
        with self.lock:
            task.bytes_completed = completed
            self._persist(task)
