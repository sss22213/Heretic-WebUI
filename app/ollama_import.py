# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, urlsplit


LLAMA_CPP_DIR = Path(os.getenv("LLAMA_CPP_DIR", "/opt/llama.cpp"))
GGUF_OUTPUT_DIR_NAME = ".gguf"
# Architectures whose safetensors import is broken in current Ollama releases;
# route them through the local llama.cpp conversion instead. Qwen3.5/3.6 VL:
# Ollama 0.31.2's converter emits block_count including the MTP layer without
# the mtp.* tensors (missing tensor 'blk.64.attn_norm.weight') and a projector
# blob its CLIP loader rejects, so the created model cannot load.
GGUF_REQUIRED_ARCHITECTURES = {
    "Gemma4UnifiedForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
QUANTIZED_SIZE_RATIOS = {
    "q2_K": 0.2,
    "q3_K_M": 0.28,
    "q4_K_M": 0.4,
    "q6_K": 0.55,
    "q8_0": 0.65,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def complete_safetensors_directory(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return bool(shards) and all((directory / shard).is_file() for shard in shards)
    return (directory / "model.safetensors").is_file()


def model_architectures(directory: Path) -> list[str]:
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        architectures = config.get("architectures", [])
        return [value for value in architectures if isinstance(value, str)]
    except (OSError, ValueError, TypeError):
        return []


def ollama_compatibility_error(version: str, architectures: list[str]) -> str | None:
    # Confirmed against real Ollama responses. Keep this version-specific so a
    # future Ollama release can add support without being blocked by the WebUI.
    # 0.31.2 + Qwen3.5/3.6: /api/create succeeds but the produced GGUF cannot
    # load (missing MTP tensor, broken projector blob).
    qwen35 = {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"}
    unsupported = {
        "0.30.6": {"Gemma4UnifiedForConditionalGeneration"},
        "0.31.2": qwen35,
    }
    matched = unsupported.get(version, set()).intersection(architectures)
    if matched:
        architecture = sorted(matched)[0]
        return (
            f"Ollama {version} 不支援 {architecture}，從 Safetensors 匯入會產生無法載入的模型。"
            "請更新至支援此架構的 Ollama，或改用 GGUF 匯入格式（由 WebUI 內建的 llama.cpp 轉換）。"
        )
    return None


def resolve_import_format(requested: str, architectures: list[str]) -> str:
    if requested not in ("auto", "safetensors", "gguf"):
        raise ValueError("匯入格式必須是 auto、safetensors 或 gguf")
    if requested == "auto":
        return "gguf" if GGUF_REQUIRED_ARCHITECTURES.intersection(architectures) else "safetensors"
    return requested


def conversion_extra_args(source: Path) -> list[str]:
    try:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    text_config = config.get("text_config") or {}
    mtp_layers = text_config.get("mtp_num_hidden_layers", config.get("mtp_num_hidden_layers", 0))
    if isinstance(mtp_layers, int) and mtp_layers > 0:
        # Heretic's transformers round-trip drops the mtp.* head tensors while
        # config.json keeps advertising them; converting with MTP then yields a
        # GGUF whose block_count/nextn metadata reference tensors that do not
        # exist ("missing tensor 'blk.N.attn_norm.weight'"). Ollama cannot use
        # the MTP head for speculative decoding anyway, so always exclude it.
        return ["--no-mtp"]
    return []


def llama_cpp_tools(llama_cpp_dir: Path | None = None) -> tuple[Path, Path]:
    llama_cpp_dir = llama_cpp_dir or LLAMA_CPP_DIR
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer_candidates = (
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
    )
    quantizer = next((path for path in quantizer_candidates if path.is_file()), quantizer_candidates[0])
    return converter, quantizer


def gguf_artifact_paths(root: Path, output_name: str, quantize: str | None) -> tuple[Path, Path]:
    directory = root / GGUF_OUTPUT_DIR_NAME / output_name
    bf16 = directory / f"{output_name}-BF16.gguf"
    if quantize:
        final = directory / f"{output_name}-{quantize.upper()}.gguf"
    else:
        final = bf16
    return bf16, final


def normalize_extra_special_tokens(config: dict) -> tuple[dict, bool]:
    values = config.get("extra_special_tokens")
    if not isinstance(values, list):
        return config, False
    normalized = dict(config)
    mapped: dict[str, str] = {}
    for index, value in enumerate(values):
        if not isinstance(value, str):
            continue
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or f"extra_{index}"
        name = f"{stem}_token"
        while name in mapped:
            name = f"{stem}_{index}_token"
        mapped[name] = value
    normalized["extra_special_tokens"] = mapped
    return normalized, True


def importable_files(directory: Path) -> list[Path]:
    allowed_suffixes = {".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken"}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in allowed_suffixes
        and not path.name.startswith(".")
    )


def _parameter_value(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_modelfile(content: str) -> dict:
    """Parse the Modelfile directives supported by Ollama's create API."""
    if len(content.encode("utf-8")) > 64 * 1024:
        raise ValueError("Modelfile 不可超過 64 KiB")
    lines = content.splitlines()
    result: dict = {}
    parameters: dict = {}
    messages: list[dict[str, str]] = []
    index = 0

    def read_argument(initial: str, directive: str) -> str:
        nonlocal index
        initial = initial.strip()
        if not initial.startswith('"""'):
            return initial
        value = initial[3:]
        if '"""' in value:
            return value.split('"""', 1)[0]
        chunks = [value] if value else []
        while index < len(lines):
            line = lines[index]
            index += 1
            if '"""' in line:
                chunks.append(line.split('"""', 1)[0])
                return "\n".join(chunks)
            chunks.append(line)
        raise ValueError(f"{directive} 的三引號內容未結束")

    while index < len(lines):
        raw_line = lines[index]
        index += 1
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z]+)(?:\s+(.*))?$", stripped)
        if not match:
            raise ValueError(f"無法解析 Modelfile 第 {index} 行：{raw_line}")
        directive = match.group(1).upper()
        argument = match.group(2) or ""

        if directive == "FROM":
            if argument.strip() not in ("", "."):
                raise ValueError("匯入 output 時 FROM 必須是 .，來源由上方 Output 模型決定")
        elif directive == "PARAMETER":
            try:
                parts = shlex.split(argument)
            except ValueError as exc:
                raise ValueError(f"PARAMETER 格式錯誤：{exc}") from exc
            if len(parts) < 2:
                raise ValueError("PARAMETER 必須包含名稱和值")
            name, value = parts[0], " ".join(parts[1:])
            parsed = _parameter_value(value)
            if name == "stop":
                parameters.setdefault(name, []).append(parsed)
            else:
                parameters[name] = parsed
        elif directive in ("TEMPLATE", "SYSTEM", "LICENSE"):
            result[directive.lower()] = read_argument(argument, directive)
        elif directive == "MESSAGE":
            role_and_content = argument.split(maxsplit=1)
            if len(role_and_content) != 2 or role_and_content[0] not in ("system", "user", "assistant"):
                raise ValueError("MESSAGE 格式必須是 MESSAGE <system|user|assistant> <內容>")
            messages.append(
                {"role": role_and_content[0], "content": read_argument(role_and_content[1], directive)}
            )
        elif directive in ("ADAPTER", "REQUIRES"):
            raise ValueError(f"目前從完整 output 匯入時不支援 {directive} 指令")
        else:
            raise ValueError(f"不支援的 Modelfile 指令：{directive}")

    if parameters:
        result["parameters"] = parameters
    if messages:
        result["messages"] = messages
    return result


@dataclass
class OllamaImport:
    id: str
    status: str
    output_name: str
    model_name: str
    base_url: str
    quantize: str | None
    created_at: str
    modelfile: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    current_file: str | None = None
    bytes_completed: int = 0
    bytes_total: int = 0
    error: str | None = None
    import_format: str = "auto"
    resolved_format: str = "safetensors"
    keep_intermediate: bool = False
    phase: str = "queued"
    artifact_path: str | None = None


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 86_400) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Ollama API 位址必須是有效的 http 或 https URL")
        self.parsed = parsed
        self.timeout = timeout
        self.prefix = parsed.path.rstrip("/")

    def _connection(self):
        cls = http.client.HTTPSConnection if self.parsed.scheme == "https" else http.client.HTTPConnection
        return cls(self.parsed.hostname, self.parsed.port, timeout=self.timeout)

    def _path(self, path: str) -> str:
        return f"{self.prefix}{path}" or "/"

    def version(self) -> dict:
        connection = self._connection()
        try:
            connection.request("GET", self._path("/api/version"))
            response = connection.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"Ollama API 回應 HTTP {response.status}: {body[:500].decode(errors='replace')}")
            return json.loads(body)
        finally:
            connection.close()

    def blob_exists(self, digest: str) -> bool:
        connection = self._connection()
        try:
            connection.request("HEAD", self._path(f"/api/blobs/{quote(digest, safe=':')}"))
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return True
            if response.status == 404:
                return False
            raise RuntimeError(f"檢查 Ollama blob 失敗：HTTP {response.status}")
        finally:
            connection.close()

    def upload_blob(self, path: Path, digest: str, progress) -> None:
        size = path.stat().st_size
        connection = self._connection()
        try:
            connection.putrequest("POST", self._path(f"/api/blobs/{quote(digest, safe=':')}"))
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            sent = 0
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    connection.send(chunk)
                    sent += len(chunk)
                    progress(sent, size)
            response = connection.getresponse()
            body = response.read()
            if response.status not in (200, 201):
                raise RuntimeError(
                    f"上傳 {path.name} 失敗，HTTP {response.status}: {body[:500].decode(errors='replace')}"
                )
        finally:
            connection.close()

    def create(
        self,
        model_name: str,
        files: dict[str, str],
        quantize: str | None,
        modelfile_options: dict | None = None,
        *,
        from_model: str | None = None,
        adapters: dict[str, str] | None = None,
    ) -> dict:
        payload: dict = {"model": model_name, "files": files, "stream": False}
        if from_model:
            payload["from"] = from_model
        if adapters:
            payload["adapters"] = adapters
        if quantize:
            payload["quantize"] = quantize
        if modelfile_options:
            payload.update(modelfile_options)
        body = json.dumps(payload).encode()
        connection = self._connection()
        try:
            connection.request(
                "POST",
                self._path("/api/create"),
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
            response = connection.getresponse()
            content = response.read()
            if response.status != 200:
                raise RuntimeError(
                    f"建立 Ollama model 失敗，HTTP {response.status}: {content[:1000].decode(errors='replace')}"
                )
            return json.loads(content)
        finally:
            connection.close()


class OllamaImportManager:
    def __init__(self, output_dir: Path, data_dir: Path) -> None:
        self.output_dir = output_dir
        self.data_dir = data_dir / "ollama_imports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.current: OllamaImport | None = None
        self._load_latest()

    def _load_latest(self) -> None:
        metadata = sorted(self.data_dir.glob("*/import.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not metadata:
            return
        try:
            item = OllamaImport(**json.loads(metadata[0].read_text(encoding="utf-8")))
            if item.status in ("queued", "running"):
                item.status = "failed"
                item.phase = "failed"
                item.finished_at = utc_now()
                item.error = "Web 服務重啟，匯入程序已中斷，請重新執行。"
                self.current = item
                self._persist(item)
            else:
                self.current = item
        except (OSError, ValueError, TypeError):
            pass

    def _directory(self, import_id: str) -> Path:
        return self.data_dir / import_id

    def _persist(self, item: OllamaImport) -> None:
        directory = self._directory(item.id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "import.json.tmp"
        temporary.write_text(json.dumps(asdict(item), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "import.json")

    def _log(self, item: OllamaImport, message: str) -> None:
        with self.lock:
            with (self._directory(item.id) / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(message.rstrip() + "\n")
            self._persist(item)

    def list_outputs(self) -> list[dict]:
        results = []
        for directory in sorted(self.output_dir.iterdir()) if self.output_dir.exists() else []:
            if complete_safetensors_directory(directory):
                files = importable_files(directory)
                results.append(
                    {
                        "name": directory.name,
                        "size": sum(path.stat().st_size for path in files),
                        "file_count": len(files),
                        "architectures": model_architectures(directory),
                        "recommended_format": resolve_import_format(
                            "auto", model_architectures(directory)
                        ),
                    }
                )
        return results

    def gguf_tools_available(self) -> bool:
        converter, quantizer = llama_cpp_tools()
        if not converter.is_file() or not quantizer.is_file():
            return False
        try:
            result = subprocess.run(
                [str(quantizer), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def delete_output(self, output_name: str) -> dict:
        if not re.fullmatch(r"[a-zA-Z0-9._-]{1,160}", output_name):
            raise ValueError("無效的 output 名稱")
        with self.lock:
            if (
                self.current
                and self.current.status in ("queued", "running")
                and self.current.output_name == output_name
            ):
                raise RuntimeError("此模型正在匯入 Ollama，無法刪除")
            source = (self.output_dir / output_name).resolve()
            if (
                source.parent != self.output_dir.resolve()
                or source.name == GGUF_OUTPUT_DIR_NAME
                or not complete_safetensors_directory(source)
            ):
                raise ValueError("找不到可刪除的完整 output 模型")
            gguf = (self.output_dir / GGUF_OUTPUT_DIR_NAME / output_name).resolve()
            gguf_root = (self.output_dir / GGUF_OUTPUT_DIR_NAME).resolve()

            def size(directory: Path) -> int:
                return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

            deleted_bytes = size(source)
            if gguf.parent == gguf_root and gguf.is_dir():
                deleted_bytes += size(gguf)
                shutil.rmtree(gguf)
            shutil.rmtree(source)
            return {"output_name": output_name, "deleted_bytes": deleted_bytes}

    def start(
        self,
        output_name: str,
        model_name: str,
        base_url: str,
        quantize: str | None,
        modelfile: str,
        import_format: str = "auto",
        keep_intermediate: bool = False,
    ) -> OllamaImport:
        with self.lock:
            if self.current and self.current.status in ("queued", "running"):
                raise RuntimeError("已有 Ollama 匯入任務正在執行")
            directory = (self.output_dir / output_name).resolve()
            if directory.parent != self.output_dir.resolve() or not complete_safetensors_directory(directory):
                raise ValueError("找不到完整的輸出模型")
            files = importable_files(directory)
            parse_modelfile(modelfile)
            resolved_format = resolve_import_format(import_format, model_architectures(directory))
            item = OllamaImport(
                id=uuid.uuid4().hex[:12],
                status="queued",
                output_name=output_name,
                model_name=model_name,
                base_url=base_url.rstrip("/"),
                quantize=quantize,
                created_at=utc_now(),
                modelfile=modelfile,
                bytes_total=(
                    sum(path.stat().st_size for path in files)
                    if resolved_format == "safetensors"
                    else 0
                ),
                import_format=import_format,
                resolved_format=resolved_format,
                keep_intermediate=keep_intermediate,
            )
            self.current = item
            self._persist(item)
            threading.Thread(target=self._run, args=(item.id,), daemon=True).start()
            return item

    def _set_phase(self, item: OllamaImport, phase: str) -> None:
        with self.lock:
            item.phase = phase
            self._persist(item)

    def _run_process(self, item: OllamaImport, command: list[str]) -> None:
        self._log(item, f"執行：{shlex.join(command)}")
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"無法啟動 GGUF 工具：{exc}") from exc
        assert process.stdout is not None
        for line in process.stdout:
            self._log(item, line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"GGUF 工具執行失敗（exit code {return_code}）")

    @contextmanager
    def _conversion_source(self, item: OllamaImport, source: Path, artifact_dir: Path):
        config_path = source / "tokenizer_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            normalized, changed = normalize_extra_special_tokens(config)
        except (OSError, ValueError, TypeError):
            changed = False
            normalized = {}
        if not changed:
            yield source
            return

        staging = artifact_dir / f".source-{item.id}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            for path in source.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative.as_posix() == "tokenizer_config.json":
                    target.write_text(
                        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                else:
                    target.symlink_to(path.resolve())
            self._log(
                item,
                "已建立相容的 tokenizer metadata 暫存視圖（原始 output 不會被修改）",
            )
            yield staging
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _ensure_gguf(self, item: OllamaImport, source: Path) -> Path:
        bf16_path, final_path = gguf_artifact_paths(self.output_dir, item.output_name, item.quantize)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.is_file() and final_path.stat().st_size > 0:
            self._log(item, f"沿用既有 GGUF：{final_path.name}")
            with self.lock:
                item.artifact_path = str(final_path)
                item.bytes_total = final_path.stat().st_size
                item.bytes_completed = 0
                self._persist(item)
            return final_path
        converter, quantizer = llama_cpp_tools()
        if not converter.is_file():
            raise RuntimeError(
                f"找不到 llama.cpp 轉換器：{converter}。請重建 WebUI image 或設定 LLAMA_CPP_DIR。"
            )
        if item.quantize and not quantizer.is_file():
            raise RuntimeError(
                f"找不到 llama-quantize：{quantizer}。請重建 WebUI image 或設定 LLAMA_CPP_DIR。"
            )
        source_size = sum(path.stat().st_size for path in importable_files(source))
        missing_bf16 = not bf16_path.is_file() or bf16_path.stat().st_size == 0
        missing_final = not final_path.is_file() or final_path.stat().st_size == 0
        required = 0
        if missing_bf16:
            required += source_size
        if item.quantize and missing_final:
            required += int(source_size * QUANTIZED_SIZE_RATIOS[item.quantize])
        free = shutil.disk_usage(final_path.parent).free
        if required and free < int(required * 1.05):
            raise RuntimeError(
                f"GGUF 轉換空間不足：預估至少需要 {format_bytes(int(required * 1.05))}，"
                f"目前只有 {format_bytes(free)} 可用。"
            )

        if missing_bf16:
            self._set_phase(item, "converting_bf16")
            self._log(item, f"轉換 Safetensors → BF16 GGUF：{bf16_path.name}")
            temporary = bf16_path.with_suffix(bf16_path.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            extra_args = conversion_extra_args(source)
            if extra_args:
                self._log(item, f"此架構需要額外轉換參數：{' '.join(extra_args)}")
            with self._conversion_source(item, source, final_path.parent) as conversion_source:
                self._run_process(
                    item,
                    [
                        sys.executable,
                        str(converter),
                        str(conversion_source),
                        "--outfile",
                        str(temporary),
                        "--outtype",
                        "bf16",
                        *extra_args,
                    ],
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("llama.cpp 未產生 BF16 GGUF 輸出")
            temporary.replace(bf16_path)
        else:
            self._log(item, f"沿用既有 BF16 GGUF：{bf16_path.name}")

        if item.quantize and missing_final:
            self._set_phase(item, "quantizing")
            self._log(item, f"量化 BF16 GGUF → {item.quantize}：{final_path.name}")
            temporary = final_path.with_suffix(final_path.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            self._run_process(
                item,
                [str(quantizer), str(bf16_path), str(temporary), item.quantize],
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("llama-quantize 未產生量化 GGUF 輸出")
            temporary.replace(final_path)
        elif item.quantize:
            self._log(item, f"沿用既有量化 GGUF：{final_path.name}")

        if item.quantize and not item.keep_intermediate and bf16_path != final_path:
            bf16_path.unlink(missing_ok=True)
            self._log(item, "已移除 BF16 中間檔（可在下次匯入時選擇保留）")

        with self.lock:
            item.artifact_path = str(final_path)
            item.bytes_total = final_path.stat().st_size
            item.bytes_completed = 0
            self._persist(item)
        return final_path

    def _upload_files(
        self, item: OllamaImport, client: OllamaClient, files: list[Path], base: Path | None = None
    ) -> dict[str, str]:
        self._set_phase(item, "uploading")
        digests: dict[str, str] = {}
        completed_before_file = 0
        for path in files:
            relative_name = path.relative_to(base).as_posix() if base else path.name
            with self.lock:
                item.current_file = relative_name
                self._persist(item)
            self._log(item, f"計算 SHA-256：{relative_name} ({format_bytes(path.stat().st_size)})")
            digest_hash = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest_hash.update(chunk)
            digest = f"sha256:{digest_hash.hexdigest()}"
            digests[relative_name] = digest
            if client.blob_exists(digest):
                self._log(item, f"Ollama 已有 blob，略過上傳：{relative_name}")
                with self.lock:
                    item.bytes_completed = completed_before_file + path.stat().st_size
                    self._persist(item)
            else:
                last_percent = -1

                def progress(sent: int, total: int) -> None:
                    nonlocal last_percent
                    percent = int(sent * 100 / total) if total else 100
                    with self.lock:
                        item.bytes_completed = completed_before_file + sent
                        self._persist(item)
                    if percent // 10 != last_percent // 10:
                        last_percent = percent
                        self._log(item, f"上傳 {relative_name}：{percent}%")

                client.upload_blob(path, digest, progress)
            completed_before_file += path.stat().st_size
        return digests

    def get(self) -> dict | None:
        with self.lock:
            if self.current is None:
                return None
            result = asdict(self.current)
            log_path = self._directory(self.current.id) / "run.log"
            result["log"] = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            return result

    def _run(self, import_id: str) -> None:
        with self.lock:
            item = self.current
            if item is None or item.id != import_id:
                return
            item.status = "running"
            item.started_at = utc_now()
            self._persist(item)

        directory = self.output_dir / item.output_name
        try:
            client = OllamaClient(item.base_url)
            version = client.version().get("version", "未知")
            self._log(item, f"已連線至 Ollama {version}：{item.base_url}")
            self._log(item, f"匯入路徑：{item.resolved_format}")
            if item.resolved_format == "safetensors":
                compatibility_error = ollama_compatibility_error(
                    version, model_architectures(directory)
                )
                if compatibility_error:
                    raise RuntimeError(compatibility_error)
                files = importable_files(directory)
                digests = self._upload_files(item, client, files, directory)
                create_quantize = item.quantize
            else:
                gguf_path = self._ensure_gguf(item, directory)
                digests = self._upload_files(item, client, [gguf_path])
                create_quantize = None

            with self.lock:
                item.current_file = None
                item.bytes_completed = item.bytes_total
                self._persist(item)
            quantize_label = item.quantize or "不量化"
            self._set_phase(item, "creating")
            self._log(item, f"建立 Ollama model：{item.model_name}（{quantize_label}）")
            response = client.create(
                item.model_name,
                digests,
                create_quantize,
                parse_modelfile(item.modelfile),
            )
            if response.get("status") != "success":
                raise RuntimeError(f"Ollama 未回報成功：{response}")
            with self.lock:
                item.status = "completed"
                item.phase = "completed"
                item.finished_at = utc_now()
                self._log(item, f"匯入完成：{item.model_name}")
        except Exception as exc:
            with self.lock:
                item.status = "failed"
                item.phase = "failed"
                item.error = str(exc)
                item.finished_at = utc_now()
                self._log(item, f"錯誤：{exc}")
