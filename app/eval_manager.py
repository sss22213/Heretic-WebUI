# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .ollama_import import OllamaClient, complete_safetensors_directory

TASK_NAME_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")
OLLAMA_MODEL_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*(?::[a-zA-Z0-9._-]+)?")

# Curated starting points shown in the UI. Any other lm-eval task name can be
# entered manually; these lists are not whitelists.
PRESET_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "mmlu",
    "gsm8k",
    "truthfulqa_mc2",
]

# The Ollama backend talks to the OpenAI-compatible chat API, which cannot
# score loglikelihood/multiple-choice tasks; only generative tasks apply.
PRESET_TASKS_OLLAMA = [
    "gsm8k",
    "gsm8k_cot",
    "minerva_math",
    "bbh_cot_fewshot",
    "mgsm_native_cot_zh",
    "drop",
    "triviaqa",
    "nq_open",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def lm_eval_available() -> bool:
    try:
        return importlib.util.find_spec("lm_eval") is not None
    except (ImportError, ValueError):
        return False


def parse_task_list(value: str) -> list[str]:
    """Split a comma/whitespace separated task string into validated names."""
    tasks: list[str] = []
    for chunk in value.replace(",", " ").split():
        if not TASK_NAME_PATTERN.fullmatch(chunk):
            raise ValueError(f"無效的評測任務名稱：{chunk}")
        if chunk not in tasks:
            tasks.append(chunk)
    if not tasks:
        raise ValueError("請至少選擇或輸入一個評測任務")
    if len(tasks) > 40:
        raise ValueError("一次最多執行 40 個評測任務")
    return tasks


def summarize_results(raw: dict) -> dict[str, dict[str, float]]:
    """Reduce an lm-eval results payload to {task: {metric: value}}.

    lm-eval reports metrics as "acc,none" / "exact_match,strict-match" pairs
    and includes stderr twins; only the primary numeric values are kept.
    """
    summary: dict[str, dict[str, float]] = {}
    results = raw.get("results")
    if not isinstance(results, dict):
        return summary
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        selected: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metric, _, metric_filter = str(key).partition(",")
            if metric.endswith("_stderr") or metric in ("alias", "sample_len"):
                continue
            # Keep non-default filters visible: gsm8k reports exact_match for
            # both strict-match and flexible-extract, which must not collide.
            name = metric if metric_filter in ("", "none") else f"{metric}({metric_filter})"
            selected[name] = round(float(value), 4)
        if selected:
            summary[str(task)] = selected
    return summary


@dataclass
class EvalRun:
    id: str
    status: str
    created_at: str
    model_source: str
    model_path: str
    tasks: list[str]
    num_fewshot: int | None = None
    limit: int | None = None
    batch_size: int = 0
    quantization: str = "none"
    backend: str = "hf"
    base_url: str | None = None
    max_gen_toks: int | None = None
    num_concurrent: int | None = None
    max_retries: int | None = None
    log_samples: bool = False
    use_cache: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    results: dict = field(default_factory=dict)


class EvalManager:
    """Runs LM Evaluation Harness benchmarks against local or HF models."""

    def __init__(self, data_dir: Path, output_dir: Path, models_dir: Path | None = None) -> None:
        self.root = data_dir / "evals"
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.lock = threading.RLock()
        self.runs: dict[str, EvalRun] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self._load()

    def _load(self) -> None:
        for metadata in self.root.glob("*/run.json"):
            try:
                run = EvalRun(**json.loads(metadata.read_text(encoding="utf-8")))
                if run.status in ("queued", "running"):
                    run.status = "failed"
                    run.error = "Web 服務重啟，評測已中斷，請重新執行。"
                    run.finished_at = utc_now()
                    self._persist(run)
                self.runs[run.id] = run
            except (OSError, ValueError, TypeError):
                continue

    def _persist(self, run: EvalRun) -> None:
        directory = self.root / run.id
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "run.json.tmp"
        temporary.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "run.json")

    def _log(self, run: EvalRun, message: str) -> None:
        with self.lock:
            directory = self.root / run.id
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "run.log").open("a", encoding="utf-8") as handle:
                handle.write(message.rstrip() + "\n")

    def list(self) -> list[dict]:
        with self.lock:
            return [
                asdict(run)
                for run in sorted(self.runs.values(), key=lambda run: run.created_at, reverse=True)
            ]

    def active(self) -> EvalRun | None:
        with self.lock:
            for run in self.runs.values():
                if run.status in ("queued", "running"):
                    return run
            return None

    def get_task(self) -> dict | None:
        """Return the active run, or the most recent one, with its log."""
        with self.lock:
            run = self.active()
            if run is None:
                candidates = sorted(self.runs.values(), key=lambda run: run.created_at)
                run = candidates[-1] if candidates else None
            if run is None:
                return None
            result = asdict(run)
            log = self.root / run.id / "run.log"
            result["log"] = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            return result

    def resolve_model(self, source: str) -> str:
        """Map a UI model source onto an lm-eval `pretrained=` value.

        Accepted forms: an outputs/ directory name, an absolute path under the
        models dir, or a Hugging Face repository ID.
        """
        source = source.strip().rstrip("/")
        if not source:
            raise ValueError("請輸入模型來源")
        if source.startswith("/"):
            if self.models_dir is None:
                raise ValueError("此部署不支援路徑形式的模型來源")
            path = Path(source).resolve()
            if not path.is_relative_to(self.models_dir.resolve()):
                raise ValueError(f"路徑形式的模型必須位於 {self.models_dir} 內")
            if not complete_safetensors_directory(path):
                raise ValueError("找不到完整的模型（需含全部 Safetensors 分片）")
            return str(path)
        if "/" in source:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", source):
                raise ValueError("Hugging Face model ID 格式應為 organization/model")
            return source
        path = (self.output_dir / source).resolve()
        if path.parent != self.output_dir.resolve() or not complete_safetensors_directory(path):
            raise ValueError("找不到完整的 output 模型（需含全部 Safetensors 分片）")
        return str(path)

    def command(self, run: EvalRun) -> list[str]:
        if run.backend == "ollama":
            model_args = [
                f"model={run.model_path}",
                f"base_url={run.base_url}/v1/chat/completions",
                f"num_concurrent={run.num_concurrent or 1}",
                f"max_retries={run.max_retries if run.max_retries is not None else 3}",
            ]
            command = [
                sys.executable, "-u", "-m", "lm_eval",
                "--model", "local-chat-completions",
                "--model_args", ",".join(model_args),
                "--tasks", ",".join(run.tasks),
                "--apply_chat_template",
                "--output_path", str(self.root / run.id / "results"),
            ]
        else:
            model_args = [f"pretrained={run.model_path}", "dtype=bfloat16"]
            if run.quantization == "bnb_4bit":
                model_args.append("load_in_4bit=True")
            command = [
                sys.executable, "-u", "-m", "lm_eval",
                "--model", "hf",
                "--model_args", ",".join(model_args),
                "--tasks", ",".join(run.tasks),
                "--batch_size", str(run.batch_size) if run.batch_size else "auto",
                "--output_path", str(self.root / run.id / "results"),
            ]
        if run.num_fewshot is not None:
            command += ["--num_fewshot", str(run.num_fewshot)]
        if run.limit:
            command += ["--limit", str(run.limit)]
        gen_kwargs = []
        if run.max_gen_toks:
            # Thinking models exhaust the per-task default (often 256 tokens)
            # on reasoning content and return an empty answer without this.
            gen_kwargs.append(f"max_gen_toks={run.max_gen_toks}")
        if run.backend == "ollama":
            # Chat-format generation already ends at the model's end-of-turn
            # token. The completion-style stop strings tasks declare (such as
            # "Question:" for gsm8k) are applied by Ollama to the raw token
            # stream, where they also match inside thinking content and cut
            # reasoning models off before any answer; neutralize them.
            gen_kwargs.append("until=<|endoftext|>")
        if gen_kwargs:
            command += ["--gen_kwargs", ",".join(gen_kwargs)]
        if run.log_samples:
            command.append("--log_samples")
        if run.use_cache:
            # Response-level cache shared across runs of the same model, so an
            # interrupted eval resumes instead of re-answering finished items.
            # lm-eval appends "_rank<N>.db" to this prefix; keep it per-model
            # because cache keys hash only the request, not the model.
            cache_root = self.root / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", run.model_source.strip()) or "model"
            command += ["--use_cache", str(cache_root / slug)]
        return command

    @staticmethod
    def validate_ollama_model(name: str) -> str:
        name = name.strip()
        if not OLLAMA_MODEL_PATTERN.fullmatch(name):
            raise ValueError("無效的 Ollama 模型名稱")
        return name

    @staticmethod
    def validate_base_url(base_url: str | None) -> str:
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            raise ValueError("GGUF（Ollama）評測需要 Ollama API 位址")
        if not base_url.startswith(("http://", "https://")) or "," in base_url or " " in base_url:
            raise ValueError("Ollama API 位址必須以 http:// 或 https:// 開頭，且不可包含逗號或空白")
        return base_url

    def start(
        self,
        model_source: str,
        tasks: list[str],
        num_fewshot: int | None,
        limit: int | None,
        batch_size: int,
        quantization: str,
        hf_token: str | None = None,
        backend: str = "hf",
        base_url: str | None = None,
        max_gen_toks: int | None = None,
        num_concurrent: int | None = None,
        max_retries: int | None = None,
        log_samples: bool = False,
        use_cache: bool = False,
    ) -> EvalRun:
        if not lm_eval_available():
            raise RuntimeError("目前映像缺少 lm-eval，請重新建置 WebUI image。")
        with self.lock:
            if self.active() is not None:
                raise RuntimeError("已有評測正在執行")
            if backend == "ollama":
                model_path = self.validate_ollama_model(model_source)
                base_url = self.validate_base_url(base_url)
                quantization = "none"
            else:
                model_path = self.resolve_model(model_source)
                base_url = None
                num_concurrent = None
                max_retries = None
            run = EvalRun(
                id=uuid.uuid4().hex[:12],
                status="queued",
                created_at=utc_now(),
                model_source=model_source.strip(),
                model_path=model_path,
                tasks=tasks,
                num_fewshot=num_fewshot,
                limit=limit,
                batch_size=batch_size,
                quantization=quantization,
                backend=backend,
                base_url=base_url,
                max_gen_toks=max_gen_toks,
                num_concurrent=num_concurrent,
                max_retries=max_retries,
                log_samples=log_samples,
                use_cache=use_cache,
            )
            self.runs[run.id] = run
            self._persist(run)
        threading.Thread(target=self._run, args=(run.id, hf_token), daemon=True).start()
        return run

    def _run(self, run_id: str, hf_token: str | None) -> None:
        with self.lock:
            run = self.runs.get(run_id)
            if run is None or run.status != "queued":
                return
            run.status = "running"
            run.started_at = utc_now()
            self._persist(run)
        try:
            if run.backend == "ollama":
                client = OllamaClient(run.base_url or "")
                version = client.version().get("version", "未知")
                models = {
                    str(item.get("name") or item.get("model"))
                    for item in client.tags().get("models", [])
                    if isinstance(item, dict)
                }
                if run.model_path not in models and f"{run.model_path}:latest" not in models:
                    raise RuntimeError(
                        f"Ollama 中找不到模型：{run.model_path}（可用：{', '.join(sorted(models)) or '無'}）"
                    )
                self._log(run, f"已連線至 Ollama {version}；評測模型：{run.model_path}")
            command = self.command(run)
            env = os.environ.copy()
            env.update({"PYTHONUNBUFFERED": "1", "TERM": "dumb", "NO_COLOR": "1"})
            if hf_token:
                env.update({"HF_TOKEN": hf_token, "HUGGING_FACE_HUB_TOKEN": hf_token})
            self._log(run, f"執行：{shlex.join(command)}")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    start_new_session=True,
                )
            except OSError as exc:
                raise RuntimeError(f"無法啟動 lm-eval 程序：{exc}") from exc
            with self.lock:
                self.processes[run_id] = process
            assert process.stdout is not None
            for line in process.stdout:
                self._log(run, line.rstrip())
            exit_code = process.wait()
            with self.lock:
                if run.status == "cancelled":
                    return
            if exit_code != 0:
                raise RuntimeError(f"lm-eval 程序失敗（exit code {exit_code}）")
            summary = self._collect_results(run)
            with self.lock:
                run.results = summary
                run.status = "completed"
                run.finished_at = utc_now()
                self._log(run, "評測完成。")
                self._persist(run)
        except Exception as exc:
            with self.lock:
                if run.status != "cancelled":
                    run.status = "failed"
                    run.error = str(exc)
                run.finished_at = utc_now()
                self._log(run, f"錯誤：{exc}")
                self._persist(run)
        finally:
            with self.lock:
                self.processes.pop(run_id, None)
                if run.finished_at is None:
                    run.finished_at = utc_now()
                self._persist(run)

    def _collect_results(self, run: EvalRun) -> dict[str, dict[str, float]]:
        results_dir = self.root / run.id / "results"
        candidates = sorted(
            results_dir.rglob("results_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ) if results_dir.exists() else []
        if not candidates:
            raise RuntimeError("lm-eval 未產生結果檔")
        raw = json.loads(candidates[0].read_text(encoding="utf-8"))
        summary = summarize_results(raw)
        if not summary:
            raise RuntimeError("結果檔中沒有可用的評測分數")
        return summary

    def cancel(self) -> EvalRun:
        with self.lock:
            run = self.active()
            if run is None:
                raise RuntimeError("目前沒有執行中的評測")
            run.status = "cancelled"
            run.error = "使用者取消評測"
            run.finished_at = utc_now()
            process = self.processes.get(run.id)
            self._log(run, "使用者取消評測。")
            self._persist(run)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return run

    def delete(self, run_id: str) -> dict:
        with self.lock:
            run = self.runs.get(run_id)
            if run is None:
                raise ValueError("找不到評測紀錄")
            if run.status in ("queued", "running"):
                raise RuntimeError("評測執行中，無法刪除")
            directory = (self.root / run_id).resolve()
            if directory.parent == self.root.resolve() and directory.is_dir():
                shutil.rmtree(directory, ignore_errors=True)
            del self.runs[run_id]
            return {"id": run_id}
