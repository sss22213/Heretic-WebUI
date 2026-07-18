from pathlib import Path
import json
import hashlib
import subprocess
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from app.main import (
    HFTokenStore,
    Job,
    JobManager,
    JobRequest,
    OllamaImportRequest,
    SettingsStore,
    UISettingsRequest,
    job_environment,
    output_artifacts_complete,
    render_config,
    safe_slug,
)
from app.ollama_import import (
    OllamaClient,
    OllamaImport,
    OllamaImportManager,
    complete_safetensors_directory,
    conversion_extra_args,
    gguf_artifact_paths,
    importable_files,
    llama_cpp_tools,
    model_architectures,
    normalize_extra_special_tokens,
    ollama_compatibility_error,
    parse_modelfile,
    resolve_import_format,
)
from app.lora_manager import LoRAManager, LoRATask, adapter_files, inspect_adapter
from app.heretic_version import HereticVersionManager


def test_safe_slug_removes_path_and_shell_characters():
    assert safe_slug("../Qwen model; rm -rf") == "Qwen-model-rm--rf"


def test_job_request_rejects_invalid_trial_counts():
    with pytest.raises(ValidationError):
        JobRequest(model="org/model", n_trials=10, n_startup_trials=11)


def test_render_config_is_non_interactive_and_escapes_strings(tmp_path: Path):
    request = JobRequest(
        model='org/model"quoted',
        n_trials=12,
        n_startup_trials=4,
        system_prompt='line one\n"line two"',
    )
    config = render_config(request, tmp_path / "output", "abc123")

    assert 'model = "org/model\\"quoted"' in config
    assert 'model_action = "save"' in config
    assert 'export_strategy = "merge"' in config
    assert "trial_index = 0" in config
    assert 'system_prompt = "line one\\n\\"line two\\""' in config

    parsed = tomllib.loads(config)
    assert parsed["n_trials"] == 12
    assert parsed["good_prompts"]["dataset"] == "mlabonne/harmless_alpaca"


def test_hf_token_is_normalized_and_never_serialized():
    request = JobRequest(model="org/private-model", hf_token="  hf_secret  ")

    assert request.hf_token == "hf_secret"
    assert "hf_token" not in request.model_dump()
    assert "hf_secret" not in render_config(request, Path("/tmp/output"), "abc123")


def test_blank_hf_token_is_treated_as_missing():
    assert JobRequest(model="org/model", hf_token="   ").hf_token is None


def test_job_environment_uses_supplied_hf_token(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HF_TOKEN", "deployment-token")

    env = job_environment("task-token")

    assert env["HF_TOKEN"] == "task-token"
    assert env["HUGGING_FACE_HUB_TOKEN"] == "task-token"
    assert job_environment(None)["HF_TOKEN"] == "deployment-token"
    slot_env = job_environment(None, tmp_path / "slot-A")
    assert slot_env["PYTHONPATH"].split(":", 1)[0] == str(tmp_path / "slot-A" / "src")


def test_hf_token_store_persists_with_private_permissions(tmp_path: Path):
    path = tmp_path / "hf_token"
    store = HFTokenStore(path)

    assert store.get() is None
    store.save("hf_first")
    assert HFTokenStore(path).get() == "hf_first"
    assert path.stat().st_mode & 0o777 == 0o600

    store.save("hf_replacement")
    assert HFTokenStore(path).get() == "hf_replacement"


def test_ui_language_settings_persist_and_reject_invalid_values(tmp_path: Path):
    path = tmp_path / "settings.json"
    store = SettingsStore(path)

    assert store.get() == {"language": "zh-TW"}
    assert store.save(UISettingsRequest(language="ja")) == {"language": "ja"}
    assert SettingsStore(path).get() == {"language": "ja"}
    path.write_text('{"language":"invalid"}')
    assert SettingsStore(path).get() == {"language": "zh-TW"}


@pytest.mark.parametrize("quantize", ["q2_K", "q3_K_M", "q4_K_M", "q6_K", "q8_0"])
def test_ollama_import_accepts_supported_quantization_levels(quantize: str):
    request = OllamaImportRequest(
        output_name="example",
        model_name="example:latest",
        base_url="http://ollama:11434",
        quantize=quantize,
    )

    assert request.quantize == quantize


def test_ollama_import_rejects_unsupported_quantization_level():
    with pytest.raises(ValidationError):
        OllamaImportRequest(
            output_name="example",
            model_name="example:latest",
            base_url="http://ollama:11434",
            quantize="q5_K_M",
        )


def test_output_artifacts_require_every_indexed_shard(tmp_path: Path):
    job = Job(
        id="abc123",
        status="completed",
        request={"export_strategy": "merge"},
        output_directory=str(tmp_path),
        created_at="2026-01-01T00:00:00+00:00",
    )
    index = {
        "weight_map": {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-00001-of-00002.safetensors").touch()

    assert not output_artifacts_complete(job)

    (tmp_path / "model-00002-of-00002.safetensors").touch()
    assert output_artifacts_complete(job)


def test_output_artifacts_reject_config_only_directory(tmp_path: Path):
    job = Job(
        id="abc123",
        status="completed",
        request={"export_strategy": "merge"},
        output_directory=str(tmp_path),
        created_at="2026-01-01T00:00:00+00:00",
    )
    (tmp_path / "config.json").write_text("{}")

    assert not output_artifacts_complete(job)


def test_deleted_output_does_not_turn_completed_job_into_failure_after_restart(
    tmp_path: Path, monkeypatch
):
    jobs_dir = tmp_path / "jobs"
    outputs_dir = tmp_path / "outputs"
    output = outputs_dir / "finished-model"
    output.mkdir(parents=True)
    (output / "model.safetensors").write_bytes(b"weights")
    job = Job(
        id="completed-job",
        status="completed",
        request={"export_strategy": "merge"},
        output_directory=str(output),
        created_at="2026-01-01T00:00:00+00:00",
    )
    metadata = jobs_dir / job.id / "job.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps(job.__dict__))
    monkeypatch.setattr("app.main.JOBS_DIR", jobs_dir)
    monkeypatch.setattr("app.main.OUTPUT_DIR", outputs_dir)

    manager = JobManager()
    manager.mark_output_deleted("finished-model")
    (output / "model.safetensors").unlink()
    output.rmdir()
    reloaded = JobManager().get(job.id)

    assert reloaded.status == "completed"
    assert reloaded.output_deleted is True


def test_complete_safetensors_directory_checks_shards(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}})
    )
    assert not complete_safetensors_directory(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").touch()
    assert complete_safetensors_directory(tmp_path)


def test_ollama_client_uploads_blob_and_creates_model(tmp_path: Path):
    uploaded = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            body = json.dumps({"version": "test"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            uploaded[self.path] = body
            response = json.dumps({"status": "success"}).encode()
            self.send_response(201 if self.path.startswith("/api/blobs/") else 200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model_file = tmp_path / "model.safetensors"
        model_file.write_bytes(b"model-weights")
        digest = f"sha256:{hashlib.sha256(model_file.read_bytes()).hexdigest()}"
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}")

        assert client.version() == {"version": "test"}
        assert not client.blob_exists(digest)
        client.upload_blob(model_file, digest, lambda _sent, _total: None)
        result = client.create(
            "example:latest",
            {model_file.name: digest},
            "q4_K_M",
            {"system": "Be concise.", "parameters": {"num_ctx": 8192}},
        )

        assert uploaded[f"/api/blobs/{digest}"] == b"model-weights"
        create_request = json.loads(uploaded["/api/create"])
        assert create_request["files"] == {"model.safetensors": digest}
        assert create_request["quantize"] == "q4_K_M"
        assert create_request["system"] == "Be concise."
        assert create_request["parameters"] == {"num_ctx": 8192}
        assert result["status"] == "success"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ollama_client_create_supports_lora_adapters(tmp_path: Path):
    requests = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            requests[self.path] = json.loads(body)
            response = json.dumps({"status": "success"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}")
        client.create(
            "custom:latest", {}, None,
            from_model="llama3.2:latest",
            adapters={"adapter_model.safetensors": "sha256:abc"},
        )

        assert requests["/api/create"]["from"] == "llama3.2:latest"
        assert requests["/api/create"]["adapters"] == {
            "adapter_model.safetensors": "sha256:abc"
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_inspect_adapter_reads_safetensors_metadata(tmp_path: Path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "meta-llama/Llama-3.2-3B"})
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")

    details = inspect_adapter(tmp_path)

    assert details["format"] == "safetensors"
    assert details["base_model"] == "meta-llama/Llama-3.2-3B"
    assert details["files"] == ["adapter_config.json", "adapter_model.safetensors"]
    assert [path.as_posix() for path in adapter_files(tmp_path)] == details["files"]


def test_lora_manager_lists_and_deletes_adapter(tmp_path: Path):
    manager = LoRAManager(tmp_path / "data")
    directory = manager.root / "example"
    directory.mkdir()
    (directory / "adapter.gguf").write_bytes(b"gguf")
    (directory / "lora.json").write_text(
        json.dumps({"repo_id": "org/example", "revision": "main"})
    )

    assert manager.list()[0]["name"] == "example"
    assert manager.list()[0]["format"] == "gguf"
    assert manager.delete("example")["deleted_bytes"] > 0
    assert not directory.exists()


def test_lora_manager_blocks_active_delete_and_path_traversal(tmp_path: Path):
    manager = LoRAManager(tmp_path / "data")
    directory = manager.root / "example"
    directory.mkdir()
    (directory / "adapter.gguf").write_bytes(b"gguf")
    manager.current = LoRATask(
        id="active", operation="import", status="running",
        created_at="2026-01-01T00:00:00+00:00", lora_name="example",
    )

    with pytest.raises(RuntimeError, match="使用中"):
        manager.delete("example")
    with pytest.raises(ValueError, match="名稱"):
        manager.delete("../example")


def test_lora_download_is_published_only_after_validation(tmp_path: Path, monkeypatch):
    manager = LoRAManager(tmp_path / "data")
    task = manager._new_task(
        "download", "example", repo_id="org/example", revision="main"
    )

    def fake_snapshot_download(**kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True)
        (destination / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "llama3.2:latest"})
        )
        (destination / "adapter_model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr("app.lora_manager.snapshot_download", fake_snapshot_download)
    manager._download(task.id, "hf_private", None)

    assert task.status == "completed"
    assert not list(manager.root.glob("*.partial"))
    assert manager.list()[0]["repo_id"] == "org/example"
    assert manager.list()[0]["base_model"] == "llama3.2:latest"
    assert "hf_private" not in json.dumps(manager.get_task())


def test_lora_import_uploads_adapter_and_uses_base_model(tmp_path: Path, monkeypatch):
    manager = LoRAManager(tmp_path / "data")
    directory = manager.root / "example"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}")
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    calls = {"uploaded": []}

    class FakeClient:
        def __init__(self, base_url):
            calls["base_url"] = base_url

        def version(self):
            return {"version": "test"}

        def blob_exists(self, _digest):
            return False

        def upload_blob(self, path, digest, progress):
            calls["uploaded"].append((path.name, digest))
            progress(path.stat().st_size, path.stat().st_size)

        def create(self, model_name, files, quantize, **kwargs):
            calls.update(model_name=model_name, files=files, quantize=quantize, **kwargs)
            return {"status": "success"}

    monkeypatch.setattr("app.lora_manager.OllamaClient", FakeClient)
    task = manager._new_task(
        "import", "example", model_name="custom:latest",
        base_model="llama3.2:latest", base_url="http://ollama:11434",
        bytes_total=7,
    )
    manager._import(task.id)

    assert task.status == "completed"
    assert calls["from_model"] == "llama3.2:latest"
    assert set(calls["adapters"]) == {
        "adapter_config.json", "adapter_model.safetensors"
    }
    assert {name for name, _digest in calls["uploaded"]} == {
        "adapter_config.json", "adapter_model.safetensors"
    }


def _git(directory: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def test_heretic_version_update_and_rollback(tmp_path: Path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    seed.mkdir()
    _git(seed, "init", "-b", "master")
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    (seed / "pyproject.toml").write_text("version = '1'\n")
    (seed / "uv.lock").write_text("lock-1\n")
    (seed / "source.py").write_text("value = 1\n")
    (seed / "src" / "heretic").mkdir(parents=True)
    (seed / "src" / "heretic" / "__init__.py").write_text("")
    (seed / "src" / "heretic" / "main.py").write_text("")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "version one")
    first = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "master")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (checkout / "source.py").write_text("value = 99\n")
    (patch_dir / "0001-local.patch").write_text(_git(checkout, "diff", "--binary") + "\n")

    (seed / "pyproject.toml").write_text("version = '2'\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "version two")
    second = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "master")

    manager = HereticVersionManager(checkout, tmp_path / "state.json", patch_dir)
    checked = manager.status(check_remote=True)
    assert checked["commit"] == first
    assert checked["latest_commit"] == second
    assert checked["update_available"] is True
    assert checked["dirty"] is False
    assert checked["managed_patches_applied"] is True
    assert checked["active_slot"] == "A"

    updated = manager.update()
    assert updated["commit"] == second
    assert updated["rollback_available"] is True
    assert updated["rebuild_required"] is True
    assert updated["active_slot"] == "B"
    assert (Path(manager.runtime_info()["path"]) / "source.py").read_text() == "value = 99\n"
    assert updated["managed_patches"][0]["status"] == "applied"

    rolled_back = manager.rollback()
    assert rolled_back["commit"] == first
    assert rolled_back["rollback_available"] is False
    assert rolled_back["active_slot"] == "A"
    assert (Path(manager.runtime_info()["path"]) / "source.py").read_text() == "value = 99\n"


def test_heretic_slots_ignore_uncommitted_source_changes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "master")
    _git(source, "config", "user.email", "test@example.com")
    _git(source, "config", "user.name", "Test")
    (source / "pyproject.toml").write_text("version = '1'\n")
    (source / "uv.lock").write_text("lock\n")
    (source / "src" / "heretic").mkdir(parents=True)
    (source / "src" / "heretic" / "__init__.py").write_text("")
    (source / "src" / "heretic" / "main.py").write_text("")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "initial")
    (source / "pyproject.toml").write_text("local patch\n")
    manager = HereticVersionManager(source, tmp_path / "state.json")

    status = manager.status()
    assert status["dirty"] is False
    assert status["active_slot"] == "A"
    assert (Path(manager.runtime_info()["path"]) / "pyproject.toml").read_text() == "version = '1'\n"


def test_heretic_update_rolls_back_when_managed_patch_is_incompatible(tmp_path: Path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    seed.mkdir()
    _git(seed, "init", "-b", "master")
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    (seed / "pyproject.toml").write_text("version = '1'\n")
    (seed / "uv.lock").write_text("lock\n")
    (seed / "source.py").write_text("value = 1\n")
    (seed / "src" / "heretic").mkdir(parents=True)
    (seed / "src" / "heretic" / "__init__.py").write_text("")
    (seed / "src" / "heretic" / "main.py").write_text("")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "version one")
    first = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "master")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    (checkout / "source.py").write_text("value = 99\n")
    (patch_dir / "0001-local.patch").write_text(_git(checkout, "diff") + "\n")

    (seed / "source.py").write_text("upstream rewrite\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "incompatible rewrite")
    _git(seed, "push", "origin", "master")
    manager = HereticVersionManager(checkout, tmp_path / "state.json", patch_dir)

    with pytest.raises(RuntimeError, match="active slot A 保持不變"):
        manager.update()

    assert manager.status()["commit"] == first
    assert manager.status()["active_slot"] == "A"
    assert (Path(manager.runtime_info()["path"]) / "source.py").read_text() == "value = 99\n"
    assert manager.status()["managed_patches_applied"] is True


def test_heretic_runtime_git_cache_is_cloned_without_submodule(tmp_path: Path):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    cache = tmp_path / "data" / "heretic_upstream"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    seed.mkdir()
    _git(seed, "init", "-b", "master")
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test")
    (seed / "pyproject.toml").write_text("version = '1'\n")
    (seed / "uv.lock").write_text("lock\n")
    (seed / "src" / "heretic").mkdir(parents=True)
    (seed / "src" / "heretic" / "__init__.py").write_text("")
    (seed / "src" / "heretic" / "main.py").write_text("")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "initial")
    commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "master")

    manager = HereticVersionManager(
        cache,
        tmp_path / "data" / "state.json",
        slots_dir=tmp_path / "data" / "slots",
        upstream_url=str(remote),
        initial_ref=commit,
    )
    status = manager.status()

    assert status["commit"] == commit
    assert status["active_slot"] == "A"
    assert (cache / ".git").is_dir()


def test_parse_modelfile_converts_supported_directives():
    result = parse_modelfile(
        '''FROM .
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
PARAMETER stop "<end>"
PARAMETER stop "USER:"
SYSTEM """You are
helpful."""
TEMPLATE """{{ .Prompt }}
{{ .Response }}"""
MESSAGE user Hello
MESSAGE assistant Hi
'''
    )

    assert result["parameters"] == {
        "temperature": 0.7,
        "num_ctx": 8192,
        "stop": ["<end>", "USER:"],
    }
    assert result["system"] == "You are\nhelpful."
    assert result["template"] == "{{ .Prompt }}\n{{ .Response }}"
    assert result["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_parse_modelfile_supports_renderer_and_parser():
    result = parse_modelfile("FROM .\nRENDERER qwen3.5\nPARSER qwen3.5")

    assert result["renderer"] == "qwen3.5"
    assert result["parser"] == "qwen3.5"
    with pytest.raises(ValueError, match="RENDERER"):
        parse_modelfile("FROM .\nRENDERER")


def test_parse_modelfile_rejects_conflicting_source_and_adapter():
    with pytest.raises(ValueError, match="FROM"):
        parse_modelfile("FROM llama3.2")
    with pytest.raises(ValueError, match="ADAPTER"):
        parse_modelfile("FROM .\nADAPTER ./adapter")


def test_gemma4_unified_is_rejected_before_upload_for_ollama_0306(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"]})
    )
    architectures = model_architectures(tmp_path)

    assert architectures == ["Gemma4UnifiedForConditionalGeneration"]
    error = ollama_compatibility_error("0.30.6", architectures)
    assert error is not None
    assert "不支援" in error
    assert ollama_compatibility_error("future-version", architectures) is None


def test_qwen35_safetensors_is_rejected_for_ollama_0312():
    architectures = ["Qwen3_5ForConditionalGeneration"]

    error = ollama_compatibility_error("0.31.2", architectures)
    assert error is not None
    assert "不支援" in error
    assert ollama_compatibility_error("0.31.2", ["Qwen3ForCausalLM"]) is None
    assert ollama_compatibility_error("future-version", architectures) is None


def test_conversion_drops_declared_mtp_head(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1}})
    )
    assert conversion_extra_args(tmp_path) == ["--no-mtp"]

    (tmp_path / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1}))
    assert conversion_extra_args(tmp_path) == ["--no-mtp"]

    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 0}})
    )
    assert conversion_extra_args(tmp_path) == []

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    assert conversion_extra_args(tmp_path) == []

    (tmp_path / "config.json").unlink()
    assert conversion_extra_args(tmp_path) == []


def test_auto_import_uses_gguf_for_broken_safetensors_architectures():
    gemma4 = ["Gemma4UnifiedForConditionalGeneration"]
    qwen35 = ["Qwen3_5ForConditionalGeneration"]

    assert resolve_import_format("auto", gemma4) == "gguf"
    assert resolve_import_format("auto", qwen35) == "gguf"
    assert resolve_import_format("auto", ["Qwen3_5MoeForConditionalGeneration"]) == "gguf"
    assert resolve_import_format("auto", ["Qwen3ForCausalLM"]) == "safetensors"
    assert resolve_import_format("safetensors", gemma4) == "safetensors"
    assert resolve_import_format("gguf", []) == "gguf"
    with pytest.raises(ValueError, match="匯入格式"):
        resolve_import_format("invalid", gemma4)


def test_extra_special_token_list_is_normalized_without_mutating_input():
    source = {"extra_special_tokens": ["<|video|>"], "bos_token": "<bos>"}

    normalized, changed = normalize_extra_special_tokens(source)

    assert changed
    assert source["extra_special_tokens"] == ["<|video|>"]
    assert normalized["extra_special_tokens"] == {"video_token": "<|video|>"}
    assert normalize_extra_special_tokens(normalized) == (normalized, False)


def test_gguf_artifacts_are_isolated_from_safetensors_uploads(tmp_path: Path):
    source = tmp_path / "example"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    bf16, quantized = gguf_artifact_paths(tmp_path, "example", "q4_K_M")
    quantized.parent.mkdir(parents=True)
    quantized.write_bytes(b"gguf")

    assert bf16 == tmp_path / ".gguf" / "example" / "example-BF16.gguf"
    assert quantized.name == "example-Q4_K_M.gguf"
    assert [path.name for path in importable_files(source)] == ["config.json", "model.safetensors"]


def test_llama_cpp_tools_find_built_quantizer(tmp_path: Path):
    converter = tmp_path / "convert_hf_to_gguf.py"
    quantizer = tmp_path / "build" / "bin" / "llama-quantize"
    converter.touch()
    quantizer.parent.mkdir(parents=True)
    quantizer.touch()

    assert llama_cpp_tools(tmp_path) == (converter, quantizer)


def test_gguf_health_check_executes_quantizer(tmp_path: Path, monkeypatch):
    tools = tmp_path / "llama.cpp"
    converter = tools / "convert_hf_to_gguf.py"
    quantizer = tools / "build" / "bin" / "llama-quantize"
    converter.parent.mkdir(parents=True)
    converter.touch()
    quantizer.parent.mkdir(parents=True)
    quantizer.write_text("#!/bin/sh\nexit 0\n")
    quantizer.chmod(0o755)
    monkeypatch.setattr("app.ollama_import.LLAMA_CPP_DIR", tools)

    manager = OllamaImportManager(tmp_path / "outputs", tmp_path / "data")

    assert manager.gguf_tools_available()
    quantizer.write_text("#!/bin/sh\nexit 1\n")
    assert manager.gguf_tools_available()
    quantizer.write_text("#!/bin/sh\nexit 127\n")
    assert not manager.gguf_tools_available()


def test_gguf_import_does_not_request_second_ollama_quantization(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "outputs"
    source = output_root / "gemma"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps({"architectures": ["Gemma4UnifiedForConditionalGeneration"]})
    )
    (source / "model.safetensors").write_bytes(b"safetensors")
    gguf = output_root / ".gguf" / "gemma" / "gemma-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"quantized-gguf")
    calls = {}

    class FakeClient:
        def __init__(self, base_url):
            calls["base_url"] = base_url

        def version(self):
            return {"version": "0.30.6"}

        def blob_exists(self, _digest):
            return True

        def create(self, model_name, files, quantize, options):
            calls.update(model_name=model_name, files=files, quantize=quantize, options=options)
            return {"status": "success"}

    monkeypatch.setattr("app.ollama_import.OllamaClient", FakeClient)
    manager = OllamaImportManager(output_root, tmp_path / "data")
    item = OllamaImport(
        id="gguf-test",
        status="queued",
        output_name="gemma",
        model_name="gemma-heretic",
        base_url="http://ollama:11434",
        quantize="q4_K_M",
        created_at="2026-01-01T00:00:00+00:00",
        modelfile="FROM .",
        resolved_format="gguf",
    )
    manager.current = item
    manager._persist(item)

    manager._run(item.id)

    assert item.status == "completed"
    assert calls["model_name"] == "gemma-heretic"
    assert list(calls["files"]) == ["gemma-Q4_K_M.gguf"]
    assert calls["files"]["gemma-Q4_K_M.gguf"].startswith("sha256:")
    assert calls["quantize"] is None


def test_gguf_conversion_uses_partial_files_and_removes_bf16_by_default(
    tmp_path: Path, monkeypatch
):
    output_root = tmp_path / "outputs"
    source = output_root / "gemma"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"source-weights")
    tools = tmp_path / "llama.cpp"
    converter = tools / "convert_hf_to_gguf.py"
    quantizer = tools / "build" / "bin" / "llama-quantize"
    quantizer.parent.mkdir(parents=True)
    converter.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "output = Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "output.write_bytes(b'bf16-gguf')\n"
    )
    quantizer.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(b'quantized-gguf')\n"
    )
    quantizer.chmod(0o755)
    monkeypatch.setattr("app.ollama_import.LLAMA_CPP_DIR", tools)
    manager = OllamaImportManager(output_root, tmp_path / "data")
    item = OllamaImport(
        id="convert-test",
        status="running",
        output_name="gemma",
        model_name="gemma-heretic",
        base_url="http://ollama:11434",
        quantize="q4_K_M",
        created_at="2026-01-01T00:00:00+00:00",
        resolved_format="gguf",
    )
    manager.current = item
    manager._persist(item)

    result = manager._ensure_gguf(item, source)
    bf16, final = gguf_artifact_paths(output_root, "gemma", "q4_K_M")

    assert result == final
    assert final.read_bytes() == b"quantized-gguf"
    assert not bf16.exists()
    assert not list(final.parent.glob("*.partial"))
    assert item.artifact_path == str(final)


def test_delete_output_removes_safetensors_and_matching_gguf(tmp_path: Path):
    output_root = tmp_path / "outputs"
    source = output_root / "gemma"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"source")
    gguf = output_root / ".gguf" / "gemma"
    gguf.mkdir(parents=True)
    (gguf / "gemma-Q4_K_M.gguf").write_bytes(b"gguf")
    manager = OllamaImportManager(output_root, tmp_path / "data")

    result = manager.delete_output("gemma")

    assert result == {"output_name": "gemma", "deleted_bytes": 12}
    assert not source.exists()
    assert not gguf.exists()


def test_delete_output_blocks_active_import_and_path_traversal(tmp_path: Path):
    output_root = tmp_path / "outputs"
    source = output_root / "gemma"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"source")
    manager = OllamaImportManager(output_root, tmp_path / "data")
    manager.current = OllamaImport(
        id="active",
        status="running",
        output_name="gemma",
        model_name="gemma",
        base_url="http://ollama:11434",
        quantize=None,
        created_at="2026-01-01T00:00:00+00:00",
    )

    with pytest.raises(RuntimeError, match="正在匯入"):
        manager.delete_output("gemma")
    with pytest.raises(ValueError, match="無效"):
        manager.delete_output("../gemma")
    assert source.exists()
