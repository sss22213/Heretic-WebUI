from pathlib import Path
import json
import math
import re
import hashlib
import subprocess
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import (
    HFTokenStore,
    Job,
    JobManager,
    JobRequest,
    OllamaHFImportRequest,
    OllamaImportRequest,
    SettingsStore,
    TrialExportRequest,
    UISettingsRequest,
    job_environment,
    output_artifacts_complete,
    parse_ara_trials,
    prompt_source,
    render_config,
    resolve_dataset_configs,
    resolved_prompt_path,
    safe_slug,
    split_range,
)
from app.dataset_resolver import (
    clean_prompt,
    list_dataset_configs,
    suggest_prompt_column,
    write_prompts,
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
    ollama_model_name_error,
    parse_modelfile,
    resolve_import_format,
)
from app.lora_manager import (
    LoRAManager,
    LoRATask,
    adapter_files,
    adapter_supported_by_ollama,
    inspect_adapter,
    suggest_merge_base,
)
from app.lora_merge import auxiliary_files, model_class_name
from app.eval_manager import (
    EvalManager,
    EvalRun,
    ollama_url_is_local,
    parse_task_list,
    summarize_results,
)
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


def test_job_request_normalizes_dataset_config():
    assert JobRequest(model="org/m", good_config="  ").good_config is None
    assert JobRequest(model="org/m", good_config="good_1000").good_config == "good_1000"
    with pytest.raises(ValidationError):
        JobRequest(model="org/m", bad_config="bad config!")


def test_render_config_without_config_passes_dataset_through():
    request = JobRequest(
        model="org/m", good_dataset="wangzhang/abliterix-datasets",
        good_split="train[:400]", good_column="prompt",
    )
    parsed = tomllib.loads(render_config(request, Path("/tmp/o"), "abc123"))
    assert parsed["good_prompts"]["dataset"] == "wangzhang/abliterix-datasets"
    assert parsed["good_prompts"]["column"] == "prompt"


def test_render_config_with_config_points_at_resolved_local_file():
    request = JobRequest(
        model="org/m",
        good_dataset="wangzhang/abliterix-datasets", good_config="good_1000",
        good_split="train[:400]", good_column="prompt",
    )
    parsed = tomllib.loads(render_config(request, Path("/tmp/o"), "abc123"))
    good = parsed["good_prompts"]
    # config.toml points at a local prompt file, not the HF id...
    assert good["dataset"] == str(resolved_prompt_path("wangzhang/abliterix-datasets", "good_1000", "train[:400]", "prompt"))
    assert good["dataset"].endswith(".txt")
    assert "datasheets" in good["dataset"] and "_resolved" in good["dataset"]
    # ...the split still applies (Heretic slices the lines), column is plain text.
    assert good["split"] == "train[:400]"
    assert good["column"] == "text"
    # The other side has no config, so it is untouched.
    assert parsed["bad_prompts"]["dataset"] == "mlabonne/harmful_behaviors"


def test_render_config_evaluates_on_the_direction_datasets():
    request = JobRequest(
        model="org/m", heretic_channel="ara",
        good_dataset="wangzhang/abliterix-datasets", good_config="good_1000",
        good_split="train[:400]", good_column="prompt", good_eval_split="train[400:500]",
        bad_dataset="wangzhang/abliterix-datasets", bad_config="harmful_1000",
        bad_split="train[:400]", bad_column="prompt", bad_eval_split="train[500:600]",
    )
    parsed = tomllib.loads(render_config(request, Path("/tmp/o"), "abc123"))

    # Evaluation reads the same files as the directions, on a held-out range.
    assert parsed["good_evaluation_prompts"]["dataset"] == parsed["good_prompts"]["dataset"]
    assert parsed["bad_evaluation_prompts"]["dataset"] == parsed["bad_prompts"]["dataset"]
    assert parsed["good_evaluation_prompts"]["split"] == "train[400:500]"
    assert parsed["bad_evaluation_prompts"]["split"] == "train[500:600]"
    # Chinese prompts need Chinese markers, or every refusal scores as a pass.
    assert "抱歉" in parsed["refusal_markers"] and "i cannot" in parsed["refusal_markers"]
    assert parsed["disable_thinking"] is True


def test_render_config_master_channel_targets_scorer_plugins():
    request = JobRequest(model="org/m", good_eval_split="test[:100]", bad_eval_split="test[:100]")
    parsed = tomllib.loads(render_config(request, Path("/tmp/o"), "abc123"))

    # On master the evaluation prompts and markers belong to the scorer plugins.
    assert parsed["scorer"]["KLDivergence"]["prompts"]["dataset"] == "mlabonne/harmless_alpaca"
    assert parsed["scorer"]["KeywordRate"]["prompts"]["dataset"] == "mlabonne/harmful_behaviors"
    assert "抱歉" in parsed["scorer"]["KeywordRate"]["keyword_markers"]
    assert "refusal_markers" not in parsed


def test_render_config_can_fall_back_to_upstream_evaluation_defaults():
    request = JobRequest(
        model="org/m", heretic_channel="ara",
        eval_follows_direction=False, include_cjk_refusal_markers=False,
    )
    parsed = tomllib.loads(render_config(request, Path("/tmp/o"), "abc123"))
    assert "good_evaluation_prompts" not in parsed
    assert "refusal_markers" not in parsed


@pytest.mark.parametrize(
    "form_id,channel_only",
    [
        ("jobForm", {"ara_lora_rank", "use_ara", "use_ara_lora", "use_piqa"}),
        ("araJobForm", {"offload_outputs_to_cpu"}),
    ],
)
def test_job_forms_only_name_real_request_fields(form_id: str, channel_only: set[str]):
    """The "load settings" button matches form controls to request fields by name."""
    html = (
        Path(__file__).resolve().parent.parent / "app" / "static" / "index.html"
    ).read_text(encoding="utf-8")
    start = html.index(f'<form id="{form_id}"')
    block = html[start : html.index("</form>", start)]
    names = set(re.findall(r'name="([a-z_]+)"', block))

    assert not names - set(JobRequest.model_fields)
    # hf_token is write-only, and these are filled in per job, not carried over.
    never_loaded = {
        "hf_token", "heretic_channel", "output_name",
        "reexport_source", "reexport_front_index", "reexport_trial_number",
    }
    assert set(JobRequest.model_fields) - never_loaded - channel_only == names - never_loaded


def test_split_range_resolves_only_what_it_can_compare():
    assert split_range("train") == (0, math.inf)
    assert split_range("train[:400]") == (0, 400)
    assert split_range("train[400:500]") == (400, 500)
    assert split_range("train[500:]") == (500, math.inf)
    # Percentages, negative indices, and multi-part specs stay Heretic's problem.
    assert split_range("train[:10%]") is None
    assert split_range("train[-100:]") is None
    assert split_range("train[:400]+train[600:]") is None


def test_job_request_rejects_empty_split_ranges():
    base = dict(model="m", good_dataset="/a.txt", bad_dataset="/b.txt")

    # This is the config that reached Heretic and died as
    # "torch.cat(): expected a non-empty list of Tensors".
    with pytest.raises(ValidationError, match="空區間"):
        JobRequest(**base, good_eval_split="train[500:500]")
    with pytest.raises(ValidationError, match="空區間"):
        JobRequest(**base, bad_eval_split="train[600:500]")
    with pytest.raises(ValidationError, match="空區間"):
        JobRequest(**base, good_split="train[500:500]")
    # Evaluation fields are not emitted at all when evaluation is upstream's.
    JobRequest(**base, eval_follows_direction=False, good_eval_split="train[500:500]")


def test_job_request_rejects_evaluating_on_the_direction_prompts():
    base = dict(model="m", good_dataset="/a.txt", bad_dataset="/b.txt")

    with pytest.raises(ValidationError, match="重疊"):
        JobRequest(**base, good_split="train", bad_split="train")
    with pytest.raises(ValidationError, match="重疊"):
        JobRequest(**base, good_split="train[:600]", bad_split="train[:600]")
    # The defaults are disjoint, and a different split name cannot overlap.
    JobRequest(**base)
    JobRequest(**base, good_split="train", good_eval_split="test[:100]",
               bad_split="train", bad_eval_split="test[:100]")


def test_job_request_rejects_eval_split_from_another_split_of_a_resolved_config():
    # Only the direction splits get materialized, so a "test" evaluation split
    # would point config.toml at a file the resolver never wrote.
    with pytest.raises(ValidationError):
        JobRequest(
            model="org/m", good_dataset="org/ds", good_config="good_1000",
            good_split="train[:400]", good_eval_split="test[:100]",
        )
    # Without a config the dataset is loaded by id, so any split is fine.
    JobRequest(model="org/m", good_split="train[:400]", good_eval_split="test[:100]")


def test_resolved_prompt_path_is_deterministic_and_selection_specific():
    a = resolved_prompt_path("org/ds", "cfg_a", "train[:400]", "prompt")
    assert a == resolved_prompt_path("org/ds", "cfg_a", "train", "prompt")  # slice ignored
    assert a != resolved_prompt_path("org/ds", "cfg_b", "train", "prompt")  # config matters
    assert a != resolved_prompt_path("org/ds", "cfg_a", "train", "text")    # column matters
    assert a.name.startswith("org-ds--cfg_a") and a.suffix == ".txt"


def test_prompt_source_switches_on_config():
    assert prompt_source("org/ds", "train", "prompt", None) == ("org/ds", "train", "prompt")
    dataset, split, column = prompt_source("org/ds", "train[:400]", "prompt", "cfg")
    assert dataset.endswith(".txt") and split == "train[:400]" and column == "text"


def test_resolve_dataset_configs_runs_resolver_and_caches(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.main.DATASET_RESOLVED_DIR", tmp_path / "resolved")
    calls = []

    class FakeCompleted:
        returncode = 0
        stdout = "已寫入 400 筆 prompt\n"

    def fake_run(command, **kwargs):
        calls.append(command)
        return FakeCompleted()

    monkeypatch.setattr("app.main.subprocess.run", fake_run)
    request = {
        "good_dataset": "wangzhang/abliterix-datasets", "good_split": "train[:400]",
        "good_column": "prompt", "good_config": "good_1000",
        "bad_dataset": "mlabonne/harmful_behaviors", "bad_split": "train[:400]",
        "bad_column": "text", "bad_config": None,
    }
    log = tmp_path / "run.log"
    resolve_dataset_configs(request, {"HF_TOKEN": "x"}, log)
    # Only the config-bearing (good) side ran; bad side passed through untouched.
    assert len(calls) == 1
    command = calls[0]
    assert "app.dataset_resolver" in command
    assert command[command.index("--config") + 1] == "good_1000"
    assert command[command.index("--split") + 1] == "train"  # slice stripped for the loader

    # Second run reuses the file the resolver produced instead of re-running.
    # (The real resolver mkdirs this; the fake subprocess does not, so create it.)
    resolved = resolved_prompt_path("wangzhang/abliterix-datasets", "good_1000", "train[:400]", "prompt")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("a\nb\n", encoding="utf-8")
    resolve_dataset_configs(request, {"HF_TOKEN": "x"}, log)
    assert len(calls) == 1
    assert "沿用既有解析結果" in log.read_text(encoding="utf-8")


def test_resolve_dataset_configs_raises_on_resolver_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.main.DATASET_RESOLVED_DIR", tmp_path / "resolved")

    class FakeCompleted:
        returncode = 1
        stdout = "解析資料集失敗：欄位「nope」不存在\n"

    monkeypatch.setattr("app.main.subprocess.run", lambda command, **kwargs: FakeCompleted())
    request = {
        "good_dataset": "org/ds", "good_split": "train", "good_column": "nope", "good_config": "cfg",
        "bad_dataset": "x/y", "bad_split": "train", "bad_column": "text", "bad_config": None,
    }
    with pytest.raises(RuntimeError, match="解析失敗"):
        resolve_dataset_configs(request, {}, tmp_path / "run.log")


def test_dataset_resolver_write_prompts_flattens_and_skips_blank(tmp_path: Path):
    assert clean_prompt("  a\nb  ") == "a b"
    assert clean_prompt(None) == ""
    output = tmp_path / "out.txt"
    written = write_prompts(["hello", "  ", "多\n行", ""], output)
    assert written == 2
    assert output.read_text(encoding="utf-8") == "hello\n多 行\n"
    with pytest.raises(ValueError):
        write_prompts(["", "   "], tmp_path / "empty.txt")


def test_suggest_prompt_column_prefers_prompt_like_names():
    assert suggest_prompt_column(["id", "prompt", "category"]) == "prompt"
    assert suggest_prompt_column(["id", "user_text"]) == "user_text"
    assert suggest_prompt_column(["id", "value"]) == "id"
    assert suggest_prompt_column([]) is None


def test_list_dataset_configs_reads_card_metadata(monkeypatch):
    import app.dataset_resolver as resolver

    class FakeInfo:
        card_data = {
            "configs": [
                {"config_name": "good_1000", "data_files": [{"split": "train", "path": "good_1000/x.json"}]},
                {"config_name": "harmful_1000", "data_files": [{"split": "train", "path": "harmful_1000/y.json"}]},
            ]
        }

    class FakeApi:
        def dataset_info(self, repo_id, token=None):
            return FakeInfo()

    monkeypatch.setattr(resolver, "HfApi", FakeApi)
    monkeypatch.setattr(resolver, "_first_rows_columns", lambda *a: ["id", "prompt", "category"])
    result = list_dataset_configs("wangzhang/abliterix-datasets")
    assert [c["name"] for c in result["configs"]] == ["good_1000", "harmful_1000"]
    assert result["configs"][0]["splits"] == ["train"]
    assert result["suggested_column"] == "prompt"


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


def test_ollama_model_name_component_limits():
    assert ollama_model_name_error("user/model:Q4_K_M") is None
    assert ollama_model_name_error("a" * 80) is None
    assert ollama_model_name_error(f"user/{'a' * 80}:tag") is None
    # 87-char model component: rejected by a live Ollama server after the
    # whole GGUF upload; must be caught at request time instead.
    long_name = "qwen3.6-27b-qwen3.6-finetune-qwen3.8-max-glm5.2-kimi-k3-distillation-a56-1168cb-heretic"
    assert "80" in ollama_model_name_error(f"n0404n0404/{long_name}:Q4_K_M")
    assert "80" in ollama_model_name_error("a" * 81)
    assert "80" in ollama_model_name_error(f"model:{'t' * 81}")
    assert ollama_model_name_error("a//b") is not None
    assert ollama_model_name_error("h/n/m/extra") is not None

    with pytest.raises(ValidationError, match="80"):
        OllamaImportRequest(
            output_name="example",
            model_name=f"n0404n0404/{long_name}:Q4_K_M",
            base_url="http://ollama:11434",
        )


def test_ollama_hf_import_request_validation():
    request = OllamaHFImportRequest(
        repo_id="Qwen/Qwen3.6-4B", model_name="qwen3.6-4b:latest", base_url="http://ollama:11434"
    )
    assert request.revision == "main"
    assert request.import_format == "auto"
    with pytest.raises(ValidationError):
        OllamaHFImportRequest(
            repo_id="no-slash", model_name="m:latest", base_url="http://ollama:11434"
        )
    # Inherits the shared model-name component limit.
    with pytest.raises(ValidationError, match="80"):
        OllamaHFImportRequest(
            repo_id="a/b", model_name="x" * 81, base_url="http://ollama:11434"
        )


def test_ollama_hf_import_start_validation(tmp_path: Path):
    manager = OllamaImportManager(tmp_path / "outputs", tmp_path / "data")
    with pytest.raises(ValueError, match="repo ID"):
        manager.start_from_hf("not-a-repo", "main", "m:latest", "http://ollama:11434", None, "FROM .")

    incomplete = tmp_path / "outputs" / "org--model"
    incomplete.mkdir(parents=True)
    (incomplete / "junk.txt").write_text("x")
    with pytest.raises(RuntimeError, match="不完整"):
        manager.start_from_hf("org/model", "main", "m:latest", "http://ollama:11434", None, "FROM .")

    manager.current = OllamaImport(
        id="busy", status="running", output_name="other", model_name="m:latest",
        base_url="http://ollama:11434", quantize=None, created_at="now",
    )
    with pytest.raises(RuntimeError, match="已有"):
        manager.start_from_hf("org/other", "main", "m:latest", "http://ollama:11434", None, "FROM .")


def test_ollama_hf_import_downloads_then_imports(tmp_path: Path, monkeypatch):
    output_root = tmp_path / "outputs"
    calls = {}

    def fake_snapshot_download(*, repo_id, revision, token, local_dir, ignore_patterns):
        calls.update(repo_id=repo_id, revision=revision, token=token,
                     ignore_patterns=list(ignore_patterns))
        staging = Path(local_dir)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "config.json").write_text("{}")
        (staging / "model.safetensors").write_bytes(b"weights")
        cache = staging / ".cache" / "huggingface"
        cache.mkdir(parents=True)
        (cache / "download.lock").write_text("")

    class FakeApi:
        def model_info(self, *args, **kwargs):
            raise RuntimeError("offline: progress total is optional")

    class FakeClient:
        def __init__(self, base_url):
            calls["base_url"] = base_url

        def version(self):
            return {"version": "0.30.6"}

        def blob_exists(self, _digest):
            return True

        def create(self, model_name, files, quantize, options):
            calls.update(model_name=model_name, files=dict(files), quantize=quantize)
            return {"status": "success"}

    monkeypatch.setattr("app.ollama_import.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("app.ollama_import.HfApi", FakeApi)
    monkeypatch.setattr("app.ollama_import.OllamaClient", FakeClient)
    manager = OllamaImportManager(output_root, tmp_path / "data")
    item = OllamaImport(
        id="hf-test", status="queued", output_name="org--model",
        model_name="model:latest", base_url="http://ollama:11434",
        quantize=None, created_at="2026-01-01T00:00:00+00:00", modelfile="FROM .",
        repo_id="org/model", revision="main",
    )
    manager.current = item
    manager._persist(item)

    manager._run(item.id, "hf_secret")

    assert item.status == "completed", item.error
    assert calls["repo_id"] == "org/model"
    assert calls["token"] == "hf_secret"
    assert "*.bin" in calls["ignore_patterns"] and "*.gguf" in calls["ignore_patterns"]
    directory = output_root / "org--model"
    assert (directory / "model.safetensors").is_file()
    assert not (directory / ".cache").exists()
    assert not (output_root / ".download-org--model").exists()
    assert sorted(calls["files"]) == ["config.json", "model.safetensors"]
    assert item.resolved_format == "safetensors"


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


def test_lora_manager_deletes_adapter_style_outputs(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manager = LoRAManager(tmp_path / "data", outputs)
    adapter = outputs / "model-heretic-abc123"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    merged = outputs / "model-merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"weights")

    listed = {item["name"]: item["source"] for item in manager.list()}
    assert listed == {"model-heretic-abc123": "outputs"}
    # A merged full model in outputs is not an adapter and stays untouchable.
    with pytest.raises(ValueError, match="adapter"):
        manager.delete("model-merged", source="outputs")
    assert merged.exists()
    with pytest.raises(ValueError, match="找不到 LoRA"):
        manager.delete("model-heretic-abc123")

    result = manager.delete("model-heretic-abc123", source="outputs")
    assert result["deleted_bytes"] > 0 and result["source"] == "outputs"
    assert not adapter.exists()


def test_lora_manager_rejects_outputs_delete_without_output_root(tmp_path: Path):
    manager = LoRAManager(tmp_path / "data")

    with pytest.raises(ValueError, match="不支援"):
        manager.delete("anything", source="outputs")


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


def test_adapter_ollama_support_and_merge_base_suggestion():
    assert adapter_supported_by_ollama(["LlamaForCausalLM"])
    assert adapter_supported_by_ollama(["Gemma2ForCausalLM"])
    assert not adapter_supported_by_ollama(["Qwen3_5ForConditionalGeneration"])
    assert not adapter_supported_by_ollama([])

    outputs = ["ThinkingCap-Qwen3.6-27B-heretic-6f87a8", "gemma-4-12B-it-heretic-b9a462"]
    assert (
        suggest_merge_base("bottlecapai/ThinkingCap-Qwen3.6-27B", outputs)
        == "ThinkingCap-Qwen3.6-27B-heretic-6f87a8"
    )
    assert suggest_merge_base("google/gemma-4-12B-it", outputs) == "gemma-4-12B-it-heretic-b9a462"
    assert suggest_merge_base("org/unrelated-model", outputs) is None
    assert suggest_merge_base(None, outputs) is None


def test_merge_auxiliary_files_exclude_weights_hidden_and_existing(tmp_path: Path):
    base = tmp_path / "base"
    output = tmp_path / "output"
    base.mkdir()
    output.mkdir()
    (base / "model-00001-of-00002.safetensors").write_bytes(b"w")
    (base / "model.safetensors.index.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    (base / "config.json").write_text("{}")
    (base / ".hidden").write_text("x")
    (output / "config.json").write_text("{}")

    assert [path.name for path in auxiliary_files(base, output)] == ["tokenizer.json"]
    assert model_class_name(base) is None
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    assert model_class_name(base) == "Qwen3_5ForConditionalGeneration"


def test_lora_merge_rejects_invalid_inputs(tmp_path: Path):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manager = LoRAManager(data)
    gguf_only = data / "loras" / "ggufonly"
    gguf_only.mkdir(parents=True)
    (gguf_only / "adapter.gguf").write_bytes(b"g")
    adapter = data / "loras" / "mylora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    base = outputs / "base-model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"weights")
    (outputs / "already").mkdir()

    with pytest.raises(ValueError, match="Safetensors"):
        manager.start_merge("ggufonly", "base-model", "merged", outputs)
    with pytest.raises(ValueError, match="基底"):
        manager.start_merge("mylora", "missing-base", "merged", outputs)
    with pytest.raises(ValueError, match="已存在"):
        manager.start_merge("mylora", "base-model", "already", outputs)
    with pytest.raises(ValueError, match="找不到 LoRA"):
        manager.start_merge("missing", "base-model", "merged", outputs)
    with pytest.raises(ValueError, match="Ollama"):
        manager.start_merge("mylora", "qwen3.6:27b", "merged", outputs)
    with pytest.raises(ValueError, match="organization/repository"):
        manager.start_merge("mylora", "a/b/c", "merged", outputs)
    models = tmp_path / "models"
    local_model = models / "my-base"
    local_model.mkdir(parents=True)
    with pytest.raises(ValueError, match="路徑形式"):
        manager.start_merge("mylora", str(outputs / "base-model"), "merged", outputs, models)
    with pytest.raises(ValueError, match="不支援路徑"):
        manager.start_merge("mylora", str(local_model), "merged", outputs, None)
    with pytest.raises(ValueError, match="完整的基底模型"):
        manager.start_merge("mylora", str(local_model), "merged", outputs, models)


def test_lora_merge_publishes_output_atomically(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manager = LoRAManager(data)
    adapter = data / "loras" / "mylora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    base = outputs / "base-model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"weights")
    (base / "config.json").write_text("{}")

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            staging = Path(command[command.index("--output") + 1])
            staging.mkdir(parents=True)
            (staging / "model.safetensors").write_bytes(b"merged")
            (staging / "config.json").write_text("{}")
            self.stdout = iter(["合併完成\n"])
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("app.lora_manager.subprocess.Popen", FakeProcess)
    task = manager.start_merge("mylora", "base-model", "merged-out", outputs)
    assert task.operation == "merge"
    for _ in range(100):
        if manager.current.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert manager.current.status == "completed", manager.current.error
    assert (outputs / "merged-out" / "model.safetensors").read_bytes() == b"merged"
    assert not (outputs / f".merge-{task.id}").exists()
    assert "合併完成" in (data / "lora_tasks" / task.id / "run.log").read_text(encoding="utf-8")


def test_lora_merge_downloads_hf_base(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manager = LoRAManager(data)
    adapter = data / "loras" / "mylora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    snapshot = tmp_path / "hf-cache" / "snap"

    def fake_snapshot_download(repo_id, token=None, ignore_patterns=None):
        assert repo_id == "Qwen/Qwen3.6-27B"
        assert token == "hf_secret"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.safetensors").write_bytes(b"weights")
        (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            assert command[command.index("--base") + 1] == str(snapshot)
            staging = Path(command[command.index("--output") + 1])
            staging.mkdir(parents=True)
            (staging / "model.safetensors").write_bytes(b"merged")
            self.stdout = iter([])
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("app.lora_manager.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("app.lora_manager.subprocess.Popen", FakeProcess)
    task = manager.start_merge(
        "mylora", "Qwen/Qwen3.6-27B", "merged-out", outputs, None, "hf_secret"
    )
    for _ in range(100):
        if manager.current.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert manager.current.status == "completed", manager.current.error
    assert (outputs / "merged-out" / "model.safetensors").read_bytes() == b"merged"
    log = (data / "lora_tasks" / task.id / "run.log").read_text(encoding="utf-8")
    assert "下載基底模型" in log


def test_lora_merge_accepts_models_dir_path_as_base(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    models = tmp_path / "models"
    outputs.mkdir()
    manager = LoRAManager(data)
    adapter = data / "loras" / "mylora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    base = models / "my-base"
    base.mkdir(parents=True)
    (base / "model.safetensors").write_bytes(b"weights")

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            assert command[command.index("--base") + 1] == str(base.resolve())
            staging = Path(command[command.index("--output") + 1])
            staging.mkdir(parents=True)
            (staging / "model.safetensors").write_bytes(b"merged")
            self.stdout = iter([])
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("app.lora_manager.subprocess.Popen", FakeProcess)
    manager.start_merge("mylora", str(base), "merged-out", outputs, models)
    for _ in range(100):
        if manager.current.status in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert manager.current.status == "completed", manager.current.error
    assert (outputs / "merged-out" / "model.safetensors").read_bytes() == b"merged"


def test_parse_task_list_validates_and_dedupes():
    assert parse_task_list("hellaswag, arc_challenge hellaswag") == ["hellaswag", "arc_challenge"]
    with pytest.raises(ValueError, match="無效的評測任務名稱"):
        parse_task_list("hellaswag;rm")
    with pytest.raises(ValueError, match="無效的評測任務名稱"):
        parse_task_list("-flag")
    with pytest.raises(ValueError, match="至少"):
        parse_task_list("  ,  ")


def test_summarize_results_keeps_primary_metrics():
    raw = {
        "results": {
            "gsm8k": {
                "alias": "gsm8k",
                "sample_len,none": 30,
                "exact_match,strict-match": 0.55123,
                "exact_match_stderr,strict-match": 0.013,
                "exact_match,flexible-extract": 0.6,
                "exact_match_stderr,flexible-extract": 0.012,
            },
            "hellaswag": {"acc,none": 0.5, "acc_norm,none": 0.66666, "acc_stderr,none": 0.01},
            "broken": "not-a-dict",
        }
    }
    summary = summarize_results(raw)
    assert summary["gsm8k"] == {
        "exact_match(strict-match)": 0.5512,
        "exact_match(flexible-extract)": 0.6,
    }
    assert summary["hellaswag"] == {"acc": 0.5, "acc_norm": 0.6667}
    assert "broken" not in summary
    assert summarize_results({}) == {}


def test_eval_resolve_model_sources(tmp_path: Path):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    models = tmp_path / "models"
    outputs.mkdir()
    complete = outputs / "my-output"
    complete.mkdir()
    (complete / "model.safetensors").write_bytes(b"w")
    (outputs / "incomplete").mkdir()
    local = models / "local-base"
    local.mkdir(parents=True)
    (local / "model.safetensors").write_bytes(b"w")

    manager = EvalManager(data, outputs, models)
    assert manager.resolve_model("my-output") == str(complete.resolve())
    assert manager.resolve_model(str(local)) == str(local.resolve())
    assert manager.resolve_model("org/model") == "org/model"
    with pytest.raises(ValueError, match="完整的 output"):
        manager.resolve_model("incomplete")
    with pytest.raises(ValueError, match="完整的 output"):
        manager.resolve_model("missing")
    with pytest.raises(ValueError, match="必須位於"):
        manager.resolve_model(str(outputs / "my-output"))
    with pytest.raises(ValueError, match="organization/model"):
        manager.resolve_model("org//model")
    no_models = EvalManager(tmp_path / "data2", outputs, None)
    with pytest.raises(ValueError, match="不支援路徑"):
        no_models.resolve_model(str(local))


def test_eval_command_reflects_options(tmp_path: Path):
    manager = EvalManager(tmp_path / "data", tmp_path / "outputs")
    run = EvalRun(
        id="abc", status="queued", created_at="now", model_source="my-output",
        model_path="/outputs/my-output", tasks=["hellaswag", "gsm8k"],
        num_fewshot=5, limit=200, batch_size=8, quantization="bnb_4bit",
    )
    command = manager.command(run)
    assert command[command.index("--model_args") + 1] == (
        "pretrained=/outputs/my-output,dtype=bfloat16,load_in_4bit=True"
    )
    assert command[command.index("--tasks") + 1] == "hellaswag,gsm8k"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--num_fewshot") + 1] == "5"
    assert command[command.index("--limit") + 1] == "200"

    auto = EvalRun(
        id="def", status="queued", created_at="now", model_source="m",
        model_path="org/model", tasks=["mmlu"],
    )
    command = manager.command(auto)
    assert command[command.index("--batch_size") + 1] == "auto"
    assert "--num_fewshot" not in command
    assert "--limit" not in command
    assert "--gen_kwargs" not in command
    assert "--log_samples" not in command
    assert "load_in_4bit" not in command[command.index("--model_args") + 1]

    thinking = EvalRun(
        id="ghi", status="queued", created_at="now", model_source="m:Q4",
        model_path="m:Q4", tasks=["gsm8k"], backend="ollama",
        base_url="http://ollama:11434", max_gen_toks=2048, log_samples=True,
    )
    command = manager.command(thinking)
    assert command[command.index("--gen_kwargs") + 1] == "max_gen_toks=2048,until=<|endoftext|>"
    assert "--log_samples" in command

    # Ollama runs neutralize task stop strings even without a token override;
    # Ollama applies them to the raw stream where they match thinking content.
    plain_ollama = EvalRun(
        id="jkl", status="queued", created_at="now", model_source="m:Q4",
        model_path="m:Q4", tasks=["gsm8k"], backend="ollama",
        base_url="http://ollama:11434",
    )
    command = manager.command(plain_ollama)
    assert command[command.index("--gen_kwargs") + 1] == "until=<|endoftext|>"


def test_eval_run_executes_and_parses_results(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    model = outputs / "my-output"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_bytes(b"w")
    manager = EvalManager(data, outputs)
    monkeypatch.setattr("app.eval_manager.lm_eval_available", lambda: True)

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            results_dir = Path(command[command.index("--output_path") + 1]) / "my-output"
            results_dir.mkdir(parents=True)
            (results_dir / "results_2026-07-26.json").write_text(json.dumps({
                "results": {"hellaswag": {"acc,none": 0.42, "acc_stderr,none": 0.01}}
            }))
            self.stdout = iter(["Running loglikelihood requests\n"])
            self.pid = 4242
            self.returncode = 0

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr("app.eval_manager.subprocess.Popen", FakeProcess)
    run = manager.start("my-output", ["hellaswag"], None, None, 0, "none")
    with pytest.raises(RuntimeError, match="已有評測正在執行"):
        manager.start("my-output", ["hellaswag"], None, None, 0, "none")
    for _ in range(100):
        if manager.runs[run.id].status in ("completed", "failed"):
            break
        time.sleep(0.05)

    stored = manager.runs[run.id]
    assert stored.status == "completed", stored.error
    assert stored.results == {"hellaswag": {"acc": 0.42}}
    log = (data / "evals" / run.id / "run.log").read_text(encoding="utf-8")
    assert "Running loglikelihood requests" in log
    assert "評測完成" in log

    task = manager.get_task()
    assert task["id"] == run.id and "log" in task
    with pytest.raises(RuntimeError, match="沒有執行中的評測"):
        manager.cancel()
    assert manager.delete(run.id) == {"id": run.id}
    assert not (data / "evals" / run.id).exists()
    with pytest.raises(ValueError, match="找不到評測紀錄"):
        manager.delete(run.id)


def test_eval_manager_marks_interrupted_runs_failed_on_load(tmp_path: Path):
    data = tmp_path / "data"
    manager = EvalManager(data, tmp_path / "outputs")
    run = EvalRun(
        id="stale1", status="running", created_at="2026-07-26T00:00:00+00:00",
        model_source="m", model_path="/outputs/m", tasks=["hellaswag"],
    )
    manager.runs[run.id] = run
    manager._persist(run)

    reloaded = EvalManager(data, tmp_path / "outputs")
    stale = reloaded.runs["stale1"]
    assert stale.status == "failed"
    assert "重啟" in stale.error


def test_eval_ollama_backend_command_and_validation(tmp_path: Path):
    manager = EvalManager(tmp_path / "data", tmp_path / "outputs")
    assert manager.validate_ollama_model("my-model:Q4") == "my-model:Q4"
    assert manager.validate_ollama_model("user/model:latest") == "user/model:latest"
    with pytest.raises(ValueError, match="Ollama 模型名稱"):
        manager.validate_ollama_model("bad name")
    with pytest.raises(ValueError, match="Ollama 模型名稱"):
        manager.validate_ollama_model(":latest")
    assert manager.validate_base_url("http://ollama:11434/") == "http://ollama:11434"
    with pytest.raises(ValueError, match="Ollama API 位址"):
        manager.validate_base_url("ollama:11434")
    with pytest.raises(ValueError, match="Ollama API 位址"):
        manager.validate_base_url("http://a,b")
    with pytest.raises(ValueError, match="需要 Ollama API 位址"):
        manager.validate_base_url(None)

    run = EvalRun(
        id="abc", status="queued", created_at="now",
        model_source="thinkingcap-q4:latest", model_path="thinkingcap-q4:latest",
        tasks=["gsm8k"], num_fewshot=5, limit=100,
        backend="ollama", base_url="http://ollama:11434",
    )
    command = manager.command(run)
    assert command[command.index("--model") + 1] == "local-chat-completions"
    assert command[command.index("--model_args") + 1] == (
        "model=thinkingcap-q4:latest,"
        "base_url=http://ollama:11434/v1/chat/completions,"
        "num_concurrent=1,max_retries=3"
    )
    assert "--apply_chat_template" in command
    assert "--batch_size" not in command
    assert command[command.index("--num_fewshot") + 1] == "5"
    assert command[command.index("--limit") + 1] == "100"

    tuned = EvalRun(
        id="def", status="queued", created_at="now",
        model_source="thinkingcap-q4:latest", model_path="thinkingcap-q4:latest",
        tasks=["gsm8k"], backend="ollama", base_url="http://ollama:11434",
        num_concurrent=8, max_retries=5,
    )
    tuned_args = manager.command(tuned)[manager.command(tuned).index("--model_args") + 1]
    assert "num_concurrent=8" in tuned_args
    assert "max_retries=5" in tuned_args


def test_eval_start_ollama_requires_base_url(tmp_path: Path, monkeypatch):
    manager = EvalManager(tmp_path / "data", tmp_path / "outputs")
    monkeypatch.setattr("app.eval_manager.lm_eval_available", lambda: True)
    with pytest.raises(ValueError, match="需要 Ollama API 位址"):
        manager.start("model:Q4", ["gsm8k"], None, None, 0, "none", backend="ollama")
    with pytest.raises(ValueError, match="Ollama 模型名稱"):
        manager.start(
            "bad//name::", ["gsm8k"], None, None, 0, "none",
            backend="ollama", base_url="http://ollama:11434",
        )


def test_ollama_model_list_uses_short_timeout_and_selected_url(monkeypatch):
    import app.main as main_module

    captured = {}

    class FakeClient:
        def __init__(self, base_url, timeout=86_400):
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def tags(self):
            return {"models": [{"name": "b:latest"}, {"model": "a:latest"}, {"name": "b:latest"}]}

    monkeypatch.setattr(main_module, "OllamaClient", FakeClient)
    result = main_module.list_ollama_models(base_url="http://10.0.0.5:11434")
    assert result == {"models": ["a:latest", "b:latest"]}
    assert captured["base_url"] == "http://10.0.0.5:11434"
    assert captured["timeout"] <= 30


def test_eval_use_cache_adds_per_model_cache_prefix(tmp_path: Path):
    manager = EvalManager(tmp_path / "data", tmp_path / "outputs")
    run = EvalRun(
        id="ghi", status="queued", created_at="now",
        model_source="user/model:latest", model_path="user/model:latest",
        tasks=["gsm8k"], backend="ollama", base_url="http://ollama:11434",
        use_cache=True,
    )
    command = manager.command(run)
    cache_prefix = Path(command[command.index("--use_cache") + 1])
    assert cache_prefix.parent == manager.root / "cache"
    assert cache_prefix.name == "user-model-latest"
    assert cache_prefix.parent.is_dir()

    run.use_cache = False
    assert "--use_cache" not in manager.command(run)


def test_eval_uses_local_gpu_only_for_hf_or_local_ollama():
    def run(backend, base_url=None):
        return EvalRun(
            id="x", status="running", created_at="now",
            model_source="m", model_path="m",
            tasks=["gsm8k"], backend=backend, base_url=base_url,
        )

    assert run("hf").uses_local_gpu()
    for local in (
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
        "http://host.docker.internal:11434",
    ):
        assert run("ollama", local).uses_local_gpu(), local
    for remote in (
        "https://abc123-11434.proxy.runpod.net",
        "http://192.168.1.50:11434",
        "http://ollama:11434",
    ):
        assert not run("ollama", remote).uses_local_gpu(), remote
    # Unparsable/empty addresses fall back to the safe assumption: local.
    assert ollama_url_is_local(None)
    assert ollama_url_is_local("not a url")


def test_job_request_ara_channel_and_render_config(tmp_path: Path):
    request = JobRequest(
        model="org/model",
        heretic_channel="ara",
        use_ara_lora=True,
        ara_lora_rank=64,
        use_piqa=True,
        export_strategy="adapter",
    )
    parsed = tomllib.loads(render_config(request, tmp_path / "output", "abc123"))

    assert parsed["use_ara"] is True
    assert parsed["use_ara_lora"] is True
    assert parsed["ara_lora_rank"] == 64
    assert parsed["use_piqa"] is True
    # ara has no offload_outputs_to_cpu setting; emitting it would make its
    # pydantic Settings reject the config file.
    assert "offload_outputs_to_cpu" not in parsed
    # The automation keys exist on ara only via the managed patch and must be
    # present so the run needs no interactive answers.
    assert parsed["model_action"] == "save"
    assert parsed["checkpoint_action"] == "continue"
    assert parsed["trial_index"] == 0
    assert parsed["export_strategy"] == "adapter"
    assert parsed["good_prompts"]["dataset"] == "mlabonne/harmless_alpaca"


def test_master_channel_config_keeps_offload_and_omits_ara_keys(tmp_path: Path):
    parsed = tomllib.loads(
        render_config(JobRequest(model="org/model"), tmp_path / "output", "abc123")
    )
    assert parsed["offload_outputs_to_cpu"] is True
    assert "use_ara" not in parsed
    assert "ara_lora_rank" not in parsed


def test_ara_full_weight_mode_rejects_adapter_export():
    with pytest.raises(ValidationError):
        JobRequest(model="org/model", heretic_channel="ara", export_strategy="adapter")
    ara_lora = JobRequest(
        model="org/model", heretic_channel="ara",
        export_strategy="adapter", use_ara_lora=True,
    )
    assert ara_lora.export_strategy == "adapter"
    assert JobRequest(model="org/model", export_strategy="adapter").export_strategy == "adapter"


def test_ara_rejects_run_ending_combinations():
    # Full-weight ARA crashes on bitsandbytes-quantized weights at trial 1.
    with pytest.raises(ValidationError):
        JobRequest(model="org/model", heretic_channel="ara", quantization="bnb_4bit")
    # ARA-LoRA merge export would finish the run with adapter-only artifacts.
    with pytest.raises(ValidationError):
        JobRequest(
            model="org/model", heretic_channel="ara",
            use_ara_lora=True, export_strategy="merge",
        )
    # The workable quantized setup: ARA-LoRA plus adapter export.
    request = JobRequest(
        model="org/model", heretic_channel="ara",
        quantization="bnb_4bit", use_ara_lora=True, export_strategy="adapter",
    )
    assert request.quantization == "bnb_4bit"
    # master is unaffected by the ara guards.
    assert JobRequest(model="org/model", quantization="bnb_4bit").quantization == "bnb_4bit"


def test_ara_patch_present_and_targets_expected_files():
    patch = Path(__file__).resolve().parent.parent / "patches" / "heretic-ara" / "0001-webui-automation.patch"
    text = patch.read_text(encoding="utf-8")
    assert "a/src/heretic/config.py" in text
    assert "a/src/heretic/main.py" in text
    assert "checkpoint_action" in text
    assert "trial_index" in text
    assert "save_directory" in text
    assert "max_shard_size" in text
    # The ara branch predates master's plain-text prompt loader, which the
    # resolved-dataset feature depends on; the patch must backport it.
    assert "a/src/heretic/utils.py" in text
    assert "os.path.isfile" in text
    # Trial re-export needs the invocation's automation keys to survive the
    # checkpoint-settings restore.
    assert "setattr(restored" in text
    # ARA-LoRA restores are only faithful from a Pareto-front adapter snapshot
    # saved during optimization; re-running LBFGS cold produces a corrupt
    # adapter (warm-start state is process-local).
    assert "trial_snapshots" in text
    assert "update_pareto_snapshots" in text
    assert "Saved adapter snapshot for trial" in text
    assert "Restored adapter snapshot for trial" in text


@pytest.mark.parametrize("channel", ["heretic", "heretic-ara"])
def test_disable_thinking_patch_present_on_both_channels(channel: str):
    patch = (
        Path(__file__).resolve().parent.parent
        / "patches" / channel / "0002-evaluate-without-thinking.patch"
    )
    text = patch.read_text(encoding="utf-8")
    # The setting config.toml writes has to exist, and generate() has to use it;
    # otherwise scoring keeps reading the opening of the chain of thought.
    assert "a/src/heretic/config.py" in text
    assert "disable_thinking: bool" in text
    assert "a/src/heretic/model.py" in text
    assert "chat_template_kwargs" in text
    assert '"enable_thinking": False' in text
    assert "**self.chat_template_kwargs," in text


def test_version_manager_tracks_configured_branch(tmp_path: Path):
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
    (seed / "src" / "heretic").mkdir(parents=True)
    (seed / "src" / "heretic" / "__init__.py").write_text("")
    (seed / "src" / "heretic" / "main.py").write_text("")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "master one")
    _git(seed, "checkout", "-b", "ara")
    (seed / "src" / "heretic" / "ara.py").write_text("ARA = 1\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "ara one")
    ara_first = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "master", "ara")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)

    manager = HereticVersionManager(
        checkout,
        tmp_path / "state.json",
        slots_dir=tmp_path / "slots",
        initial_ref="origin/ara",
        branch="ara",
    )
    status = manager.status()
    assert status["branch"] == "ara"
    assert status["commit"] == ara_first
    assert (Path(manager.runtime_info()["path"]) / "src" / "heretic" / "ara.py").is_file()

    _git(seed, "checkout", "ara")
    (seed / "src" / "heretic" / "ara.py").write_text("ARA = 2\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "ara two")
    ara_second = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "ara")

    checked = manager.status(check_remote=True)
    assert checked["latest_commit"] == ara_second
    assert checked["update_available"] is True

    updated = manager.update()
    assert updated["commit"] == ara_second
    assert updated["branch"] == "ara"


def test_version_manager_rebuilds_when_patch_set_changes(tmp_path: Path):
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
    _git(seed, "commit", "-m", "one")
    first = _git(seed, "rev-parse", "HEAD")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "master")
    subprocess.run(["git", "clone", str(remote), str(checkout)], check=True, capture_output=True)

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    manager = HereticVersionManager(checkout, tmp_path / "state.json", patch_dir)
    assert manager.status()["patch_update_available"] is False
    assert manager.update()["changed"] is False

    # A new managed patch appears after the slot was built: the same upstream
    # commit must be rebuilt so the patch takes effect.
    (checkout / "source.py").write_text("value = 99\n")
    (patch_dir / "0001-local.patch").write_text(_git(checkout, "diff", "--binary") + "\n")
    _git(checkout, "checkout", "--", "source.py")

    assert manager.status()["patch_update_available"] is True
    updated = manager.update()
    assert updated["changed"] is True
    assert updated["commit"] == first
    assert updated["patch_update_available"] is False
    assert (Path(manager.runtime_info()["path"]) / "source.py").read_text() == "value = 99\n"


def test_basic_auth_guard_blocks_and_admits(monkeypatch):
    import base64

    from fastapi.testclient import TestClient

    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_BASIC_AUTH", "user:secret")
    client = TestClient(main_module.app)

    assert client.get("/api/settings").status_code == 401
    good = {"Authorization": "Basic " + base64.b64encode(b"user:secret").decode()}
    assert client.get("/api/settings", headers=good).status_code == 200
    bad = {"Authorization": "Basic " + base64.b64encode(b"user:wrong").decode()}
    assert client.get("/api/settings", headers=bad).status_code == 401
    # The Docker HEALTHCHECK probes /api/health without credentials.
    assert client.get("/api/health").status_code == 200

    monkeypatch.setattr(main_module, "APP_BASIC_AUTH", "")
    assert client.get("/api/settings").status_code == 200


def test_delete_lora_endpoint_handles_outputs_adapters(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_module

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    adapter = outputs / "model-heretic-abc123"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    class FakeJobs:
        def __init__(self) -> None:
            self.in_use = True
            self.deleted: list[str] = []

        def output_in_use(self, name: str) -> bool:
            return self.in_use

        def mark_output_deleted(self, name: str) -> None:
            self.deleted.append(name)

    jobs = FakeJobs()
    monkeypatch.setattr(main_module, "lora_manager", LoRAManager(tmp_path / "data", outputs))
    monkeypatch.setattr(main_module, "manager", jobs)
    client = TestClient(main_module.app)

    assert client.delete("/api/loras/x?source=elsewhere").status_code == 400
    assert client.delete("/api/loras/model-heretic-abc123?source=outputs").status_code == 409
    assert adapter.exists()

    jobs.in_use = False
    # Without the source the library root is searched, so the adapter survives.
    assert client.delete("/api/loras/model-heretic-abc123").status_code == 404
    assert adapter.exists()

    response = client.delete("/api/loras/model-heretic-abc123?source=outputs")
    assert response.status_code == 200 and response.json()["deleted_bytes"] > 0
    assert not adapter.exists()
    assert jobs.deleted == ["model-heretic-abc123"]


def test_parse_ara_trials_builds_pareto_front():
    log = """
* Initial refusals: 96/100

Running trial 1 of 4...
  * KL divergence: 0.0370
  * Refusals: 92/100
Running trial 2 of 4...
  * KL divergence: 0.2000
  * Refusals: 2/100
Running trial 3 of 4...
  * KL divergence: 0.3500
  * Refusals: 0/100
Running trial 4 of 4...
  * KL divergence: 15.0
  * Refusals: 0/100
Running trial 5 of 4...
  * KL divergence: 0.5
"""
    data = parse_ara_trials(log)
    assert data["completed"] == 4
    assert data["total"] == 4
    front = data["front"]
    # Fewest refusals first, then strictly improving KL; the dominated
    # trial 4 (same refusals as 3 but far worse KL) is excluded, as is the
    # truncated trial 5.
    assert [trial["trial"] for trial in front] == [3, 2, 1]
    assert [trial["front_index"] for trial in front] == [0, 1, 2]
    assert front[0]["refusals"] == 0 and front[0]["kl"] == 0.35


def test_render_config_reexport_uses_source_checkpoint_and_front_index(tmp_path: Path):
    request = JobRequest(
        model="org/m", heretic_channel="ara", use_ara_lora=True,
        export_strategy="adapter", reexport_source="abcdefabcdef",
        reexport_front_index=3, reexport_trial_number=159,
    )
    parsed = tomllib.loads(render_config(request, tmp_path / "out", "123456789012"))
    assert parsed["trial_index"] == 3
    assert parsed["study_checkpoint_dir"].endswith("abcdefabcdef")

    plain = JobRequest(model="org/m")
    parsed = tomllib.loads(render_config(plain, tmp_path / "out", "123456789012"))
    assert parsed["trial_index"] == 0
    assert parsed["study_checkpoint_dir"].endswith("123456789012")


def test_trial_export_request_validation():
    assert TrialExportRequest(front_indices=[0, 2, 1]).front_indices == [0, 2, 1]
    with pytest.raises(ValidationError):
        TrialExportRequest(front_indices=[])
    with pytest.raises(ValidationError):
        TrialExportRequest(front_indices=[1, 1])
    with pytest.raises(ValidationError):
        TrialExportRequest(front_indices=[-1])


def test_job_creation_refuses_a_slot_built_from_older_patches(tmp_path: Path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()

    class StaleVersionManager:
        def runtime_info(self, slot=None):
            return {
                "slot": "A", "commit": "deadbeef", "path": str(tmp_path),
                "patches_signature": "built-before",
            }

        def expected_patches_signature(self):
            return "current"

    monkeypatch.setattr(
        main_module, "heretic_version_managers",
        {"master": StaleVersionManager(), "ara": StaleVersionManager()},
    )
    manager = main_module.JobManager()
    # config.toml now carries disable_thinking, which a slot built before the
    # patch would reject after the model has already loaded.
    with pytest.raises(HTTPException) as error:
        manager.create(main_module.JobRequest(model="org/model"))
    assert error.value.status_code == 409
    assert "patch" in error.value.detail
    assert not manager.jobs


def test_reexport_jobs_queue_and_start_sequentially(tmp_path: Path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(main_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main_module, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    (tmp_path / "jobs").mkdir()

    class FakeVersionManager:
        def runtime_info(self, slot=None):
            return {
                "slot": "A", "commit": "deadbeef", "path": str(tmp_path),
                "patches_signature": "sig",
            }

        def expected_patches_signature(self):
            return "sig"

    monkeypatch.setattr(
        main_module, "heretic_version_managers",
        {"master": FakeVersionManager(), "ara": FakeVersionManager()},
    )
    ran = []
    monkeypatch.setattr(main_module.JobManager, "_run", lambda self, job_id: ran.append(job_id))

    manager = main_module.JobManager()
    source_request = main_module.JobRequest(
        model="org/model", heretic_channel="ara",
        use_ara_lora=True, export_strategy="adapter",
    )
    source = main_module.Job(
        id="a" * 12, status="completed", request=source_request.model_dump(),
        output_directory=str(tmp_path / "outputs" / "model-heretic-aaaaaa"),
        created_at=main_module.utc_now(), heretic_channel="ara",
    )
    manager.jobs[source.id] = source

    selections = [
        {"front_index": 0, "trial": 162, "refusals": 0, "denominator": 100, "kl": 0.35},
        {"front_index": 1, "trial": 159, "refusals": 2, "denominator": 100, "kl": 0.20},
    ]
    created = manager.create_reexports(source, selections)
    assert len(created) == 2
    time.sleep(0.2)

    # Only the first export owns a worker; the second waits its turn.
    assert ran == [created[0].id]
    assert created[1].id not in manager.started
    assert created[0].request["reexport_front_index"] == 0
    assert created[1].request["reexport_trial_number"] == 159
    assert Path(created[1].output_directory).name.startswith("model-heretic-aaaaaa-t159")

    second_config = tomllib.loads(
        (tmp_path / "jobs" / created[1].id / "config.toml").read_text(encoding="utf-8")
    )
    assert second_config["trial_index"] == 1
    assert second_config["study_checkpoint_dir"].endswith(source.id)

    # While exports are queued, nothing else may claim the GPU.
    with pytest.raises(RuntimeError):
        manager.create_reexports(source, selections)
    with pytest.raises(RuntimeError):
        manager.create(source_request)

    # The drain picks up the waiting export exactly once.
    manager._start_next_queued()
    time.sleep(0.2)
    assert ran == [created[0].id, created[1].id]


def test_log_confirms_trial_snapshot_requirement(tmp_path: Path):
    manager = JobManager.__new__(JobManager)

    restored = tmp_path / "restored.log"
    restored.write_text(
        "Restoring model from trial 159...\n"
        "* Restored adapter snapshot for trial 159.\n",
        encoding="utf-8",
    )
    assert manager._log_confirms_trial(restored, 159)
    assert manager._log_confirms_trial(restored, 159, require_snapshot=True)

    # An old slot re-runs the ablation instead of loading the snapshot;
    # for ARA-LoRA that export is corrupt and must be rejected.
    rerun = tmp_path / "rerun.log"
    rerun.write_text(
        "Restoring model from trial 159...\n"
        "* Abliterating (Arbitrary-Rank Ablation with LoRA)...\n",
        encoding="utf-8",
    )
    assert manager._log_confirms_trial(rerun, 159)
    assert not manager._log_confirms_trial(rerun, 159, require_snapshot=True)


def test_reexport_endpoint_requires_adapter_snapshots(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_module

    monkeypatch.setattr(main_module, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    job_id = "b" * 12
    request = main_module.JobRequest(
        model="org/m", heretic_channel="ara",
        use_ara_lora=True, export_strategy="adapter",
    )
    job = main_module.Job(
        id=job_id, status="completed", request=request.model_dump(),
        output_directory=str(tmp_path / "out"),
        created_at=main_module.utc_now(), heretic_channel="ara",
    )
    monkeypatch.setitem(main_module.manager.jobs, job_id, job)
    log = (
        "Running trial 3 of 4...\n"
        "  * KL divergence: 0.3500\n"
        "  * Refusals: 0/100\n"
    )
    monkeypatch.setattr(main_module.JobManager, "log_text", lambda self, jid: log)
    (tmp_path / "checkpoints" / job_id).mkdir(parents=True)
    client = TestClient(main_module.app)

    data = client.get(f"/api/jobs/{job_id}/trials").json()
    assert data["needs_snapshot"] is True
    assert data["front"][0]["has_snapshot"] is False

    response = client.post(f"/api/jobs/{job_id}/reexport", json={"front_indices": [0]})
    assert response.status_code == 409
    assert "snapshot" in response.json()["detail"]

    snapshot = main_module.ara_snapshot_file(job_id, 3)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"stub")
    queued = []

    def fake_create_reexports(self, source, selections):
        queued.append(selections)
        return []

    monkeypatch.setattr(main_module.JobManager, "create_reexports", fake_create_reexports)
    data = client.get(f"/api/jobs/{job_id}/trials").json()
    assert data["front"][0]["has_snapshot"] is True
    response = client.post(f"/api/jobs/{job_id}/reexport", json={"front_indices": [0]})
    assert response.status_code == 202
    assert queued and queued[0][0]["trial"] == 3


def test_lora_manager_lists_and_merges_outputs_adapters(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    exported = outputs / "model-heretic-abc123-t159"
    exported.mkdir()
    (exported / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    (exported / "adapter_model.safetensors").write_bytes(b"w")
    # A merged full model in outputs must not appear as an adapter.
    full = outputs / "full-model"
    full.mkdir()
    (full / "model.safetensors").write_bytes(b"weights")
    (full / "config.json").write_text("{}")
    base = outputs / "base-model"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"weights")
    (base / "config.json").write_text("{}")

    manager = LoRAManager(data, outputs)
    entries = manager.list()
    assert [(entry["name"], entry["source"]) for entry in entries] == [
        ("model-heretic-abc123-t159", "outputs")
    ]
    assert entries[0]["base_model"] == "org/base"
    assert entries[0]["format"] == "safetensors"

    captured = {}

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            captured["adapter"] = command[command.index("--adapter") + 1]
            staging = Path(command[command.index("--output") + 1])
            staging.mkdir(parents=True)
            (staging / "model.safetensors").write_bytes(b"merged")
            (staging / "config.json").write_text("{}")
            self.stdout = iter(["ok\n"])
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr("app.lora_manager.subprocess.Popen", FakeProcess)
    manager.start_merge(
        "model-heretic-abc123-t159", "base-model", "merged-out", outputs, source="outputs"
    )
    for _ in range(100):
        if manager.current.status in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert manager.current.status == "completed", manager.current.error
    assert captured["adapter"] == str(exported)
    assert (outputs / "merged-out" / "model.safetensors").read_bytes() == b"merged"

    # The library source must not resolve into the outputs directory.
    with pytest.raises(ValueError):
        manager.start_merge(
            "model-heretic-abc123-t159", "base-model", "merged-two", outputs
        )
