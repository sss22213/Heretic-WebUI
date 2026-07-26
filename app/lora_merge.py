# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge a PEFT LoRA adapter into a full model.

Runs as a subprocess (`python -u -m app.lora_merge`) so the heavy
torch/transformers/peft imports and the multi-GB weight load never happen
inside the web worker process.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def auxiliary_files(base: Path, output: Path) -> list[Path]:
    """Base-model files to copy alongside the merged weights.

    save_pretrained only writes config and weights; tokenizer, processor and
    chat template files must come from the base model. Weight shards and
    files already produced by save_pretrained are excluded.
    """
    results = []
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        if (output / path.name).exists():
            continue
        results.append(path)
    return results


def model_class_name(base: Path) -> str | None:
    try:
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        architectures = config.get("architectures", [])
        return architectures[0] if architectures and isinstance(architectures[0], str) else None
    except (OSError, ValueError, TypeError, IndexError):
        return None


def load_base_model(base: Path):
    import torch
    import transformers

    candidates = []
    architecture = model_class_name(base)
    if architecture and hasattr(transformers, architecture):
        # Instantiate the exact class from config.json so the module tree (and
        # therefore the PEFT target-module paths) matches the one the adapter
        # was trained against. Auto classes may resolve a multimodal config to
        # its text-only variant and silently drop weights.
        candidates.append(getattr(transformers, architecture))
    candidates.extend(
        [transformers.AutoModelForCausalLM, transformers.AutoModelForImageTextToText, transformers.AutoModel]
    )
    last_error: Exception | None = None
    for cls in candidates:
        try:
            print(f"載入基底模型（{cls.__name__}, bfloat16, CPU）…", flush=True)
            return cls.from_pretrained(
                base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True
            )
        except Exception as exc:  # noqa: BLE001 - try the next loader class
            last_error = exc
            print(f"{cls.__name__} 無法載入：{exc}", flush=True)
    raise RuntimeError(f"無法載入基底模型：{last_error}")


def merge(base: Path, adapter: Path, output: Path) -> None:
    from peft import PeftModel

    model = load_base_model(base)
    print("套用 LoRA adapter…", flush=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    print("合併權重（merge_and_unload）…", flush=True)
    model = model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    print(f"寫入合併後模型：{output}", flush=True)
    model.save_pretrained(output, safe_serialization=True)
    for path in auxiliary_files(base, output):
        print(f"複製附屬檔案：{path.name}", flush=True)
        shutil.copy2(path, output / path.name)
    print("合併完成", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="基底模型目錄")
    parser.add_argument("--adapter", type=Path, required=True, help="PEFT adapter 目錄")
    parser.add_argument("--output", type=Path, required=True, help="輸出目錄")
    args = parser.parse_args()
    try:
        merge(args.base, args.adapter, args.output)
    except Exception as exc:  # noqa: BLE001 - report to the task log
        print(f"合併失敗：{exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
