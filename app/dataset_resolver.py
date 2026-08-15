# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolve a Hugging Face dataset *config* into a local prompt file.

Heretic's dataset loader has no `config` field, so a dataset that ships
multiple configs with no default (e.g. wangzhang/abliterix-datasets) cannot
be loaded by ID. This runs as a subprocess (`python -u -m app.dataset_resolver`)
that downloads one config/split with the canonical `datasets` loader — which
resolves configs and splits correctly — and writes the chosen column out as
one prompt per line. Heretic then reads that file via its plain-text path,
which is unambiguous (no split/column inference).

The heavy `datasets` import is kept inside main() so importing this module
(and unit-testing the pure helpers) does not require datasets to be present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

# Column names that typically hold the prompt text, best first.
PROMPT_COLUMN_HINTS = ("prompt", "text", "instruction", "question", "content", "message", "input")


def _first_rows_columns(repo_id: str, config: str, split: str) -> list[str]:
    """Best-effort column names from the datasets-server (no download)."""
    query = urllib.parse.urlencode({"dataset": repo_id, "config": config, "split": split})
    url = f"https://datasets-server.huggingface.co/first-rows?{query}"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:  # noqa: S310 - fixed HTTPS host
            payload = json.loads(response.read())
        return [str(feature["name"]) for feature in payload.get("features", []) if feature.get("name")]
    except Exception:  # noqa: BLE001 - columns are a convenience; the UI degrades gracefully
        return []


def suggest_prompt_column(columns: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for hint in PROMPT_COLUMN_HINTS:
        for lower, original in lowered.items():
            if hint == lower or hint in lower:
                return original
    return columns[0] if columns else None


def list_dataset_configs(repo_id: str, token: str | None = None) -> dict:
    """List a dataset's configs (+ best-effort splits/columns) without downloading.

    Configs come from the dataset card metadata, which is authoritative even
    for datasets the datasets-server has not converted; columns are fetched
    best-effort and may be empty for private/unconverted datasets.
    """
    info = HfApi().dataset_info(repo_id, token=token)
    card_data = getattr(info, "card_data", None)
    raw_configs = (card_data.get("configs") if card_data else None) or []

    configs: list[dict] = []
    all_columns: list[str] = []
    for entry in raw_configs:
        name = entry.get("config_name")
        if not name:
            continue
        splits = []
        for data_file in entry.get("data_files", []) or []:
            split = data_file.get("split") if isinstance(data_file, dict) else None
            if split and split not in splits:
                splits.append(split)
        columns = _first_rows_columns(repo_id, name, splits[0]) if splits else []
        all_columns.extend(columns)
        configs.append({"name": name, "splits": splits or ["train"], "columns": columns})

    return {
        "configs": configs,
        "suggested_column": suggest_prompt_column(all_columns),
    }


def clean_prompt(value: object) -> str:
    """Flatten one dataset cell into a single trimmed line.

    Newlines would split one prompt across several lines in the output file,
    so they are collapsed to spaces.
    """
    text = "" if value is None else str(value)
    return " ".join(text.split())


def write_prompts(rows: list, output: Path) -> int:
    """Write non-empty prompts one per line; return how many were written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = clean_prompt(row)
            if line:
                handle.write(line + "\n")
                written += 1
    if written == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("解析後沒有任何非空白的 prompt，請確認欄位名稱是否正確")
    temporary.replace(output)
    return written


def resolve(repo: str, config: str, split: str, column: str, output: Path) -> int:
    from datasets import load_dataset

    print(f"載入 Hugging Face dataset：{repo}（config={config}, split={split}, column={column}）", flush=True)
    dataset = load_dataset(repo, config, split=split)
    if column not in dataset.column_names:
        raise ValueError(
            f"欄位「{column}」不存在，可用欄位：{', '.join(dataset.column_names)}"
        )
    written = write_prompts(list(dataset[column]), output)
    print(f"已寫入 {written} 筆 prompt：{output}", flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--config", required=True, help="Dataset config 名稱")
    parser.add_argument("--split", required=True, help="Split 名稱（不含切片，例如 train）")
    parser.add_argument("--column", required=True, help="要取用的欄位")
    parser.add_argument("--output", type=Path, required=True, help="輸出的純文字 prompt 檔")
    args = parser.parse_args()
    try:
        resolve(args.repo, args.config, args.split, args.column, args.output)
    except Exception as exc:  # noqa: BLE001 - report to the task log
        print(f"解析資料集失敗：{exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
