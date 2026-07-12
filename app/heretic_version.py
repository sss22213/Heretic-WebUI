# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class HereticVersionManager:
    """Build and atomically activate patched Heretic A/B source slots."""

    def __init__(
        self,
        source_dir: Path,
        state_file: Path,
        patch_dir: Path | None = None,
        slots_dir: Path | None = None,
        upstream_url: str = "https://github.com/p-e-w/heretic.git",
        initial_ref: str | None = None,
    ) -> None:
        self.source_dir = source_dir.resolve()
        self.state_file = state_file
        self.patch_dir = patch_dir.resolve() if patch_dir else None
        self.slots_dir = (slots_dir or state_file.parent / "heretic_slots").resolve()
        self.lock = threading.RLock()
        self.remote = os.getenv("HERETIC_UPDATE_REMOTE", "origin")
        self.branch = os.getenv("HERETIC_UPDATE_BRANCH", "master")
        self.upstream_url = upstream_url
        self.initial_ref = initial_ref

    def _run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> str:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"無法執行 {' '.join(command[:2])}：{exc}") from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(message or f"指令結束碼：{result.returncode}")
        return result.stdout.strip()

    def _git_source(self, *arguments: str, timeout: int = 60) -> str:
        return self._run(
            ["git", "-C", str(self.source_dir), *arguments], timeout=timeout
        )

    def _ensure_source_repo(self) -> None:
        if (self.source_dir / ".git").exists():
            return
        if self.source_dir.exists() and any(self.source_dir.iterdir()):
            raise RuntimeError(
                f"Heretic Git cache 不是空目錄且缺少 .git：{self.source_dir}"
            )
        self.source_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.source_dir.with_name(
            f".{self.source_dir.name}.{uuid.uuid4().hex}.cloning"
        )
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            self._run(
                ["git", "clone", "--quiet", self.upstream_url, str(temporary)],
                timeout=300,
            )
            temporary.replace(self.source_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _git_slot(self, slot_dir: Path, *arguments: str) -> str:
        return self._run(["git", "-C", str(slot_dir), *arguments])

    def _load_state(self) -> dict:
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("schema") == 2:
                return value
        except (FileNotFoundError, OSError, ValueError, TypeError):
            pass
        return {}

    def _save_state(self, value: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_name(
            f".{self.state_file.name}.{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_file)

    def _patches(self) -> list[Path]:
        if self.patch_dir is None or not self.patch_dir.is_dir():
            return []
        return sorted(self.patch_dir.glob("*.patch"))

    def _commit_info(self, revision: str) -> dict:
        values = self._git_source(
            "show", "-s", "--format=%H%n%h%n%s%n%cI", revision
        ).splitlines()
        if len(values) < 4:
            raise RuntimeError("無法讀取 Heretic commit 資訊")
        return {
            "commit": values[0],
            "short_commit": values[1],
            "subject": values[2],
            "committed_at": values[3],
        }

    def _dependency_signature(self, directory: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        for filename in ("pyproject.toml", "uv.lock"):
            path = directory / filename
            digest.update(filename.encode())
            digest.update(b"\0")
            if path.is_file():
                digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _slot_path(self, name: str) -> Path:
        if name not in ("A", "B"):
            raise RuntimeError("無效的 Heretic slot")
        return self.slots_dir / name

    def _apply_patches(self, slot_dir: Path) -> list[dict[str, str]]:
        results = []
        for patch in self._patches():
            try:
                self._git_slot(slot_dir, "apply", "--check", str(patch))
            except RuntimeError as apply_error:
                try:
                    self._git_slot(
                        slot_dir, "apply", "--reverse", "--check", str(patch)
                    )
                except RuntimeError:
                    raise RuntimeError(
                        f"Patch 無法套用：{patch.name}：{apply_error}"
                    ) from apply_error
                results.append({"name": patch.name, "status": "included_upstream"})
                continue
            self._git_slot(slot_dir, "apply", str(patch))
            results.append({"name": patch.name, "status": "applied"})
        return results

    def _validate_slot(self, slot_dir: Path) -> None:
        source = slot_dir / "src"
        package = source / "heretic"
        if not package.is_dir():
            raise RuntimeError("更新內容缺少 src/heretic package")
        self._run(
            [sys.executable, "-m", "compileall", "-q", str(package)], timeout=120
        )
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(source) + (os.pathsep + existing if existing else "")
        imported = self._run(
            [
                sys.executable,
                "-c",
                "import pathlib, heretic.main; print(pathlib.Path(heretic.main.__file__).resolve())",
            ],
            timeout=120,
            env=env,
        )
        if not Path(imported).is_relative_to(slot_dir):
            raise RuntimeError("Smoke test 載入的不是待啟用 slot")

    def _build_slot(self, name: str, revision: str) -> dict:
        destination = self._slot_path(name)
        temporary = self.slots_dir / f".{name}.{uuid.uuid4().hex}.building"
        self.slots_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            self._run(
                [
                    "git", "clone", "--quiet", "--no-checkout", "--shared",
                    str(self.source_dir), str(temporary),
                ],
                timeout=180,
            )
            self._git_slot(temporary, "checkout", "--quiet", "--detach", revision)
            patch_results = self._apply_patches(temporary)
            self._validate_slot(temporary)
            info = self._commit_info(revision)
            metadata = {
                **info,
                "slot": name,
                "built_at": utc_now(),
                "path": str(destination),
                "patches": patch_results,
                "dependency_signature": self._dependency_signature(temporary),
            }
            shutil.rmtree(destination, ignore_errors=True)
            temporary.replace(destination)
            return metadata
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _bootstrap(self) -> dict:
        state = self._load_state()
        if state:
            active = state.get("active_slot")
            if active in ("A", "B") and self._slot_path(active).is_dir():
                return state
        self._ensure_source_repo()
        head = self._git_source("rev-parse", self.initial_ref or "HEAD")
        metadata = self._build_slot("A", head)
        state = {
            "schema": 2,
            "active_slot": "A",
            "previous_slot": None,
            "slots": {"A": metadata},
            "switched_at": utc_now(),
            "rebuild_required": False,
        }
        self._save_state(state)
        return state

    def runtime_info(self, slot: str | None = None) -> dict:
        with self.lock:
            state = self._bootstrap()
            selected = slot or state["active_slot"]
            metadata = state.get("slots", {}).get(selected)
            path = self._slot_path(selected)
            if not metadata or not path.is_dir():
                raise RuntimeError(f"Heretic slot {selected} 不存在")
            return {**metadata, "path": str(path)}

    def status(self, *, check_remote: bool = False) -> dict:
        with self.lock:
            try:
                state = self._bootstrap()
            except RuntimeError as exc:
                return {
                    "available": False,
                    "error": str(exc),
                }
            active_name = state["active_slot"]
            active = state["slots"][active_name]
            inactive_name = "B" if active_name == "A" else "A"
            inactive = state.get("slots", {}).get(inactive_name)
            result = {
                "available": True,
                **active,
                "active_slot": active_name,
                "inactive_slot": inactive,
                "remote": self.remote,
                "branch": self.branch,
                "dirty": False,
                "working_tree_dirty": False,
                "dirty_files": [],
                "managed_patches_applied": any(
                    item.get("status") == "applied" for item in active.get("patches", [])
                ),
                "managed_patches": active.get("patches", []),
                "rollback_available": bool(
                    state.get("previous_slot")
                    and state.get("slots", {}).get(state["previous_slot"])
                ),
                "previous_commit": (
                    state.get("slots", {}).get(state.get("previous_slot"), {}).get("commit")
                ),
                "previous_short_commit": (
                    state.get("slots", {}).get(state.get("previous_slot"), {}).get("short_commit")
                ),
                "updated_at": state.get("switched_at"),
                "rebuild_required": bool(state.get("rebuild_required")),
            }
            if check_remote:
                self._ensure_source_repo()
                reference = f"refs/heads/{self.branch}"
                output = self._git_source(
                    "ls-remote", self.remote, reference, timeout=30
                )
                if not output:
                    raise RuntimeError(f"遠端找不到 branch：{self.branch}")
                latest = output.split()[0]
                result["latest_commit"] = latest
                result["update_available"] = latest != active["commit"]
            return result

    def update(self) -> dict:
        with self.lock:
            state = self._bootstrap()
            self._ensure_source_repo()
            active_name = state["active_slot"]
            active = state["slots"][active_name]
            self._git_source(
                "fetch", "--no-tags", self.remote, self.branch, timeout=180
            )
            target = self._git_source("rev-parse", "FETCH_HEAD")
            if target == active["commit"]:
                return {
                    **self.status(),
                    "changed": False,
                    "message": "目前已是最新版本",
                }
            inactive_name = "B" if active_name == "A" else "A"
            try:
                candidate = self._build_slot(inactive_name, target)
            except Exception as exc:
                raise RuntimeError(
                    f"更新未啟用；active slot {active_name} 保持不變：{exc}"
                ) from exc
            rebuild_required = (
                active.get("dependency_signature")
                != candidate.get("dependency_signature")
            )
            new_state = {
                "schema": 2,
                "active_slot": inactive_name,
                "previous_slot": active_name,
                "slots": {**state.get("slots", {}), inactive_name: candidate},
                "switched_at": utc_now(),
                "rebuild_required": rebuild_required,
            }
            self._save_state(new_state)
            return {
                **self.status(),
                "changed": True,
                "message": f"Heretic 已切換至 slot {inactive_name}",
            }

    def rollback(self) -> dict:
        with self.lock:
            state = self._bootstrap()
            previous = state.get("previous_slot")
            if previous not in ("A", "B") or previous not in state.get("slots", {}):
                raise RuntimeError("沒有可退回的上一個版本")
            current = state["active_slot"]
            current_meta = state["slots"][current]
            previous_meta = state["slots"][previous]
            if not self._slot_path(previous).is_dir():
                raise RuntimeError("上一個 Heretic slot 已遺失")
            new_state = {
                **state,
                "active_slot": previous,
                "previous_slot": None,
                "switched_at": utc_now(),
                "rebuild_required": (
                    current_meta.get("dependency_signature")
                    != previous_meta.get("dependency_signature")
                ),
            }
            self._save_state(new_state)
            return {
                **self.status(),
                "changed": True,
                "message": f"已退回 Heretic slot {previous}",
            }
