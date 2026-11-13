"""Shared fixtures: synthetic task directories, never the organisers' samples."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from src.core.tasks import Task, load_tasks

SECRET_EXPECTED = "expected-value-must-never-leak"

MakeTask = Callable[..., Path]


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    root.mkdir()
    return root


@pytest.fixture
def make_task(tasks_dir: Path) -> MakeTask:
    def build(
        name: str,
        *,
        task_type: str = "math",
        check: str | None = "exact_int",
        reward: object = 2,
        workspace: bool = False,
        **extra: object,
    ) -> Path:
        task_dir = tasks_dir / name
        task_dir.mkdir()
        payload: dict[str, object] = {
            "task_id": name,
            "task_type": task_type,
            "instruction": f"Instruction of {name}.",
            "reward": reward,
            "check": check,
            "expected": SECRET_EXPECTED,
        }
        payload.update(extra)
        (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")
        if workspace:
            (task_dir / "workspace").mkdir()
        return task_dir

    return build


@pytest.fixture
def isolated_process_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Let `cli.main` change cwd, env and tempfile state, then restore all of it."""
    saved_env = dict(os.environ)
    saved_tempdir = tempfile.tempdir
    monkeypatch.chdir(tmp_path)
    for name in ("TASKS", "OUT", "BUDGET", "RUN_DEADLINE_S", "PHASE0_CAP_S"):
        monkeypatch.delenv(name, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    tempfile.tempdir = saved_tempdir


def list_files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


ANSWER_TAIL = (
    " Work only inside the current directory. When you are done, write ONLY the final answer to "
    "the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)
DONE_TAIL = (
    " Work only inside the current directory. When you are done, write ONLY the word DONE to "
    "the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)


def write_files(root: Path, files: dict[str, str]) -> None:
    """Create text files (LF line endings) under `root`."""
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))


def load_one(task_dir: Path) -> Task:
    (task,) = [t for t in load_tasks(task_dir.parent) if t.task_dir == task_dir.resolve()]
    return task
