from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pytest

from src.core.tasks import UNKNOWN_TYPE, Task, load_tasks
from tests.conftest import SECRET_EXPECTED, MakeTask


def test_loads_tasks_sorted_with_fields_from_json(tasks_dir: Path, make_task: MakeTask) -> None:
    make_task("b_dir", task_type="search", check="exact_str", reward=5, workspace=True)
    make_task("a_dir", task_id="math_0000")

    tasks = load_tasks(tasks_dir)

    assert [t.task_id for t in tasks] == ["math_0000", "b_dir"]
    first, second = tasks
    assert first.task_type == "math" and first.check == "exact_int" and first.reward == 2.0
    assert first.workspace is None
    assert first.task_dir == (tasks_dir / "a_dir").resolve()
    assert second.workspace == (tasks_dir / "b_dir" / "workspace").resolve()


def test_expected_never_reaches_the_task_or_the_log(
    tasks_dir: Path, make_task: MakeTask, caplog: pytest.LogCaptureFixture
) -> None:
    make_task("t1")
    (tasks_dir / "broken").mkdir()
    (tasks_dir / "broken" / "task.json").write_text(
        f'{{"expected": "{SECRET_EXPECTED}", ', encoding="utf-8"
    )

    with caplog.at_level(logging.DEBUG):
        tasks = load_tasks(tasks_dir)

    assert len(tasks) == 1
    assert "expected" not in {f.name for f in dataclasses.fields(Task)}
    assert SECRET_EXPECTED not in repr(tasks)
    assert SECRET_EXPECTED not in caplog.text


def test_skips_junk_and_malformed_entries(tasks_dir: Path, make_task: MakeTask) -> None:
    make_task("good")
    make_task("__MACOSX")
    make_task("._good")
    (tasks_dir / ".DS_Store").write_bytes(b"\x00\x01")
    (tasks_dir / "stray.json").write_text("{}", encoding="utf-8")
    (tasks_dir / "no_task_file").mkdir()
    (tasks_dir / "not_object").mkdir()
    (tasks_dir / "not_object" / "task.json").write_text("[1, 2]", encoding="utf-8")
    (tasks_dir / "no_instruction").mkdir()
    (tasks_dir / "no_instruction" / "task.json").write_text(
        json.dumps({"task_id": "x", "task_type": "math"}), encoding="utf-8"
    )

    assert [t.task_id for t in load_tasks(tasks_dir)] == ["good"]


def test_unknown_type_and_bad_fields_get_defaults(tasks_dir: Path, make_task: MakeTask) -> None:
    make_task("odd", task_type="Puzzle", check=None, reward="many", task_id=17)
    make_task("typeless", task_type=None, reward=True)

    odd, typeless = load_tasks(tasks_dir)

    assert odd.task_id == "odd" and odd.task_type == "puzzle" and odd.check is None
    assert odd.reward == 1.0
    assert typeless.task_type == UNKNOWN_TYPE and typeless.reward == 1.0


def test_missing_directory_yields_no_tasks(tmp_path: Path) -> None:
    assert load_tasks(tmp_path / "absent") == []


def test_loader_does_not_write_into_the_tasks_directory(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("t1", workspace=True)
    before = sorted(p.as_posix() for p in tasks_dir.rglob("*"))

    load_tasks(tasks_dir)

    assert sorted(p.as_posix() for p in tasks_dir.rglob("*")) == before
