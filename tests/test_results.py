from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.core import results
from src.core.results import (
    AnswerFormatError,
    ResultWriter,
    TrajectoryLog,
    atomic_write_text,
    format_answer,
    resolve_aggregate_path,
)
from src.core.tasks import load_tasks
from tests.conftest import MakeTask, list_files


@pytest.mark.parametrize(
    ("value", "check", "expected"),
    [
        (158, "exact_int", "158"),
        (158.0, "exact_int", "158"),
        ("158.0", "exact_int", "158"),
        (" -7 ", "exact_int", "-7"),
        ("007", "exact_int", "7"),
        ("Osaka", "exact_str", "Osaka"),
        (" Osaka \n", "exact_str", "Osaka"),
        (1983, "exact_str", "1983"),
        ("1983", "exact_str", "1983"),
        ("anything", "unittest", "DONE"),
        (None, "unittest", "DONE"),
        ("free text", None, "free text"),
    ],
)
def test_format_answer(value: object, check: str | None, expected: str) -> None:
    assert format_answer(value, check) == expected


@pytest.mark.parametrize(
    ("value", "check"),
    [
        (158.5, "exact_int"),
        ("12 apples", "exact_int"),
        ("1,752", "exact_int"),
        (True, "exact_int"),
        (None, "exact_int"),
        (float("nan"), "exact_int"),
        ("", "exact_str"),
        ("two\nlines", "exact_str"),
        (None, "exact_str"),
    ],
)
def test_format_answer_rejects(value: object, check: str) -> None:
    with pytest.raises(AnswerFormatError):
        format_answer(value, check)


def test_atomic_write_replaces_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "answer.txt"
    target.write_bytes(b"old\n")

    atomic_write_text(target, "42\n")

    assert target.read_bytes() == b"42\n"
    assert list_files(tmp_path) == ["answer.txt"]


def test_atomic_write_failure_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "answer.txt"
    target.write_bytes(b"old\n")

    def broken_replace(_src: object, _dst: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(results.os, "replace", broken_replace)
    with pytest.raises(PermissionError):
        atomic_write_text(target, "new\n")

    assert target.read_bytes() == b"old\n"
    assert list_files(tmp_path) == ["answer.txt"]


def test_answer_is_written_next_to_task_json_and_mirrored(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("with_ws", workspace=True)
    make_task("without_ws")
    with_ws, without_ws = load_tasks(tasks_dir)
    writer = ResultWriter()

    assert writer.write_answer(with_ws, 42)
    assert writer.write_answer(without_ws, "17.0")

    assert (with_ws.task_dir / "answer.txt").read_bytes() == b"42\n"
    assert (with_ws.task_dir / "workspace" / "answer.txt").read_bytes() == b"42\n"
    assert (without_ws.task_dir / "answer.txt").read_bytes() == b"17\n"
    assert not (without_ws.task_dir / "workspace").exists()
    assert writer.answers == {"with_ws": "42", "without_ws": "17"}


def test_existing_answer_is_overwritten_and_bad_value_keeps_it(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    task_dir = make_task("t1")
    (task_dir / "answer.txt").write_bytes(b"stale\n")
    (task,) = load_tasks(tasks_dir)
    writer = ResultWriter()

    assert not writer.write_answer(task, "not a number")
    assert (task_dir / "answer.txt").read_bytes() == b"stale\n"

    assert writer.write_answer(task, 5)
    assert (task_dir / "answer.txt").read_bytes() == b"5\n"


def test_falls_back_to_workspace_when_task_dir_is_not_writable(
    tasks_dir: Path, make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_task("t1", workspace=True)
    (task,) = load_tasks(tasks_dir)
    real_write = results.atomic_write_text

    def guarded(path: Path, text: str) -> None:
        if path.parent == task.task_dir:
            raise PermissionError("no CAP_DAC_OVERRIDE")
        real_write(path, text)

    monkeypatch.setattr(results, "atomic_write_text", guarded)
    writer = ResultWriter()

    assert writer.write_answer(task, 9)
    assert not (task.task_dir / "answer.txt").exists()
    assert (task.task_dir / "workspace" / "answer.txt").read_bytes() == b"9\n"


def test_write_failure_everywhere_is_reported_not_raised(
    tasks_dir: Path, make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_task("t1")
    (task,) = load_tasks(tasks_dir)

    def denied(_path: Path, _text: str) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(results, "atomic_write_text", denied)
    writer = ResultWriter()

    assert not writer.write_answer(task, 9)
    assert writer.answers == {}


def test_aggregate_is_rewritten_after_every_answer(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path
) -> None:
    make_task("a")
    make_task("b", task_type="search", check="exact_str")
    first, second = load_tasks(tasks_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aggregate = resolve_aggregate_path(str(out_dir))
    assert aggregate == out_dir / "answers.json"
    writer = ResultWriter(aggregate)

    writer.write_answer(first, 1)
    assert json.loads(aggregate.read_text(encoding="utf-8")) == {"a": "1"}
    writer.write_answer(second, "Osaka")
    assert json.loads(aggregate.read_text(encoding="utf-8")) == {"a": "1", "b": "Osaka"}


def test_aggregate_failure_does_not_lose_the_answer(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path
) -> None:
    make_task("a")
    (task,) = load_tasks(tasks_dir)
    writer = ResultWriter(tmp_path / "missing_dir" / "answers.json")

    assert writer.write_answer(task, 3)
    assert (task.task_dir / "answer.txt").read_bytes() == b"3\n"


def test_resolve_aggregate_path(tmp_path: Path) -> None:
    assert resolve_aggregate_path(None) is None
    assert resolve_aggregate_path("  ") is None
    assert resolve_aggregate_path(str(tmp_path / "file.json")) == tmp_path / "file.json"


def test_trajectory_log_appends_json_lines_and_survives_a_bad_path(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "trajectory.jsonl"
    log = TrajectoryLog(path)
    log.record(task_id="a", verdict="unparsed", cost=0)
    log.record(task_id="b", path=Path("x"))
    log.close()
    log.record(task_id="after close")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["task_id"] for line in lines] == ["a", "b"]

    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    broken = TrajectoryLog(blocker / "sub" / "t.jsonl")
    broken.record(task_id="c")
    broken.close()
    assert os.path.isfile(blocker)
