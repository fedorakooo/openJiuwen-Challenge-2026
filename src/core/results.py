"""Atomic answer files, the optional aggregate file and the trajectory log."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.core.tasks import Task

logger = logging.getLogger(__name__)

ANSWER_FILE = "answer.txt"
AGGREGATE_FILE = "answers.json"
TRAJECTORY_FILE = "trajectory.jsonl"
DONE_MARKER = "DONE"
_INT_RE = re.compile(r"[+-]?\d+")
_INT_WITH_ZERO_FRACTION_RE = re.compile(r"([+-]?\d+)\.0*")


class AnswerFormatError(ValueError):
    """The value cannot be written in the format required by `check`."""


def _format_int(value: object) -> str:
    if isinstance(value, bool):
        raise AnswerFormatError("boolean is not an integer answer")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")) or value != int(value):
            raise AnswerFormatError("non-integral number")
        return str(int(value))
    if isinstance(value, str):
        text = value.strip()
        match = _INT_WITH_ZERO_FRACTION_RE.fullmatch(text)
        if match:
            text = match.group(1)
        if _INT_RE.fullmatch(text):
            return str(int(text))
    raise AnswerFormatError("not an integer")


def format_answer(value: object, check: str | None) -> str:
    """Render the bare answer value for the given checker, without the trailing newline."""
    if check == "unittest":
        return DONE_MARKER
    if check == "exact_int":
        return _format_int(value)
    if value is None or isinstance(value, bool):
        raise AnswerFormatError("no scalar value")
    text = str(value).strip()
    if not text or "\n" in text or "\r" in text:
        raise AnswerFormatError("answer must be one non-empty line")
    return text


def atomic_write_text(path: Path, text: str) -> None:
    """Write through a temp file in the same directory, fsync, then rename over the target."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def resolve_aggregate_path(out: str | None) -> Path | None:
    """Map `OUT`/`--out` to a file path: a directory gets `answers.json` inside it."""
    if not out or not out.strip():
        return None
    path = Path(out.strip())
    return path / AGGREGATE_FILE if path.is_dir() else path


class ResultWriter:
    """Publishes answers the moment they are known; never raises."""

    def __init__(self, aggregate_path: Path | None = None) -> None:
        self._aggregate_path = aggregate_path
        self._answers: dict[str, str] = {}

    @property
    def answers(self) -> dict[str, str]:
        return dict(self._answers)

    def write_answer(self, task: Task, value: object) -> bool:
        """Write `answer.txt` next to `task.json` and mirror it into `workspace/`."""
        try:
            answer = format_answer(value, task.check)
        except AnswerFormatError as exc:
            logger.warning("task %s: answer rejected (%s)", task.task_id, exc)
            return False

        targets = [task.task_dir / ANSWER_FILE]
        if task.workspace is not None:
            targets.append(task.workspace / ANSWER_FILE)
        written = 0
        for target in targets:
            try:
                atomic_write_text(target, answer + "\n")
                written += 1
            except OSError as exc:
                logger.error(
                    "task %s: cannot write %s/%s (%s)",
                    task.task_id,
                    target.parent.name,
                    target.name,
                    type(exc).__name__,
                )
        if written == 0:
            return False
        self._answers[task.task_id] = answer
        self._write_aggregate()
        return True

    def _write_aggregate(self) -> None:
        if self._aggregate_path is None:
            return
        try:
            payload = json.dumps(self._answers, ensure_ascii=False, indent=2, sort_keys=True)
            atomic_write_text(self._aggregate_path, payload + "\n")
        except (OSError, ValueError) as exc:
            logger.error("cannot write the aggregate file (%s)", type(exc).__name__)


class TrajectoryLog:
    """Append-only JSONL log; a failing disk never fails the run."""

    def __init__(self, path: Path | None) -> None:
        self._handle = None
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8", newline="\n")
        except OSError as exc:
            logger.error("trajectory log disabled (%s)", type(exc).__name__)

    @property
    def enabled(self) -> bool:
        return self._handle is not None

    def record(self, **fields: Any) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(json.dumps(fields, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            logger.error("trajectory record dropped (%s)", type(exc).__name__)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError:
            pass
        self._handle = None
