"""Task model, verdicts and the task directory loader; standard library only."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

TASK_FILE = "task.json"
WORKSPACE_DIR = "workspace"
UNKNOWN_TYPE = "unknown"
_JUNK_NAMES = frozenset({"__MACOSX", ".DS_Store"})
# Only these keys are ever copied out of task.json; `expected` is deliberately not among them.
_ALLOWED_KEYS = ("task_id", "task_type", "instruction", "reward", "check")
_MAX_TASK_FILE_BYTES = 4 * 1024 * 1024


class Verdict(Enum):
    """Outcome of a solver attempt."""

    SOLVED = "solved"  # answer computed, evidence recorded
    UNPARSED = "unparsed"  # the solver did not understand the wording -> next tier
    UNGROUNDED = "ungrounded"  # wording understood, required fact is absent


@dataclass(frozen=True)
class Outcome:
    """Verdict of one solver with the answer value when there is one."""

    verdict: Verdict
    answer: object | None = None
    reason: str | None = None
    evidence: str | None = None


# Tool hints and the answer-file reminder that close every instruction. They are cut before
# parsing; `echo 42` inside them is an example command, never an answer.
_TAIL_SENTENCES = tuple(
    re.compile(r"\s*" + pattern + r"\s*\Z")
    for pattern in (
        r"Compute it with the bash tool \(python3 -c\)\.",
        r"Use the file tools or bash \(ls, grep, cat\) to find it\.",
        r"Use bash with python3 to compute it\.",
        r"Work only inside the current directory\.",
        r"When you are done, write ONLY (?:the final answer|the word DONE) to the file "
        r"answer\.txt in the current directory \(for example with bash: echo \d+ > answer\.txt\)\.",
    )
)


def strip_boilerplate(instruction: str) -> str:
    """Cut the known service sentences from the end of an instruction."""
    text = instruction.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _TAIL_SENTENCES:
            match = pattern.search(text)
            if match:
                text = text[: match.start()]
                changed = True
    return text.strip()


@dataclass(frozen=True)
class Task:
    """One task directory; never carries the reference answer."""

    task_id: str
    task_type: str
    instruction: str
    reward: float
    check: str | None
    task_dir: Path
    workspace: Path | None


def _is_junk(name: str) -> bool:
    return name in _JUNK_NAMES or name.startswith("._")


def _parse_reward(raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return 1.0
    value = float(raw)
    return value if math.isfinite(value) and value >= 0 else 1.0


def _load_one(task_dir: Path) -> Task | None:
    task_file = task_dir / TASK_FILE
    if not task_file.is_file():
        return None
    if task_file.stat().st_size > _MAX_TASK_FILE_BYTES:
        logger.warning("skip %s: task.json is too large", task_dir.name)
        return None
    try:
        raw = json.loads(task_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning("skip %s: unreadable task.json (%s)", task_dir.name, type(exc).__name__)
        return None
    if not isinstance(raw, dict):
        logger.warning("skip %s: task.json is not an object", task_dir.name)
        return None
    fields = {key: raw.get(key) for key in _ALLOWED_KEYS}
    del raw

    instruction = fields["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        logger.warning("skip %s: no instruction", task_dir.name)
        return None
    task_id = fields["task_id"]
    task_type = fields["task_type"]
    check = fields["check"]
    workspace = task_dir / WORKSPACE_DIR
    return Task(
        task_id=task_id if isinstance(task_id, str) and task_id.strip() else task_dir.name,
        task_type=(
            task_type.strip().lower()
            if isinstance(task_type, str) and task_type.strip()
            else UNKNOWN_TYPE
        ),
        instruction=instruction,
        reward=_parse_reward(fields["reward"]),
        check=check.strip().lower() if isinstance(check, str) and check.strip() else None,
        task_dir=task_dir,
        workspace=workspace if workspace.is_dir() else None,
    )


def load_tasks(tasks_dir: Path) -> list[Task]:
    """Return tasks found in the immediate subdirectories of `tasks_dir`, sorted by name."""
    try:
        root = tasks_dir.resolve()
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        logger.error("cannot list tasks directory: %s", type(exc).__name__)
        return []

    tasks: list[Task] = []
    for entry in entries:
        if _is_junk(entry.name):
            continue
        try:
            if not entry.is_dir() or entry.resolve().parent != root:
                continue
            task = _load_one(entry)
        except OSError as exc:
            logger.warning("skip %s: %s", entry.name, type(exc).__name__)
            continue
        if task is not None:
            tasks.append(task)
    return tasks
