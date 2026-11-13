"""Offline judge, dev only. The single place in this repository allowed to read `expected`.

Usage:
    python scripts/check.py <answers_dir> [--reference <dir>] [--by-family] [--quiet] [--json F]

`answers_dir` is a copy of the tasks where the harness wrote its answers. `--reference` holds
the pristine tasks with `expected` (default: `answers_dir` itself): the organisers' samples or
the `--ref` directory of scripts/gen_tasks.py. For coding the judge also runs every
`hidden_test_*.py` found in the reference workspace, like the organisers' wider hidden tests.

Infeasible reference tasks (`"feasible": false`) carry no reward; the judge only counts how
many of them received an answer. Exit code is 0 only if every feasible task is correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

ANSWER_FILE = "answer.txt"
TEST_TIMEOUT_S = 60


@dataclass
class Verdict:
    """Judgement of one task."""

    task_id: str
    task_type: str
    family: str
    reward: float
    feasible: bool
    answered: bool
    correct: bool
    note: str


def _read_answer(task_dir: Path) -> str | None:
    path = task_dir / ANSWER_FILE
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _run_tests(directory: Path, names: list[str]) -> str | None:
    for name in names:
        try:
            proc = subprocess.run(
                [sys.executable, "-B", name],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"{name} timed out"
        if proc.returncode != 0:
            return f"{name} fails"
    return None


def _check_unittest(task_dir: Path, reference_dir: Path) -> tuple[bool, str]:
    workspace = task_dir / "workspace"
    reference = reference_dir / "workspace"
    visible = sorted(reference.glob("test_*.py"))
    hidden = sorted(reference.glob("hidden_test_*.py"))
    if not visible:
        return False, "no test files in the reference"
    for test in visible:
        candidate = workspace / test.name
        if not candidate.is_file() or _digest(candidate) != _digest(test):
            return False, f"{test.name} was modified"
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "workspace"
        shutil.copytree(workspace, copy, ignore=shutil.ignore_patterns("__pycache__"))
        failure = _run_tests(copy, [test.name for test in visible])
        if failure:
            return False, failure
        for test in hidden:
            shutil.copyfile(test, copy / test.name)
        failure = _run_tests(copy, [test.name for test in hidden])
        if failure:
            return False, f"visible tests pass, {failure}"
    return True, "tests pass" + (" (visible and hidden)" if hidden else "")


def check_task(task_dir: Path, reference_dir: Path) -> Verdict:
    """Judge one task against its reference."""
    spec = json.loads((reference_dir / "task.json").read_text(encoding="utf-8"))
    answer = _read_answer(task_dir)
    verdict = Verdict(
        task_id=reference_dir.name,
        task_type=str(spec.get("task_type", "?")),
        family=str(spec.get("family", "sample")),
        reward=float(spec.get("reward", 0)),
        feasible=bool(spec.get("feasible", True)),
        answered=answer is not None,
        correct=False,
        note="",
    )
    if not verdict.feasible:
        verdict.note = "infeasible, answered" if verdict.answered else "infeasible, skipped"
        return verdict
    if answer is None:
        verdict.note = "no answer.txt"
        return verdict
    check = spec.get("check")
    if check == "unittest":
        # `expected` is only a marker here; the instruction asks for the word DONE.
        if answer != "DONE":
            verdict.note = "answer is not DONE"
        else:
            verdict.correct, verdict.note = _check_unittest(task_dir, reference_dir)
        return verdict
    expected = str(spec["expected"]).strip()
    if check == "exact_int":
        try:
            verdict.correct = int(answer) == int(expected)
        except ValueError:
            verdict.correct = False
    else:
        verdict.correct = answer == expected
    verdict.note = "match" if verdict.correct else "wrong answer"
    return verdict


def _summary(verdicts: list[Verdict], key: str) -> list[str]:
    groups: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        name = verdict.task_type if key == "type" else f"{verdict.task_type}/{verdict.family}"
        groups.setdefault(name, []).append(verdict)
    lines = []
    for name in sorted(groups):
        group = groups[name]
        feasible = [v for v in group if v.feasible]
        correct = [v for v in feasible if v.correct]
        wrong = [v for v in feasible if v.answered and not v.correct]
        infeasible = [v for v in group if not v.feasible]
        lines.append(
            f"{name:<40} solved {len(correct):3d}/{len(feasible):<3d} "
            f"reward {sum(v.reward for v in correct):5g}/{sum(v.reward for v in feasible):<5g} "
            f"wrong {len(wrong):3d}  no answer {len(feasible) - len(correct) - len(wrong):3d}  "
            f"infeasible answered {sum(v.answered for v in infeasible)}/{len(infeasible)}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("answers_dir", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--by-family", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="print only failures and totals")
    parser.add_argument("--json", type=Path, default=None, help="write all verdicts to a file")
    args = parser.parse_args()
    reference_root = args.reference or args.answers_dir

    verdicts: list[Verdict] = []
    for reference_dir in sorted(p.parent for p in reference_root.glob("*/task.json")):
        try:
            verdict = check_task(args.answers_dir / reference_dir.name, reference_dir)
        except (OSError, KeyError, ValueError) as exc:
            verdict = Verdict(reference_dir.name, "?", "?", 0.0, True, False, False, "")
            verdict.note = f"judge error: {type(exc).__name__}"
        verdicts.append(verdict)
        failed = verdict.feasible and not verdict.correct
        if failed or not args.quiet:
            mark = "FAIL" if failed else ("OK  " if verdict.feasible else "--  ")
            print(f"{mark} {verdict.task_id:<24} reward={verdict.reward:g}  {verdict.note}")

    feasible = [v for v in verdicts if v.feasible]
    correct = [v for v in feasible if v.correct]
    wrong = [v for v in feasible if v.answered and not v.correct]
    infeasible = [v for v in verdicts if not v.feasible]
    print()
    for line in _summary(verdicts, "family" if args.by_family else "type"):
        print(line)
    print(
        f"\nsolved {len(correct)}/{len(feasible)}, "
        f"reward {sum(v.reward for v in correct):g}/{sum(v.reward for v in feasible):g}, "
        f"wrong answers {len(wrong)}, "
        f"infeasible answered {sum(v.answered for v in infeasible)}/{len(infeasible)}"
    )
    if args.json:
        payload = json.dumps([asdict(v) for v in verdicts], ensure_ascii=False, indent=1)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0 if feasible and len(correct) == len(feasible) else 1


if __name__ == "__main__":
    sys.exit(main())
