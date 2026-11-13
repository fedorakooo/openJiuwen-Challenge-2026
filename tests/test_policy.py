from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest

from src import cli
from src.config import Settings
from src.core.agent import Tier1
from src.core.budget import Ledger, Usage
from src.core.results import ResultWriter, TrajectoryLog
from src.core.runner import Clock, run_phase0, run_phase1
from src.core.scheduler import (
    BREAKER_PAUSE_S,
    ProxyGuard,
    density,
    is_qualification,
    order_by_density,
)
from src.core.tasks import Outcome, Task, Verdict, load_tasks
from src.foundation.llm import Completion, LLMError
from src.solvers import coding
from tests.conftest import ANSWER_TAIL, DONE_TAIL, MakeTask, load_one, write_files
from tests.test_solver_coding import BODY, GOOD_LIB, TESTS
from tests.test_solver_data import HEADER, ROWS

MODEL = "qwen3.8-27b-noreason"


def _phase0(tasks_dir: Path, solvers, **options):
    writer = ResultWriter()
    stats = run_phase0(
        load_tasks(tasks_dir),
        solvers,
        writer,
        TrajectoryLog(None),
        Clock(Settings()),
        threading.Event(),
        **options,
    )
    return stats, writer


# --- scheduler ----------------------------------------------------------------------------


def test_qualification_is_a_handful_of_tasks() -> None:
    assert [is_qualification(n) for n in (0, 1, 2, 5, 6, 100)] == [
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_paid_attempts_are_ordered_by_density_not_by_reward(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("coding_big", task_type="coding", reward=3)
    make_task("math_small", task_type="math", reward=2)
    make_task("search_tiny", task_type="search", reward=1)
    make_task("math_zero", task_type="math", reward=0)
    tasks = load_tasks(tasks_dir)
    ledger = Ledger(100_000)

    ordered = [task.task_id for task in order_by_density(tasks, ledger)]

    # 2*0.85/70 > 3*0.75/160 > 1*0.90/75 > 0
    assert ordered == ["math_small", "coding_big", "search_tiny", "math_zero"]
    by_id = {task.task_id: task for task in tasks}
    assert density(by_id["math_small"], ledger, "T1") == pytest.approx(2 * 0.85 / 70)

    ledger.observe("math", "T1", 2000.0)  # math turned out to be expensive in this run
    assert order_by_density(tasks, ledger)[0].task_id == "coding_big"


# --- proxy guard --------------------------------------------------------------------------


def test_guard_lowers_concurrency_on_429_and_raises_it_back_slowly() -> None:
    guard = ProxyGuard(Ledger(1000), max_concurrency=5)
    assert guard.limit == 3

    guard.rate_limited()
    assert guard.limit == 1
    guard.rate_limited()
    assert guard.limit == 1
    for _ in range(45):
        guard.success()
    assert guard.limit == 5

    assert ProxyGuard(Ledger(1000), max_concurrency=2).limit == 2


def test_guard_slots_bound_concurrent_calls() -> None:
    async def scenario() -> int:
        guard = ProxyGuard(Ledger(1000), max_concurrency=5)
        guard.rate_limited()
        guard.limit = 2
        active = peak = 0

        async def call() -> None:
            nonlocal active, peak
            async with guard.slot():
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(call() for _ in range(8)))
        return peak

    assert asyncio.run(scenario()) == 2


def test_breaker_makes_calls_wait_or_give_up_near_the_deadline() -> None:
    now = [100.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    patient = ProxyGuard(Ledger(1000), sleep=sleep, now=lambda: now[0])
    assert asyncio.run(patient.wait_until_closed()) and slept == []
    patient.open_breaker()
    assert asyncio.run(patient.wait_until_closed())
    assert slept == [BREAKER_PAUSE_S]

    hurried = ProxyGuard(Ledger(1000), sleep=sleep, now=lambda: now[0], can_wait=lambda _s: False)
    hurried.open_breaker()
    assert not asyncio.run(hurried.wait_until_closed())
    assert slept == [BREAKER_PAUSE_S]


def test_failure_above_ninety_percent_of_the_ledger_means_exhausted() -> None:
    ledger = Ledger(1000)
    guard = ProxyGuard(ledger)
    ledger.reserve(500)
    ledger.settle(500, Usage(0, 500))
    guard.failure()
    assert guard.dead_reason is None

    ledger.reserve(420)
    ledger.settle(420, Usage(0, 420))
    guard.failure()
    assert guard.dead_reason is not None and "exhausted" in guard.dead_reason


# --- unsure answers and qualification in phase 0 -------------------------------------------


def _solvers():
    return {
        "math": lambda _t: Outcome(Verdict.SOLVED, answer=7),
        "data_analysis": lambda _t: Outcome(Verdict.UNGROUNDED, answer=0, reason="value absent"),
        "search": lambda _t: Outcome(Verdict.UNGROUNDED, reason="entity absent"),
        "coding": lambda _t: Outcome(Verdict.UNPARSED, answer="DONE", reason="partial"),
    }


def _four_tasks(make_task: MakeTask) -> None:
    make_task("m", task_type="math")
    make_task("d", task_type="data_analysis")
    make_task("s", task_type="search", check="exact_str")
    make_task("c", task_type="coding", check="unittest")


def test_free_answers_are_written_and_infeasible_tasks_cost_nothing(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    _four_tasks(make_task)

    stats, writer = _phase0(tasks_dir, _solvers())

    assert writer.answers == {"m": "7", "d": "0", "c": "DONE"}
    assert not (tasks_dir / "s" / "answer.txt").exists()
    assert (stats.solved, stats.ungrounded, stats.unsure) == (1, 2, 2)
    assert [task.task_id for task in stats.deferred] == ["c"]  # only the unparsed one is paid for
    assert stats.verify == []


def test_unsure_answers_can_be_switched_off(tasks_dir: Path, make_task: MakeTask) -> None:
    _four_tasks(make_task)

    stats, writer = _phase0(tasks_dir, _solvers(), answer_when_unsure=False)

    assert writer.answers == {"m": "7"}
    assert stats.unsure == 0 and [task.task_id for task in stats.deferred] == ["c"]


def test_qualification_escalates_ungrounded_tasks_and_verifies_solved_ones(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    _four_tasks(make_task)

    stats, writer = _phase0(tasks_dir, _solvers(), qualification=True)

    assert writer.answers == {"m": "7", "d": "0", "c": "DONE"}
    assert sorted(task.task_id for task in stats.deferred) == ["c", "d", "s"]
    assert [task.task_id for task in stats.verify] == ["m"]


# --- phase 1 ------------------------------------------------------------------------------


def _phase1(tasks: list[Task], solve, writer: ResultWriter, **options) -> int:
    return asyncio.run(
        run_phase1(
            tasks,
            solve,
            writer,
            TrajectoryLog(None),
            Clock(Settings()),
            threading.Event(),
            3,
            60.0,
            **options,
        )
    )


def test_verification_never_overwrites_the_zero_token_answer(
    tasks_dir: Path, make_task: MakeTask, caplog: pytest.LogCaptureFixture
) -> None:
    make_task("agree")
    make_task("differ")
    tasks = load_tasks(tasks_dir)
    writer = ResultWriter()
    for task in tasks:
        writer.write_answer(task, 7)

    async def solve(task: Task) -> Outcome:
        return Outcome(Verdict.SOLVED, answer="7.0" if task.task_id == "agree" else 8)

    package_logger = logging.getLogger("src")  # does not propagate once the CLI configured it
    package_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("INFO", logger="src"):
            solved = _phase1([], solve, writer, verify=tasks)
    finally:
        package_logger.removeHandler(caplog.handler)

    assert solved == 0
    assert (tasks_dir / "differ" / "answer.txt").read_bytes() == b"7\n"
    assert "task agree: model agrees" in caplog.text
    assert "task differ: model DISAGREES" in caplog.text


def test_model_tier_free_answers_follow_the_same_policy(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("d", task_type="data_analysis")
    tasks = load_tasks(tasks_dir)

    async def solve(_task: Task) -> Outcome:
        return Outcome(Verdict.UNGROUNDED, answer=0, reason="value absent")

    assert _phase1(tasks, solve, ResultWriter(), answer_when_unsure=False) == 0
    assert not (tasks_dir / "d" / "answer.txt").exists()
    assert _phase1(tasks, solve, ResultWriter()) == 0
    assert (tasks_dir / "d" / "answer.txt").read_bytes() == b"0\n"


class _Scripted:
    name = "fake"

    def __init__(self, answers: list[str | Exception]) -> None:
        self._answers = answers
        self.systems: list[str] = []

    async def complete(self, *, model, system, user, max_tokens, stop=None, timeout=60.0):
        self.systems.append(system)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return Completion(answer, 0, "stop", Usage(100, 10), 0.01, model)


def _tier(answers: list[str | Exception], **options) -> tuple[Tier1, _Scripted]:
    async def no_sleep(_seconds: float) -> None:
        return None

    gateway = _Scripted(answers)
    tier = Tier1(gateway, Ledger(100_000), TrajectoryLog(None), MODEL, sleep=no_sleep, **options)
    return tier, gateway


def test_model_x_is_final_in_a_full_run_but_challenged_in_qualification(
    make_task: MakeTask,
) -> None:
    task = load_one(make_task("m", instruction="How many loaves remain of 9 after 2 are sold?"))

    tier, gateway = _tier(["X"])
    assert asyncio.run(tier.solve(task)).verdict is Verdict.UNGROUNDED
    assert len(gateway.systems) == 1

    tier, gateway = _tier(["X", "s=9-2"], qualification=True)
    outcome = asyncio.run(tier.solve(task))
    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, 7)
    assert "known to be answerable" not in gateway.systems[0]
    assert "known to be answerable" in gateway.systems[1]

    tier, gateway = _tier(["X", "X"], qualification=True)
    assert asyncio.run(tier.solve(task)).verdict is Verdict.UNGROUNDED


def test_breaker_opens_after_repeated_server_errors_and_stops_other_tasks(
    make_task: MakeTask,
) -> None:
    task = load_one(make_task("m", instruction="How many loaves remain of 9 after 2 are sold?"))
    broken = LLMError("server", 503, "unavailable")
    ledger = Ledger(100_000)

    async def no_sleep(_seconds: float) -> None:
        return None

    guard = ProxyGuard(ledger, sleep=no_sleep, can_wait=lambda _s: False)
    gateway = _Scripted([broken] * 4)
    tier = Tier1(gateway, ledger, TrajectoryLog(None), MODEL, sleep=no_sleep, guard=guard)

    first = asyncio.run(tier.solve(task))
    second = asyncio.run(tier.solve(task))

    assert first.reason == "model call failed: server" and len(gateway.systems) == 4
    assert second.reason == "proxy unavailable until the deadline"
    assert len(gateway.systems) == 4  # no call was made while the breaker is open


# --- real solvers under the policy ---------------------------------------------------------


def test_partial_coding_repair_is_kept_with_done_and_still_escalated(
    make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = GOOD_LIB.replace("s += x", "s -= x").replace(
        "return ' '.join(reversed(s.split()))", "return s.upper().title()"
    )
    half_fixed = broken.replace("s -= x", "s += x")
    task_dir = make_task(
        "c", task_type="coding", check="unittest", workspace=True, instruction=BODY + DONE_TAIL
    )
    write_files(task_dir / "workspace", {"mylib.py": broken, "test_mylib.py": TESTS})
    lib = task_dir / "workspace" / "mylib.py"

    monkeypatch.setenv("ANSWER_WHEN_UNSURE", "0")
    strict = coding.solve(load_one(task_dir))
    assert (strict.verdict, strict.answer) == (Verdict.UNPARSED, None)
    assert lib.read_text(encoding="utf-8") == broken

    monkeypatch.setenv("ANSWER_WHEN_UNSURE", "1")
    lenient = coding.solve(load_one(task_dir))
    assert (lenient.verdict, lenient.answer) == (Verdict.UNPARSED, "DONE")
    assert "best partial repair kept" in (lenient.reason or "")
    assert lib.read_text(encoding="utf-8") == half_fixed


@pytest.mark.usefixtures("isolated_process_state")
@pytest.mark.parametrize(("flag", "expected"), [("1", b"0\n"), ("0", None)])
def test_absent_filter_value_gets_a_free_zero_end_to_end(
    tasks_dir: Path,
    make_task: MakeTask,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    expected: bytes | None,
) -> None:
    question = (
        f"The file sales.csv has columns {HEADER} (6 rows). How many units of product 'gizmo' "
        "were sold in months 7 through 8 inclusive? Use bash with python3 to compute it."
    )
    for index in range(6):  # more than five tasks: a full run, not qualification
        task_dir = make_task(
            f"d{index}",
            task_type="data_analysis",
            workspace=True,
            instruction=question + ANSWER_TAIL,
        )
        write_files(task_dir / "workspace", {"sales.csv": "\n".join([HEADER, *ROWS]) + "\n"})
    monkeypatch.setenv("ANSWER_WHEN_UNSURE", flag)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    assert cli.main(["--tasks-dir", str(tasks_dir)]) == 0

    answer = tasks_dir / "d0" / "answer.txt"
    assert (answer.read_bytes() if answer.exists() else None) == expected
