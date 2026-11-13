"""Run clock, stop flag and the zero-token phase over all tasks."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from src.config import DEADLINE_SLACK_S, Settings
from src.core.results import AnswerFormatError, ResultWriter, TrajectoryLog, format_answer
from src.core.tasks import Outcome, Task, Verdict

logger = logging.getLogger(__name__)

Solver = Callable[[Task], Outcome]
_STOP_POLL_S = 0.5


class Clock:
    """Monotonic time since process start with the internal deadline."""

    def __init__(self, settings: Settings, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._start = now()
        self._deadline_s = settings.deadline_s
        self._phase0_cap_s = min(settings.phase0_cap_s, settings.deadline_s)

    def elapsed(self) -> float:
        return self._now() - self._start

    def remaining(self) -> float:
        return self._deadline_s - self.elapsed()

    def can_start(self, attempt_timeout_s: float) -> bool:
        """An attempt may start only if its timeout plus the slack fits before the deadline."""
        return self.remaining() >= attempt_timeout_s + DEADLINE_SLACK_S

    def phase0_expired(self) -> bool:
        return self.elapsed() >= self._phase0_cap_s or self.remaining() <= 0


@dataclass
class RunStats:
    """Counters of one run."""

    total: int = 0
    solved: int = 0
    ungrounded: int = 0
    unsure: int = 0
    deferred: list[Task] = field(default_factory=list)
    verify: list[Task] = field(default_factory=list)


def _solve_one(task: Task, solvers: Mapping[str, Solver]) -> Outcome:
    solver = solvers.get(task.task_type)
    if solver is None:
        return Outcome(Verdict.UNPARSED, reason="no solver")
    try:
        return solver(task)
    except Exception as exc:  # one task must never take the run down
        logger.exception("task %s: solver crashed", task.task_id)
        return Outcome(Verdict.UNPARSED, reason=f"solver error: {type(exc).__name__}")


def run_phase0(
    tasks: Sequence[Task],
    solvers: Mapping[str, Solver],
    writer: ResultWriter,
    trajectory: TrajectoryLog,
    clock: Clock,
    stop: threading.Event,
    answer_when_unsure: bool = True,
    qualification: bool = False,
) -> RunStats:
    """Try the free solver of every task; answers go to disk as soon as they exist."""
    stats = RunStats(total=len(tasks))
    for index, task in enumerate(tasks):
        if stop.is_set() or clock.phase0_expired():
            reason = "stop requested" if stop.is_set() else "phase 0 time cap"
            logger.warning("phase 0 ends early (%s), %d tasks left", reason, len(tasks) - index)
            stats.deferred.extend(tasks[index:])
            break
        try:
            started = clock.elapsed()
            outcome = _solve_one(task, solvers)
            published = False
            if outcome.verdict is Verdict.SOLVED:
                published = writer.write_answer(task, outcome.answer)
                if published:
                    stats.solved += 1
                    if qualification:
                        stats.verify.append(task)
            elif outcome.answer is not None and answer_when_unsure:
                # A defined value costs nothing and a wrong answer scores like no answer.
                published = writer.write_answer(task, outcome.answer)
                stats.unsure += published
            if outcome.verdict is Verdict.UNGROUNDED:
                stats.ungrounded += 1
            # In qualification every task is feasible, so "ungrounded" is our own mistake.
            escalate = outcome.verdict is Verdict.UNPARSED or (
                outcome.verdict is Verdict.UNGROUNDED and qualification
            )
            if escalate or (outcome.verdict is Verdict.SOLVED and not published):
                stats.deferred.append(task)
            trajectory.record(
                task_id=task.task_id,
                task_type=task.task_type,
                reward=task.reward,
                tier="T0",
                cost=0,
                latency_s=round(clock.elapsed() - started, 4),
                verdict=outcome.verdict.value,
                published=published,
                reason=outcome.reason,
                evidence=outcome.evidence,
            )
        except Exception:  # keep going with the next task
            logger.exception("task %s: unexpected failure", task.task_id)
            stats.deferred.append(task)
    return stats


async def run_phase1(
    tasks: Sequence[Task],
    solve: Callable[[Task], Awaitable[Outcome]],
    writer: ResultWriter,
    trajectory: TrajectoryLog,
    clock: Clock,
    stop: threading.Event,
    concurrency: int,
    attempt_timeout_s: float,
    answer_when_unsure: bool = True,
    verify: Sequence[Task] = (),
    tier: str = "T1",
    outcomes: dict[str, Outcome] | None = None,
    disagreements: dict[str, str] | None = None,
) -> int:
    """Send tasks to the model tier in the given order; return the number solved.

    `verify` tasks already have a zero-token answer on disk (qualification mode): the model
    answers them independently and a disagreement is only reported, never written.
    """
    semaphore = asyncio.Semaphore(concurrency)
    solved = 0

    async def one(task: Task, check_only: bool) -> None:
        nonlocal solved
        async with semaphore:
            if stop.is_set() or not clock.can_start(attempt_timeout_s):
                return
            try:
                outcome = await solve(task)
                published = False
                if not check_only and outcomes is not None:
                    outcomes[task.task_id] = outcome
                if check_only:
                    theirs = _report_agreement(task, outcome, writer)
                    if theirs is not None and disagreements is not None:
                        disagreements[task.task_id] = theirs
                elif outcome.verdict is Verdict.SOLVED:
                    published = writer.write_answer(task, outcome.answer)
                    solved += published
                elif outcome.answer is not None and answer_when_unsure:
                    writer.write_answer(task, outcome.answer)
                if not check_only and (not published or outcome.reason):
                    logger.info(
                        "task %s: %s %s, published=%s, reason: %s, evidence: %s",
                        task.task_id,
                        tier,
                        outcome.verdict.value,
                        published,
                        outcome.reason,
                        (outcome.evidence or "")[:200],
                    )
                trajectory.record(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    reward=task.reward,
                    tier=tier,
                    step="verification" if check_only else "result",
                    verdict=outcome.verdict.value,
                    published=published,
                    reason=outcome.reason,
                    evidence=outcome.evidence,
                )
            except Exception:  # one task must never take the run down
                logger.exception("task %s: model tier failed", task.task_id)

    jobs = [one(task, False) for task in tasks] + [one(task, True) for task in verify]
    work = asyncio.gather(*jobs)
    try:
        while not work.done():
            await asyncio.wait({work}, timeout=_STOP_POLL_S)
            if stop.is_set() and not work.done():
                logger.warning("stop requested: cancelling in-flight model calls")
                work.cancel()
        await work
    except asyncio.CancelledError:
        pass
    return solved


def _report_agreement(task: Task, outcome: Outcome, writer: ResultWriter) -> str | None:
    """Compare a model answer with the zero-token one on disk; return it if they differ."""
    ours = writer.answers.get(task.task_id)
    if outcome.verdict is not Verdict.SOLVED:
        logger.info("task %s: verification gave no answer (%s)", task.task_id, outcome.reason)
        return None
    try:
        theirs = format_answer(outcome.answer, task.check)
    except AnswerFormatError:
        return None
    if theirs == ours:
        logger.info("task %s: model agrees with the zero-token answer", task.task_id)
        return None
    # The zero-token answer stays unless the reasoning tier sides with the model.
    logger.warning("task %s: model DISAGREES with the zero-token answer", task.task_id)
    return theirs


async def arbitrate(
    tasks: Sequence[Task],
    disagreements: dict[str, str],
    solve: Callable[[Task], Awaitable[Outcome]],
    writer: ResultWriter,
    stop: threading.Event,
) -> int:
    """Qualification mode: replace a zero-token answer only if two model tiers agree on another."""
    replaced = 0
    for task in tasks:
        model_answer = disagreements.get(task.task_id)
        if model_answer is None or stop.is_set():
            continue
        backed = False
        try:
            outcome = await solve(task)
            if outcome.verdict is Verdict.SOLVED:
                backed = format_answer(outcome.answer, task.check) == model_answer
        except Exception:  # an arbiter that fails means "keep ours"
            logger.exception("task %s: arbiter failed", task.task_id)
        if backed:
            replaced += writer.write_answer(task, model_answer)
            logger.warning("task %s: both model tiers agree, our answer replaced", task.task_id)
        else:
            logger.info("task %s: arbiter does not back the model, kept ours", task.task_id)
    return replaced
