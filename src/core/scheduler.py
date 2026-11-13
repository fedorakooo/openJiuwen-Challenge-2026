"""Ordering of paid attempts and the proxy guard shared by all model sessions."""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from src.config import QUALIFICATION_MAX_TASKS
from src.core.budget import Ledger
from src.core.tasks import Outcome, Task, Verdict

logger = logging.getLogger(__name__)

START_CONCURRENCY = 3
BREAKER_PAUSE_S = 120.0
EXHAUSTION_SHARE = 0.9
_RAISE_AFTER_SUCCESSES = 10
_MIN_PROBABILITY = 0.01
_INFRASTRUCTURE = ("budget", "model call failed", "proxy", "agent stopped")


def is_qualification(task_count: int) -> bool:
    """The qualification run has at most a handful of tasks and all of them are feasible."""
    return 0 < task_count <= QUALIFICATION_MAX_TASKS


def density(task: Task, ledger: Ledger, tier: str) -> float:
    """Expected reward per output equivalent: reward x success probability / expected cost."""
    probability = max(_MIN_PROBABILITY, ledger.success_prior(task.task_type, tier))
    return task.reward * probability / max(1.0, ledger.expected_cost(task.task_type, tier))


def order_by_density(tasks: Sequence[Task], ledger: Ledger, tier: str = "T1") -> list[Task]:
    """Greedy order for paid attempts; never by reward alone."""
    return sorted(tasks, key=lambda task: (-density(task, ledger, tier), task.task_id))


def complaint_for_reasoning(outcome: Outcome, qualification: bool) -> str | None:
    """Why a cheaper tier's result deserves the reasoning tier, or None if it does not."""
    reason = outcome.reason or ""
    if reason.startswith(_INFRASTRUCTURE):
        return None  # the proxy or the budget failed, not the model: reasoning cannot help
    if outcome.verdict is Verdict.UNPARSED:
        return reason or "the answer was rejected"
    if outcome.verdict is Verdict.SOLVED:
        return reason if reason.startswith("needs review") else None
    return (reason or "the task was judged infeasible") if qualification else None


def worth_an_agent(task: Task, outcome: Outcome | None, rewards: Sequence[float]) -> bool:
    """The agent is the last resort: unsolved tasks whose reward is at least the median."""
    if outcome is None or outcome.verdict is not Verdict.UNPARSED:
        return False
    if (outcome.reason or "").startswith(_INFRASTRUCTURE):
        return False
    return task.reward >= statistics.median(rewards) if rewards else True


class ProxyGuard:
    """One circuit breaker and one adaptive concurrency limit for every model session."""

    def __init__(
        self,
        ledger: Ledger,
        max_concurrency: int = START_CONCURRENCY,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        now: Callable[[], float] = time.monotonic,
        can_wait: Callable[[float], bool] = lambda _seconds: True,
    ) -> None:
        self._ledger = ledger
        self._max = max(1, max_concurrency)
        self.limit = min(START_CONCURRENCY, self._max)
        self._sleep = sleep
        self._now = now
        self._can_wait = can_wait
        self._active = 0
        self._changed = asyncio.Condition()
        self._open_until = 0.0
        self._successes = 0
        self.dead_reason: str | None = None

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one of the currently allowed concurrent calls."""
        async with self._changed:
            while self._active >= self.limit:
                await self._changed.wait()
            self._active += 1
        try:
            yield
        finally:
            async with self._changed:
                self._active -= 1
                self._changed.notify_all()

    async def wait_until_closed(self) -> bool:
        """Sit out an open breaker; False if the run's deadline does not allow the wait."""
        remaining = self._open_until - self._now()
        if remaining <= 0:
            return True
        if not self._can_wait(remaining):
            return False
        await self._sleep(remaining)
        return True

    def success(self) -> None:
        self._successes += 1
        if self._successes >= _RAISE_AFTER_SUCCESSES and self.limit < self._max:
            self.limit += 1
            self._successes = 0
            logger.info("no rate limiting for a while: concurrency raised to %d", self.limit)

    def rate_limited(self) -> None:
        self.limit = max(1, self.limit // 2)
        self._successes = 0
        logger.warning("HTTP 429: concurrency lowered to %d", self.limit)

    def open_breaker(self) -> None:
        self._open_until = self._now() + BREAKER_PAUSE_S
        logger.warning("proxy keeps failing: no calls for %.0f s", BREAKER_PAUSE_S)

    def stop(self, reason: str) -> None:
        if self.dead_reason is None:
            self.dead_reason = reason
            logger.error("model tiers stopped: %s", reason)

    def failure(self) -> None:
        """A failure once most of the budget is committed means the proxy has cut us off."""
        if self._ledger.committed > EXHAUSTION_SHARE * self._ledger.budget:
            self.stop("budget presumably exhausted: the proxy fails above 90% of the ledger")
