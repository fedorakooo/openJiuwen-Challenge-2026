"""Tier T1: one no-reasoning call translates the task into a tiny program; local code runs it.

Every step is a plain function so that a failure inside the SDK can never cost a task more
than the single call that failed.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from src import prompts
from src.core.budget import Ledger
from src.core.results import TrajectoryLog
from src.core.scheduler import ProxyGuard
from src.core.tasks import Outcome, Task, Verdict, strip_boilerplate
from src.foundation.llm import (
    NOREASON_TIMEOUT_S,
    REASON_TIMEOUT_S,
    Completion,
    Gateway,
    LLMError,
)
from src.solvers import coding, data, search
from src.solvers import math as math_solver
from src.solvers.sqlworld import (
    SqlError,
    SqlWorld,
    clean_sql,
    normalise,
    run_select,
    string_literals,
)

logger = logging.getLogger(__name__)

TIER = "T1"
_ATTEMPTS = 2
_MATH_MAX_TOKENS = 96
_DATA_MAX_TOKENS = 120
_SEARCH_MAX_TOKENS = 160
_CODING_MAX_TOKENS = 200
_SHORT_MAX_TOKENS = 60
_REASONING_MAX_TOKENS = 1400
_MAX_FRAGMENT_CHARS = 18_000
_RATE_LIMIT_RETRIES = 2
_SERVER_PAUSES_S = (5.0, 15.0, 45.0)
_TIMEOUT_RETRIES = 1
_CONNECT_RETRIES = 3
_CONNECT_RETRY_PAUSE_S = 1.0
_AGGREGATE = re.compile(r"(?i)\b(sum|count|total)\s*\(")
_ANSWER_LINE = re.compile(r"ANSWER:\s*(?P<value>.+)")


class _Stop(Exception):
    """The task ends here with this outcome."""

    def __init__(self, verdict: Verdict, reason: str) -> None:
        super().__init__(reason)
        self.outcome = Outcome(verdict, reason=reason)


def _first_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _is_infeasible(answer: str) -> bool:
    return answer.strip().strip(".`'\"").upper() == prompts.INFEASIBLE


def _free_aggregate(world: SqlWorld, answer: str) -> object | None:
    """Value of a SUM/COUNT over a filter that matches nothing: a defined, free answer."""
    try:
        sql = clean_sql(answer)
        if not _AGGREGATE.search(sql):
            return None
        rows = run_select(world, sql)
    except SqlError:
        return None
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    value = rows[0][0]
    return 0 if value is None else value if isinstance(value, int) else None


class Tier1:
    """Model tier shared by all tasks of a run; holds the gateway, ledger and failure state."""

    def __init__(
        self,
        gateway: Gateway,
        ledger: Ledger,
        trajectory: TrajectoryLog,
        model: str,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        guard: ProxyGuard | None = None,
        qualification: bool = False,
        tier: str = TIER,
        reasoning: bool = False,
    ) -> None:
        self.tier = tier
        self._reasoning = reasoning
        self._attempts = 1 if reasoning else _ATTEMPTS
        self._timeout = REASON_TIMEOUT_S if reasoning else NOREASON_TIMEOUT_S
        self._complaints: dict[str, str] = {}
        self._gateway = gateway
        self._ledger = ledger
        self._trajectory = trajectory
        self._model = model
        self._sleep = sleep
        self._guard = guard if guard is not None else ProxyGuard(ledger, sleep=sleep)
        self._qualification = qualification
        self._insist: set[str] = set()
        self.task_cost: dict[str, float] = {}

    @property
    def dead_reason(self) -> str | None:
        return self._guard.dead_reason

    # --- one accounted call -------------------------------------------------------------

    async def _call(
        self, task: Task, system: str, user: str, max_tokens: int, stop: str | None
    ) -> Completion:
        prompt_chars = len(system) + len(user)
        upper = self._ledger.upper_bound(prompt_chars, max_tokens)
        if not self._ledger.reserve(upper):
            raise _Stop(Verdict.UNPARSED, "budget exhausted")
        try:
            completion = await self._gateway.complete(
                model=self._model,
                system=system,
                user=user,
                max_tokens=max_tokens,
                stop=stop,
                timeout=self._timeout,
            )
        except LLMError as error:
            if error.maybe_billed:
                self._ledger.lose(upper)
            else:
                self._ledger.release(upper)
            self._trajectory.record(
                task_id=task.task_id,
                task_type=task.task_type,
                tier=self.tier,
                model=self._model,
                est_cost=upper,
                error=error.kind,
                status=error.status,
            )
            raise
        except BaseException:
            self._ledger.lose(upper)
            raise
        cost = self._ledger.settle(upper, completion.usage, prompt_chars)
        self.task_cost[task.task_id] = self.task_cost.get(task.task_id, 0.0) + cost
        usage = completion.usage
        self._trajectory.record(
            task_id=task.task_id,
            task_type=task.task_type,
            reward=task.reward,
            tier=self.tier,
            model=self._model,
            est_cost=upper,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
            cost=round(cost, 1),
            latency_s=completion.latency_s,
            finish_reason=completion.finish_reason,
        )
        return completion

    async def _ask(
        self, task: Task, system: str, user: str, max_tokens: int, stop: str | None = None
    ) -> str:
        """A call with the proxy failure policy; returns the cleaned answer text."""
        if self._reasoning:
            # A reasoning model starts its answer with blank lines, so a newline stop would
            # cut it off; its budget must also cover the reasoning itself.
            stop, max_tokens = None, _REASONING_MAX_TOKENS
        complaint = self._complaints.get(task.task_id)
        if complaint:
            system += f"\nA previous answer was rejected: {complaint}. Do not repeat that mistake."
        if task.task_id in self._insist:
            system += "\nThis task is known to be answerable: never output X."
        rate_limited = server_errors = refused = timeouts = 0
        guard = self._guard
        while True:
            if guard.dead_reason is not None:
                raise _Stop(Verdict.UNPARSED, guard.dead_reason)
            if not await guard.wait_until_closed():
                raise _Stop(Verdict.UNPARSED, "proxy unavailable until the deadline")
            try:
                async with guard.slot():
                    completion = await self._call(task, system, user, max_tokens, stop)
                guard.success()
                return completion.text
            except LLMError as error:
                guard.failure()
                if error.kind == "auth":
                    guard.stop(f"proxy rejected the credentials ({error.status})")
                elif error.kind == "rate_limit" and rate_limited < _RATE_LIMIT_RETRIES:
                    rate_limited += 1
                    guard.rate_limited()
                    await self._sleep(rate_limited * random.uniform(2.0, 5.0))
                    continue
                elif error.kind == "connect" and refused < _CONNECT_RETRIES:
                    refused += 1
                    await self._sleep(_CONNECT_RETRY_PAUSE_S)
                    continue
                elif error.kind == "server" and server_errors < len(_SERVER_PAUSES_S):
                    await self._sleep(_SERVER_PAUSES_S[server_errors])
                    server_errors += 1
                    continue
                elif error.kind == "server":
                    guard.open_breaker()
                elif error.kind == "timeout" and timeouts < _TIMEOUT_RETRIES:
                    timeouts += 1
                    continue
                raise _Stop(Verdict.UNPARSED, f"model call failed: {error.kind}") from None

    # --- task types ---------------------------------------------------------------------

    async def solve(self, task: Task, complaint: str | None = None) -> Outcome:
        """Run the model tier for one task; never raises for task-level problems."""
        if complaint:
            self._complaints[task.task_id] = complaint[:200]
        handlers = {
            "math": self._math,
            "data_analysis": self._data,
            "search": self._search,
            "coding": self._coding,
        }
        handler = handlers.get(task.task_type, self._generic)
        try:
            outcome = await handler(task)
            if self._qualification and outcome.verdict is Verdict.UNGROUNDED:
                # Qualification tasks are feasible by the rules, so "infeasible" is our error.
                self._insist.add(task.task_id)
                second = await handler(task)
                if second.verdict is Verdict.SOLVED or second.answer is not None:
                    outcome = second
        except _Stop as stop:
            outcome = stop.outcome
        self._ledger.observe(task.task_type, self.tier, self.task_cost.get(task.task_id, 0.0))
        return outcome

    async def _math(self, task: Task) -> Outcome:
        question = strip_boilerplate(task.instruction)
        system, user = prompts.math_prompt(question)
        complaint = "no answer"
        doubted: int | None = None
        for _attempt in range(self._attempts):
            program = _first_line(await self._ask(task, system, user, _MATH_MAX_TOKENS, "\n"))
            if _is_infeasible(program):
                return Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly")
            try:
                value = math_solver.evaluate_program(program)
            except math_solver.ProgramError as error:
                complaint = str(error)
                user = prompts.with_complaint(user, program, complaint)
                continue
            doubt = math_solver.number_coverage_complaint(question, program)
            if doubt is None and value < 0:
                doubt = "negative result"
            if doubt is None:
                return Outcome(Verdict.SOLVED, answer=value, evidence=program)
            if doubted is None and _attempt + 1 < self._attempts:
                # A cheap second opinion first: the same model, told what looks odd. Only if
                # the two answers differ is the expensive reasoning tier worth its tokens.
                doubted = value
                hint = f"{doubt}; if that is intended, repeat the same statements"
                user = prompts.with_complaint(user, program, hint)
                continue
            review = None if doubted == value else f"needs review: {doubt}"
            return Outcome(Verdict.SOLVED, answer=value, reason=review, evidence=program)
        return Outcome(Verdict.UNPARSED, reason=f"program rejected: {complaint}")

    async def _sql(
        self,
        task: Task,
        world: SqlWorld,
        system: str,
        user: str,
        max_tokens: int,
        grounded: bool = False,
    ) -> Outcome:
        """`grounded` = the question names things that do exist in the data."""
        known = world.literals()
        complaint = "no answer"
        for _attempt in range(self._attempts):
            answer = _first_line(await self._ask(task, system, user, max_tokens, "\n"))
            if _is_infeasible(answer):
                if grounded and task.task_id not in self._insist:
                    # The data does contain what the question names: do not take "X" at once.
                    self._insist.add(task.task_id)
                    continue
                return Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly")
            try:
                sql = clean_sql(answer)
                unknown = [lit for lit in string_literals(sql) if normalise(lit) not in known]
                if unknown:
                    raise SqlError(f"value {unknown[0]!r} does not exist in the data")
                rows = list(dict.fromkeys(run_select(world, sql)))
                if len(rows) > 1 or (rows and len(rows[0]) != 1):
                    shape = f"{len(rows)} rows, {len(rows[0])} columns"
                    raise SqlError(f"result has {shape}, expected one value")
            except SqlError as error:
                complaint = str(error)
                user = prompts.with_complaint(user, answer, complaint)
                continue
            if not rows:
                return Outcome(Verdict.UNGROUNDED, reason="query selects nothing", evidence=sql)
            value = rows[0][0]
            if value is None:
                if not _AGGREGATE.search(sql):
                    return Outcome(Verdict.UNGROUNDED, reason="value is absent", evidence=sql)
                value = 0
            if isinstance(value, float) and value == int(value):
                value = int(value)
            if task.check == "exact_int" and not isinstance(value, int):
                return Outcome(Verdict.UNPARSED, reason="result is not an integer", evidence=sql)
            return Outcome(Verdict.SOLVED, answer=value, evidence=sql)
        if "does not exist" not in complaint:
            return Outcome(Verdict.UNPARSED, reason=f"query rejected: {complaint}")
        return Outcome(
            Verdict.UNGROUNDED,
            answer=_free_aggregate(world, answer),
            reason=f"query rejected: {complaint}",
        )

    async def _data(self, task: Task) -> Outcome:
        if task.workspace is None:
            return Outcome(Verdict.UNGROUNDED, reason="no workspace")
        world = await asyncio.to_thread(data.build_world, task.workspace)
        if world is None:
            return Outcome(Verdict.UNGROUNDED, reason="no readable table")
        try:
            question = strip_boilerplate(task.instruction)
            system, user = prompts.data_prompt(world.describe(typed=True), question)
            quoted = string_literals(question)
            known = world.literals()
            grounded = bool(quoted) and all(normalise(value) in known for value in quoted)
            return await self._sql(task, world, system, user, _DATA_MAX_TOKENS, grounded)
        finally:
            world.close()

    async def _search(self, task: Task) -> Outcome:
        if task.workspace is None:
            return Outcome(Verdict.UNGROUNDED, reason="no workspace")
        question = strip_boilerplate(task.instruction)
        kb = await asyncio.to_thread(search.load_knowledge_base, task.workspace)
        if kb.total_files == 0:
            return Outcome(Verdict.UNGROUNDED, reason="knowledge base is empty")
        if not search.is_structured(kb):
            return await self._fragments(task, question, task.workspace)
        world = search.build_world(kb)
        try:
            system, user = prompts.search_prompt(world.describe(typed=False), question)
            text = normalise(question)
            grounded = any(normalise(name) in text for name in world.names)
            return await self._sql(task, world, system, user, _SEARCH_MAX_TOKENS, grounded)
        finally:
            world.close()

    async def _fragments(self, task: Task, question: str, workspace: Path) -> Outcome:
        documents = await asyncio.to_thread(search.read_documents, workspace, _MAX_FRAGMENT_CHARS)
        if documents is None:
            return Outcome(Verdict.UNPARSED, reason="knowledge base is too large for one request")
        system, user = prompts.fragments_prompt(documents, question)
        answer = _first_line(await self._ask(task, system, user, _SHORT_MAX_TOKENS, "\n"))
        if _is_infeasible(answer):
            return Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly")
        match = _ANSWER_LINE.fullmatch(answer)
        value = match["value"].strip() if match else ""
        if not value or value not in documents:
            return Outcome(Verdict.UNPARSED, reason="answer is not quoted from the documents")
        return Outcome(Verdict.SOLVED, answer=value, evidence="fragments")

    async def _coding(self, task: Task) -> Outcome:
        if task.workspace is None:
            return Outcome(Verdict.UNGROUNDED, reason="no workspace")
        located = await asyncio.to_thread(coding.locate_files, task.workspace)
        if located is None:
            return Outcome(Verdict.UNPARSED, reason="library and test file not identified")
        lib_name, tests_name = located
        try:
            bench = await asyncio.to_thread(coding.Workbench, task.workspace, lib_name, tests_name)
        except (OSError, ValueError) as error:
            return Outcome(Verdict.UNPARSED, reason=f"workspace not usable: {type(error).__name__}")
        try:
            return await self._repair(task, bench, lib_name)
        finally:
            await asyncio.to_thread(bench.close)

    async def _repair(self, task: Task, bench: coding.Workbench, lib_name: str) -> Outcome:
        run, report = await asyncio.to_thread(bench.run, bench.original)
        if run.status == "ok":
            refusal = await asyncio.to_thread(bench.publish, bench.original)
            if refusal is not None:
                return Outcome(Verdict.UNPARSED, reason=refusal)
            return Outcome(Verdict.SOLVED, answer="DONE", evidence="tests already pass")
        extra = ""
        complaint = "no answer"
        for _attempt in range(self._attempts):
            methods = {key.split(" ", 1)[0] for key in run.failing}
            failures = "\n".join(coding.failure_summary(report)) or report.strip()[-400:]
            system, user = prompts.coding_prompt(
                strip_boilerplate(task.instruction),
                lib_name,
                coding.number_lines(bench.original),
                coding.failing_test_sources(bench.tests_text, methods),
                failures + extra,
            )
            patch = await self._ask(task, system, user, _CODING_MAX_TOKENS)
            try:
                patched = coding.apply_line_patch(bench.original, patch)
            except coding.PatchError as error:
                complaint = str(error)
            else:
                if coding.copies_assert_literals(bench.original, patched, bench.tests_text):
                    complaint = "the patch hard-codes values from the tests"
                else:
                    patched_run, patched_report = await asyncio.to_thread(bench.run, patched)
                    if patched_run.status == "ok":
                        refusal = await asyncio.to_thread(bench.publish, patched)
                        if refusal is None:
                            return Outcome(Verdict.SOLVED, answer="DONE", evidence=patch.strip())
                        complaint = refusal
                    else:
                        still = coding.failure_summary(patched_report)
                        complaint = "tests still fail: " + ("; ".join(still) or patched_run.status)
            extra = f"\n\nA previous patch was rejected ({complaint}):\n{patch.strip()[:400]}"
        return Outcome(Verdict.UNPARSED, reason=f"patch rejected: {complaint}")

    async def _generic(self, task: Task) -> Outcome:
        system, user = prompts.generic_prompt(strip_boilerplate(task.instruction))
        answer = _first_line(await self._ask(task, system, user, _SHORT_MAX_TOKENS, "\n"))
        if not answer:
            return Outcome(Verdict.UNPARSED, reason="empty answer")
        if _is_infeasible(answer):
            return Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly")
        return Outcome(Verdict.SOLVED, answer=answer, evidence="generic")
