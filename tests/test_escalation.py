from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from src import cli
from src.core.agent import Tier1
from src.core.budget import Ledger, Usage
from src.core.results import ResultWriter, TrajectoryLog
from src.core.runner import arbitrate
from src.core.scheduler import ProxyGuard, complaint_for_reasoning, worth_an_agent
from src.core.tasks import Outcome, Task, Verdict, load_tasks
from src.foundation.llm import Completion
from tests.conftest import ANSWER_TAIL, DONE_TAIL, MakeTask, load_one, write_files
from tests.fake_proxy import completion, fake_proxy
from tests.test_solver_coding import GOOD_LIB, TESTS
from tests.test_solver_search import KB

HAS_SDK = importlib.util.find_spec("openjiuwen") is not None
needs_sdk = pytest.mark.skipif(not HAS_SDK, reason="openjiuwen is not installed here")
REASONER = "qwen3.8-27b"
KEY = "sk-test-key-123456"
QUESTION = "A bakery has 120 loaves and sells 15 loaves each hour for 4 hours. How many are left?"


@pytest.fixture(autouse=True)
def writable_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


class _Recorder:
    name = "fake"

    def __init__(self, answers: list[str]) -> None:
        self._answers = answers
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, model, system, user, max_tokens, stop=None, timeout=60.0):
        self.calls.append(
            {
                "model": model,
                "system": system,
                "max_tokens": max_tokens,
                "stop": stop,
                "timeout": timeout,
            }
        )
        return Completion(self._answers.pop(0), 0, "stop", Usage(300, 400, 380), 0.01, model)


# --- T2 -----------------------------------------------------------------------------------


def test_reasoning_tier_gets_the_complaint_one_attempt_and_no_newline_stop(
    make_task: MakeTask,
) -> None:
    task = load_one(make_task("m", instruction=QUESTION + ANSWER_TAIL))
    gateway = _Recorder(["s=120; s-=15*4"])
    ledger = Ledger(100_000)
    tier = Tier1(gateway, ledger, TrajectoryLog(None), REASONER, tier="T2", reasoning=True)

    outcome = asyncio.run(tier.solve(task, "number 4 from the question is unused"))

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, 60)
    call = gateway.calls[0]
    assert (call["model"], call["stop"], call["max_tokens"], call["timeout"]) == (
        REASONER,
        None,
        1400,
        180.0,
    )
    assert "A previous answer was rejected: number 4 from the question is unused" in call["system"]
    assert ledger.spent == pytest.approx(430.0)  # 300/10 + 400, reasoning already inside
    assert ledger.expected_cost("math", "T2") < 1200

    failing = Tier1(
        _Recorder(["nonsense"]),
        Ledger(100_000),
        TrajectoryLog(None),
        REASONER,
        tier="T2",
        reasoning=True,
    )
    assert asyncio.run(failing.solve(task)).verdict is Verdict.UNPARSED  # one attempt only


@pytest.mark.parametrize(
    ("outcome", "qualification", "expected"),
    [
        (Outcome(Verdict.UNPARSED, reason="program rejected: not valid syntax"), False, True),
        (Outcome(Verdict.SOLVED, answer=1, reason="needs review: number 3 is unused"), False, True),
        (Outcome(Verdict.SOLVED, answer=1), False, False),
        (Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly"), False, False),
        (Outcome(Verdict.UNGROUNDED, reason="model: cannot be answered exactly"), True, True),
        (Outcome(Verdict.UNPARSED, reason="budget exhausted"), True, False),
        (Outcome(Verdict.UNPARSED, reason="model call failed: server"), False, False),
        (Outcome(Verdict.UNPARSED, reason="proxy unavailable until the deadline"), False, False),
    ],
)
def test_only_model_mistakes_are_escalated_to_reasoning(
    outcome: Outcome, qualification: bool, expected: bool
) -> None:
    assert (complaint_for_reasoning(outcome, qualification) is not None) is expected


def test_agent_is_reserved_for_unsolved_tasks_at_or_above_the_median_reward(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("cheap", reward=1)
    make_task("rich", reward=4)
    cheap, rich = load_tasks(tasks_dir)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    failed = Outcome(Verdict.UNPARSED, reason="query rejected: 3 rows")

    assert worth_an_agent(rich, failed, rewards)
    assert not worth_an_agent(cheap, failed, rewards)
    assert not worth_an_agent(rich, Outcome(Verdict.SOLVED, answer=1), rewards)
    assert not worth_an_agent(rich, Outcome(Verdict.UNGROUNDED, reason="absent"), rewards)
    assert not worth_an_agent(rich, Outcome(Verdict.UNPARSED, reason="budget exhausted"), rewards)
    assert not worth_an_agent(rich, None, rewards)


def test_arbiter_replaces_the_zero_token_answer_only_when_both_tiers_agree(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    for name in ("backed", "split", "silent"):
        make_task(name)
    tasks = load_tasks(tasks_dir)
    writer = ResultWriter()
    for task in tasks:
        writer.write_answer(task, 7)
    disagreements = {"backed": "8", "split": "8", "silent": "8"}

    async def solve(task: Task) -> Outcome:
        if task.task_id == "backed":
            return Outcome(Verdict.SOLVED, answer=8)
        if task.task_id == "split":
            return Outcome(Verdict.SOLVED, answer=9)
        return Outcome(Verdict.UNPARSED, reason="no answer")

    replaced = asyncio.run(arbitrate(tasks, disagreements, solve, writer, threading.Event()))

    assert replaced == 1
    assert (tasks_dir / "backed" / "answer.txt").read_bytes() == b"8\n"
    assert (tasks_dir / "split" / "answer.txt").read_bytes() == b"7\n"
    assert (tasks_dir / "silent" / "answer.txt").read_bytes() == b"7\n"


# --- end to end against the fake proxy (needs the SDK) -------------------------------------


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> tuple[int, dict[str, Any]]:
    status, payload = completion("", 400, 30)
    payload["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    payload["choices"][0]["finish_reason"] = "tool_calls"
    return status, payload


def _agent(proxy_url: str, budget: int = 100_000):
    from src.core.react import AgentTier

    ledger = Ledger(budget)
    tier = AgentTier(proxy_url, KEY, REASONER, ledger, TrajectoryLog(None), ProxyGuard(ledger))
    return tier, ledger


@needs_sdk
def test_agent_answers_a_search_task_with_one_tool_call(make_task: MakeTask) -> None:
    task_dir = make_task(
        "s", task_type="search", check="exact_str", workspace=True,
        instruction="Where is the company Irina Smith works for based?" + ANSWER_TAIL,
    )  # fmt: skip
    write_files(task_dir / "workspace", KB)
    query = (
        "SELECT c.headquarters FROM person p JOIN company c ON c.name=p.employer "
        "WHERE p.name='Irina Smith'"
    )

    def respond(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if body["messages"][-1]["role"] == "tool":
            return completion(f"ANSWER: {body['messages'][-1]['content'][3:8]}", 600, 12)
        return _tool_call("sql", {"query": query}, "call_1")

    with fake_proxy(respond) as proxy:
        tier, ledger = _agent(proxy.base_url)
        outcome = asyncio.run(tier.solve(load_one(task_dir)))

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, "Osaka")
    first = proxy.requests[0]
    assert first["model"] == REASONER and first["max_tokens"] == 600
    assert sorted(tool["function"]["name"] for tool in first["tools"]) == ["read_file", "sql"]
    assert "person(name, born, employer, mentor -> person.name)" in first["messages"][1]["content"]
    assert ledger.calls == 2 and ledger.spent == pytest.approx(40 + 30 + 60 + 12)
    assert tier.task_cost["s"] == pytest.approx(ledger.spent)


@needs_sdk
def test_agent_stops_at_the_iteration_limit_and_at_the_cost_cap(make_task: MakeTask) -> None:
    task_dir = make_task(
        "s", task_type="search", check="exact_str", workspace=True,
        instruction="Where is the company Irina Smith works for based?" + ANSWER_TAIL,
    )  # fmt: skip
    write_files(task_dir / "workspace", KB)

    def loop(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return _tool_call("sql", {"query": "SELECT 1"}, "call_x")

    with fake_proxy(loop) as proxy:
        tier, _ledger = _agent(proxy.base_url)
        endless = asyncio.run(tier.solve(load_one(task_dir)))
        assert endless.verdict is Verdict.UNPARSED and endless.answer is None
        assert len(proxy.requests) <= 4

    with fake_proxy(loop) as proxy:
        tier, ledger = _agent(proxy.base_url, budget=1000)  # cap = 3% of the budget = 30
        capped = asyncio.run(tier.solve(load_one(task_dir)))
        assert capped.verdict is Verdict.UNPARSED and "agent" in (capped.reason or "")
        assert proxy.requests == [] and ledger.committed == 0


@needs_sdk
def test_agent_repairs_code_through_tools_and_publishes_only_green_code(
    make_task: MakeTask,
) -> None:
    broken = GOOD_LIB.replace("return ' '.join(reversed(s.split()))", "return s.upper().title()")
    task_dir = make_task(
        "c", task_type="coding", check="unittest", workspace=True,
        instruction="The tests are red. Repair the library." + DONE_TAIL,
    )  # fmt: skip
    write_files(task_dir / "workspace", {"mylib.py": broken, "test_mylib.py": TESTS})
    steps = [
        _tool_call("replace_lines", {"patch": "2|    return ' '.join(reversed(s.split()))"}, "c1"),
        _tool_call("run_tests", {}, "c2"),
        completion("ANSWER: DONE", 700, 8),
    ]

    with fake_proxy(lambda _body: steps.pop(0)) as proxy:
        tier, _ledger = _agent(proxy.base_url)
        outcome = asyncio.run(tier.solve(load_one(task_dir)))

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, "DONE")
    assert (task_dir / "workspace" / "mylib.py").read_text(encoding="utf-8") == GOOD_LIB
    assert proxy.requests[2]["messages"][-1]["content"] == "OK: all tests pass"

    (task_dir / "workspace" / "mylib.py").write_bytes(broken.encode("utf-8"))
    with fake_proxy(lambda _body: completion("ANSWER: DONE", 700, 8)) as proxy:
        tier, _ledger = _agent(proxy.base_url)
        lazy = asyncio.run(tier.solve(load_one(task_dir)))
    assert lazy.verdict is Verdict.UNPARSED  # claimed DONE without fixing anything
    assert (task_dir / "workspace" / "mylib.py").read_text(encoding="utf-8") == broken


@needs_sdk
@pytest.mark.usefixtures("isolated_process_state")
def test_run_escalates_from_t1_to_the_reasoning_model(
    tasks_dir: Path, make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(6):  # a full run, not qualification
        make_task(f"m{index}", instruction=QUESTION + ANSWER_TAIL)

    def respond(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if body["model"].endswith("-noreason"):
            return completion("I think the answer is sixty", 300, 9)
        return completion("\n\ns=120; s-=15*4", 320, 500, reasoning_content="thinking")

    with fake_proxy(respond) as proxy:
        monkeypatch.setenv("LLM_BASE_URL", proxy.base_url)
        monkeypatch.setenv("LLM_API_KEY", KEY)
        monkeypatch.setenv("DISABLE_T3", "1")
        assert cli.main(["--tasks-dir", str(tasks_dir)]) == 0

    assert (tasks_dir / "m0" / "answer.txt").read_bytes() == b"60\n"
    models = [body["model"] for body in proxy.requests]
    assert models.count("qwen3.8-27b-noreason") == 12 and models.count(REASONER) == 6
    reasoning_call = next(body for body in proxy.requests if body["model"] == REASONER)
    assert "stop" not in reasoning_call and reasoning_call["max_tokens"] == 1400
    assert "A previous answer was rejected" in reasoning_call["messages"][0]["content"]
