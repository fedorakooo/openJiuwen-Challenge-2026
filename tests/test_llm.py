from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from src import cli
from src.foundation import llm
from tests.conftest import ANSWER_TAIL, MakeTask
from tests.fake_proxy import completion, fake_proxy

HAS_SDK = importlib.util.find_spec("openjiuwen") is not None
needs_sdk = pytest.mark.skipif(not HAS_SDK, reason="openjiuwen is not installed here")
MODEL = "qwen3.8-27b-noreason"
KEY = "sk-test-key-123456"


@pytest.fixture(autouse=True)
def writable_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK creates ./logs on import and fails in a read-only working directory."""
    monkeypatch.chdir(tmp_path)


def test_clean_text_drops_reasoning_blocks() -> None:
    assert llm.clean_text("<think></think>\n\ns=1") == "s=1"
    assert llm.clean_text("<think>step one\nstep two</think> SELECT 1 ") == "SELECT 1"
    assert llm.clean_text("answer <think>never closed") == "answer"
    assert llm.clean_text(None) == ""


def test_normalise_base_url() -> None:
    assert llm.normalise_base_url(" https://proxy.example/v1/ ") == "https://proxy.example/v1"
    assert (
        llm.normalise_base_url("https://proxy.example/v1/chat/completions")
        == "https://proxy.example/v1"
    )


@pytest.mark.parametrize(
    ("message", "kind", "billed"),
    [
        ("APIStatusError: Error code: 401", "auth", False),
        ("Error code: 403 - forbidden", "auth", False),
        ("Error code: 429 - slow down", "rate_limit", False),
        ("Error code: 502", "server", True),
        ("APIConnectionError: Connection error.", "server", True),
        ("Error code: 308", "bad_request", False),
        ("Error code: 400 - bad model", "bad_request", False),
        ("APITimeoutError: Request timed out (timeout)", "timeout", True),
        ("something odd", "other", True),
    ],
)
def test_classify_error(message: str, kind: str, billed: bool) -> None:
    error = llm.classify_error(RuntimeError(message))

    assert (error.kind, error.maybe_billed) == (kind, billed)


def test_classify_error_follows_causes_and_redacts_the_key() -> None:
    class Inner(Exception):
        status_code = 429

    try:
        try:
            raise Inner("inner")
        except Inner as inner:
            raise RuntimeError(f"wrapped, key {KEY}") from inner
    except RuntimeError as outer:
        error = llm.classify_error(outer, KEY)

    assert error.kind == "rate_limit" and error.status == 429
    assert KEY not in error.detail and KEY not in str(error)
    assert llm.classify_error(TimeoutError()).kind == "timeout"


def test_build_gateway_needs_both_variables() -> None:
    assert llm.build_gateway({}, sdk_importable=True) is None
    assert llm.build_gateway({"LLM_BASE_URL": "http://x/v1"}, sdk_importable=True) is None


def _complete(gateway: Any, **overrides: Any) -> llm.Completion:
    params = {"model": MODEL, "system": "SYS", "user": "USER", "max_tokens": 32, "stop": "\n"}
    params.update(overrides)
    return asyncio.run(gateway.complete(**params))


@needs_sdk
def test_sdk_sends_exactly_what_we_asked_for_and_usage_is_normalised() -> None:
    def respond(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return completion("<think></think>s=1", prompt_tokens=321, completion_tokens=7)

    with fake_proxy(respond) as proxy:
        gateway = llm.OpenJiuwenGateway(proxy.base_url + "/", KEY)
        result = _complete(gateway)

    assert proxy.paths == ["/v1/chat/completions"]
    body = proxy.requests[0]
    assert body["model"] == MODEL
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert (body["max_tokens"], body["temperature"], body["stop"]) == (32, 0.0, "\n")
    # Nothing hidden: no reasoning switches, tools or streaming added for qwen models.
    allowed = {"model", "messages", "max_tokens", "temperature", "stop", "stream"}
    assert set(body) <= allowed, sorted(set(body) - allowed)
    assert not body.get("stream")
    assert proxy.headers[0]["authorization"] == f"Bearer {KEY}"
    assert result.text == "s=1"
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (321, 7)
    assert result.usage.cost == pytest.approx(39.1)


@needs_sdk
def test_sdk_does_not_retry_and_errors_are_classified() -> None:
    def respond(_body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return 429, {"error": {"message": "rate limit", "type": "rate_limit"}}

    with fake_proxy(respond) as proxy:
        gateway = llm.OpenJiuwenGateway(proxy.base_url, KEY)
        with pytest.raises(llm.LLMError) as caught:
            _complete(gateway)

    assert caught.value.kind == "rate_limit"
    assert len(proxy.requests) == 1
    assert KEY not in str(caught.value)


@needs_sdk
def test_one_gateway_serves_reasoning_and_plain_models_in_one_loop() -> None:
    def respond(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if body["model"].endswith("-noreason"):
            return completion("plain")
        return completion("thought", completion_tokens=400, reasoning_content="long reasoning")

    async def both(gateway: llm.OpenJiuwenGateway) -> list[llm.Completion]:
        return list(
            await asyncio.gather(
                gateway.complete(model=MODEL, system="s", user="u", max_tokens=8),
                gateway.complete(model="qwen3.8-27b", system="s", user="u", max_tokens=8),
            )
        )

    with fake_proxy(respond) as proxy:
        plain, reasoned = asyncio.run(both(llm.OpenJiuwenGateway(proxy.base_url, KEY)))

    assert (plain.text, plain.reasoning_chars) == ("plain", 0)
    assert (reasoned.text, reasoned.reasoning_chars) == ("thought", len("long reasoning"))
    assert sorted(body["model"] for body in proxy.requests) == sorted([MODEL, "qwen3.8-27b"])


@needs_sdk
def test_fallback_gateway_speaks_the_same_protocol() -> None:
    with fake_proxy(lambda _body: completion("ok", 50, 2)) as proxy:
        result = _complete(llm.OpenAIFallbackGateway(proxy.base_url, KEY))

    assert result.text == "ok" and result.usage is not None and result.usage.cost == 7.0
    assert proxy.requests[0]["stop"] == "\n"


@needs_sdk
@pytest.mark.usefixtures("isolated_process_state")
def test_run_solves_through_the_model_when_t0_is_disabled(
    tasks_dir: Path, make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        "A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
        "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?"
    )
    make_task("m1", instruction=body + ANSWER_TAIL)
    make_task("m2", instruction=body.replace("1669", "2000") + ANSWER_TAIL)

    def respond(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        start = "2000" if "2000 boxes" in request["messages"][-1]["content"] else "1669"
        return completion(f"s={start}; s-=36*3; s+=2*87", 300, 14)

    with fake_proxy(respond) as proxy:
        monkeypatch.setenv("LLM_BASE_URL", proxy.base_url)
        monkeypatch.setenv("LLM_API_KEY", KEY)
        monkeypatch.setenv("DISABLE_T0", "1")
        assert cli.main(["--tasks-dir", str(tasks_dir)]) == 0

    assert (tasks_dir / "m1" / "answer.txt").read_bytes() == b"1735\n"
    assert (tasks_dir / "m2" / "answer.txt").read_bytes() == b"2066\n"
    assert len(proxy.requests) == 2
    log = (Path.cwd() / "logs" / "trajectory.jsonl").read_text(encoding="utf-8")
    assert '"tier": "T1"' in log and '"cost": 44.0' in log
    assert KEY not in log and KEY not in os.environ.get("TMPDIR", "")


@pytest.mark.usefixtures("isolated_process_state")
def test_run_without_proxy_variables_only_uses_zero_token_solvers(
    tasks_dir: Path, make_task: MakeTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_task("m1", instruction="An unknown kind of puzzle." + ANSWER_TAIL)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    assert cli.main(["--tasks-dir", str(tasks_dir)]) == 0
    assert not (tasks_dir / "m1" / "answer.txt").exists()


def test_refused_connection_is_recognised_through_the_cause_chain() -> None:
    class ConnectError(Exception):
        pass

    try:
        try:
            raise ConnectError("refused")
        except ConnectError as inner:
            raise RuntimeError("APIConnectionError: Connection error.") from inner
    except RuntimeError as outer:
        error = llm.classify_error(outer)

    assert (error.kind, error.maybe_billed) == ("connect", False)
