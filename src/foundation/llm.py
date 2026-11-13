"""Model gateway on the openJiuwen `Model`, usage normalisation and a degraded fallback.

Importing this module is cheap; `openjiuwen` and `openai` are imported only when a gateway
is built, after the zero-token phase has already written its answers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import DEFAULT_T1_MODEL
from src.core.budget import Usage
from src.core.sandbox import run_sandboxed

logger = logging.getLogger(__name__)

NOREASON_TIMEOUT_S = 60.0
REASON_TIMEOUT_S = 180.0
_CANARY_TIMEOUT_S = 120.0
_TIMEOUT_GRACE_S = 5.0
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.DOTALL)
_STATUS = re.compile(r"Error code: (\d{3})")
_CONNECT_ERRORS = frozenset({"ConnectError", "ConnectTimeout", "ConnectionRefusedError"})


class LLMError(Exception):
    """A failed call, classified for the retry policy."""

    def __init__(self, kind: str, status: int | None, detail: str) -> None:
        super().__init__(f"{kind} ({status}): {detail}")
        self.kind = kind  # auth | rate_limit | connect | timeout | server | bad_request | other
        self.status = status
        self.detail = detail

    @property
    def maybe_billed(self) -> bool:
        """True when the request may have reached the model before failing."""
        return self.kind in ("timeout", "server", "other")


@dataclass(frozen=True)
class Completion:
    """One model answer with everything the ledger and the log need."""

    text: str
    reasoning_chars: int
    finish_reason: str | None
    usage: Usage | None
    latency_s: float
    model: str


class Gateway(Protocol):
    """Anything that can answer one system+user prompt."""

    name: str

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stop: str | None = None,
        timeout: float = NOREASON_TIMEOUT_S,
    ) -> Completion: ...


def clean_text(content: object) -> str:
    """Drop reasoning blocks (also unterminated ones) and surrounding whitespace."""
    text = content if isinstance(content, str) else ""
    return _OPEN_THINK.sub("", _THINK_BLOCK.sub("", text)).strip()


def normalise_base_url(raw: str) -> str:
    """`<LLM_BASE_URL>/chat/completions` is the endpoint; return the part before it."""
    base = raw.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _status_of(exc: BaseException) -> int | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status
        match = _STATUS.search(str(current))
        if match:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def _never_sent(exc: BaseException) -> bool:
    """True if the TCP connection was never established, so nothing could be billed."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _CONNECT_ERRORS:
            return True
        current = current.__cause__ or current.__context__
    return False


def classify_error(exc: BaseException, api_key: str = "") -> LLMError:
    """Map SDK and transport exceptions to a retry-policy category; never leaks the key."""
    detail = f"{type(exc).__name__}: {exc}"
    if api_key:
        detail = detail.replace(api_key, "<redacted>")
    detail = detail[:300]
    status = _status_of(exc)
    lowered = detail.lower()
    if _never_sent(exc):
        return LLMError("connect", status, detail)
    if isinstance(exc, asyncio.TimeoutError | TimeoutError) or "timeout" in lowered:
        return LLMError("timeout", status, detail)
    if status in (401, 403):
        return LLMError("auth", status, detail)
    if status == 429:
        return LLMError("rate_limit", status, detail)
    if (status is not None and status >= 500) or "connection" in lowered:
        return LLMError("server", status, detail)
    if status is not None and 300 <= status < 500:
        return LLMError("bad_request", status, detail)
    return LLMError("other", status, detail)


def quiet_sdk_logging() -> None:
    """The SDK logs every request, prompt included, to stdout and files; keep only our log."""
    for sdk_logger in list(logging.root.manager.loggerDict.values()):
        handlers = getattr(sdk_logger, "handlers", [])
        if any(type(handler).__name__ == "SafeRotatingFileHandler" for handler in handlers):
            sdk_logger.setLevel(logging.CRITICAL + 1)


class _Sink:
    """A write-only stream that discards everything."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        return None


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class OpenJiuwenGateway:
    """Calls through `openjiuwen` `Model`, built explicitly (never via `init_model`)."""

    name = "openjiuwen"

    def __init__(self, base_url: str, api_key: str, default_model: str = DEFAULT_T1_MODEL) -> None:
        # The SDK attaches stdout handlers while it is imported and logs every request with
        # its prompt; bind those handlers to a sink so that only our own log reaches stdout.
        with contextlib.redirect_stdout(_Sink()):  # type: ignore[type-var]
            from openjiuwen.core.common.logging import llm_logger
            from openjiuwen.core.foundation.llm import (
                Model,
                ModelClientConfig,
                ModelRequestConfig,
            )

            llm_logger.debug("gateway start")
            self._build(
                Model, ModelClientConfig, ModelRequestConfig, base_url, api_key, default_model
            )

    def _build(
        self,
        model_class: Any,
        client_config_class: Any,
        request_config_class: Any,
        base_url: str,
        api_key: str,
        default_model: str,
    ) -> None:
        self._api_key = api_key
        client_config = client_config_class(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=normalise_base_url(base_url),
            timeout=REASON_TIMEOUT_S,
            max_retries=0,
            verify_ssl=True,
        )
        request_config = request_config_class(model=default_model, temperature=0.0)
        self._model = model_class(model_client_config=client_config, model_config=request_config)
        quiet_sdk_logging()

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stop: str | None = None,
        timeout: float = NOREASON_TIMEOUT_S,
    ) -> Completion:
        started = time.monotonic()
        try:
            message = await asyncio.wait_for(
                self._model.invoke(
                    _messages(system, user),
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    stop=stop,
                    timeout=timeout,
                ),
                timeout=timeout + _TIMEOUT_GRACE_S,
            )
        except Exception as exc:
            raise classify_error(exc, self._api_key) from None
        finally:
            quiet_sdk_logging()
        metadata = message.usage_metadata
        usage = None
        if metadata is not None and (metadata.input_tokens or metadata.output_tokens):
            usage = Usage(
                prompt_tokens=int(metadata.input_tokens or 0),
                completion_tokens=int(metadata.output_tokens or 0),
                reasoning_tokens=int(metadata.reasoning_tokens or 0),
            )
        return Completion(
            text=clean_text(message.content),
            reasoning_chars=len(message.reasoning_content or ""),
            finish_reason=message.finish_reason,
            usage=usage,
            latency_s=round(time.monotonic() - started, 3),
            model=model,
        )


class OpenAIFallbackGateway:
    """Degraded mode: the `openai` package directly, used only if the SDK cannot be imported."""

    name = "openai-fallback"

    def __init__(self, base_url: str, api_key: str) -> None:
        from openai import AsyncOpenAI

        self._api_key = api_key
        self._client = AsyncOpenAI(
            base_url=normalise_base_url(base_url), api_key=api_key, max_retries=0
        )

    async def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stop: str | None = None,
        timeout: float = NOREASON_TIMEOUT_S,
    ) -> Completion:
        started = time.monotonic()
        params: dict[str, Any] = {
            "model": model,
            "messages": _messages(system, user),
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if stop is not None:
            params["stop"] = stop
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**params),
                timeout=timeout + _TIMEOUT_GRACE_S,
            )
        except Exception as exc:
            raise classify_error(exc, self._api_key) from None
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        usage = None
        if response.usage is not None:
            details = getattr(response.usage, "completion_tokens_details", None)
            usage = Usage(
                prompt_tokens=int(response.usage.prompt_tokens or 0),
                completion_tokens=int(response.usage.completion_tokens or 0),
                reasoning_tokens=int(getattr(details, "reasoning_tokens", 0) or 0),
            )
        return Completion(
            text=clean_text(getattr(message, "content", "")),
            reasoning_chars=len(getattr(message, "reasoning_content", None) or ""),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            latency_s=round(time.monotonic() - started, 3),
            model=model,
        )


def start_canary() -> Future[bool]:
    """Check in a child process, in parallel with phase 0, that the SDK can be imported."""

    def probe() -> bool:
        result = run_sandboxed(
            [sys.executable, "-c", "import openjiuwen.core.foundation.llm"],
            cwd=Path.cwd(),
            timeout=_CANARY_TIMEOUT_S,
            limits=False,  # a memory cap could fail a heavy import that works in the parent
        )
        if result.returncode != 0:
            logger.error("SDK canary failed: %s", (result.stderr or "timeout")[-300:])
        return result.returncode == 0

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="canary")
    future = pool.submit(probe)
    pool.shutdown(wait=False)
    return future


def build_gateway(env: Mapping[str, str], sdk_importable: bool) -> Gateway | None:
    """Gateway for this run, or None when the proxy is not configured or nothing imports."""
    base_url = env.get("LLM_BASE_URL", "").strip()
    api_key = env.get("LLM_API_KEY", "").strip()
    if not base_url or not api_key:
        logger.warning("LLM_BASE_URL or LLM_API_KEY is not set: model tiers are skipped")
        return None
    if sdk_importable:
        try:
            return OpenJiuwenGateway(base_url, api_key)
        except Exception as exc:
            logger.error("openJiuwen gateway failed to start: %s", type(exc).__name__)
    try:
        gateway = OpenAIFallbackGateway(base_url, api_key)
    except Exception as exc:
        logger.error("fallback gateway failed to start: %s", type(exc).__name__)
        return None
    logger.warning("degraded mode: calling the proxy through the openai package directly")
    return gateway
