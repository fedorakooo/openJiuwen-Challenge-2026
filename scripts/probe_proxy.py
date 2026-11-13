"""Proxy contract probe, for development only.

Every call spends the team's key quota, so the probe is small (about 25 calls) and stops
calling a reasoning model as soon as `max_tokens` turns out not to cap its reasoning.

Usage (inside the image, which has openjiuwen):
    python scripts/probe_proxy.py [--out probe.json] [--skip-sdk] [--skip-concurrency]

Reads LLM_BASE_URL and LLM_API_KEY from the environment and never prints them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

NOREASON_MODELS = ("qwen3.8-27b-noreason", "qwen3.6-35b-a3b-noreason")
REASON_MODELS = ("qwen3.8-27b", "qwen3.6-35b-a3b")
TIMEOUT_S = 120
CONCURRENCY = 5
REASONING_OVERRUN_FACTOR = 3  # completion_tokens above max_tokens * factor = cap not enforced

BASIC = "Reply with the single word: ready"
COUNT = "Count from 1 to 10, numbers separated by commas, nothing else."
LONG = "Write the integers from 1 to 60 separated by single spaces, nothing else."
THINK = "What is 17 * 23 + 9? Reply with the number only."


def _base_url() -> str:
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return base


def _redact(text: str) -> str:
    key = os.environ.get("LLM_API_KEY", "")
    return text.replace(key, "<redacted>") if key else text


def _cost(usage: dict[str, Any]) -> float | None:
    if not usage:
        return None
    return usage.get("prompt_tokens", 0) / 10 + usage.get("completion_tokens", 0)


def _describe_redirect(source: str, location: str | None) -> dict[str, Any] | None:
    """What a redirect changes, without printing the host."""
    if not location:
        return None
    before = urllib.parse.urlsplit(source)
    after = urllib.parse.urlsplit(urllib.parse.urljoin(source, location))
    return {
        "scheme": f"{before.scheme} -> {after.scheme}",
        "host_changed": before.hostname != after.hostname,
        "port": f"{before.port} -> {after.port}",
        "path": f"{before.path} -> {after.path}",
    }


def raw_call(model: str, prompt: str, **params: Any) -> dict[str, Any]:
    """One chat completion over plain HTTP; returns only non-secret facts."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], **params}
    request = urllib.request.Request(
        _base_url() + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", ""),
        },
        method="POST",
    )
    started = time.monotonic()
    record: dict[str, Any] = {"model": model, "params": params}
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
            record["status"] = response.status
            record["rate_headers"] = {
                name.lower(): value
                for name, value in response.headers.items()
                if "ratelimit" in name.lower() or name.lower() == "retry-after"
            }
    except urllib.error.HTTPError as exc:
        record["status"] = exc.code
        record["redirect"] = _describe_redirect(request.full_url, exc.headers.get("Location"))
        record["error"] = _redact(exc.read().decode("utf-8", errors="replace"))[:300]
        record["retry_after"] = exc.headers.get("Retry-After")
        payload = {}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        record["status"] = None
        record["error"] = _redact(f"{type(exc).__name__}: {exc}")[:300]
        payload = {}
    record["latency_s"] = round(time.monotonic() - started, 2)

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    usage = payload.get("usage") or {}
    record.update(
        finish_reason=choice.get("finish_reason"),
        content=content[:200],
        content_has_think_tag="<think>" in content,
        message_keys=sorted(message),
        reasoning_chars=len(reasoning),
        usage=usage,
        cost=_cost(usage),
    )
    return record


def probe_noreason(model: str) -> dict[str, Any]:
    basic = raw_call(model, BASIC, max_tokens=16, temperature=0)
    stop = raw_call(model, COUNT, max_tokens=40, temperature=0, stop="5")
    capped = raw_call(model, LONG, max_tokens=8, temperature=0)
    return {"basic": basic, "stop": stop, "max_tokens": capped}


def probe_reason(model: str) -> dict[str, Any]:
    """Start with the cheapest call that shows whether max_tokens caps reasoning."""
    limit = 24
    capped = raw_call(model, THINK, max_tokens=limit, temperature=0)
    result: dict[str, Any] = {"max_tokens_caps_reasoning_call": capped}
    spent = (capped.get("usage") or {}).get("completion_tokens", 0)
    if capped.get("status") != 200 or spent > limit * REASONING_OVERRUN_FACTOR:
        result["stopped_early"] = "max_tokens does not cap reasoning or the call failed"
        return result
    result["full_answer"] = raw_call(model, THINK, max_tokens=600, temperature=0)
    result["stop"] = raw_call(model, COUNT, max_tokens=300, temperature=0, stop="5")
    return result


def probe_concurrency(model: str) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        started = time.monotonic()
        calls = list(
            pool.map(lambda _i: raw_call(model, BASIC, max_tokens=4, temperature=0), range(5))
        )
    return {
        "parallel_calls": CONCURRENCY,
        "statuses": [call["status"] for call in calls],
        "latencies_s": [call["latency_s"] for call in calls],
        "wall_s": round(time.monotonic() - started, 2),
        "errors": [call["error"] for call in calls if call.get("error")],
        "cost": sum(call["cost"] or 0 for call in calls),
    }


async def _sdk_call(model_name: str) -> dict[str, Any]:
    from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig

    client_config = ModelClientConfig(
        client_provider="OpenAI",
        api_key=os.environ.get("LLM_API_KEY", ""),
        api_base=_base_url(),
        timeout=float(TIMEOUT_S),
        max_retries=0,
        verify_ssl=True,
    )
    request_config = ModelRequestConfig(model=model_name, temperature=0, max_tokens=16)
    model = Model(model_client_config=client_config, model_config=request_config)
    started = time.monotonic()
    message = await model.invoke(
        [{"role": "user", "content": BASIC}],
        model=model_name,
        temperature=0,
        max_tokens=16,
        timeout=float(TIMEOUT_S),
    )
    usage = message.usage_metadata
    return {
        "model": model_name,
        "latency_s": round(time.monotonic() - started, 2),
        "content": str(message.content or "")[:200],
        "reasoning_chars": len(message.reasoning_content or ""),
        "finish_reason": message.finish_reason,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
    }


def probe_sdk(models: tuple[str, ...]) -> list[dict[str, Any]]:
    """All SDK calls share one event loop: the SDK's HTTP client is bound to the first one."""

    async def run_all() -> list[dict[str, Any]]:
        results = []
        for name in models:
            try:
                results.append(await _sdk_call(name))
            except Exception as exc:  # the probe reports, it does not crash
                detail = _redact(f"{type(exc).__name__}: {exc}")[:300]
                results.append({"model": name, "error": detail})
        return results

    return asyncio.run(run_all())


def _total_cost(node: Any) -> float:
    if isinstance(node, dict):
        own = node["cost"] if isinstance(node.get("cost"), int | float) else 0.0
        nested = sum(_total_cost(value) for key, value in node.items() if key != "cost")
        return own + nested
    if isinstance(node, list):
        return sum(_total_cost(item) for item in node)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="also write the JSON report to this file")
    parser.add_argument("--skip-sdk", action="store_true")
    parser.add_argument("--skip-concurrency", action="store_true")
    args = parser.parse_args()
    if not _base_url() or not os.environ.get("LLM_API_KEY"):
        print("LLM_BASE_URL and LLM_API_KEY must be set", file=sys.stderr)
        return 2

    report: dict[str, Any] = {"path": "<LLM_BASE_URL>/chat/completions", "raw": {}}
    for model in NOREASON_MODELS:
        report["raw"][model] = probe_noreason(model)
    for model in REASON_MODELS:
        report["raw"][model] = probe_reason(model)
    if not args.skip_concurrency:
        report["concurrency"] = probe_concurrency(NOREASON_MODELS[0])
    report["raw_cost_total"] = round(_total_cost(report), 1)
    if not args.skip_sdk:
        report["sdk"] = probe_sdk(NOREASON_MODELS + REASON_MODELS)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
