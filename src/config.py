"""Run settings read from the environment; standard library only."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_BUDGET = 100_000
DEFAULT_TASKS_DIR = "/data/tasks"
DEFAULT_DEADLINE_S = 50 * 60.0
DEFAULT_PHASE0_CAP_S = 8 * 60.0
DEADLINE_SLACK_S = 10.0
MIN_LLM_CONCURRENCY = 1
MAX_LLM_CONCURRENCY = 5
DEFAULT_LLM_CONCURRENCY = 3
QUALIFICATION_MAX_TASKS = 5
DEFAULT_T1_MODEL = "qwen3.8-27b-noreason"
DEFAULT_T2_MODEL = "qwen3.8-27b"
ALLOWED_MODELS = frozenset(
    {"qwen3.8-27b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"}
)

_FALSE_WORDS = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class Settings:
    """Immutable run configuration."""

    budget: int = DEFAULT_BUDGET
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY
    deadline_s: float = DEFAULT_DEADLINE_S
    phase0_cap_s: float = DEFAULT_PHASE0_CAP_S
    answer_when_unsure: bool = True
    t0_enabled: bool = True
    t1_model: str = DEFAULT_T1_MODEL
    t2_model: str = DEFAULT_T2_MODEL
    t2_enabled: bool = True
    t3_enabled: bool = True


def _positive_float(raw: str | None, default: float) -> float:
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(raw.strip()) if raw is not None else default
    except ValueError:
        return default
    return value if value > 0 else default


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from the environment; malformed values fall back to defaults."""
    source = os.environ if env is None else env
    concurrency = _positive_int(source.get("LLM_CONCURRENCY"), DEFAULT_LLM_CONCURRENCY)
    model = source.get("T1_MODEL", "").strip()
    reasoner = source.get("T2_MODEL", "").strip()
    return Settings(
        budget=_positive_int(source.get("BUDGET"), DEFAULT_BUDGET),
        llm_concurrency=max(MIN_LLM_CONCURRENCY, min(MAX_LLM_CONCURRENCY, concurrency)),
        deadline_s=_positive_float(source.get("RUN_DEADLINE_S"), DEFAULT_DEADLINE_S),
        phase0_cap_s=_positive_float(source.get("PHASE0_CAP_S"), DEFAULT_PHASE0_CAP_S),
        answer_when_unsure=source.get("ANSWER_WHEN_UNSURE", "1").strip().lower()
        not in _FALSE_WORDS,
        t0_enabled=source.get("DISABLE_T0", "0").strip().lower() in _FALSE_WORDS | {""},
        t1_model=model if model in ALLOWED_MODELS else DEFAULT_T1_MODEL,
        t2_model=reasoner if reasoner in ALLOWED_MODELS else DEFAULT_T2_MODEL,
        t2_enabled=source.get("DISABLE_T2", "0").strip().lower() in _FALSE_WORDS | {""},
        t3_enabled=source.get("DISABLE_T3", "0").strip().lower() in _FALSE_WORDS | {""},
    )
