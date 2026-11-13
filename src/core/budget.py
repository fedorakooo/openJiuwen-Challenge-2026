"""Token ledger and the cost table; one asyncio loop mutates it, so there are no locks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

INPUT_TOKENS_PER_OUTPUT = 10
_INITIAL_CHARS_PER_TOKEN = 3.0
_MIN_CHARS_PER_TOKEN = 1.5
_MAX_CHARS_PER_TOKEN = 6.0
_SMOOTHING = 0.2

# (task type, tier) -> (expected cost in output equivalents, prior success probability).
# Expected costs are corrected online by `observe`.
_PRIORS: dict[tuple[str, str], tuple[float, float]] = {
    ("math", "T1"): (70.0, 0.85),
    ("data_analysis", "T1"): (80.0, 0.85),
    ("search", "T1"): (75.0, 0.90),
    ("coding", "T1"): (160.0, 0.75),
}
_DEFAULT_PRIOR = (120.0, 0.6)
_TIER_PRIORS = {"T2": (1200.0, 0.5), "T3": (2500.0, 0.3)}


@dataclass(frozen=True)
class Usage:
    """Token counts of one call as reported by the proxy."""

    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0

    @property
    def cost(self) -> float:
        """Output equivalents: reasoning counts as output even if reported separately."""
        completion = self.completion_tokens
        if self.reasoning_tokens > completion:
            completion += self.reasoning_tokens
        return self.prompt_tokens / INPUT_TOKENS_PER_OUTPUT + completion


@dataclass
class Ledger:
    """Reserve an upper bound before a call, settle it by the reported usage after."""

    budget: int
    margin: float = 0.05
    spent: float = 0.0
    reserved: float = 0.0
    unknown: float = 0.0
    calls: int = 0
    chars_per_token: float = _INITIAL_CHARS_PER_TOKEN
    _expected: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def committed(self) -> float:
        return self.spent + self.reserved + self.unknown

    def estimate_prompt_tokens(self, prompt_chars: int) -> int:
        return math.ceil(prompt_chars / self.chars_per_token)

    def upper_bound(self, prompt_chars: int, max_tokens: int) -> float:
        tokens = self.estimate_prompt_tokens(prompt_chars)
        return math.ceil(tokens / INPUT_TOKENS_PER_OUTPUT) + max_tokens

    def can_afford(self, upper: float) -> bool:
        return self.committed + upper <= self.budget * (1 - self.margin)

    def reserve(self, upper: float) -> bool:
        if not self.can_afford(upper):
            return False
        self.reserved += upper
        return True

    def settle(self, upper: float, usage: Usage | None, prompt_chars: int = 0) -> float:
        """Replace a reservation with the fact; without usage the reservation stays spent."""
        self.reserved = max(0.0, self.reserved - upper)
        self.calls += 1
        cost = upper if usage is None else usage.cost
        self.spent += cost
        if usage is not None and usage.prompt_tokens > 0 and prompt_chars > 0:
            observed = prompt_chars / usage.prompt_tokens
            blended = (1 - _SMOOTHING) * self.chars_per_token + _SMOOTHING * observed
            self.chars_per_token = min(_MAX_CHARS_PER_TOKEN, max(_MIN_CHARS_PER_TOKEN, blended))
        return cost

    def lose(self, upper: float) -> None:
        """The request was sent but its fate is unknown (timeout): keep the bound charged."""
        self.reserved = max(0.0, self.reserved - upper)
        self.unknown += upper
        self.calls += 1

    def release(self, upper: float) -> None:
        """The request was rejected before any generation (bad request, auth)."""
        self.reserved = max(0.0, self.reserved - upper)

    @staticmethod
    def _prior(key: tuple[str, str]) -> tuple[float, float]:
        return _PRIORS.get(key, _TIER_PRIORS.get(key[1], _DEFAULT_PRIOR))

    def expected_cost(self, task_type: str, tier: str) -> float:
        key = (task_type, tier)
        return self._expected.get(key, self._prior(key)[0])

    def success_prior(self, task_type: str, tier: str) -> float:
        return self._prior((task_type, tier))[1]

    def observe(self, task_type: str, tier: str, cost: float) -> None:
        """Move the expected cost of this (type, tier) towards what a task really cost."""
        previous = self.expected_cost(task_type, tier)
        self._expected[(task_type, tier)] = (1 - _SMOOTHING) * previous + _SMOOTHING * cost
