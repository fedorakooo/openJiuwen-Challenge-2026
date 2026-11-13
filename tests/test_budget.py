from __future__ import annotations

import pytest

from src.core.budget import Ledger, Usage


def test_cost_formula_counts_ten_input_tokens_as_one_output_token() -> None:
    assert Usage(prompt_tokens=412, completion_tokens=31).cost == pytest.approx(72.2)
    # Reasoning reported inside completion_tokens is not counted twice ...
    assert Usage(400, 900, reasoning_tokens=850).cost == pytest.approx(940.0)
    # ... but reasoning reported on top of the visible answer is added.
    assert Usage(400, 50, reasoning_tokens=850).cost == pytest.approx(940.0)


def test_reservation_is_replaced_by_the_reported_usage() -> None:
    ledger = Ledger(budget=1000)
    upper = ledger.upper_bound(prompt_chars=900, max_tokens=96)
    assert upper == 30 + 96  # 900 chars / 3 chars per token = 300 tokens = 30 equivalents

    assert ledger.reserve(upper)
    assert ledger.committed == upper
    cost = ledger.settle(upper, Usage(250, 20), prompt_chars=900)

    assert cost == pytest.approx(45.0)
    assert (ledger.spent, ledger.reserved, ledger.calls) == (pytest.approx(45.0), 0.0, 1)


def test_missing_usage_and_timeouts_keep_the_upper_bound_charged() -> None:
    ledger = Ledger(budget=1000)
    ledger.reserve(100)
    ledger.settle(100, None)
    ledger.reserve(60)
    ledger.lose(60)
    ledger.reserve(40)
    ledger.release(40)

    assert (ledger.spent, ledger.unknown, ledger.reserved) == (100.0, 60.0, 0.0)
    assert ledger.committed == 160.0


def test_calls_are_refused_once_the_margin_is_reached() -> None:
    ledger = Ledger(budget=1000, margin=0.05)
    assert ledger.reserve(900)
    assert not ledger.can_afford(51)
    assert not ledger.reserve(51)
    assert ledger.reserve(50)
    assert ledger.reserved == 950


def test_chars_per_token_is_calibrated_and_clamped() -> None:
    ledger = Ledger(budget=10_000)
    for _ in range(40):
        ledger.reserve(10)
        ledger.settle(10, Usage(prompt_tokens=100, completion_tokens=1), prompt_chars=400)
    assert ledger.chars_per_token == pytest.approx(4.0, abs=0.01)

    for _ in range(60):
        ledger.reserve(10)
        ledger.settle(10, Usage(prompt_tokens=10_000, completion_tokens=1), prompt_chars=100)
    assert ledger.chars_per_token == 1.5


def test_expected_cost_moves_towards_observations() -> None:
    ledger = Ledger(budget=1000)
    assert ledger.expected_cost("math", "T1") == 70.0
    assert ledger.success_prior("search", "T1") == 0.9

    ledger.observe("math", "T1", 170.0)

    assert ledger.expected_cost("math", "T1") == pytest.approx(90.0)
    assert ledger.expected_cost("puzzle", "T1") == 120.0
