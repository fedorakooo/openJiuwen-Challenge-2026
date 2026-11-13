"""Reference implementations by function name, used only as oracles for differential testing.

An oracle never replaces task code: candidate edits of the task's own library are compared
with it on random inputs, and the edit that agrees everywhere is preferred.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Oracle:
    """A trusted implementation with inputs to start from when the tests give none."""

    function: Callable[..., object]
    samples: tuple[tuple[object, ...], ...]


def _product(xs: list[int]) -> int:
    result = 1
    for x in xs:
        result *= x
    return result


def _last_n(xs: list[object], n: int) -> list[object]:
    return xs[len(xs) - n :] if 0 < n <= len(xs) else ([] if n <= 0 else list(xs))


_NUMBERS = (([3, 9, 2, 7],), ([1, 2, 4, 6, 7],))
_SLICES = (([5, 6, 7, 8], 3), ([1, 2, 3], 1))

ORACLES: dict[str, Oracle] = {
    "total": Oracle(lambda xs: sum(xs), _NUMBERS),
    "product": Oracle(_product, _NUMBERS),
    "maximum": Oracle(lambda xs: max(xs), _NUMBERS),
    "minimum": Oracle(lambda xs: min(xs), _NUMBERS),
    "average": Oracle(lambda xs: sum(xs) / len(xs), _NUMBERS),
    "count_even": Oracle(lambda xs: sum(1 for x in xs if x % 2 == 0), _NUMBERS),
    "count_odd": Oracle(lambda xs: sum(1 for x in xs if x % 2 == 1), _NUMBERS),
    "count_positive": Oracle(lambda xs: sum(1 for x in xs if x > 0), _NUMBERS),
    "count_negative": Oracle(lambda xs: sum(1 for x in xs if x < 0), _NUMBERS),
    "first_n": Oracle(lambda xs, n: xs[:n], _SLICES),
    "last_n": Oracle(_last_n, _SLICES),
    "reverse_words": Oracle(lambda s: " ".join(reversed(s.split())), (("c a b",), ("one two",))),
    "clamp": Oracle(lambda x, lo, hi: max(lo, min(x, hi)), ((15, 0, 10), (4, 0, 10))),
}


def expected_output(name: str, arguments: tuple[object, ...]) -> str | None:
    """What the probe would print for a correct implementation, or None without an oracle."""
    oracle = ORACLES.get(name)
    if oracle is None:
        return None
    try:
        return repr(oracle.function(*copy.deepcopy(arguments)))
    except Exception as exc:  # the oracle's own failure mode is part of its behaviour
        return "EXC:" + type(exc).__name__
