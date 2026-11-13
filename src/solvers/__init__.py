"""Strict zero-token solvers (tier T0); standard library only."""

from __future__ import annotations

from collections.abc import Callable

from src.core.tasks import Outcome, Task
from src.solvers import coding, data, math, search

SOLVERS: dict[str, Callable[[Task], Outcome]] = {
    "math": math.solve,
    "data_analysis": data.solve,
    "search": search.solve,
    "coding": coding.solve,
}
