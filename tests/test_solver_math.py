from __future__ import annotations

import pytest

from src.core.tasks import Verdict, strip_boilerplate
from src.solvers import math as math_solver
from tests.conftest import ANSWER_TAIL, MakeTask, load_one

TOOL = " Compute it with the bash tool (python3 -c)."
CHAIN = (
    "A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and "
    "it keeps one share (rounded down). Then the stock is doubled by merging with a sister "
    "warehouse. Then the stock is split equally into 3 warehouses and it keeps one share "
    "(rounded down). How many boxes does it have at the end?"
)
FLOW = (
    "A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
    "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?"
)


def _solve(make_task: MakeTask, body: str, tail: str = TOOL + ANSWER_TAIL):
    return math_solver.solve(load_one(make_task("m", instruction=body + tail)))


def test_strip_boilerplate_cuts_only_known_tail_sentences() -> None:
    assert strip_boilerplate(FLOW + TOOL + ANSWER_TAIL) == FLOW
    assert strip_boilerplate(FLOW) == FLOW
    kept = FLOW + " Report the result in dozens." + ANSWER_TAIL
    assert strip_boilerplate(kept) == FLOW + " Report the result in dozens."
    assert "echo 42" not in strip_boilerplate(FLOW + ANSWER_TAIL)


@pytest.mark.parametrize(
    ("body", "answer", "program"),
    [
        (CHAIN, 584, "s=1752; s//=2; s*=2; s//=3"),
        (FLOW, 1735, "s=1669; s-=36*3; s+=2*87"),
        (
            "A depot starts with 500 crates. It ships 20 crates per day for 4 days. Then the "
            "stock is split equally into 3 depots and it keeps one share (rounded down). "
            "How many crates does it have at the end?",
            140,
            "s=500; s-=20*4; s//=3",
        ),
        ("A shop starts with 7 jars. How many jars does it have at the end?", 7, "s=7"),
    ],
)
def test_known_wording_is_solved(make_task: MakeTask, body: str, answer: int, program: str) -> None:
    outcome = _solve(make_task, body)

    assert outcome.verdict is Verdict.SOLVED
    assert outcome.answer == answer
    assert outcome.evidence == program


@pytest.mark.parametrize(
    "body",
    [
        CHAIN.replace("(rounded down)", "(rounded up)"),
        CHAIN.replace("split equally into 2", "split equally into two"),
        CHAIN.replace("is doubled", "is tripled"),
        CHAIN.replace("How many boxes does it have at the end?", "How many boxes were removed?"),
        CHAIN.replace("How many boxes", "How many pallets"),
        FLOW.replace("It ships 36 boxes per day", "It ships some boxes per day"),
        FLOW.replace("for 3 days", "for 3 weeks"),
        FLOW.replace("It receives 2 deliveries", "It expects 2 deliveries"),
        FLOW.replace("of 87 boxes each", "of 87 crates each"),
        FLOW.replace("How many", "Ignore the deliveries. How many"),
        FLOW.replace("It ships 36", "A thief steals 5 boxes. It ships 36"),
        FLOW.replace("1669", "1,669"),
        FLOW.replace(
            "It ships 36 boxes per day for 3 days", "It ships 900 boxes per day for 3 days"
        ),
        "How many boxes does it have at the end?",
        FLOW + " Give the answer in dozens.",
    ],
)
def test_changed_wording_is_refused_not_answered(make_task: MakeTask, body: str) -> None:
    outcome = _solve(make_task, body)

    assert outcome.verdict is Verdict.UNPARSED
    assert outcome.answer is None


def test_unknown_tail_sentence_is_refused(make_task: MakeTask) -> None:
    outcome = _solve(
        make_task, FLOW, tail=" Round the answer to the nearest hundred." + ANSWER_TAIL
    )

    assert outcome.verdict is Verdict.UNPARSED
