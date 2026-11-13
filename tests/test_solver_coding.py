from __future__ import annotations

from pathlib import Path

import pytest

from src.core.tasks import Outcome, Verdict
from src.solvers import coding, oracles
from tests.conftest import DONE_TAIL, MakeTask, list_files, load_one, write_files

BODY = (
    "Run `python3 test_mylib.py` in the current directory; some tests fail. "
    "Fix the bugs in mylib.py (do not edit the tests) until all tests pass."
)
GOOD_LIB = """def reverse_words(s):
    return ' '.join(reversed(s.split()))


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def first_n(xs, n):
    return xs[:n]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)
"""
TESTS = """import unittest

import mylib as m


class T(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(m.reverse_words('c a b'), 'b a c')

    def test_total(self):
        self.assertEqual(m.total([1, 2, 3]), 6)

    def test_maximum(self):
        self.assertEqual(m.maximum([3, 9, 2]), 9)

    def test_first_n(self):
        self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])

    def test_count_even(self):
        self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)


if __name__ == '__main__':
    unittest.main()
"""


def _task(
    make_task: MakeTask, lib: str | None, tests: str | None = TESTS, body: str = BODY
) -> Path:
    task_dir: Path = make_task(
        "c", task_type="coding", check="unittest", workspace=True, instruction=body + DONE_TAIL
    )
    files = {name: text for name, text in (("mylib.py", lib), ("test_mylib.py", tests)) if text}
    write_files(task_dir / "workspace", files)
    return task_dir


def _solve(make_task: MakeTask, lib: str | None, tests: str | None = TESTS, body: str = BODY):
    task_dir = _task(make_task, lib, tests, body)
    before = {p: (task_dir / p).read_bytes() for p in list_files(task_dir)}
    outcome: Outcome = coding.solve(load_one(task_dir))
    after = {p: (task_dir / p).read_bytes() for p in list_files(task_dir)}
    changed = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    return outcome, changed, (task_dir / "workspace" / "mylib.py")


def test_two_independent_bugs_are_repaired(make_task: MakeTask) -> None:
    broken = GOOD_LIB.replace("reversed(", "sorted(").replace("s += x", "s -= x")

    outcome, changed, lib = _solve(make_task, broken)

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, "DONE")
    assert changed == ["workspace/mylib.py"]
    assert lib.read_bytes().decode() == GOOD_LIB


def test_crlf_source_keeps_its_line_endings(make_task: MakeTask) -> None:
    broken = GOOD_LIB.replace("x > best", "x < best").replace("\n", "\r\n")

    outcome, _changed, lib = _solve(make_task, broken)

    assert outcome.verdict is Verdict.SOLVED
    assert lib.read_bytes() == GOOD_LIB.replace("\n", "\r\n").encode()


def test_semantically_wrong_candidate_is_not_chosen(make_task: MakeTask) -> None:
    broken = GOOD_LIB.replace("x % 2 == 0", "x % 2 == 1")

    outcome, _changed, lib = _solve(make_task, broken)

    fixed = lib.read_text(encoding="utf-8")
    assert outcome.verdict is Verdict.SOLVED
    assert "% 3" not in fixed
    assert "x % 2 != 1" in fixed or "x % 2 == 0" in fixed
    assert "ambiguous" not in (outcome.evidence or "")  # the count_even oracle settles it


def test_without_an_oracle_the_choice_is_marked_ambiguous(make_task: MakeTask) -> None:
    lib = GOOD_LIB.replace("count_even", "tally").replace("x % 2 == 0", "x % 2 == 1")
    tests = TESTS.replace("count_even", "tally")

    outcome, _changed, path = _solve(make_task, lib, tests)

    assert outcome.verdict is Verdict.SOLVED
    assert "% 3" not in path.read_text(encoding="utf-8")
    assert "ambiguous: 2 behaviours among 3 candidates" in (outcome.evidence or "")


def test_audit_repairs_a_function_that_no_visible_test_covers(make_task: MakeTask) -> None:
    broken = GOOD_LIB.replace("x > best", "x < best").replace("s += x", "s -= x")
    tests = TESTS.replace(
        "    def test_total(self):\n        self.assertEqual(m.total([1, 2, 3]), 6)\n\n", ""
    )
    assert "test_total" not in tests

    outcome, changed, path = _solve(make_task, broken, tests)

    assert outcome.verdict is Verdict.SOLVED and changed == ["workspace/mylib.py"]
    assert path.read_text(encoding="utf-8") == GOOD_LIB
    assert "audit L" in (outcome.evidence or "")


def test_audit_leaves_alone_a_function_whose_tests_contradict_the_oracle(
    make_task: MakeTask,
) -> None:
    # Here `total` is specified by its test as the negated sum: not a bug, just not our oracle.
    lib = GOOD_LIB.replace("s += x", "s -= x")
    tests = TESTS.replace("m.total([1, 2, 3]), 6", "m.total([1, 2, 3]), -6")

    outcome, changed, _path = _solve(make_task, lib, tests)

    assert outcome.verdict is Verdict.SOLVED and changed == []


def test_oracle_outputs_match_what_the_probe_prints() -> None:
    assert oracles.expected_output("total", ([1, 2, 3],)) == "6"
    assert oracles.expected_output("average", ([1, 2],)) == "1.5"
    assert oracles.expected_output("average", ([],)) == "EXC:ZeroDivisionError"
    assert oracles.expected_output("last_n", ([1, 2, 3], 2)) == "[2, 3]"
    assert oracles.expected_output("last_n", ([1, 2, 3], 0)) == "[]"
    assert oracles.expected_output("reverse_words", ("c a b",)) == "'b a c'"
    assert oracles.expected_output("no_such_function", (1,)) is None


def test_slice_bug_is_repaired(make_task: MakeTask) -> None:
    outcome, _changed, lib = _solve(make_task, GOOD_LIB.replace("xs[:n]", "xs[:n - 1]"))

    assert outcome.verdict is Verdict.SOLVED
    fixed = lib.read_text(encoding="utf-8")
    assert "xs[:(n - 1) + 1]" in fixed or "xs[:n - 0]" in fixed


def test_green_tests_need_no_change(make_task: MakeTask) -> None:
    outcome, changed, _lib = _solve(make_task, GOOD_LIB)

    assert (outcome.verdict, outcome.answer) == (Verdict.SOLVED, "DONE")
    assert changed == []


@pytest.mark.parametrize(
    "broken",
    [
        GOOD_LIB.replace(
            "    return ' '.join(reversed(s.split()))", "    return s.upper().title()"
        ),
        GOOD_LIB.replace("def total(xs):", "def total(xs:"),
        GOOD_LIB.replace("def total(xs):", "import no_such_module_42\n\n\ndef total(xs):"),
    ],
)
def test_bugs_beyond_single_edits_are_refused_and_nothing_is_published(
    make_task: MakeTask, broken: str
) -> None:
    outcome, changed, _lib = _solve(make_task, broken)

    assert outcome.verdict is Verdict.UNPARSED
    assert outcome.answer is None
    assert changed == []


def test_hanging_tests_are_refused(make_task: MakeTask, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coding, "_TEST_TIMEOUT_S", 1.5)
    hanging = GOOD_LIB.replace("    s = 0\n", "    s = 0\n    while True:\n        pass\n")

    outcome, changed, _lib = _solve(make_task, hanging)

    assert outcome.verdict is Verdict.UNPARSED
    assert changed == []


@pytest.mark.parametrize(
    ("lib", "tests"),
    [
        (None, TESTS),
        (GOOD_LIB, None),
        (
            GOOD_LIB.replace("s += x", "s -= x"),
            TESTS.replace(
                "self.assertEqual(m.total([1, 2, 3]), 6)",
                "self.assertEqual(m.total([1, 2, 3]), 6)\n"
                "        self.assertEqual(m.total([1, 2, 3]), 7)",
            ),
        ),
        (
            GOOD_LIB.replace("s += x", "s -= x"),
            TESTS.replace(
                "self.assertEqual(m.total([1, 2, 3]), 6)",
                "self.assertEqual(m.total([1, 2, 3]), 6)\n"
                "        self.assertEqual(m.total([1, 2, 3]), 6 + 1)",
            ),
        ),
        (
            GOOD_LIB,
            TESTS.replace(
                "    def test_total(self):",
                "    def test_arithmetic(self):\n        self.assertEqual(1 + 1, 3)\n\n"
                "    def test_total(self):",
            ),
        ),
    ],
)
def test_impossible_setups_are_ungrounded(
    make_task: MakeTask, lib: str | None, tests: str | None
) -> None:
    outcome, changed, _lib = _solve(make_task, lib, tests)

    assert outcome.verdict is Verdict.UNGROUNDED
    assert changed == []


@pytest.mark.parametrize(
    "body",
    [
        BODY.replace(
            "Fix the bugs in mylib.py", "Fix the bugs in mylib.py and add a median function"
        ),
        BODY.replace("(do not edit the tests)", "(you may edit the tests)"),
        BODY.replace("until all tests pass", "until test_total passes"),
        BODY + " Do not change the function total.",
        "Make the project work.",
        BODY.replace("mylib.py", "../mylib.py"),
    ],
)
def test_changed_wording_is_refused_and_nothing_is_touched(make_task: MakeTask, body: str) -> None:
    outcome, changed, _lib = _solve(make_task, GOOD_LIB.replace("s += x", "s -= x"), body=body)

    assert outcome.verdict is Verdict.UNPARSED
    assert changed == []


def test_no_bytecode_or_scratch_files_appear_in_the_workspace(make_task: MakeTask) -> None:
    task_dir = _task(make_task, GOOD_LIB.replace("x > best", "x < best"))
    before = list_files(task_dir)

    coding.solve(load_one(task_dir))

    assert list_files(task_dir) == before


def test_enumerate_mutations_covers_the_documented_classes() -> None:
    source = (
        "def f(xs, n):\n"
        "    best = xs[0]\n"
        "    total = 0\n"
        "    for x in xs[:n]:\n"
        "        if x < best and not x == 3:\n"
        "            total += x * 2\n"
        "    return sorted(xs) if total else best\n"
        "\n"
        "def g(a, b):\n"
        "    return a\n"
    )

    mutated = {m.apply(source) for m in coding.enumerate_mutations(source)}

    for old, new in [
        ("x < best", "x > best"),
        ("x < best", "x <= best"),
        ("x == 3", "x != 3"),
        ("total += x", "total -= x"),
        ("x * 2", "x // 2"),
        ("best and not", "best or not"),
        ("not x == 3", "(x == 3)"),
        ("xs[0]", "xs[1]"),
        ("total = 0", "total = 1"),
        ("xs[:n]", "xs[:n - 1]"),
        ("xs[:n]", "xs[n:]"),
        ("sorted(xs)", "reversed(xs)"),
        ("    return a\n", "    return b\n"),
    ]:
        assert source.replace(old, new) in mutated, (old, new)
    assert all(m.owner in {"f", "g"} for m in coding.enumerate_mutations(source))


def test_parse_test_output() -> None:
    report = (
        "test_a (__main__.T.test_a) ... ok\n"
        "test_b (__main__.T.test_b) ... FAIL\n"
        "test_c (__main__.T.test_c) ... ERROR\n\n"
        "======\nERROR: test_c (__main__.T.test_c)\n------\nTraceback...\n\n"
        "======\nFAIL: test_b (__main__.T.test_b)\n------\nTraceback...\n\n"
        "------\nRan 3 tests in 0.001s\n\nFAILED (failures=1, errors=1)\n"
    )

    failed = coding.parse_test_output(1, report, timed_out=False)
    assert failed.status == "failed"
    assert {key.split(" ")[0] for key in failed.failing} == {"test_b", "test_c"}
    assert coding.parse_test_output(0, "Ran 2 tests in 0.0s\n\nOK\n", False).status == "ok"
    assert coding.parse_test_output(1, "Traceback\nSyntaxError: bad\n", False).status == "broken"
    assert coding.parse_test_output(None, "", True).status == "timeout"
