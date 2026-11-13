"""Training task generator, dev only. Mirrors the structure of the organisers' samples.

Usage:
    python scripts/gen_tasks.py --out /tmp/batch/tasks --ref /tmp/batch/ref [--count 100] [--seed 1]

`--out` gets what the harness may see (task.json without `expected`, workspace/).
`--ref` gets what only the judge may see: `<task_id>/task.json` with `expected`, `feasible`
and `family`, and hidden tests for coding. Keep `--ref` outside the tasks directory.

Families per type: wording exactly as in the samples, paraphrases, tricks (distractors, wrong
declared sizes, bugs without a visible test) and infeasible variants with their feasible twins.
Paraphrases are hand-written; none of them was produced by a model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_TAIL = (
    " Work only inside the current directory. When you are done, write ONLY the final answer to "
    "the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)
DONE_TAIL = (
    " Work only inside the current directory. When you are done, write ONLY the word DONE to "
    "the file answer.txt in the current directory (for example with bash: echo 42 > answer.txt)."
)
ALT_ANSWER_TAIL = " Stay in this folder. Put just the answer, nothing else, into answer.txt."
ALT_DONE_TAIL = " Stay in this folder. Once the tests are green, put the word DONE into answer.txt."
MATH_TOOL = " Compute it with the bash tool (python3 -c)."
DATA_TOOL = " Use bash with python3 to compute it."
SEARCH_TOOL = " Use the file tools or bash (ls, grep, cat) to find it."


@dataclass
class Generated:
    """One task before it is written to disk."""

    task_type: str
    family: str
    instruction: str
    check: str
    expected: str | None
    feasible: bool = True
    files: dict[str, str] = field(default_factory=dict)
    hidden: dict[str, str] = field(default_factory=dict)
    free_answer: str | None = None


def _tail(rng: random.Random, tool: str, paraphrased: bool, done: bool = False) -> str:
    if paraphrased and rng.random() < 0.4:
        return ALT_DONE_TAIL if done else ALT_ANSWER_TAIL
    return tool + (DONE_TAIL if done else ANSWER_TAIL)


# --- math -------------------------------------------------------------------------------

DOMAINS = [("warehouse", "boxes"), ("depot", "crates"), ("shop", "jars"), ("farm", "sacks")]


def _math_ops(rng: random.Random, length: int, extended: bool) -> tuple[int, list[tuple], int]:
    """A random chain that never goes negative: (start, operations, final value)."""
    kinds = ["split", "double", "ship", "receive"]
    if extended:
        kinds += ["split_up", "triple", "lose"]
    while True:
        start = rng.randint(200, 3000)
        stock, ops = start, []
        for _ in range(length):
            kind = rng.choice(kinds)
            if kind in ("split", "split_up"):
                k = rng.randint(2, 5)
                stock = stock // k if kind == "split" else -(-stock // k)
                ops.append((kind, k))
            elif kind == "double":
                stock *= 2
                ops.append((kind,))
            elif kind == "triple":
                stock *= 3
                ops.append((kind,))
            elif kind == "ship":
                a, d = rng.randint(5, 60), rng.randint(2, 6)
                stock -= a * d
                ops.append((kind, a, d))
            elif kind == "receive":
                k, m = rng.randint(2, 6), rng.randint(10, 120)
                stock += k * m
                ops.append((kind, k, m))
            else:
                a = rng.randint(5, 90)
                stock -= a
                ops.append((kind, a))
            if stock < 0:
                break
        if stock >= 0:
            return start, ops, stock


def _canonical_math(start: int, ops: list[tuple], place: str, items: str) -> str:
    parts = [f"A {place} starts with {start} {items}."]
    for op in ops:
        if op[0] == "split":
            parts.append(
                f"Then the stock is split equally into {op[1]} {place}s and it keeps one share "
                "(rounded down)."
            )
        elif op[0] == "double":
            parts.append(f"Then the stock is doubled by merging with a sister {place}.")
        elif op[0] == "ship":
            parts.append(f"It ships {op[1]} {items} per day for {op[2]} days.")
        else:
            parts.append(f"It receives {op[1]} deliveries of {op[2]} {items} each.")
    parts.append(f"How many {items} does it have at the end?")
    return " ".join(parts)


def _paraphrased_math(
    rng: random.Random, start: int, ops: list[tuple], place: str, items: str
) -> str:
    opening = rng.choice(
        [f"A {place} has {start} {items} in stock.", f"There are {start} {items} in a {place}."]
    )
    parts = [opening]
    for op in ops:
        if op[0] in ("split", "split_up"):
            rounding = "down" if op[0] == "split" else "up"
            parts.append(
                f"The {items} are then divided evenly among {op[1]} {place}s, and this {place} "
                f"keeps one part (round {rounding})."
            )
        elif op[0] == "double":
            parts.append("Its stock then doubles.")
        elif op[0] == "triple":
            parts.append("Its stock then triples.")
        elif op[0] == "ship":
            parts.append(f"It sends out {op[1]} {items} every day for {op[2]} days.")
        elif op[0] == "receive":
            parts.append(f"{op[1]} shipments of {op[2]} {items} each arrive.")
        else:
            parts.append(f"It loses {op[1]} {items} to damage.")
    parts.append(
        rng.choice([f"How many {items} are left?", f"What is the final number of {items}?"])
    )
    return " ".join(parts)


def gen_math(rng: random.Random) -> Generated:
    family = rng.choices(
        ["canonical", "domain", "paraphrase", "distractor", "infeasible_missing"],
        weights=[25, 15, 35, 12, 13],
    )[0]
    place, items = DOMAINS[0] if family == "canonical" else rng.choice(DOMAINS)
    extended = family == "paraphrase"
    start, ops, final = _math_ops(rng, rng.randint(1, 6), extended)
    if family in ("canonical", "domain"):
        body = _canonical_math(start, ops, place, items)
    elif family == "paraphrase":
        body = _paraphrased_math(rng, start, ops, place, items)
    elif family == "distractor":
        body = _canonical_math(start, ops, place, items)
        noise = rng.choice(
            [
                f"The {place} was opened in {rng.randint(1950, 2020)}.",
                f"It employs {rng.randint(3, 40)} people.",
                f"A truck holds {rng.randint(100, 400)} {items}.",
            ]
        )
        sentences = body.split(". ")
        sentences.insert(rng.randint(1, len(sentences) - 1), noise.rstrip("."))
        body = ". ".join(sentences)
    else:
        body = _canonical_math(start, [*ops, ("ship", 7, 3)], place, items).replace(
            f"It ships 7 {items} per day for 3 days.", f"It ships some {items} per day for 3 days."
        )
        return Generated(
            "math", family, body + _tail(rng, MATH_TOOL, False), "exact_int", None, False
        )
    paraphrased = family == "paraphrase"
    return Generated(
        "math", family, body + _tail(rng, MATH_TOOL, paraphrased), "exact_int", str(final)
    )


# --- data_analysis ------------------------------------------------------------------------

REGIONS = ["east", "north", "south", "west"]
PRODUCTS = ["gadget", "gizmo", "sprocket", "widget"]
HEADER = "order_id,region,product,month,units,price"


def _sales(rng: random.Random, rows: int, skip: tuple[str, int, int] | None = None) -> list[tuple]:
    """Random sales rows; `skip` = (product, low, high) keeps that product out of a range."""
    data = []
    for order_id in range(1, rows + 1):
        while True:
            row = (
                order_id,
                rng.choice(REGIONS),
                rng.choice(PRODUCTS),
                rng.randint(1, 12),
                rng.randint(1, 50),
                rng.randint(5, 99),
            )
            if skip is None or not (row[2] == skip[0] and skip[1] <= row[3] <= skip[2]):
                break
        data.append(row)
    return data


def _csv(data: list[tuple]) -> str:
    return "\r\n".join([HEADER, *(",".join(str(cell) for cell in row) for row in data)]) + "\r\n"


def gen_data(rng: random.Random) -> Generated:
    family = rng.choices(
        [
            "canonical",
            "wrong_rowcount",
            "empty_range",
            "paraphrase_sum",
            "single_month",
            "count_orders",
            "revenue",
            "region_filter",
            "infeasible_product",
            "infeasible_column",
            "infeasible_month",
        ],
        weights=[22, 6, 6, 14, 8, 8, 8, 8, 8, 6, 6],
    )[0]
    rows = rng.choice([20, 50, 120, 300])
    product = rng.choice(PRODUCTS)
    low = rng.randint(1, 11)
    high = min(12, low + rng.randint(0, 3))
    if family == "single_month":
        high = low
    data = _sales(rng, rows, (product, low, high) if family == "empty_range" else None)
    declared = rows + rng.choice([-7, 5, 100]) if family == "wrong_rowcount" else rows
    preamble = f"The file sales.csv has columns {HEADER} ({declared} rows). "
    in_range = [row for row in data if low <= row[3] <= high]
    matching = [row for row in in_range if row[2] == product]
    units = sum(row[4] for row in matching)
    canonical = (
        f"How many units of product '{product}' were sold in months {low} through {high} inclusive?"
    )
    feasible, expected, free = True, str(units), None
    paraphrased = family not in ("canonical", "wrong_rowcount", "empty_range")
    if family in ("canonical", "wrong_rowcount", "empty_range"):
        question = canonical
    elif family == "paraphrase_sum":
        question = (
            f"What is the total number of units sold for product '{product}' from month {low} "
            f"to month {high} (both inclusive)?"
        )
    elif family == "single_month":
        question = f"How many units of product '{product}' were sold in month {low}?"
    elif family == "count_orders":
        question = (
            f"How many orders of product '{product}' were placed in months {low} through {high} "
            "inclusive?"
        )
        expected = str(len(matching))
    elif family == "revenue":
        question = (
            f"What is the total revenue (units times price) of product '{product}' in months "
            f"{low} through {high} inclusive?"
        )
        expected = str(sum(row[4] * row[5] for row in matching))
    elif family == "region_filter":
        region = rng.choice(REGIONS)
        question = (
            f"How many units were sold in region '{region}' in months {low} through {high} "
            "inclusive?"
        )
        expected = str(sum(row[4] for row in in_range if row[1] == region))
    elif family == "infeasible_product":
        question = canonical.replace(product, rng.choice(["doohickey", "flange", "rotor"]))
        feasible, expected, free, paraphrased = False, None, "0", False
    elif family == "infeasible_column":
        question = canonical.replace("How many units", "How many returns")
        feasible, expected, paraphrased = False, None, False
    else:
        question = canonical.replace(f"{low} through {high}", "13 through 14")
        feasible, expected, free, paraphrased = False, None, "0", False
    return Generated(
        "data_analysis",
        family,
        preamble + question + _tail(rng, DATA_TOOL, paraphrased),
        "exact_int",
        expected,
        feasible,
        {"sales.csv": _csv(data)},
        free_answer=free,
    )


# --- search -------------------------------------------------------------------------------

COMPANIES = [
    "Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio",
    "Fjord Data", "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods",
]  # fmt: skip
CITIES = ["Osaka", "Quito", "Lagos", "Oslo", "Lima", "Hanoi", "Perth", "Turin", "Quebec", "Tunis"]
FIRST = [
    "Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira",
    "Maya", "Nikolai", "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya",
]  # fmt: skip
LAST = [
    "Garcia", "Haddad", "Ivanova", "Kowalski", "Muller", "Novak", "Petrov", "Sato", "Smith", "Wang",
]  # fmt: skip
PREAMBLE = "The directory kb/ contains a small knowledge base of markdown files. "


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def gen_search(rng: random.Random) -> Generated:
    family = rng.choices(
        [
            "born", "hq_employer", "founded_employer", "mentor", "two_hop",
            "paraphrase_born", "paraphrase_hq", "paraphrase_mentor",
            "infeasible_unknown", "infeasible_missing_key", "infeasible_dangling", "near_miss",
        ],
        weights=[14, 14, 8, 8, 8, 9, 9, 6, 7, 6, 5, 6],
    )[0]  # fmt: skip
    cities = rng.sample(CITIES, len(COMPANIES))
    companies = {
        name: {"Headquarters": city, "Founded": str(rng.randint(1950, 2015))}
        for name, city in zip(COMPANIES, cities, strict=True)
    }
    names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], rng.randint(20, 40))
    people = {
        name: {
            "Born": str(rng.randint(1955, 2000)),
            "Employer": rng.choice(COMPANIES),
            "Mentor": rng.choice([other for other in names if other != name]),
        }
        for name in names
    }
    person = rng.choice(names)
    facts = people[person]
    employer = companies[facts["Employer"]]
    mentor = people[facts["Mentor"]]
    feasible, free, paraphrased = True, None, family.startswith("paraphrase")
    if family == "born":
        question, expected = f"In which year was {person} born?", facts["Born"]
    elif family == "hq_employer":
        question = f"In which city is the headquarters of {person}'s employer?"
        expected = employer["Headquarters"]
    elif family == "founded_employer":
        question, expected = f"In which year was {person}'s employer founded?", employer["Founded"]
    elif family == "mentor":
        question, expected = f"Who is the mentor of {person}?", facts["Mentor"]
    elif family == "two_hop":
        question = f"In which city is the headquarters of {person}'s mentor's employer?"
        expected = companies[mentor["Employer"]]["Headquarters"]
    elif family == "paraphrase_born":
        question = rng.choice([f"What year was {person} born?", f"When was {person} born (year)?"])
        expected = facts["Born"]
    elif family == "paraphrase_hq":
        question = rng.choice(
            [
                f"Where is the company that {person} works for headquartered (city)?",
                f"Which city hosts the head office of {person}'s employer?",
            ]
        )
        expected = employer["Headquarters"]
    elif family == "paraphrase_mentor":
        question, expected = f"Who mentors {person}? Give the full name.", facts["Mentor"]
    elif family == "infeasible_unknown":
        ghost = next(f"{a} {b}" for a in FIRST for b in ["Okafor", "Lindqvist"] if a != "Alice")
        question, expected, feasible = f"In which year was {ghost} born?", None, False
    elif family == "infeasible_missing_key":
        del facts["Mentor"]
        question, expected, feasible = f"Who is the mentor of {person}?", None, False
    elif family == "infeasible_dangling":
        facts["Employer"] = "Zenith Cargo"
        question = f"In which city is the headquarters of {person}'s employer?"
        expected, feasible = None, False
    else:
        first, last = person.split(" ")
        typo = f"{first} {last[:-1]}{'z' if last[-1] != 'z' else 'x'}"
        question, expected, feasible, free = (
            f"In which year was {typo} born?", None, False, facts["Born"],
        )  # fmt: skip
    files = {}
    for name, fields_ in companies.items():
        body = "\n".join(f"{key}: {value}" for key, value in fields_.items())
        files[f"kb/company_{_slug(name)}.md"] = f"# {name}\n\n{body}\n"
    for name, fields_ in people.items():
        body = "\n".join(f"{key}: {value}" for key, value in fields_.items())
        files[f"kb/person_{_slug(name)}.md"] = f"# {name}\n\n{body}\n"
    return Generated(
        "search",
        family,
        PREAMBLE + question + _tail(rng, SEARCH_TOOL, paraphrased),
        "exact_str",
        expected,
        feasible,
        files,
        free_answer=free,
    )


# --- coding -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Function:
    """A library function with its tests and the bugs the generator can plant in it."""

    name: str
    source: str
    visible: tuple[str, ...]
    hidden: tuple[str, ...]
    single_edit: tuple[str, str]
    hard: tuple[str, str] | None = None


FUNCTIONS = [
    Function(
        "total",
        "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
        ("m.total([1, 2, 3]), 6", "m.total([4, 5]), 9"),
        ("m.total([]), 0", "m.total([-2, 2, 7]), 7", "m.total([10]), 10"),
        ("s += x", "s -= x"),
        ("    return s\n", "    return len(xs)\n"),
    ),
    Function(
        "maximum",
        "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n"
        "            best = x\n    return best\n",
        ("m.maximum([3, 9, 2]), 9", "m.maximum([1, 5, 4]), 5"),
        ("m.maximum([-3, -9, -2]), -2", "m.maximum([7]), 7", "m.maximum([2, 2, 8, 8]), 8"),
        ("x > best", "x < best"),
    ),
    Function(
        "minimum",
        "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n"
        "            best = x\n    return best\n",
        ("m.minimum([3, 9, 2]), 2", "m.minimum([6, 5, 8]), 5"),
        ("m.minimum([-3, -9, -2]), -9", "m.minimum([7]), 7", "m.minimum([4, 4, 1, 1]), 1"),
        ("x < best", "x > best"),
    ),
    Function(
        "average",
        "def average(xs):\n    return sum(xs) / len(xs)\n",
        ("m.average([1, 2]), 1.5", "m.average([2, 4, 9]), 5"),
        ("m.average([5]), 5", "m.average([1, 2, 2]), 5 / 3", "m.average([-1, 1]), 0"),
        ("sum(xs) / len(xs)", "sum(xs) // len(xs)"),
        ("sum(xs) / len(xs)", "sum(xs) / (len(xs) + 1)"),
    ),
    Function(
        "reverse_words",
        "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
        ("m.reverse_words('c a b'), 'b a c'", "m.reverse_words('one two'), 'two one'"),
        (
            "m.reverse_words('z y x w'), 'w x y z'",
            "m.reverse_words('solo'), 'solo'",
            "m.reverse_words('b a b'), 'b a b'",
        ),
        ("reversed(", "sorted("),
        ("' '.join(reversed(s.split()))", "' '.join(s.split())"),
    ),
    Function(
        "first_n",
        "def first_n(xs, n):\n    return xs[:n]\n",
        ("m.first_n([5, 6, 7, 8], 3), [5, 6, 7]", "m.first_n([1, 2, 3], 1), [1]"),
        ("m.first_n([1, 2], 2), [1, 2]", "m.first_n([9, 8, 7], 0), []", "m.first_n([], 2), []"),
        ("xs[:n]", "xs[n:]"),
    ),
    Function(
        "count_even",
        "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
        ("m.count_even([1, 2, 4, 6, 7]), 3", "m.count_even([2, 3]), 1"),
        ("m.count_even([3, 6, 9]), 1", "m.count_even([]), 0", "m.count_even([0, -2, 5]), 2"),
        ("x % 2 == 0", "x % 2 == 1"),
    ),
    Function(
        "count_odd",
        "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
        ("m.count_odd([1, 2, 4, 6, 7]), 2", "m.count_odd([3, 5, 8]), 2"),
        ("m.count_odd([3, 6, 9]), 2", "m.count_odd([]), 0", "m.count_odd([2, 4]), 0"),
        ("x % 2 == 1", "x % 2 == 0"),
    ),
    Function(
        "clamp",
        "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
        ("m.clamp(15, 0, 10), 10", "m.clamp(-4, 0, 10), 0"),
        ("m.clamp(5, 0, 10), 5", "m.clamp(0, 0, 10), 0", "m.clamp(11, 2, 3), 3"),
        ("max(lo, min(x, hi))", "max(lo, max(x, hi))"),
    ),
]
CODING_BODY = (
    "Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in "
    "mylib.py (do not edit the tests) until all tests pass."
)
CODING_PARAPHRASES = [
    "The unit tests in test_mylib.py are failing. Repair mylib.py so that every test passes; "
    "the test file must stay untouched.",
    "mylib.py contains bugs. Make `python3 test_mylib.py` succeed without changing the tests.",
]


def _test_file(asserts: list[str]) -> str:
    lines = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    for index, text in enumerate(asserts):
        name = text.split("(", 1)[0].removeprefix("m.")
        lines += [f"    def test_{name}_{index}(self):", f"        self.assertEqual({text})", ""]
    lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def gen_coding(rng: random.Random) -> Generated:
    family = rng.choices(
        ["single_edit", "untested_bug", "hard_bug", "paraphrase", "infeasible_contradiction"],
        weights=[40, 18, 17, 15, 10],
    )[0]
    chosen = rng.sample(FUNCTIONS, rng.randint(4, 6))
    rng.shuffle(chosen)
    buggy = rng.sample(chosen, rng.randint(1, min(3, len(chosen))))
    sources = {function.name: function.source for function in chosen}
    tested = list(chosen)
    broken: set[str] = set()
    if family == "hard_bug":
        hard = [function for function in chosen if function.hard]
        target = rng.choice(hard) if hard else None
        buggy = [function for function in buggy if function is not target][:1]
        if target is not None and target.hard is not None:
            sources[target.name] = target.source.replace(*target.hard)
            broken.add(target.name)
    for function in buggy:
        sources[function.name] = function.source.replace(*function.single_edit)
        broken.add(function.name)
    if family == "untested_bug" and len(buggy) >= 1:
        if len(buggy) == 1:
            extra = rng.choice([function for function in chosen if function is not buggy[0]])
            sources[extra.name] = extra.source.replace(*extra.single_edit)
            broken.add(extra.name)
            buggy.append(extra)
        tested = [function for function in chosen if function is not buggy[0]]
    # The first visible assertion of every function is the one that exposes its bug.
    visible = [
        function.visible[0] if function.name in broken else rng.choice(function.visible)
        for function in tested
    ]
    feasible = True
    if family == "infeasible_contradiction":
        victim = rng.choice(tested)
        call, value = victim.visible[0].rsplit(", ", 1)
        visible.append(f"{call}, {value} + 1" if value[0].isdigit() else f"{call}, 'zzz'")
        feasible = False
    hidden = [text for function in chosen for text in (*function.visible, *function.hidden)]
    paraphrased = family == "paraphrase"
    body = rng.choice(CODING_PARAPHRASES) if paraphrased else CODING_BODY
    return Generated(
        "coding",
        family,
        body + _tail(rng, "", paraphrased, done=True),
        "unittest",
        "tests_pass" if feasible else None,
        feasible,
        {
            "mylib.py": "\n\n".join(sources[function.name] for function in chosen),
            "test_mylib.py": _test_file(visible),
        },
        {"hidden_test_mylib.py": _test_file(hidden)},
    )


GENERATORS: dict[str, Callable[[random.Random], Generated]] = {
    "math": gen_math,
    "data_analysis": gen_data,
    "search": gen_search,
    "coding": gen_coding,
}


def write_batch(out: Path, ref: Path, count: int, seed: int) -> dict[str, int]:
    """Write `count` tasks, cycling through the four types; returns counts per family."""
    rng = random.Random(seed)
    families: dict[str, int] = {}
    types = list(GENERATORS)
    for index in range(count):
        task_type = types[index % len(types)]
        task = GENERATORS[task_type](rng)
        task_id = f"{task_type}_{index:04d}"
        public = {
            "task_id": task_id,
            "task_type": task.task_type,
            "instruction": task.instruction,
            "reward": rng.randint(1, 5),
            "check": task.check,
        }
        private = {
            **public,
            "expected": task.expected,
            "feasible": task.feasible,
            "family": task.family,
            "free_answer": task.free_answer,
        }
        for root, payload in ((out, public), (ref, private)):
            (root / task_id).mkdir(parents=True, exist_ok=False)
            text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            (root / task_id / "task.json").write_bytes(text.encode("utf-8"))
        for root, files in ((out, task.files), (ref, {**task.files, **task.hidden})):
            for relative, content in files.items():
                path = root / task_id / "workspace" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.encode("utf-8"))
        key = f"{task_type}/{task.family}"
        families[key] = families.get(key, 0) + 1
    return families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.ref.resolve().is_relative_to(args.out.resolve()):
        print("--ref must not be inside --out: the harness would see the answers", file=sys.stderr)
        return 2
    families = write_batch(args.out, args.ref, args.count, args.seed)
    for key in sorted(families):
        print(f"{families[key]:4d}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
