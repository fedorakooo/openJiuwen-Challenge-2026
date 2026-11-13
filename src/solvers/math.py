"""Strict solver for stock word problems: every sentence must match a known template."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable

from src.core.tasks import Outcome, Task, Verdict, strip_boilerplate

_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")
_START = re.compile(r"An? (?P<place>[A-Za-z]+) starts with (?P<n>\d+) (?P<items>[A-Za-z]+)\.")
_QUESTION = re.compile(r"How many (?P<items>[A-Za-z]+) does it have at the end\?")
_SPLIT = re.compile(
    r"Then the stock is split equally into (?P<k>\d+) (?P<places>[A-Za-z]+) "
    r"and it keeps one share \(rounded down\)\."
)
_DOUBLE = re.compile(r"Then the stock is doubled by merging with a sister (?P<place>[A-Za-z]+)\.")
_SHIP = re.compile(r"It ships (?P<a>\d+) (?P<items>[A-Za-z]+) per day for (?P<d>\d+) days?\.")
_RECEIVE = re.compile(
    r"It receives (?P<k>\d+) deliver(?:y|ies) of (?P<m>\d+) (?P<items>[A-Za-z]+) each\."
)


class _Refusal(Exception):
    """The wording is outside the grammar."""


def _plurals(word: str) -> set[str]:
    return {word + "s", word + "es"}


def _require(condition: bool, why: str) -> None:
    if not condition:
        raise _Refusal(why)


def _step(sentence: str, place: str, items: str) -> tuple[Callable[[int], int], str]:
    """Translate one operation sentence into a function over the stock and its notation."""
    if match := _SPLIT.fullmatch(sentence):
        divisor = int(match["k"])
        _require(match["places"] in _plurals(place), "unexpected place word")
        _require(divisor > 0, "division by zero")
        return (lambda s: s // divisor), f"s//={divisor}"
    if match := _DOUBLE.fullmatch(sentence):
        _require(match["place"] == place, "unexpected place word")
        return (lambda s: s * 2), "s*=2"
    if match := _SHIP.fullmatch(sentence):
        per_day, days = int(match["a"]), int(match["d"])
        _require(match["items"] == items, "unexpected item word")
        return (lambda s: s - per_day * days), f"s-={per_day}*{days}"
    if match := _RECEIVE.fullmatch(sentence):
        count, size = int(match["k"]), int(match["m"])
        _require(match["items"] == items, "unexpected item word")
        return (lambda s: s + count * size), f"s+={count}*{size}"
    raise _Refusal("unknown sentence")


def _solve_body(body: str) -> tuple[int, str]:
    sentences = _SENTENCE_BREAK.split(body)
    _require(len(sentences) >= 2, "too few sentences")
    start = _START.fullmatch(sentences[0])
    question = _QUESTION.fullmatch(sentences[-1])
    _require(start is not None, "unknown opening sentence")
    _require(question is not None, "unknown question")
    assert start is not None and question is not None
    place, items = start["place"], start["items"]
    _require(question["items"] == items, "question asks about another item")

    stock = int(start["n"])
    program = [f"s={stock}"]
    for sentence in sentences[1:-1]:
        operation, notation = _step(sentence, place, items)
        stock = operation(stock)
        program.append(notation)
        _require(stock >= 0, "stock goes negative")
    return stock, "; ".join(program)


def solve(task: Task) -> Outcome:
    """Answer only when the whole instruction is consumed by the sentence grammar."""
    try:
        answer, program = _solve_body(strip_boilerplate(task.instruction))
    except _Refusal as refusal:
        return Outcome(Verdict.UNPARSED, reason=str(refusal))
    return Outcome(Verdict.SOLVED, answer=answer, evidence=program)


# --- T1: safe evaluation of a model-written program over integers -----------------------

_MAX_STATEMENTS = 40
_MAX_MAGNITUDE = 10**30
_MAX_EXPONENT = 6
_VARIABLE = re.compile(r"[a-z_][a-z0-9_]{0,15}")
_FUNCTIONS = {"min": min, "max": max, "abs": abs}
_NUMBER = re.compile(r"\d+")
_FREE_CONSTANTS = frozenset({0, 1, 2, 10, 100})


class ProgramError(ValueError):
    """The program is outside the allowed subset or fails while running."""


def _bounded(value: int) -> int:
    if abs(value) > _MAX_MAGNITUDE:
        raise ProgramError("value out of range")
    return value


def _binary(operator: ast.operator, left: int, right: int) -> int:
    if isinstance(operator, ast.Add):
        return left + right
    if isinstance(operator, ast.Sub):
        return left - right
    if isinstance(operator, ast.Mult):
        return left * right
    if isinstance(operator, ast.FloorDiv | ast.Mod):
        if right == 0:
            raise ProgramError("division by zero")
        return left // right if isinstance(operator, ast.FloorDiv) else left % right
    if isinstance(operator, ast.Pow):
        if not 0 <= right <= _MAX_EXPONENT:
            raise ProgramError("exponent out of range")
        return left**right
    raise ProgramError(f"operator {type(operator).__name__} is not allowed")


def _evaluate(node: ast.expr, variables: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return _bounded(node.value)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id not in variables:
            raise ProgramError(f"name {node.id!r} is not defined")
        return variables[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        value = _evaluate(node.operand, variables)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, variables)
        return _bounded(_binary(node.op, left, _evaluate(node.right, variables)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = _FUNCTIONS.get(node.func.id)
        if function is None or not node.args:
            raise ProgramError("call is not allowed")
        return _bounded(function(*(_evaluate(argument, variables) for argument in node.args)))
    raise ProgramError(f"{type(node).__name__} is not allowed")


def evaluate_program(text: str) -> int:
    """Run `s=...; s-=...` statements with our own AST walker; return the final `s`."""
    try:
        tree = ast.parse(text.strip())
    except (SyntaxError, ValueError) as exc:
        raise ProgramError("not valid syntax") from exc
    if not 0 < len(tree.body) <= _MAX_STATEMENTS:
        raise ProgramError("unexpected number of statements")
    variables: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], _evaluate(statement.value, variables)
        elif isinstance(statement, ast.AugAssign):
            target = statement.target
            current = _evaluate(ast.Name(id=getattr(target, "id", ""), ctx=ast.Load()), variables)
            value = _bounded(_binary(statement.op, current, _evaluate(statement.value, variables)))
        else:
            raise ProgramError(f"{type(statement).__name__} is not allowed")
        if not isinstance(target, ast.Name) or not _VARIABLE.fullmatch(target.id):
            raise ProgramError("assignment target is not a simple variable")
        variables[target.id] = value
    if "s" not in variables:
        raise ProgramError("the program never assigns s")
    return variables["s"]


def number_coverage_complaint(question: str, program: str) -> str | None:
    """Numbers of the question the program ignores, or numbers it invents."""
    asked = {int(number) for number in _NUMBER.findall(question)}
    used = {int(number) for number in _NUMBER.findall(program)}
    unused = sorted(asked - used)
    # `(s + k - 1) // k` is the usual way to round a division up: k - 1 is not invented.
    ceiling = {number - 1 for number in asked}
    invented = sorted(used - asked - _FREE_CONSTANTS - ceiling)
    if unused:
        return f"number {unused[0]} from the question is unused"
    if invented:
        return f"number {invented[0]} is not in the question"
    return None
