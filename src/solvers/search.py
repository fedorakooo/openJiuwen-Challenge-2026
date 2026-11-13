"""Strict solver for knowledge-base lookups: entity linking and typed relation chains."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.core.sandbox import PathEscapeError, ensure_within
from src.core.tasks import Outcome, Task, Verdict, strip_boilerplate
from src.solvers.sqlworld import SqlWorld, TableInfo, create_table

_MAX_FILES = 5000
_MAX_FILE_BYTES = 1024 * 1024
_MIN_PARSED_SHARE = 0.9
_MAX_NAME_DISTANCE = 2
_MIN_FUZZY_LENGTH = 5
_LINK_SHARE = 0.8
_IDENTIFIER = re.compile(r"[a-z_]\w*")

_BODY = re.compile(
    r"The directory (?P<dir>[\w-]+)/ contains a small knowledge base of markdown files\. "
    r"(?P<question>[^?]+\?)"
)
# (question pattern, requested key, value pattern)
_QUESTIONS: tuple[tuple[re.Pattern[str], str, re.Pattern[str]], ...] = (
    (re.compile(r"In which year was (?P<np>.+) born\?"), "born", re.compile(r"\d{1,4}")),
    (re.compile(r"In which year was (?P<np>.+) founded\?"), "founded", re.compile(r"\d{1,4}")),
    (
        re.compile(r"In which city is the headquarters of (?P<np>.+)\?"),
        "headquarters",
        re.compile(r".+"),
    ),
    (re.compile(r"Who is the mentor of (?P<np>.+)\?"), "mentor", re.compile(r".+")),
)
_TITLE = re.compile(r"#\s+(?P<name>\S.*)")
_FIELD = re.compile(r"(?P<key>[^:#][^:]*):\s*(?P<value>.*)")
_PROPER_NAME = re.compile(r"[A-Z][\w.'-]*(?: [A-Z][\w.'-]*)*")
_OF_FORM = re.compile(r"the (?P<relation>[\w ]+?) of (?P<inner>.+)")


class _Stop(Exception):
    """Evaluation ends with a verdict other than SOLVED."""

    def __init__(self, verdict: Verdict, reason: str) -> None:
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


@dataclass
class Entity:
    """One knowledge-base file."""

    name: str
    kind: str
    fields: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class KnowledgeBase:
    """Entities of one task; nothing is shared between tasks."""

    by_name: dict[str, list[Entity]] = field(default_factory=dict)
    keys: set[str] = field(default_factory=set)
    total_files: int = 0
    parsed_files: int = 0


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _parse_entity(path: Path) -> Entity | None:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        lines = [ln.strip() for ln in path.read_bytes().decode("utf-8-sig").splitlines()]
    except (OSError, UnicodeDecodeError):
        return None
    lines = [line for line in lines if line]
    title = _TITLE.fullmatch(lines[0]) if lines else None
    if title is None:
        return None
    entity = Entity(name=title["name"].strip(), kind=path.stem.split("_", 1)[0].casefold())
    for line in lines[1:]:
        match = _FIELD.fullmatch(line)
        if match is None:
            return None
        values = entity.fields.setdefault(_normalise(match["key"]), [])
        value = match["value"].strip()
        if value and value not in values:
            values.append(value)
    return entity


def load_knowledge_base(directory: Path) -> KnowledgeBase:
    """Parse every markdown file under `directory` into an entity."""
    kb = KnowledgeBase()
    for path in sorted(directory.rglob("*.md"))[:_MAX_FILES]:
        if not path.is_file():
            continue
        kb.total_files += 1
        entity = _parse_entity(path)
        if entity is None:
            continue
        kb.parsed_files += 1
        kb.by_name.setdefault(_normalise(entity.name), []).append(entity)
        kb.keys.update(entity.fields)
    return kb


def _distance(left: str, right: str, limit: int) -> int:
    """Levenshtein distance, cut off above `limit`."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


class _Evaluator:
    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self.near_match: str | None = None
        self.path: list[str] = []

    def _unique(self, name: str, missing: str) -> Entity:
        matches = self._kb.by_name.get(_normalise(name), [])
        if len(matches) > 1:
            raise _Stop(Verdict.UNGROUNDED, f"several entities are named {name!r}")
        if not matches:
            raise _Stop(Verdict.UNGROUNDED, missing)
        return matches[0]

    def _nearest(self, name: str) -> Entity:
        wanted = _normalise(name)
        if len(wanted) < _MIN_FUZZY_LENGTH:
            raise _Stop(Verdict.UNGROUNDED, f"entity {name!r} is absent")
        scored = sorted(
            (_distance(wanted, known, _MAX_NAME_DISTANCE), known) for known in self._kb.by_name
        )
        close = [known for score, known in scored if score == scored[0][0]] if scored else []
        if not scored or scored[0][0] > _MAX_NAME_DISTANCE or len(close) != 1:
            raise _Stop(Verdict.UNGROUNDED, f"entity {name!r} is absent")
        entity = self._unique(close[0], f"entity {name!r} is absent")
        self.near_match = entity.name
        return entity

    def value(self, entity: Entity, key: str) -> str:
        values = entity.fields.get(key, [])
        if not values:
            raise _Stop(Verdict.UNGROUNDED, f"{entity.name!r} has no {key!r}")
        if len(values) > 1:
            raise _Stop(Verdict.UNGROUNDED, f"{entity.name!r} has conflicting {key!r} values")
        self.path.append(key)
        return values[0]

    def _follow(self, entity: Entity, relation: str) -> Entity:
        target = self.value(entity, relation)
        return self._unique(target, f"{relation!r} of {entity.name!r} points to a missing entity")

    def entity(self, text: str) -> Entity:
        """NP := ENTITY | NP "'s" REL | "the" REL "of" NP."""
        text = text.strip()
        if _normalise(text) in self._kb.by_name:
            entity = self._unique(text, "")
            self.path.append(entity.name)
            return entity

        relational = False
        for found in reversed(list(re.finditer(r"'s ", text))):
            relational = True
            relation = _normalise(text[found.end() :])
            if relation in self._kb.keys:
                return self._follow(self.entity(text[: found.start()]), relation)
        of_form = _OF_FORM.fullmatch(text)
        if of_form is not None:
            relational = True
            relation = _normalise(of_form["relation"])
            if relation in self._kb.keys:
                return self._follow(self.entity(of_form["inner"]), relation)

        if not relational and _PROPER_NAME.fullmatch(text):
            entity = self._nearest(text)
            self.path.append(entity.name)
            return entity
        raise _Stop(Verdict.UNPARSED, "noun phrase not recognised")


def _answer(kb: KnowledgeBase, question: str) -> Outcome:
    for pattern, key, value_pattern in _QUESTIONS:
        match = pattern.fullmatch(question)
        if match is None:
            continue
        evaluator = _Evaluator(kb)
        value = evaluator.value(evaluator.entity(match["np"]), key)
        if not value_pattern.fullmatch(value):
            raise _Stop(Verdict.UNPARSED, f"value of {key!r} has an unexpected shape")
        evidence = " -> ".join(evaluator.path)
        if evaluator.near_match is not None:
            reason = f"entity is absent, nearest is {evaluator.near_match!r}"
            return Outcome(Verdict.UNGROUNDED, answer=value, reason=reason, evidence=evidence)
        return Outcome(Verdict.SOLVED, answer=value, evidence=evidence)
    raise _Stop(Verdict.UNPARSED, "question not recognised")


def solve(task: Task) -> Outcome:
    """Answer only when the whole instruction matches the question grammar."""
    match = _BODY.fullmatch(strip_boilerplate(task.instruction))
    if match is None:
        return Outcome(Verdict.UNPARSED, reason="instruction not recognised")
    if task.workspace is None:
        return Outcome(Verdict.UNGROUNDED, reason="no workspace")
    try:
        directory = ensure_within(task.workspace / match["dir"], task.workspace)
    except PathEscapeError:
        return Outcome(Verdict.UNGROUNDED, reason="knowledge base is outside the workspace")
    if not directory.is_dir():
        return Outcome(Verdict.UNGROUNDED, reason="knowledge base directory is missing")
    kb = load_knowledge_base(directory)
    if kb.total_files == 0:
        return Outcome(Verdict.UNGROUNDED, reason="knowledge base is empty")
    if not is_structured(kb):
        return Outcome(Verdict.UNPARSED, reason="knowledge base is not in key-value form")
    try:
        return _answer(kb, match["question"])
    except _Stop as stop:
        return Outcome(stop.verdict, reason=stop.reason)


def _column(key: str) -> str:
    name = re.sub(r"\W+", "_", key).strip("_") or "field"
    return name + "_" if name == "name" or not _IDENTIFIER.fullmatch(name) else name


def build_world(kb: KnowledgeBase) -> SqlWorld:
    """Entities as one table per kind, with reference columns inferred from the values."""
    by_kind: dict[str, list[Entity]] = {}
    for entities in kb.by_name.values():
        for entity in entities:
            kind = entity.kind if _IDENTIFIER.fullmatch(entity.kind) else "entity"
            by_kind.setdefault(kind, []).append(entity)
    canonical = {
        kind: {_normalise(entity.name): entity.name for entity in entities}
        for kind, entities in by_kind.items()
    }

    world = SqlWorld(sqlite3.connect(":memory:", check_same_thread=False))
    for kind in sorted(by_kind):
        entities = sorted(by_kind[kind], key=lambda entity: entity.name)
        keys = sorted({key for entity in entities for key in entity.fields})
        info = TableInfo(kind, ["name", *[_column(key) for key in keys]])
        columns: dict[str, list[str | None]] = {}
        for key in keys:
            # A key with conflicting values in one file is unknown, not a guess.
            values = [
                e.fields[key][0] if len(e.fields.get(key, [])) == 1 else None for e in entities
            ]
            present = [value for value in values if value is not None]
            for target, names in canonical.items():
                hits = sum(1 for value in present if _normalise(value) in names)
                if present and hits >= _LINK_SHARE * len(present):
                    info.links[_column(key)] = f"{target}.name"
                    values = [
                        names.get(_normalise(v), v) if v is not None else None for v in values
                    ]
                    break
            columns[key] = values
        records = [
            [entity.name, *[columns[key][index] for key in keys]]
            for index, entity in enumerate(entities)
        ]
        create_table(world.connection, kind, info.columns, set(), records)
        world.tables.append(info)
        world.names.extend(entity.name for entity in entities)
    world.lock()
    return world


def is_structured(kb: KnowledgeBase) -> bool:
    """True when enough files follow the `# Name` / `Key: Value` layout to trust the model."""
    return kb.total_files > 0 and kb.parsed_files >= _MIN_PARSED_SHARE * kb.total_files


def read_documents(directory: Path, max_chars: int) -> str | None:
    """All markdown files as one text for the fragments mode; None if they do not fit."""
    parts: list[str] = []
    total = 0
    for path in sorted(directory.rglob("*.md"))[:_MAX_FILES]:
        try:
            text = path.read_bytes().decode("utf-8-sig").strip()
        except (OSError, UnicodeDecodeError):
            continue
        part = f"[{path.name}]\n{text}"
        total += len(part) + 2
        if total > max_chars:
            return None
        parts.append(part)
    return "\n\n".join(parts) if parts else None
