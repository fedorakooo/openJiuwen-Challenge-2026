"""Strict solver for single-table CSV questions: CSV -> SQLite, slot grammar, grounding."""

from __future__ import annotations

import csv
import io
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from src.core.sandbox import PathEscapeError, ensure_within
from src.core.tasks import Outcome, Task, Verdict, strip_boilerplate
from src.solvers.sqlworld import SqlWorld, TableInfo, create_table

logger = logging.getLogger(__name__)

_MAX_CSV_BYTES = 200 * 1024 * 1024
_MAX_TABLES = 5
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_INTEGER = re.compile(r"[+-]?\d+")
_DELIMITERS = ",;\t|"
# Columns whose legal values are known without looking at the data.
_KNOWN_DOMAINS = {"month": (1, 12)}

_BODY = re.compile(
    r"The file (?P<file>[\w.-]+\.csv) has columns (?P<columns>\w+(?:,\w+)*) "
    r"\((?P<rows>\d+) rows\)\. "
    r"How many (?P<measure>\w+) of (?P<dimension>\w+) '(?P<literal>[^']*)' were sold "
    r"in (?P<range>\w+) (?P<low>\d+) through (?P<high>\d+) inclusive\?"
)


class TableError(Exception):
    """The file exists but is not a table this solver understands."""


@dataclass
class Table:
    """One CSV file loaded into an in-memory SQLite table named `t`."""

    columns: list[str]
    integer_columns: set[str]
    row_count: int
    connection: sqlite3.Connection
    name: str = "t"

    def distinct(self, column: str) -> list[str]:
        cursor = self.connection.execute(f'SELECT DISTINCT "{column}" FROM t')
        return [str(row[0]) for row in cursor]

    def bounds(self, column: str) -> tuple[int, int] | None:
        low, high = self.connection.execute(
            f'SELECT MIN("{column}"), MAX("{column}") FROM t'
        ).fetchone()
        return None if low is None else (int(low), int(high))


def _delimiter(header_line: str) -> str:
    try:
        return csv.Sniffer().sniff(header_line, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        return ","


def load_table(path: Path, name: str = "t") -> Table | None:
    """Load a CSV file; None means there is no data at all (missing, empty, unreadable)."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        if path.stat().st_size > _MAX_CSV_BYTES:
            raise TableError("file is too large")
        text = path.read_bytes().decode("utf-8-sig")
    except OSError:
        return None
    except UnicodeDecodeError as exc:
        raise TableError("not UTF-8") from exc

    lines = text.splitlines()
    if not lines or not lines[0].strip():
        return None
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=_delimiter(lines[0]))
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    columns, records = rows[0], rows[1:]
    if len(set(columns)) != len(columns) or not all(_IDENTIFIER.fullmatch(c) for c in columns):
        raise TableError("unusable header")
    if any(len(record) != len(columns) for record in records):
        raise TableError("ragged rows")

    integer_columns = {
        column
        for index, column in enumerate(columns)
        if records and all(_INTEGER.fullmatch(record[index]) for record in records)
    }
    connection = sqlite3.connect(":memory:")
    create_table(
        connection, name, columns, integer_columns, _typed(columns, integer_columns, records)
    )
    return Table(columns, integer_columns, len(records), connection, name)


def _typed(
    columns: list[str], integer_columns: set[str], records: list[list[str]]
) -> list[list[object]]:
    return [
        [
            int(cell) if column in integer_columns else cell
            for column, cell in zip(columns, record, strict=True)
        ]
        for record in records
    ]


def build_world(workspace: Path) -> SqlWorld | None:
    """All CSV files of a workspace as one read-only SQLite world for the model tier."""
    paths = sorted(p for p in workspace.rglob("*.csv") if p.is_file())[:_MAX_TABLES]
    # Built in a worker thread, queried from the event loop thread, never concurrently.
    world = SqlWorld(sqlite3.connect(":memory:", check_same_thread=False))
    used: set[str] = set()
    for path in paths:
        name = re.sub(r"\W+", "_", path.stem).strip("_").lower() or "data"
        if not _IDENTIFIER.fullmatch(name) or name in used:
            continue
        try:
            table = load_table(ensure_within(path, workspace), name="t")
        except (TableError, PathEscapeError):
            continue
        if table is None:
            continue
        rows = table.connection.execute("SELECT * FROM t").fetchall()
        table.connection.close()
        create_table(world.connection, name, table.columns, table.integer_columns, rows)
        world.tables.append(TableInfo(name, table.columns, table.integer_columns))
        used.add(name)
    if not world.tables:
        world.close()
        return None
    world.lock()
    return world


def _normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def _range_column(word: str, columns: list[str]) -> str | None:
    if word in columns:
        return word
    if word.endswith("s") and word[:-1] in columns:
        return word[:-1]
    return None


def _answer(table: Table, match: re.Match[str], task_id: str) -> Outcome:
    measure, dimension = match["measure"], match["dimension"]
    range_column = _range_column(match["range"], table.columns)
    declared = match["columns"].split(",")
    if declared != table.columns or int(match["rows"]) != table.row_count:
        logger.info("task %s: the instruction misdescribes the file, the file wins", task_id)

    for column in (measure, dimension):
        if column not in table.columns:
            return Outcome(Verdict.UNGROUNDED, reason=f"column {column!r} is absent")
    if range_column is None:
        return Outcome(Verdict.UNGROUNDED, reason=f"column for {match['range']!r} is absent")
    if table.row_count and not {measure, range_column} <= table.integer_columns:
        return Outcome(Verdict.UNPARSED, reason="measure or range column is not integer")

    low, high = int(match["low"]), int(match["high"])
    if low > high:
        return Outcome(Verdict.UNPARSED, reason="reversed range")
    domain = _KNOWN_DOMAINS.get(range_column)
    if domain is not None and not (domain[0] <= low and high <= domain[1]):
        return Outcome(Verdict.UNGROUNDED, answer=0, reason="range outside the column domain")
    bounds = table.bounds(range_column)
    if domain is None and bounds is not None and (high < bounds[0] or low > bounds[1]):
        return Outcome(Verdict.UNGROUNDED, answer=0, reason="range outside the column domain")

    wanted = _normalise(match["literal"])
    raw_values = [value for value in table.distinct(dimension) if _normalise(value) == wanted]
    if not raw_values:
        return Outcome(Verdict.UNGROUNDED, answer=0, reason="filter value is absent")
    if raw_values != [match["literal"]]:
        logger.info("task %s: filter value matched after normalisation", task_id)

    marks = ", ".join("?" for _ in raw_values)
    sql = (
        f'SELECT COALESCE(SUM("{measure}"), 0) FROM t '
        f'WHERE "{dimension}" IN ({marks}) AND "{range_column}" BETWEEN ? AND ?'
    )
    (total,) = table.connection.execute(sql, [*raw_values, low, high]).fetchone()
    evidence = (
        f"SUM({measure}) WHERE {dimension}={match['literal']!r} AND {range_column} {low}..{high}"
    )
    return Outcome(Verdict.SOLVED, answer=int(total), evidence=evidence)


def solve(task: Task) -> Outcome:
    """Answer only when the whole instruction matches the slot grammar."""
    match = _BODY.fullmatch(strip_boilerplate(task.instruction))
    if match is None:
        return Outcome(Verdict.UNPARSED, reason="instruction not recognised")
    if task.workspace is None:
        return Outcome(Verdict.UNGROUNDED, reason="no workspace")
    try:
        path = ensure_within(task.workspace / match["file"], task.workspace)
        table = load_table(path)
    except PathEscapeError:
        return Outcome(Verdict.UNGROUNDED, reason="data file is outside the workspace")
    except TableError as exc:
        return Outcome(Verdict.UNPARSED, reason=f"table not understood: {exc}")
    if table is None:
        return Outcome(Verdict.UNGROUNDED, reason="data file is missing or empty")
    try:
        return _answer(table, match, task.task_id)
    finally:
        table.connection.close()
