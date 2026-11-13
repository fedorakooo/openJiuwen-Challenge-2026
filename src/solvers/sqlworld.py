"""In-memory SQLite world model shared by data_analysis and search: build, describe, query."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

_MAX_DICTIONARY = 12
_SAMPLE_ROWS = 3
_MAX_NAMES = 80
_PROGRESS_INTERVAL = 10_000
_MAX_PROGRESS_CALLS = 500
_MAX_RESULT_ROWS = 50
# Only plain words reach a prompt; anything that could read as an instruction is left out.
_SAFE_VALUE = re.compile(r"[\w .&-]{1,40}")
_STRING_LITERAL = re.compile(r"'((?:[^']|'')*)'")
_CODE_FENCE = re.compile(r"^```(?:sql|sqlite)?\s*|\s*```$", re.IGNORECASE)
_ALLOWED_ACTIONS = frozenset({sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION})


class SqlError(Exception):
    """The statement is not a single read-only SELECT or failed to run."""


@dataclass
class TableInfo:
    """Schema of one table as shown to the model."""

    name: str
    columns: list[str]
    integer_columns: set[str] = field(default_factory=set)
    links: dict[str, str] = field(default_factory=dict)  # column -> "table.column"


@dataclass
class SqlWorld:
    """Tables of one task; never shared between tasks."""

    connection: sqlite3.Connection
    tables: list[TableInfo] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.connection.close()

    def lock(self) -> None:
        """After loading, allow nothing but reads and bound the work of one query."""

        def authorize(action: int, target: object = None, *_details: object) -> int:
            internal = action == sqlite3.SQLITE_READ and str(target).lower().startswith("sqlite_")
            allowed = action in _ALLOWED_ACTIONS and not internal
            return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

        self.connection.set_authorizer(authorize)

    def literals(self) -> set[str]:
        """Every text value in the world, normalised; a query may only mention these."""
        known = {normalise(name) for name in self.names}
        for table in self.tables:
            for column in table.columns:
                if column in table.integer_columns:
                    continue
                cursor = self.connection.execute(f'SELECT DISTINCT "{column}" FROM "{table.name}"')
                known.update(normalise(str(row[0])) for row in cursor if row[0] is not None)
        return known

    def _dictionary(self, table: TableInfo, column: str) -> str | None:
        if column in table.integer_columns:
            low, high = self.connection.execute(
                f'SELECT MIN("{column}"), MAX("{column}") FROM "{table.name}"'
            ).fetchone()
            return None if low is None else f"{column} {low}..{high}"
        rows = self.connection.execute(
            f'SELECT DISTINCT "{column}" FROM "{table.name}" WHERE "{column}" IS NOT NULL '
            f"ORDER BY 1 LIMIT {_MAX_DICTIONARY + 1}"
        ).fetchall()
        values = [str(row[0]) for row in rows]
        if not values or len(values) > _MAX_DICTIONARY:
            return None
        if not all(_SAFE_VALUE.fullmatch(value) for value in values):
            return None
        return f"{column} in {{{', '.join(values)}}}"

    def _sample_rows(self, table: TableInfo) -> str | None:
        rows = self.connection.execute(
            f'SELECT * FROM "{table.name}" LIMIT {_SAMPLE_ROWS}'
        ).fetchall()
        rendered = []
        for row in rows:
            cells = []
            for cell in row:
                if isinstance(cell, int):
                    cells.append(str(cell))
                elif cell is not None and _SAFE_VALUE.fullmatch(str(cell)):
                    cells.append(f"'{cell}'")
                else:
                    return None
            rendered.append("(" + ",".join(cells) + ")")
        return "Rows: " + ", ".join(rendered) if rendered else None

    def describe(self, typed: bool) -> str:
        """Schema, value dictionaries and a few rows; never whole files."""
        lines: list[str] = []
        for table in self.tables:
            columns = []
            for column in table.columns:
                text = column
                if typed:
                    text += " INT" if column in table.integer_columns else " TEXT"
                if column in table.links:
                    text += f" -> {table.links[column]}"
                columns.append(text)
            prefix = "Table " if typed else ""
            lines.append(f"{prefix}{table.name}({', '.join(columns)})")
            if typed:
                dictionaries = [self._dictionary(table, column) for column in table.columns]
                shown = [entry for entry in dictionaries if entry]
                if shown:
                    lines.append("; ".join(shown))
                sample = self._sample_rows(table)
                if sample:
                    lines.append(sample)
        safe_names = [name for name in self.names if _SAFE_VALUE.fullmatch(name)]
        if safe_names and len(self.names) <= _MAX_NAMES and len(safe_names) == len(self.names):
            lines.append("Names: " + ", ".join(safe_names))
        return "\n".join(lines)


def normalise(value: str) -> str:
    return " ".join(value.split()).casefold()


def create_table(
    connection: sqlite3.Connection,
    name: str,
    columns: Sequence[str],
    integer_columns: set[str],
    records: Sequence[Sequence[object]],
) -> None:
    """Create and fill one table; identifiers must already be validated by the caller."""
    definitions = ", ".join(
        f'"{column}" {"INTEGER" if column in integer_columns else "TEXT"}' for column in columns
    )
    connection.execute(f'CREATE TABLE "{name}" ({definitions})')
    marks = ", ".join("?" for _ in columns)
    connection.executemany(f'INSERT INTO "{name}" VALUES ({marks})', records)


def clean_sql(text: str) -> str:
    """Strip code fences and a trailing semicolon; insist on one SELECT statement."""
    sql = _CODE_FENCE.sub("", text.strip()).strip().rstrip(";").strip()
    if not re.match(r"(?i)(select|with)\b", sql):
        raise SqlError("not a SELECT statement")
    if ";" in _STRING_LITERAL.sub("''", sql):
        raise SqlError("more than one statement")
    return sql


def string_literals(sql: str) -> list[str]:
    return [match.replace("''", "'") for match in _STRING_LITERAL.findall(sql)]


def run_select(world: SqlWorld, sql: str) -> list[tuple[object, ...]]:
    """Run one read-only statement with a step budget; returns at most a few rows."""
    calls = 0

    def tick() -> int:
        nonlocal calls
        calls += 1
        return 1 if calls > _MAX_PROGRESS_CALLS else 0

    world.connection.set_progress_handler(tick, _PROGRESS_INTERVAL)
    try:
        cursor = world.connection.execute(sql)
        return [tuple(row) for row in cursor.fetchmany(_MAX_RESULT_ROWS)]
    except (sqlite3.Error, sqlite3.Warning) as exc:
        raise SqlError(str(exc)[:160]) from exc
    finally:
        world.connection.set_progress_handler(None, 0)
