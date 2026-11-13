"""openJiuwen tools of the last-resort agent: bounded, read-only views of one task.

Every tool is a closure over a single task's world; nothing is shared between tasks and no
tool can reach outside the task's workspace. Importing this module imports the SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openjiuwen.core.foundation.tool import tool

from src.core.sandbox import PathEscapeError, ensure_within
from src.solvers import coding
from src.solvers.sqlworld import SqlError, SqlWorld, clean_sql, run_select

MAX_OUTPUT_CHARS = 600
_MAX_FILE_BYTES = 256 * 1024


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else text[: MAX_OUTPUT_CHARS - 12] + " ...[cut]"


def build_tools(
    workspace: Path, world: SqlWorld | None, bench: coding.Workbench | None, state: dict[str, str]
) -> list[Any]:
    """Tools for one task; `state["source"]` carries the library text between coding calls."""
    tools: list[Any] = []

    @tool(name="read_file", description="Read a text file of the task by its relative path.")
    def read_file(path: str, start_line: int = 1) -> str:
        try:
            target = ensure_within(workspace / path, workspace)
            if not target.is_file() or target.stat().st_size > _MAX_FILE_BYTES:
                return "error: no such file or file too large"
            lines = target.read_bytes().decode("utf-8", errors="replace").splitlines()
        except (OSError, PathEscapeError):
            return "error: path is outside the task or unreadable"
        first = max(1, int(start_line))
        numbered = [f"{first + i}|{line}" for i, line in enumerate(lines[first - 1 :])]
        return _clip("\n".join(numbered) or "(empty)")

    tools.append(read_file)

    if world is not None:

        @tool(name="sql", description="Run one read-only SQLite SELECT over the task's tables.")
        def sql(query: str) -> str:
            try:
                rows = run_select(world, clean_sql(query))
            except SqlError as error:
                return _clip(f"error: {error}")
            return _clip(repr(rows))

        tools.append(sql)

    if bench is not None:

        @tool(name="run_tests", description="Run the task's unit tests on the current library.")
        def run_tests() -> str:
            run, report = bench.run(state["source"])
            if run.status == "ok":
                return "OK: all tests pass"
            failures = coding.failure_summary(report)
            return _clip("FAILED\n" + ("\n".join(failures) or report.strip()[-300:]))

        @tool(
            name="replace_lines",
            description='Replace library lines. Argument: lines of "<line number>|<new text>".',
        )
        def replace_lines(patch: str) -> str:
            try:
                state["source"] = coding.apply_line_patch(state["source"], patch)
            except coding.PatchError as error:
                return _clip(f"error: {error}")
            return "applied; call run_tests"

        tools.extend([run_tests, replace_lines])
    return tools
