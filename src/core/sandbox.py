"""Subprocess runner with a cleaned environment, limits and a hard timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

DEFAULT_TIMEOUT_S = 10.0
MAX_OUTPUT_BYTES = 64 * 1024
_CPU_SLACK_S = 2
_MEMORY_LIMIT_BYTES = 2 * 1024**3
_FILE_SIZE_LIMIT_BYTES = 64 * 1024**2
_PROCESS_LIMIT = 256
# Variables a child may inherit. LLM_*, BUDGET and everything else are dropped.
_PASSTHROUGH = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL")


class PathEscapeError(ValueError):
    """A path resolves outside the directory it must stay in."""


@dataclass(frozen=True)
class SandboxResult:
    """Result of one sandboxed command."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def ensure_within(path: Path, root: Path) -> Path:
    """Resolve symlinks and return the path, or raise if it leaves `root`."""
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise PathEscapeError(f"{path.name} escapes {root.name}")
    return resolved


def clean_env() -> dict[str, str]:
    """Environment for child processes: no secrets, no budget, no bytecode writes."""
    env = {name: os.environ[name] for name in _PASSTHROUGH if name in os.environ}
    env["HOME"] = tempfile.gettempdir()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _limits(cpu_seconds: int) -> Callable[[], None] | None:
    try:
        import resource
    except ImportError:  # Windows: the timeout is the only limit
        return None

    wanted = (
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_AS", _MEMORY_LIMIT_BYTES),
        ("RLIMIT_FSIZE", _FILE_SIZE_LIMIT_BYTES),
        ("RLIMIT_NPROC", _PROCESS_LIMIT),
    )
    plan = [(getattr(resource, name), value) for name, value in wanted if hasattr(resource, name)]

    def apply() -> None:
        for kind, value in plan:
            try:
                _soft, hard = resource.getrlimit(kind)
                limit = value if hard == resource.RLIM_INFINITY else min(value, hard)
                resource.setrlimit(kind, (limit, limit))
            except (OSError, ValueError):
                pass

    return apply


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _read_capped(handle: IO[bytes]) -> str:
    handle.seek(0)
    return handle.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")


def run_sandboxed(
    argv: Sequence[str], cwd: Path, timeout: float = DEFAULT_TIMEOUT_S, limits: bool = True
) -> SandboxResult:
    """Run `argv` (never through a shell) in `cwd` inside its own process group."""
    popen_kwargs: dict[str, object] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
        if limits:
            popen_kwargs["preexec_fn"] = _limits(int(timeout) + _CPU_SLACK_S)
    else:
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(  # argv list, shell is never used
                list(argv),
                cwd=str(cwd),
                env=clean_env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                shell=False,
                **popen_kwargs,
            )
        except OSError as exc:
            return SandboxResult(None, "", f"spawn failed: {type(exc).__name__}", False)

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Kill the whole group even on success so that stray children cannot linger.
            _kill_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return SandboxResult(
            returncode=None if timed_out else proc.returncode,
            stdout=_read_capped(out),
            stderr=_read_capped(err),
            timed_out=timed_out,
        )
