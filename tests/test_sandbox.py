from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from src.core.sandbox import (
    MAX_OUTPUT_BYTES,
    PathEscapeError,
    clean_env,
    ensure_within,
    run_sandboxed,
)

SECRET = "sk-test-secret-value"


@pytest.fixture(autouse=True)
def secrets_in_parent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    monkeypatch.setenv("LLM_BASE_URL", "http://proxy.invalid/v1")
    monkeypatch.setenv("BUDGET", "12345")


def test_clean_env_drops_secrets_and_budget() -> None:
    env = clean_env()

    assert not any(name.startswith("LLM_") for name in env)
    assert "BUDGET" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert set(env) <= {"PATH", "SYSTEMROOT", "LANG", "LC_ALL", "HOME", "PYTHONDONTWRITEBYTECODE"}


def test_child_does_not_see_secrets_and_runs_in_cwd(tmp_path: Path) -> None:
    code = (
        "import os; "
        "print(sorted(k for k in os.environ if k.startswith('LLM_') or k == 'BUDGET')); "
        "print(os.getcwd())"
    )

    result = run_sandboxed([sys.executable, "-c", code], cwd=tmp_path)

    assert result.returncode == 0 and not result.timed_out
    lines = result.stdout.splitlines()
    assert lines[0] == "[]"
    assert Path(lines[1]).resolve() == tmp_path.resolve()
    assert SECRET not in result.stdout + result.stderr


def test_nonzero_exit_and_stderr_are_reported(tmp_path: Path) -> None:
    code = "import sys; sys.stderr.write('boom'); sys.exit(3)"

    result = run_sandboxed([sys.executable, "-c", code], cwd=tmp_path)

    assert result.returncode == 3
    assert "boom" in result.stderr


def test_timeout_kills_the_process(tmp_path: Path) -> None:
    started = time.monotonic()

    result = run_sandboxed([sys.executable, "-c", "while True: pass"], cwd=tmp_path, timeout=1.0)

    assert result.timed_out and result.returncode is None
    assert time.monotonic() - started < 15


@pytest.mark.skipif(os.name != "posix", reason="process groups are killed on POSIX only")
def test_timeout_kills_grandchildren(tmp_path: Path) -> None:
    marker = tmp_path / "late.txt"
    grandchild = f"import time; time.sleep(3); open({str(marker)!r}, 'w').close()"
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(30)"
    )

    result = run_sandboxed([sys.executable, "-c", code], cwd=tmp_path, timeout=1.0)
    time.sleep(4)

    assert result.timed_out
    assert not marker.exists()


def test_output_is_capped(tmp_path: Path) -> None:
    code = f"print('x' * {MAX_OUTPUT_BYTES * 3})"

    result = run_sandboxed([sys.executable, "-c", code], cwd=tmp_path)

    assert result.returncode == 0
    assert len(result.stdout) == MAX_OUTPUT_BYTES


def test_missing_executable_is_reported_not_raised(tmp_path: Path) -> None:
    result = run_sandboxed(["definitely-not-a-real-binary-42"], cwd=tmp_path)

    assert result.returncode is None and not result.timed_out
    assert result.stderr.startswith("spawn failed")


def test_ensure_within_accepts_inner_and_rejects_outer_paths(tmp_path: Path) -> None:
    root = tmp_path / "task"
    (root / "workspace").mkdir(parents=True)

    assert ensure_within(root / "workspace" / "mylib.py", root).parent.name == "workspace"
    assert ensure_within(root, root) == root.resolve()
    with pytest.raises(PathEscapeError):
        ensure_within(root / ".." / "other", root)
    with pytest.raises(PathEscapeError):
        ensure_within(tmp_path / "task_sibling" / "x", root)


@pytest.mark.skipif(os.name != "posix", reason="symlink creation needs privileges on Windows")
def test_ensure_within_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "task"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    with pytest.raises(PathEscapeError):
        ensure_within(root / "link.txt", root)
