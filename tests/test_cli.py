from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src import cli
from src.config import DEFAULT_BUDGET, Settings, load_settings
from src.core.results import ResultWriter, TrajectoryLog
from src.core.runner import Clock, run_phase0
from src.core.tasks import Outcome, Task, Verdict, load_tasks
from tests.conftest import SECRET_EXPECTED, MakeTask, list_files

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT = REPO_ROOT / "run_agent.py"


def _resolve(argv: list[str], env: dict[str, str] | None = None) -> str:
    return cli.resolve_tasks_dir(cli.parse_args(argv), env or {}).as_posix()


def test_tasks_dir_precedence() -> None:
    env = {"TASKS": "/from/env"}
    assert _resolve(["--tasks-dir", "/a", "--tasks", "/b", "/c"], env) == "/a"
    assert _resolve(["--tasks", "/b", "/c"], env) == "/b"
    assert _resolve(["/c"], env) == "/c"
    assert _resolve([], env) == "/from/env"
    assert _resolve([]) == "/data/tasks"


def test_unknown_and_broken_arguments_do_not_abort() -> None:
    assert _resolve(["--tasks-dir", "/a", "--unknown-flag", "--other=1"]) == "/a"
    assert _resolve(["--tasks-dir"]) == "/data/tasks"
    assert _resolve(["--help"]) == "/data/tasks"


def test_settings_defaults_and_malformed_values() -> None:
    assert load_settings({}).budget == DEFAULT_BUDGET
    assert load_settings({"BUDGET": "2500"}).budget == 2500
    assert load_settings({"BUDGET": "lots"}).budget == DEFAULT_BUDGET
    assert load_settings({"BUDGET": "-5"}).budget == DEFAULT_BUDGET
    assert load_settings({"LLM_CONCURRENCY": "40"}).llm_concurrency == 5
    assert load_settings({}).deadline_s == 50 * 60
    assert load_settings({"ANSWER_WHEN_UNSURE": "off"}).answer_when_unsure is False


def test_clock_refuses_attempts_that_do_not_fit_before_the_deadline() -> None:
    now = [0.0]
    clock = Clock(Settings(deadline_s=100.0, phase0_cap_s=30.0), now=lambda: now[0])

    assert clock.can_start(60.0) and not clock.phase0_expired()
    now[0] = 31.0
    assert clock.phase0_expired()
    assert not clock.can_start(60.0)  # 69 s left < 60 s timeout + 10 s slack
    assert clock.can_start(59.0)


@pytest.mark.usefixtures("isolated_process_state")
def test_main_walks_tasks_solves_nothing_and_returns_zero(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    make_task("m1")
    make_task("c1", task_type="coding", check="unittest", workspace=True)
    before = list_files(tasks_dir)

    assert cli.main(["--tasks-dir", str(tasks_dir), "--surprise"]) == 0

    assert list_files(tasks_dir) == before
    assert Path.cwd() != tasks_dir
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    log_text = (Path.cwd() / "logs" / "trajectory.jsonl").read_text(encoding="utf-8")
    assert '"task_id": "m1"' in log_text
    assert SECRET_EXPECTED not in log_text


@pytest.mark.usefixtures("isolated_process_state")
@pytest.mark.usefixtures("isolated_process_state")
def test_trajectory_dir_redirects_the_log_and_falls_back_when_unusable(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_task("m1")
    collected = tmp_path / "collected"
    collected.mkdir()
    monkeypatch.setenv("TRAJECTORY_DIR", str(collected))
    assert cli.main(["--tasks-dir", str(tasks_dir)], solvers={}) == 0
    assert '"task_id": "m1"' in (collected / "trajectory.jsonl").read_text(encoding="utf-8")

    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path / "missing"))
    assert cli.main(["--tasks-dir", str(tasks_dir)], solvers={}) == 0
    assert (Path.cwd() / "logs" / "trajectory.jsonl").is_file()
    assert not (tmp_path / "missing").exists()


def test_main_returns_zero_for_missing_or_empty_tasks_dir(tmp_path: Path) -> None:
    assert cli.main(["--tasks-dir", str(tmp_path / "absent")]) == 0
    assert cli.main([]) == 0


@pytest.mark.usefixtures("isolated_process_state")
def test_main_isolates_failing_solvers_and_publishes_the_rest(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path
) -> None:
    make_task("a_crash", task_type="boom")
    make_task("b_ok", task_type="math")
    make_task("c_absent", task_type="search", check="exact_str")
    out = tmp_path / "answers.json"

    def crash(_task: Task) -> Outcome:
        raise RuntimeError("solver bug")

    solvers = {
        "boom": crash,
        "math": lambda _t: Outcome(Verdict.SOLVED, answer=42),
        "search": lambda _t: Outcome(Verdict.UNGROUNDED, reason="entity absent"),
    }

    assert cli.main(["--tasks-dir", str(tasks_dir), "--out", str(out)], solvers=solvers) == 0

    assert (tasks_dir / "b_ok" / "answer.txt").read_bytes() == b"42\n"
    assert not (tasks_dir / "a_crash" / "answer.txt").exists()
    assert not (tasks_dir / "c_absent" / "answer.txt").exists()
    assert out.read_text(encoding="utf-8").strip() == '{\n  "b_ok": "42"\n}'


def test_stop_flag_keeps_written_answers_and_skips_the_rest(
    tasks_dir: Path, make_task: MakeTask
) -> None:
    for name in ("a", "b", "c"):
        make_task(name)
    tasks = load_tasks(tasks_dir)
    stop = threading.Event()

    def solve(task: Task) -> Outcome:
        if task.task_id == "b":
            stop.set()
        return Outcome(Verdict.SOLVED, answer=7)

    stats = run_phase0(
        tasks, {"math": solve}, ResultWriter(), TrajectoryLog(None), Clock(Settings()), stop
    )

    assert stats.solved == 2
    assert [t.task_id for t in stats.deferred] == ["c"]
    assert (tasks_dir / "a" / "answer.txt").read_bytes() == b"7\n"
    assert (tasks_dir / "b" / "answer.txt").read_bytes() == b"7\n"
    assert not (tasks_dir / "c" / "answer.txt").exists()


def test_phase0_time_cap_defers_remaining_tasks(tasks_dir: Path, make_task: MakeTask) -> None:
    for name in ("a", "b"):
        make_task(name)
    now = [0.0]
    clock = Clock(Settings(phase0_cap_s=5.0), now=lambda: now[0])

    def slow(_task: Task) -> Outcome:
        now[0] += 6.0
        return Outcome(Verdict.SOLVED, answer=1)

    stats = run_phase0(
        load_tasks(tasks_dir),
        {"math": slow},
        ResultWriter(),
        TrajectoryLog(None),
        clock,
        threading.Event(),
    )

    assert stats.solved == 1 and [t.task_id for t in stats.deferred] == ["b"]


def test_signal_handler_sets_the_stop_flag() -> None:
    stop = threading.Event()
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        cli.install_signal_handlers(stop)
        signal.raise_signal(signal.SIGTERM)
        assert stop.is_set()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _subprocess_env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
    env.pop("TASKS", None)
    env.pop("OUT", None)
    env["TEMP"] = env["TMP"] = str(tmp_path)
    return env


def test_run_agent_exits_zero_end_to_end(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path
) -> None:
    make_task("m1")
    make_task("s1", task_type="search", check="exact_str", workspace=True)
    before = list_files(tasks_dir)

    for argv in (["--tasks-dir", str(tasks_dir)], ["--tasks-dir", str(tmp_path / "nope")], ["-x"]):
        proc = subprocess.run(
            [sys.executable, str(RUN_AGENT), *argv],
            cwd=tmp_path,
            env=_subprocess_env(tmp_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert SECRET_EXPECTED not in proc.stdout + proc.stderr

    assert list_files(tasks_dir) == before


def test_run_agent_exits_zero_even_when_the_package_is_broken(tmp_path: Path) -> None:
    (tmp_path / "run_agent.py").write_bytes(RUN_AGENT.read_bytes())
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "cli.py").write_text("raise RuntimeError('broken import')\n")

    proc = subprocess.run(
        [sys.executable, str(tmp_path / "run_agent.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0
    assert "broken import" in proc.stderr


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM cannot be delivered gracefully on Windows")
def test_sigterm_mid_run_keeps_answers_and_exits_zero(
    tasks_dir: Path, make_task: MakeTask, tmp_path: Path
) -> None:
    for name in ("a", "b", "c"):
        make_task(name)
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src import cli\n"
        "from src.core.tasks import Outcome, Verdict\n"
        "def solve(task):\n"
        "    if task.task_id != 'a':\n"
        "        time.sleep(30)\n"
        "    return Outcome(Verdict.SOLVED, answer=5)\n"
        "sys.exit(cli.main(sys.argv[1:], solvers={'math': solve}))\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver), "--tasks-dir", str(tasks_dir)],
        cwd=tmp_path,
        env=_subprocess_env(tmp_path),
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while not (tasks_dir / "a" / "answer.txt").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    _out, err = proc.communicate(timeout=60)

    assert proc.returncode == 0, err
    assert (tasks_dir / "a" / "answer.txt").read_bytes() == b"5\n"
    assert not (tasks_dir / "c" / "answer.txt").exists()
    assert "stopping" in err
