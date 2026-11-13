"""Entry point: organisers run `python /app/run_agent.py --tasks-dir /data/tasks`."""

import sys
import traceback


def _run() -> None:
    try:
        from src.cli import main

        main(sys.argv[1:])
    except BaseException:  # the process must exit with code 0 whatever happens
        traceback.print_exc()


if __name__ == "__main__":
    _run()
    sys.exit(0)
