"""Flushed progress messages captured by notebooks, run logs and HtmlCatcher."""
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from contextvars import ContextVar
from pathlib import Path
import sys
import time

_console = ContextVar('rl_progress_console', default=None)


@contextmanager
def quiet_output(path: Path, heartbeat_seconds: float = 3600):
    """Keep full diagnostics on disk; expose milestones and a bounded heartbeat.

    Heartbeats require actual output/progress, so a hung process is not kept alive.
    Both streams share a timer. Numerical CSV/TensorBoard logging is unchanged.
    """
    console = sys.stdout
    last = [time.monotonic()]

    class Capture:
        def write(self, text):
            written = log.write(text)
            if text.strip() and time.monotonic() - last[0] >= heartbeat_seconds:
                console.write('[RL/heartbeat] Still making progress; details: ' + str(path) + '\n')
                console.flush()
                last[0] = time.monotonic()
            return written

        def flush(self):
            log.flush()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as log:
        token = _console.set((console, log, last))
        try:
            with redirect_stdout(Capture()), redirect_stderr(Capture()):
                yield
        finally:
            _console.reset(token)


def progress(stage: str, message: str, *, warning: bool = False, milestone: bool = False) -> None:
    level = 'WARNING' if warning else 'INFO'
    text = f'[{datetime.now().isoformat(timespec="seconds")}] [RL/{stage}] {level}: {message}'
    if milestone and (context := _console.get()):
        console, log, last = context
        print(text, file=console, flush=True)
        print(text, file=log, flush=True)
        last[0] = time.monotonic()
    else:
        print(text, file=sys.stderr if warning else sys.stdout, flush=True)
