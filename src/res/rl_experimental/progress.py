"""Flushed progress messages captured by notebooks, run logs and HtmlCatcher."""
from datetime import datetime
import sys


def progress(stage: str, message: str, *, warning: bool = False) -> None:
    level = 'WARNING' if warning else 'INFO'
    print(f'[{datetime.now().isoformat(timespec="seconds")}] [RL/{stage}] {level}: {message}',
          file=sys.stderr if warning else sys.stdout, flush=True)
