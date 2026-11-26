"""Absolute nightly deadlines, shared by the scheduler and its worker."""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


@dataclass(frozen=True)
class NightlyWindow:
    cutoff: float
    hard_deadline: float
    tasks: tuple[str, ...]


def load_window(root: Path, scheduled_at: float) -> NightlyWindow:
    settings = yaml.safe_load((root / 'configs/automation/nightly.yaml').read_text())
    timezone = ZoneInfo(settings['timezone'])
    slot = datetime.fromtimestamp(scheduled_at, timezone)
    end = time.fromisoformat(settings['cutoff'])
    grace = settings['grace_minutes']
    if end.tzinfo is not None or type(grace) is not int or grace < 0:
        raise ValueError('Nightly maintenance requires a local cutoff and nonnegative grace_minutes')
    cutoff = datetime.combine(slot.date(), end, timezone)
    if end < time(21):
        cutoff += timedelta(days=1)
    return NightlyWindow(cutoff.timestamp(), (cutoff + timedelta(minutes=grace)).timestamp(),
                         tuple(settings['tasks']))
