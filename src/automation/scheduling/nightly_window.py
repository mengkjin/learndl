"""Absolute nightly deadlines, shared by the scheduler and its worker."""
from __future__ import annotations

__all__ = [
    'NIGHTLY_ENTRYPOINTS',
    'NightlyWindow',
    'latest_nightly_slot',
    'manual_nightly_slot',
    'load_window',
]

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

NIGHTLY_ENTRYPOINTS = ('nightly_maintenance', 'rcquant_sec_none')


@dataclass(frozen=True)
class NightlyWindow:
    cutoff: float
    hard_deadline: float
    tasks: tuple[str, ...]


def latest_nightly_slot(now: float) -> float:
    local = datetime.fromtimestamp(now, ZoneInfo('Asia/Shanghai'))
    evening = datetime.combine(local.date(), time(21), local.tzinfo)
    if local < evening:
        evening -= timedelta(days=1)
    return evening.timestamp()


def manual_nightly_slot(root: Path, now: float) -> float:
    """Use the open overnight window, or today's window for a daytime start."""
    previous = latest_nightly_slot(now)
    return previous if now < load_window(root, previous).cutoff else now


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
