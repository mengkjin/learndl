"""RCQuant per-day exclusion and atomic publication; no provider calls here."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from pathlib import Path
import uuid

import portalocker

sample_size = ContextVar('rcquant_sample_size', default=-1)


@contextmanager
def date_lock(root, data_type, date):
    directory = Path(root) / '.locks'
    directory.mkdir(parents=True, exist_ok=True)
    # Different dates can progress independently; one date cannot be fetched twice.
    with (directory / f'{data_type}-{int(date)}.lock').open('a+') as stream:
        portalocker.lock(stream, portalocker.LOCK_EX)
        yield


def locked_day(func):
    @wraps(func)
    def wrapped(self, date, data_type, first_n=-1, **kwargs):
        from .bar_min import RQ_PATH
        with date_lock(RQ_PATH, data_type, date):
            token = sample_size.set(first_n)
            try:
                return func(self, date, data_type, first_n, **kwargs)
            finally:
                sample_size.reset(token)
    return wrapped


def atomic_frame(frame, path, **kwargs):
    from src.proj import Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the final extension so the existing serializer chooses the same format.
    temp = path.with_name(f'.{path.stem}-{uuid.uuid4().hex}.tmp{path.suffix}')
    try:
        result = Save.df(frame, temp, async_save=False, **kwargs)
        if result is False or not temp.exists():
            raise OSError(f'RCQuant publication failed: {path}')
        with temp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return True
