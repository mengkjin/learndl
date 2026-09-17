"""Shared lifecycle guard against automatic in-place code replacement."""
from contextlib import contextmanager

import portalocker

from src.proj.env import PATH


@contextmanager
def training_code_guard():
    """Concurrent pipelines share this lock; automatic Git updates need exclusivity."""
    path = PATH.runtime / 'scheduling' / 'git_update.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        portalocker.lock(stream, portalocker.LOCK_SH)
        try:
            yield
        finally:
            portalocker.unlock(stream)
