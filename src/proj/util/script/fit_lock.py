"""Shared fit occupancy for all trainers and a separate exclusive NN fit lock."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path

import portalocker

from src.proj.env import MACHINE, PATH

__all__ = ['FitLock', 'FitLockNN']


def _occupied(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        try:
            portalocker.lock(stream, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except (portalocker.LockException, OSError):
            return True
        portalocker.unlock(stream)
        return False


class FitLockNN:
    """Serialize NN/NNBoost fits; retain the old filename for live old processes."""

    LOCK_DIR = PATH.runtime / 'script_lock'

    @classmethod
    def enabled(cls) -> bool:
        return bool(MACHINE.preference('gpu', 'fit_lock_nn/enabled', default=
                    MACHINE.preference('gpu', 'fit_lock/enabled', default=True)))

    @classmethod
    def lock_name(cls) -> str:
        return str(MACHINE.preference('gpu', 'fit_lock_nn/lock_name', default=
                   MACHINE.preference('gpu', 'fit_lock/lock_name', default='train_fit')))

    @classmethod
    def lock_path(cls) -> Path:
        return cls.LOCK_DIR / f'{cls.lock_name()}.lock'

    @classmethod
    def guard(cls, try_cuda: bool = True):
        if not cls.enabled() or not try_cuda:
            return nullcontext()
        return cls._guard()

    @classmethod
    @contextmanager
    def _guard(cls):
        path = cls.lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+') as stream:
            portalocker.lock(stream, portalocker.LOCK_EX)
            try:
                yield
            finally:
                portalocker.unlock(stream)

    @classmethod
    def is_held(cls) -> bool:
        return _occupied(cls.lock_path())


class FitLock:
    """Unlimited shared fit occupancy; the OS releases it even after SIGKILL."""

    LOCK_DIR = PATH.runtime / 'script_lock'

    @classmethod
    def lock_path(cls) -> Path:
        return cls.LOCK_DIR / 'train_fit_active.lock'

    @classmethod
    @contextmanager
    def guard(cls):
        path = cls.lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+') as stream:
            portalocker.lock(stream, portalocker.LOCK_SH)
            try:
                yield
            finally:
                portalocker.unlock(stream)

    @classmethod
    def is_held(cls) -> bool:
        # Old code still holds only the original NN lock during rolling upgrades.
        return _occupied(cls.lock_path()) or FitLockNN.is_held()
