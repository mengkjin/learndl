"""Shared fit occupancy for all trainers and a separate exclusive NN fit lock."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
import time

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
    _production = ContextVar('production_fit_lock', default=False)

    @classmethod
    def lock_path(cls) -> Path:
        return cls.LOCK_DIR / 'train_fit_active.lock'

    @classmethod
    def production_lock_path(cls) -> Path:
        return cls.LOCK_DIR / 'daily_production_fit.lock'

    @classmethod
    def production_requested(cls) -> bool:
        try:
            with cls.production_lock_path().open('r') as stream:
                try:
                    portalocker.lock(stream, portalocker.LOCK_SH | portalocker.LOCK_NB)
                except portalocker.LockException:
                    return True
        except FileNotFoundError:
            pass
        return False

    @classmethod
    @contextmanager
    def production_guard(cls, *, on_wait=None):
        """Announce priority, then wait for existing fits and hold the batch slot."""
        path = cls.production_lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+') as priority, cls.lock_path().open('a+') as fit:
            portalocker.lock(priority, portalocker.LOCK_EX)
            started = time.monotonic()
            next_report = started
            while True:
                try:
                    portalocker.lock(fit, portalocker.LOCK_EX | portalocker.LOCK_NB)
                    break
                except portalocker.AlreadyLocked:
                    now = time.monotonic()
                    if on_wait is not None and now >= next_report:
                        on_wait(now - started)
                        next_report = now + 300
                    time.sleep(.25)
            token = cls._production.set(True)
            try:
                yield time.monotonic() - started
            finally:
                cls._production.reset(token)
                portalocker.unlock(fit)
                portalocker.unlock(priority)

    @classmethod
    @contextmanager
    def guard(cls, *, respect_production=False):
        if cls._production.get():
            yield
            return
        path = cls.lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+') as stream:
            while True:
                portalocker.lock(stream, portalocker.LOCK_SH)
                if not respect_production or not cls.production_requested():
                    break
                portalocker.unlock(stream)
                time.sleep(.25)
            try:
                yield
            finally:
                portalocker.unlock(stream)

    @classmethod
    @contextmanager
    def trainer_guard(cls, *, try_cuda, respect_production):
        while True:
            with cls.guard(respect_production=respect_production), FitLockNN.guard(try_cuda=try_cuda):
                # Priority can arrive while this trainer is queued for the NN lock.
                if not respect_production or not cls.production_requested():
                    yield
                    return

    @classmethod
    def is_held(cls) -> bool:
        # Old code still holds only the original NN lock during rolling upgrades.
        return _occupied(cls.lock_path()) or FitLockNN.is_held()
