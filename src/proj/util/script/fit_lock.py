"""Exclusive lock for NN ``stage_fit``; inference queries ``is_held`` without acquiring."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import portalocker

from src.proj.env import MACHINE, PATH
from src.proj.util.script.script_lock import ScriptLock

__all__ = ['FitLock']


class FitLock:
    """Serialize NN ``stage_fit`` across processes; expose a non-blocking occupancy probe."""

    LOCK_DIR = PATH.runtime.joinpath('script_lock')

    @classmethod
    def enabled(cls) -> bool:
        return bool(MACHINE.preference('gpu', 'fit_lock/enabled', default=True))

    @classmethod
    def lock_name(cls) -> str:
        return str(MACHINE.preference('gpu', 'fit_lock/lock_name', default='train_fit'))

    @classmethod
    def nn_only(cls) -> bool:
        return bool(MACHINE.preference('gpu', 'fit_lock/nn_only', default=True))

    @classmethod
    def lock_path(cls) -> Path:
        cls.LOCK_DIR.mkdir(parents=True, exist_ok=True)
        return cls.LOCK_DIR.joinpath(f'{cls.lock_name()}.lock')

    @classmethod
    def guard(cls, try_cuda: bool = True) -> Any:
        """Context manager: acquire fit lock, or no-op when disabled / non-NN."""
        if not cls.enabled():
            return ScriptLock(None)
        if cls.nn_only() and not try_cuda:
            return ScriptLock(None)
        return ScriptLock(cls.lock_name(), timeout=None)

    @classmethod
    def is_held(cls) -> bool:
        """True if another process currently holds the fit lock. Does not acquire."""
        if not cls.enabled():
            return False
        lock_path = cls.lock_path()
        lock_file = open(lock_path, 'a+')
        try:
            portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
            portalocker.unlock(lock_file)
            return False
        except (portalocker.AlreadyLocked, BlockingIOError, OSError):
            return True
        finally:
            lock_file.close()
