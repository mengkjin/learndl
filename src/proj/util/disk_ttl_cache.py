"""Disk-backed TTL cache for simple JSON-serializable values, shared across scripts."""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, TypeVar

import portalocker

from src.proj.cal import BJ_TZ
from src.proj.env import PATH

_T = TypeVar('_T')
_SAFE_NAMESPACE = re.compile(r'[^\w\-.]+')
_MAX_TTL_HOURS = 24

def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)

def _ensure_json_serializable(value: Any) -> None:
    json.dumps(value, ensure_ascii=False)

def _validate_ttl_hours(ttl_hours: float) -> float:
    if not isinstance(ttl_hours, (int, float)):
        raise TypeError(f'ttl_hours must be numeric, got {type(ttl_hours).__name__}')
    hours = float(ttl_hours)
    if not (0 < hours <= _MAX_TTL_HOURS + 1e-6):
        raise ValueError(f'ttl_hours must be in (0, {_MAX_TTL_HOURS}], got {ttl_hours!r}')
    return hours

@dataclass(frozen=True, slots=True)
class DiskTTLCacheEntry:
    """One cache record returned by :meth:`DiskTTLCache.get`."""
    key: str
    value: Any
    ttl_hours: float
    created_at: datetime
    expires_at: datetime

    def __bool__(self) -> bool:
        return self.is_valid

    def to_dict(self) -> dict[str, Any]:
        """Serialise this entry for JSON storage."""
        return {
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'ttl_hours': self.ttl_hours,
        }

    @classmethod
    def create(cls, key: str, value: Any, ttl_hours: float) -> DiskTTLCacheEntry:
        return cls(
            key=key,
            value=value,
            ttl_hours=ttl_hours,
            created_at=datetime.now(tz=BJ_TZ),
            expires_at=datetime.now(tz=BJ_TZ) + timedelta(hours=ttl_hours),
        )

    @classmethod
    def from_dict(cls, key: str, value: Any | None = None, created_at: str | None = None , expires_at: str | None = None, ttl_hours: float | None = None) -> DiskTTLCacheEntry | None:
        """Build from on-disk dict; return ``None`` if required fields are invalid."""
        if value is None or created_at is None or expires_at is None or ttl_hours is None:
            return None
        _ttl_hours = float(ttl_hours)
        _expires_at = _parse_datetime(expires_at)
        _created_at = _parse_datetime(created_at)

        return cls(
            key=key,
            value=value,
            created_at=_created_at,
            expires_at=_expires_at,
            ttl_hours=_ttl_hours,
        )

    @property
    def is_valid(self) -> bool:
        """Return whether this entry is still before ``expires_at``."""
        return datetime.now(tz=BJ_TZ) <= self.expires_at

    @property
    def create_time_str(self) -> str:
        return self.created_at.strftime('%Y-%m-%d %H:%M:%S')


class DiskTTLCache:
    """Disk JSON cache with per-entry TTL (hours), callable via classmethods.

    Each ``put`` sets ``ttl_hours`` (0 < hours ≤ 24). ``created_at`` and
    ``expires_at`` are stored in JSON and exposed on :class:`DiskTTLCacheEntry`.

    Uses a per-namespace :class:`threading.Lock` (same process) and
    ``portalocker`` (cross-process) around all reads and writes.

    Storage layout: ``PATH.runtime/disk_ttl_cache/{namespace}.json``.

    Example::

        entry = DiskTTLCache.get('factor_daily_update', 'download_trade')
        if entry is None:
            run_download()
            DiskTTLCache.put(
                'factor_daily_update',
                'download_trade',
                {'last_date': '2025-05-24'},
                ttl_hours=24,
            )
        else:
            entry.created_at
            entry.value
    """

    _ROOT = PATH.runtime.joinpath('disk_ttl_cache')
    _locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _namespace_locks: ClassVar[dict[str, threading.Lock]] = {}

    @classmethod
    def put(
        cls,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_hours: float,
    ) -> DiskTTLCacheEntry:
        """Persist ``value`` under ``namespace``/``key`` with the given TTL in hours."""
        _ensure_json_serializable(value)
        hours = _validate_ttl_hours(ttl_hours)
        entry = DiskTTLCacheEntry.create(key=key, value=value, ttl_hours=hours)

        def mutate(entries: dict[str, Any]) -> None:
            cls._prune_entries(entries)
            entries[key] = entry.to_dict()

        cls._mutate_store(namespace, mutate)
        return entry

    @classmethod
    def get(cls, namespace: str, key: str) -> DiskTTLCacheEntry | None:
        """Return the cache entry, or ``None`` if missing or past ``expires_at``."""
        with cls._thread_lock(namespace):
            entries = cls._load_entries_unlocked(namespace)
        raw = entries.get(key)
        if raw is None or not isinstance(raw, dict):
            return None
        entry = DiskTTLCacheEntry.from_dict(key, **raw)
        if entry is None or not entry.is_valid:
            return None
        return entry

    @classmethod
    def cache_file(cls, namespace: str) -> Path:
        safe_name = _SAFE_NAMESPACE.sub('_', namespace.strip()) or 'default'
        return cls._ROOT.joinpath(f'{safe_name}.json')

    @classmethod
    def keys(cls, namespace: str, *, valid_only: bool = True) -> list[str]:
        """List keys in ``namespace``; when ``valid_only``, omit expired entries."""
        with cls._thread_lock(namespace):
            entries = cls._load_entries_unlocked(namespace)
        if valid_only:
            cls._prune_entries(entries)
        return sorted(entries)

    @classmethod
    def clear(cls, namespace: str, *, expired_only: bool = False) -> None:
        """Drop all entries in ``namespace``, or only expired ones."""
        if expired_only:
            def mutate(entries: dict[str, Any]) -> None:
                cls._prune_entries(entries)
        else:
            def mutate(entries: dict[str, Any]) -> None:
                entries.clear()

        cls._mutate_store(namespace, mutate)

    @classmethod
    def _load_entries_unlocked(cls, namespace: str) -> dict[str, Any]:
        cache_file = cls.cache_file(namespace)
        if not cache_file.exists():
            return {}
        with cls._locked_file(cache_file, 'r') as f:
            store = json.load(f)
        if not isinstance(store, dict):
            cache_file.unlink()
            return {}
        entries = store.get('entries', {})
        return entries if isinstance(entries, dict) else {}

    @classmethod
    def _mutate_store(cls, namespace: str, mutate: Callable[[dict[str, Any]], _T]) -> _T:
        with cls._thread_lock(namespace):
            cache_file = cls.cache_file(namespace)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cls._locked_file(cache_file, 'a+') as f:
                f.seek(0)
                raw = f.read()
                if raw.strip():
                    store = json.loads(raw)
                else:
                    store = {}
                if not isinstance(store, dict):
                    store = {}
                entries = store.get('entries')
                if not isinstance(entries, dict):
                    entries = {}
                result = mutate(entries)
                f.seek(0)
                f.truncate()
                json.dump(
                    {'namespace': namespace, 'entries': entries},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write('\n')
                return result

    @classmethod
    @contextmanager
    def _thread_lock(cls, namespace: str) -> Iterator[None]:
        lock = cls._get_thread_lock(namespace)
        with lock:
            yield

    @classmethod
    def _get_thread_lock(cls, namespace: str) -> threading.Lock:
        path_key = str(cls.cache_file(namespace))
        with cls._locks_guard:
            if path_key not in cls._namespace_locks:
                cls._namespace_locks[path_key] = threading.Lock()
            return cls._namespace_locks[path_key]

    @classmethod
    @contextmanager
    def _locked_file(cls, cache_file: Path, mode: str) -> Iterator[Any]:
        with open(cache_file, mode, encoding='utf-8') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            try:
                yield f
            finally:
                portalocker.unlock(f)

    @classmethod
    def _prune_entries(cls, entries: dict[str, Any]) -> None:
        stale = [key for key, entry in entries.items() if not DiskTTLCacheEntry.from_dict(key, **entry)]
        for key in stale:
            entries.pop(key)
