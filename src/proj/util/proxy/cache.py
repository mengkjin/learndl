"""Persist discovered proxies under ``PATH.local_machine/proxies`` with TTL-style refresh."""
from __future__ import annotations
import threading
from dataclasses import dataclass
from datetime import datetime , timedelta
from typing import Literal , Any , Iterable

from src.proj.env import PATH
from src.proj.cal import BJ_TZ
from .core import Proxy , ProxySet

_default_time = datetime.strptime('1900-01-01 00:00:00', "%Y-%m-%d %H:%M:%S").replace(tzinfo=BJ_TZ)

@dataclass
class CachedProxies:
    """One cache entry: a target URL together with its verified proxy list and the time it was refreshed."""

    url: str
    update_time: datetime
    candidates: ProxySet

    @classmethod
    def from_input(cls, name: str, input: Any) -> CachedProxies:
        """Deserialise a cache entry from a dict (with ``update_time`` / ``candidates`` keys) or a plain list."""
        if isinstance(input, dict):
            return cls(name, datetime.fromisoformat(input.get('update_time', _default_time.isoformat())), ProxySet(input.get('candidates', []) , source='cache'))
        elif isinstance(input, list):
            return cls(name, _default_time, ProxySet(input, source='cache'))
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

    def to_dict(self) -> dict[str, Any]:
        """Serialise this cache entry to a JSON-safe dict."""
        return {
            'update_time': self.update_time.isoformat(),
            'candidates': self.candidates.to_urls(),
        }

class ProxyCache:
    """
    Process-level disk-backed cache of verified proxies keyed by target URL.

    Proxies are persisted to ``PATH.local_machine/proxies/proxies.json`` so that a fresh process
    can skip the discovery/verification cycle when the file is recent enough.
    """

    _cached_proxies = None
    _proxies_file = PATH.local_machine.joinpath('proxies' , 'proxies.json')
    _locks = {
        'cache': threading.Lock(),
        'file': threading.Lock(),
    }
    _refresh_time : dict[str, timedelta] = {
        'cache': timedelta(seconds=300),
        'file': timedelta(days=1),
    }

    @classmethod
    def load_file(cls) -> dict[str, Any]:
        """Load proxies from proxies.json file"""
        if not cls._proxies_file.exists():
            return {}
        with cls._locks['file']:
            return PATH.read_json(cls._proxies_file)

    @classmethod
    def load(cls) -> dict[str, CachedProxies]:
        """Load proxies cache from cache file"""
        if cls._cached_proxies is None:
            cls._cached_proxies = {url: CachedProxies.from_input(url , sub_cache) for url, sub_cache in cls.load_file().items()}
        return cls._cached_proxies

    @classmethod
    def update(cls , target_url: str, verified_proxies: Iterable[Proxy | str]) -> None:
        """Save proxies cache to cache file"""
        with cls._locks['cache']:
            if cls._cached_proxies is None:
                cls.load()
            assert cls._cached_proxies is not None, 'Proxies cache is not loaded'
            cls._cached_proxies[target_url] = CachedProxies(target_url, datetime.now(tz=BJ_TZ), ProxySet(verified_proxies))
        with cls._locks['file']:
            PATH.dump_json({name: cache.to_dict() for name, cache in cls._cached_proxies.items()}, cls._proxies_file, overwrite=True)

    @classmethod
    def get_cached_proxies(cls , target_url: str | Literal['all']) -> ProxySet:
        """Get cached proxies from cache , return is if_obsolete and proxies"""
        proxies = cls.load()
        if target_url == 'all':
            return ProxySet([proxy for cache in proxies.values() for proxy in cache.candidates])
        elif target_url in proxies:
            return proxies[target_url].candidates
        else:
            return ProxySet()