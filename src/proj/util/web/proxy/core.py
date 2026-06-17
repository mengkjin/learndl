"""Core types: ``Proxy``, ``ProxySet``, and statistics helpers for proxy pools."""
from __future__ import annotations
import re
import random
from functools import cached_property
from typing import Literal , TypeAlias
from collections.abc import Iterable

from src.proj.core import lit
from src.proj.log import Logger

INVALID_THRESHOLD = 3
PROXY_MAX_CONCURRENT = 2

__all__ = ['Proxy' , 'ProxySet' , 'ProxyStats' , 'ProxyStatsSet' , 'INVALID_THRESHOLD' , 'PROXY_MAX_CONCURRENT']

strProxy : TypeAlias = 'str | Proxy'
ProxyStatsType : TypeAlias = Literal['running', 'error', 'success']

class Proxy:
    """Single proxy address (``protocol://host:port``) with source tracking and verification history."""
    def __new__(cls, url: strProxy, source: str = 'unknown'):
        proxy = super().__new__(cls)
        proxy.set_url(url)
        proxy.set_source(source)
        return proxy

    @cached_property
    def verified(self) -> list[str]:
        """URLs for which this proxy has been successfully verified."""
        return []

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url})"

    def __eq__(self, other: Proxy) -> bool:
        return self.url == other.url

    def __hash__(self) -> int:
        return hash(self.url)

    def __bool__(self) -> bool:
        return bool(self.url)

    def set_url(self, url: strProxy) -> Proxy:
        """Parse and store the proxy URL, copying source from another Proxy when given."""
        if isinstance(url, Proxy):
            self.set_source(url.source)
            url_addr = url.url
        else:
            url_addr = url
        self.protocal, self.host, self.port = self.url_segregate(url_addr)
        self.url : str = url_addr
        return self

    def set_source(self, source: str) -> Proxy:
        """Set the origin label only if it has not been set yet (or is still 'unknown')."""
        if not hasattr(self, 'source') or self.source == 'unknown':
            self.source = source
        return self

    @classmethod
    def url_segregate(cls, url: str) -> tuple[str, str, int]:
        """Segregate the url into protocal, host and port"""
        m = re.match(r"^(http|https|socks4|socks5)://([^:]+):(\d+)$", url)
        if not m:
            raise ValueError(f"Invalid proxy address: {url}")
        protocal, host, port = m.groups()
        return protocal, host, int(port)

    @classmethod
    def unique(cls, proxies: Iterable[Proxy]) -> list[Proxy]:
        """Deduplicate proxies by URL identity."""
        return list(set(proxies))

class ProxySet:
    """Unordered, deduplicated collection of :class:`Proxy` objects."""

    def __init__(self , proxies: Iterable[strProxy] | None = None , source: str = 'unknown'):
        if proxies is None:
            proxies = []
        proxies = [proxy.set_source(source) if isinstance(proxy, Proxy) else Proxy(proxy , source) for proxy in proxies]
        self.proxies = Proxy.unique(proxies)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={len(self)})'

    def __getitem__(self , index: int) -> Proxy:
        return self.proxies[index]

    def __len__(self) -> int:
        return len(self.proxies)

    def __iter__(self):
        return iter(self.proxies)

    def __contains__(self , proxy: Proxy) -> bool:
        return proxy in self.proxies

    def __bool__(self) -> bool:
        return bool(self.proxies)

    def extend(self , proxies: Iterable[strProxy]):
        """Append proxies and deduplicate in place."""
        self.proxies = Proxy.unique(self.proxies + [Proxy(proxy) for proxy in proxies])
        return self

    def to_urls(self) -> list[str]:
        """Return a plain list of URL strings for serialisation."""
        return [proxy.url for proxy in self.proxies]

    def set_source(self , source: str) -> ProxySet:
        """Propagate an origin label to every proxy in the set."""
        [proxy.set_source(source) for proxy in self.proxies]
        return self

    def pick_one(self) -> Proxy | None:
        """Get available proxies"""
        if self.proxies:
            return random.choice(self.proxies)
        return None

class ProxyStats(Proxy):
    """A stated proxy, can be used to track the state of a proxy"""
    _instances : dict[str, ProxyStats] = {}

    def __new__(cls, url: strProxy, source: str = 'unknown'):
        if str(url) not in cls._instances:
            cls._instances[str(url)] = super().__new__(cls, url, source = source)
        return cls._instances[str(url)]

    def __bool__(self) -> bool:
        return self.valid

    @cached_property
    def stats(self) -> dict[ProxyStatsType, int]:
        """Lazy-initialised usage counters: running (in-flight), error, and success counts."""
        return {
            'running': 0,
            'error': 0,
            'success': 0,
        }

    @property
    def valid(self) -> bool:
        """Whether the proxy is valid"""
        return self.stats['error'] < INVALID_THRESHOLD

    @property
    def invalid(self) -> bool:
        """Whether the proxy is invalid (error count >= invalid threshold)"""
        return self.stats['error'] >= INVALID_THRESHOLD

    @property
    def num_running(self) -> int:
        """The number of running proxies"""
        return self.stats['running']

    @property
    def num_slots(self) -> int:
        """The number of slots available"""
        return PROXY_MAX_CONCURRENT - self.num_running

    @property
    def available(self) -> bool:
        """Whether the proxy is available (valid and running < max concurrent)"""
        return self.valid and self.num_running < PROXY_MAX_CONCURRENT

    @property
    def total_count(self) -> int:
        """The total count of the proxy (success + error)"""
        return self.stats['success'] + self.stats['error']

    def acquire(self):
        """Acquired, increment the running count"""
        self.stats['running'] += 1
        return self

    def release(self, success: bool, *, counted: bool = True , vb_level : lit.VerbosityLevel = 4):
        """Released, decrement running; optionally update success/error counters."""
        self.stats['running'] -= 1
        if not counted:
            return
        if success:
            self.stats['success'] += 1
        else:
            self.stats['error'] += 1
            if self.stats['error'] >= INVALID_THRESHOLD:
                Logger.alert1(f"Proxy {self.url} has reached INVALID_THRESHOLD ({INVALID_THRESHOLD}), and will be unavailable." , vb_level=vb_level)

    @classmethod
    def unique(cls, proxies: Iterable[ProxyStats]) -> list[ProxyStats]:
        """Deduplicate ProxyStats instances (same URL → same singleton)."""
        return list(set(proxies))

class ProxyStatsSet(ProxySet):
    """A :class:`ProxySet` whose members are :class:`ProxyStats` singletons with live usage tracking."""

    def __init__(self , proxies: Iterable[ProxyStats | strProxy] | None = None , source: str = 'unknown' , **kwargs):
        if proxies is None:
            proxies = []
        self.proxies : list[ProxyStats] = list(set([ProxyStats(proxy , source) for proxy in proxies]))

    def __iter__(self):
        return iter(self.proxies)

    def __bool__(self) -> bool:
        return self.valid_count > 0

    def extend(self , proxies: Iterable[ProxyStats | strProxy]):
        """Append proxies (auto-converting to ProxyStats singletons) and deduplicate in place."""
        self.proxies = ProxyStats.unique(self.proxies + [ProxyStats(proxy) for proxy in proxies])
        return self

    @property
    def valid_ratio(self) -> float:
        """The ratio of invalid proxies"""
        return 0.0 if not self.proxies else self.valid_count / len(self.proxies)

    @property
    def valid_count(self) -> int:
        """Number of proxies whose error count has not yet reached INVALID_THRESHOLD."""
        return sum(proxy.valid for proxy in self.proxies)

    def pick_one(self) -> ProxyStats | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies if proxy.available]
        if not available_proxies:
            return None
        max_slots = max(proxy.num_slots for proxy in available_proxies)
        options = [proxy for proxy in available_proxies if proxy.num_slots == max_slots] or available_proxies
        if options:
            proxy = random.choice(options)
            proxy.acquire()
            return proxy
        return None