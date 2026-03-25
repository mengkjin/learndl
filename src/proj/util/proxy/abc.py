import re
import random
from typing import Iterable , Literal

class Proxy:
    """A basic proxy class"""
    def __new__(cls, url: 'str | Proxy', source: str = 'unknown'):
        proxy = super().__new__(cls)
        proxy.set_url(url)
        proxy.set_source(source)
        return proxy

    @property
    def verified(self) -> list[str]:
        if not hasattr(self, '_verified'):
            self._verified: list[str] = []
        return self._verified

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url})"

    def __eq__(self, other: 'Proxy') -> bool:
        return self.url == other.url

    def __hash__(self) -> int:
        return hash(self.url)

    def set_url(self, url: 'str | Proxy') -> 'Proxy':
        if isinstance(url, Proxy):
            self.set_source(url.source)
            url_addr = url.url
        else:
            url_addr = url
        self.protocal, self.host, self.port = self.url_segregate(url_addr)
        self.url = url_addr
        return self

    def set_source(self, source: str) -> 'Proxy':
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
    def unique(cls, proxies: Iterable['Proxy']) -> list['Proxy']:
        return list(set(proxies))

class ProxySet:
    def __init__(self , proxies: Iterable[Proxy | str] | None = None , source: str = 'unknown'):
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

    def append(self , proxy: Proxy):
        self.proxies = Proxy.unique(self.proxies + [proxy])

    def extend(self , proxies: Iterable[Proxy]):
        self.proxies = Proxy.unique(self.proxies + list(proxies))

    def to_urls(self) -> list[str]:
        return [proxy.url for proxy in self.proxies]

    def set_source(self , source: str) -> 'ProxySet':
        [proxy.set_source(source) for proxy in self.proxies]
        return self

    def pick_one(self) -> Proxy | None:
        """Get available proxies"""
        if self.proxies:
            return random.choice(self.proxies)
        return None

class ProxyStats(Proxy):
    """A stated proxy, can be used to track the state of a proxy"""
    invalid_threshold : int = 3
    max_concurrent : int = 2
    _instances : dict[str, 'ProxyStats'] = {}

    def __new__(cls, url: str | Proxy, source: str = 'unknown'):
        if str(url) not in cls._instances:
            cls._instances[str(url)] = super().__new__(cls, url, source = source)
        return cls._instances[str(url)]

    @property
    def stats(self) -> dict[Literal['occupied', 'error', 'success'], int]:
        if not hasattr(self, '_stats'):
            self._stats : dict[Literal['occupied', 'error', 'success'], int] = {
                'occupied': 0,
                'error': 0,
                'success': 0,
            }
        return self._stats

    @classmethod
    def set_class_attrs(cls, invalid_threshold: int | None = None, max_concurrent: int | None = None):
        if invalid_threshold is not None:
            cls.invalid_threshold = invalid_threshold
        if max_concurrent is not None:
            cls.max_concurrent = max_concurrent

    @property
    def valid(self) -> bool:
        """Whether the proxy is valid"""
        return self.stats['error'] < self.invalid_threshold

    @property
    def invalid(self) -> bool:
        """Whether the proxy is invalid (error count >= invalid threshold)"""
        return self.stats['error'] >= self.invalid_threshold

    @property
    def available(self) -> bool:
        """Whether the proxy is available (valid and occupied < max concurrent)"""
        return self.valid and self.stats['occupied'] < self.max_concurrent

    @property
    def total_count(self) -> int:
        """The total count of the proxy (success + error)"""
        return self.stats['success'] + self.stats['error']

    def acquire(self):
        """Acquired, increment the occupied count"""
        self.stats['occupied'] += 1
        return self

    def release(self, success: bool):
        """Released, decrement the occupied count, and update the success or error count"""
        self.stats['occupied'] -= 1
        if success:
            self.stats['success'] += 1
        else:
            self.stats['error'] += 1

    def unique(self, proxies: Iterable['ProxyStats']) -> list['ProxyStats']:
        return list(set(proxies))

class ProxyStatsSet(ProxySet):
    def __init__(self , proxies: Iterable[ProxyStats | Proxy | str] | None = None , source: str = 'unknown'):
        if proxies is None:
            proxies = []
        self.proxies : list[ProxyStats] = list(set([ProxyStats(proxy , source) for proxy in proxies]))

    def __iter__(self):
        return iter(self.proxies)

    @property
    def valid_ratio(self) -> float:
        """The ratio of invalid proxies"""
        return 0.0 if not self.proxies else sum(proxy.valid for proxy in self.proxies) / len(self.proxies)

    def pick_one(self) -> ProxyStats | None:
        """Get available proxies"""
        available_proxies = [proxy for proxy in self.proxies if proxy.available]
        if available_proxies:
            proxy = random.choice(available_proxies)
            proxy.acquire()
            return proxy
        return None