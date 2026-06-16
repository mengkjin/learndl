"""Facade: cached working proxies and shared ``AdaptiveProxyPool`` instances per target URL set."""

from __future__ import annotations
from src.proj import Base
from .finder import FreeProxyFinder as ProxyFinder
from .verifier import ProxyVerifier
from .cache import ProxyCache
from .ppool import AdaptiveProxyPool , WorkingProxies , UrlsType

__all__ = ['ProxyAPI' , 'ProxyFinder' , 'ProxyVerifier' , 'ProxyCache' , 'AdaptiveProxyPool']

class ProxyAPI(Base.BoundLogger):
    """Process-wide registry of ``AdaptiveProxyPool`` keyed by sorted target URL tuples."""

    proxy_pools : dict[tuple[str, ...], AdaptiveProxyPool] = {}

    @classmethod
    def get_working_proxies(
        cls , target_url: str , min_count: int = 5, * , max_round: int = 3 , timeout: float = 10.0, workers: int = 50):
        """
        return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
        if force_refresh is True, ignore the expired cache.
        """
        return WorkingProxies.get(target_url, min_count=min_count, max_round=max_round, timeout=timeout,  workers=workers)

    @classmethod
    def get_proxy_pool(cls , target_urls: UrlsType , go_with_cached_proxies: bool = False, * , refresh_interval: int = 5 , refresh_max_attempts: int = 10 , refresh_threshold: float = 0.2) -> AdaptiveProxyPool:
        """Return or create an ``AdaptiveProxyPool`` for the given URL(s)."""
        if isinstance(target_urls, str):
            pool_key = (target_urls,)
        else:
            pool_key = tuple(sorted(set(target_urls)))
        if pool_key not in cls.proxy_pools:
            cls.proxy_pools[pool_key] = AdaptiveProxyPool(
                target_urls, 
                go_with_cached_proxies=go_with_cached_proxies, 
                refresh_interval=refresh_interval, 
                refresh_max_attempts=refresh_max_attempts,
                refresh_threshold=refresh_threshold
            )
        return cls.proxy_pools[pool_key]

    @classmethod
    def verification_stats(cls):
        """Aggregate verification stats from ``ProxyVerifier``."""
        return ProxyVerifier.stats()