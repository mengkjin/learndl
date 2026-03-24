from .finder import FreeProxyFinder as ProxyFinder
from .verifier import ProxyVerifier
from .cache import ProxyCache
from .ppool import ProxyPoolAutoRefresh

__all__ = ['ProxyAPI' , 'ProxyFinder' , 'ProxyVerifier' , 'ProxyCache' , 'ProxyPoolAutoRefresh']

class ProxyAPI:
    proxy_pools : dict[tuple[str, ...], ProxyPoolAutoRefresh] = {}

    @classmethod
    def get_working_proxies(
        cls , target_url: str , min_count: int = 5, * , max_round: int = 3 , timeout: float = 10.0, workers: int = 50) -> list[str]:
        """
        return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
        if force_refresh is True, ignore the expired cache.
        """
        return ProxyVerifier.get_working_proxies(target_url, min_count=min_count, max_round=max_round, timeout=timeout,  workers=workers)

    @classmethod
    def get_proxy_pool(cls , verify_urls: list[str] , go_with_cached_proxies: bool = False, * , refresh_interval: int = 180 , refresh_max_attempts: int = 10 , refresh_threshold: float = 0.2) -> ProxyPoolAutoRefresh:
        pool_key = tuple(sorted(verify_urls))
        if pool_key not in cls.proxy_pools:
            cls.proxy_pools[pool_key] = ProxyPoolAutoRefresh(
                list(pool_key), 
                go_with_cached_proxies=go_with_cached_proxies, 
                refresh_interval=refresh_interval, 
                refresh_max_attempts=refresh_max_attempts, 
                refresh_threshold=refresh_threshold
            )
        return cls.proxy_pools[pool_key]