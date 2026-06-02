"""Scrape public proxy lists (e.g. Zdaye) into ``ProxySet`` instances."""
from __future__ import annotations
import time
import random
from curl_cffi import requests
from typing import Callable, Any , Literal , cast

from abc import ABC, abstractmethod
from threading import RLock

from src.proj.core import Silence
from src.proj.env import MACHINE
from src.proj.bases import BaseClass
from src.proj.util.error_handler import retry_call
from .core import ProxySet

class ProxiesCache:
    """Proxy Cache"""
    @classmethod
    def _default_condition(cls, value: Any) -> bool:
        """Default cache condition: not None and not empty container (has __len__ and length > 0)"""
        if value is None:
            return False
        if hasattr(value, '__len__'):
            return len(value) > 0
        return True  # objects that are not None and have no length (e.g. int, float) are cached by default

    @classmethod
    def cached_function(
        cls,
        ttl_seconds: int,
        maxsize: int = 128,
        condition: Callable[[Any], bool] | None = None,
    ) -> Callable:
        """
        Factory function to create a wrapper function with conditional caching capability for a business function.

        Args:
            ttl_seconds: Cache item lifespan (seconds)
            maxsize:     Maximum number of cache entries
            condition:   Optional function to determine if the result should be cached. Default is to cache all non-None and non-empty containers.

        Returns:
            A decorator to decorate the target function. The decorated function accepts an additional bool keyword argument
            `use_cache` (default is False), if set to False, force not to read the cache, but the calculation result
            will still try to write to the cache (if the condition is met).
        """
        from cachetools import TTLCache
        from cachetools.keys import hashkey
        cache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        lock = RLock()

        def decorator(func: Callable) -> Callable:
            def wrapper(decorated_cls , *args, **kwargs) -> Any:
                # generate cache key (based on all positional and keyword arguments, excluding use_cache)
                use_cache = kwargs.pop('use_cache', False)
                key = (decorated_cls , *hashkey(*args, **kwargs))

                # 1. if allowed to use cache and key is in cache, return directly
                with lock:
                    if use_cache and key in cache:
                        return cache[key]

                # 2. call the real function (exceptions are not cached)
                result = func(decorated_cls, *args, **kwargs)

                # 3. determine if the result should be cached
                should_cache = condition(result) if condition else cls._default_condition(result)

                # 4. if the condition is met, write to cache (whether the cache read fails or the calculation is forced)
                if should_cache:
                    with lock:
                        cache[key] = result

                return result

            return wrapper
        return decorator

class BaseProxiesFinder(ABC, BaseClass.BoundLogger):
    _find_cached_impl: Callable[..., ProxySet] | None = None

    @classmethod
    @abstractmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        raise NotImplementedError

    @classmethod
    def _find_impl(cls, level_type: Literal["any", "anonymous"] = "anonymous") -> ProxySet:
        try:
            proxies = retry_call(cls.find_candidates, kwargs={"level_type": level_type}, attempts=3, base_delay=1.0)
            if isinstance(proxies, Exception):
                raise proxies
            return proxies.set_source(cls.__name__)
        except Exception as e:
            cls.logger.alert1(f"[!] Error occurred while finding proxies through {cls.__name__} level_type={level_type}: {e}")
            return ProxySet()

    @classmethod
    def _get_find_cached_impl(cls) -> Callable[..., ProxySet]:
        cached_impl = cls._find_cached_impl
        if cached_impl is None:
            cached_impl = ProxiesCache.cached_function(
                ttl_seconds=60,
                condition=lambda x: len(x) > 0,
            )(cls._find_impl)
            cls._find_cached_impl = cached_impl
        return cast(Callable[..., ProxySet], cached_impl)

    @classmethod
    def find(
        cls,
        level_type: Literal["any", "anonymous"] = "anonymous",
        use_cache: bool = False,
    ) -> ProxySet:
        return cls._get_find_cached_impl()(cls, level_type=level_type, use_cache=use_cache)

class ZDAYEFinder(BaseProxiesFinder):
    """Get proxies from Zdaye API"""
    MAIN_PAGE = "https://www.zdaye.com/"
    API_URL = "http://www.zdopen.com/FreeProxy/Get/"
    INTERVAL = 1.2
    last_request_time = 0

    @classmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Get free proxies from Zdaye API"""
        candidates = []
        for protocol in ["https" , "socks5" , "https" , "https"]:
            candidates += cls._get_zdaye_proxies(None, protocol, level_type , silent = True)
        return ProxySet(candidates)

    @classmethod
    def _zdaye_api_url(
        cls, count: int | None = 100 , protocol_type: Literal["any" , "socks4" , "socks5" , "http", "https"] | str = "http" , 
        level_type: Literal["any" , "anonymous"] = "anonymous") -> str:
        """URL of Zdaye API"""
        kwargs = {
            "app_id": MACHINE.secret.get('accounts' , 'zdaye/app_id'),
            "akey": MACHINE.secret.get('accounts' , 'zdaye/akey'),
            "count": count,
            "dalu": 1,
            "level_type": 3 if level_type == "anonymous" else None,
            "protocol_type": {
                "any": None,
                "http": 1,
                "socks4": 2,
                "socks5": 3,
                "https": 4,
            }[protocol_type],
            "return_type": 3,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        url = f"{cls.API_URL}?" + "&".join([f"{k}={v}" for k, v in kwargs.items()])
        return url

    @classmethod
    def _get_zdaye_proxies(cls, count: int | None = None , protocol_type: Literal["any" , "socks4" , "socks5" , "http", "https"] | str = "http" , 
                           level_type: Literal["any" , "anonymous"] = "anonymous" , silent: bool = False) -> list[str]:
        """Get proxies from Zdaye API"""
        with Silence(silent):
            url = cls._zdaye_api_url(count, protocol_type, level_type)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            try:
                if (wait_time := cls.INTERVAL - time.time() + cls.last_request_time) > 0:
                    time.sleep(wait_time) # wait for the interval
                cls.logger.stdout("[*] Getting proxies from Zdaye API...")
                response = requests.get(url, headers=headers, timeout=15)
                cls.last_request_time = time.time()
                if response.status_code != 200:
                    cls.logger.alert1(f"[!] API request failed, status code: {response.status_code}")
                    return []
                
                data = response.json()
                code = data.get('code')
                
                if int(code) != 10001:
                    cls.logger.alert1(f"[!] API returned error: {data.get('msg', 'unknown error')} (code: {code})")
                    return []
                
                proxy_list = data.get('data', {}).get('proxy_list', [])
                proxies = []
                for item in proxy_list:
                    if isinstance(item, dict):
                        proxy_str = "{}://{}:{}".format(item.get('protocol'), item.get('ip'), item.get('port'))
                        proxies.append(proxy_str)
                    elif isinstance(item, str):
                        proxies.append(item)
                
                cls.logger.stdout(f"[+] Successfully got {len(proxies)} proxies")
                return proxies
                
            except requests.exceptions.RequestException as e:
                cls.logger.alert1(f"[!] Error occurred while requesting API: {e}")
                return []
            except ValueError as e:
                cls.logger.alert1(f"[!] Failed to parse JSON response: {e}")
                return []

class FreeProxyListNetFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    MAIN_PAGE = "https://free-proxy-list.net/"
    LIST_URLS = ["https://free-proxy-list.net/zh-cn/us-proxy.html"]

    @classmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in cls.LIST_URLS:
            r = requests.get(url)
            r.raise_for_status()
            candidates = cls._parse_proxy_table(r.text , level_type)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , level_type: Literal["any" , "anonymous"] = "anonymous") -> list[str]:
        """Parse the HTML proxy table from free-proxy-list.net and return a list of ``http://ip:port`` strings."""
        from bs4 import BeautifulSoup , Tag
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            return []
        out: list[str] = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td") # type: ignore[attr-defined]
            if len(cols) < 7:
                continue
            ip = cols[0].get_text(strip=True)
            port = cols[1].get_text(strip=True)
            anonymous = cols[4].get_text(strip=True).lower() in ["anonymous", "elite proxy"]
            https_ok = cols[6].get_text(strip=True).lower() == "yes"
            if not ip or not port or not https_ok:
                continue
            if level_type == "anonymous" and not anonymous:
                continue
            out.append(f"http://{ip}:{port}")
        return out

class FreeProxyListCCFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    MAIN_PAGE = "https://freeproxylist.cc/"
    LIST_URLS = ["https://freeproxylist.cc/servers/" , *[f"https://freeproxylist.cc/servers/{i}.html" for i in range(2, 6)]]

    @classmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in cls.LIST_URLS:
            r = requests.get(url)
            r.raise_for_status()
            candidates += cls._parse_proxy_table(r.text , level_type)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , level_type: Literal["any" , "anonymous"] = "anonymous") -> list[str]:
        """Parse the HTML proxy table from freeproxylist.cc and return a list of ``http://ip:port`` strings."""
        from bs4 import BeautifulSoup , Tag
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            return []
        out: list[str] = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td") # type: ignore[attr-defined]
            if len(cols) < 7:
                continue
            ip = cols[0].get_text(strip=True)
            port = cols[1].get_text(strip=True)
            anonymous = cols[4].get_text(strip=True).lower() in ["anonymous", "elite"]
            https_ok = cols[5].get_text(strip=True).lower() == "yes"
            if not ip or not port or not https_ok:
                continue
            if level_type == "anonymous" and not anonymous:
                continue
            out.append(f"http://{ip}:{port}")
        return out

class FreeProxyWorldFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    MAIN_PAGE = "freeproxy.world"
    FORMAT_URL = "freeproxy.world/?type={protocol}&anonymity={anonymity_level}&country=&speed=1000&port="

    @classmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for protocol in ['https' , 'socks5']:
            url = cls.FORMAT_URL.format(protocol = protocol , anonymity_level = 4 if level_type == "anonymous" else '')
            r = requests.get(url)
            r.raise_for_status()
            candidates += cls._parse_proxy_table(r.text , protocol)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , protocol: Literal["http" , "socks4" , "socks5" , "https"] | str = "https") -> list[str]:
        """Parse the HTML proxy table from freeproxy.world and return a list of ``<protocol>://ip:port`` strings."""
        from bs4 import BeautifulSoup , Tag
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            return []
        out: list[str] = []
        prefix = "http://" if protocol == 'https' else f"{protocol}://"
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td") # type: ignore[attr-defined]
            if len(cols) < 8:
                continue
            
            ip = cols[0].get_text(strip=True)
            port = cols[1].get_text(strip=True)
            if not ip or not port:
                continue
            out.append(f"{prefix}{ip}:{port}")
        return out
class ProxiflyFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    MAIN_PAGE = "https://github.com/proxifly/free-proxy-list"
    PROXY_SOURCES = {
        "https": "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/https/data.txt",
        "socks5": "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/socks5/data.txt",
        "http": "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/http/data.txt",
        "socks4": "https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/socks4/data.txt",
    }
    LIST_URLS = [f"https://cdn.jsdelivr.net/gh/proxifly/free-proxy-list@main/proxies/protocols/{protocol}/data.txt" for protocol in ["socks5" , "https"]]

    @classmethod
    def find_candidates(cls , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in cls.LIST_URLS:
            resp = requests.get(url, timeout=10.0)
            lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
            candidates += random.sample(lines , min(100 , len(lines) // 2))
        return ProxySet(candidates)

class FreeProxyFinder:
    """Aggregate proxy finder that queries all registered :class:`BaseProxiesFinder` implementations."""

    FINDERS_TYPE = {
        'zdaye': ZDAYEFinder,
        'fplcc':  FreeProxyListCCFinder,
        'fpw': FreeProxyWorldFinder,
    }
    def __init__(self):
        self.finders : dict[str, type[BaseProxiesFinder]] = {name: finder_type for name, finder_type in self.FINDERS_TYPE.items()}

    def __len__(self) -> int:
        return len(self.finders)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} finders)"

    def find(self , level_type: Literal["any" , "anonymous"] = "anonymous" , use_cache: bool = False) -> ProxySet:
        """Collect proxies from all registered finders and return the deduplicated union."""
        proxies = ProxySet()
        for finder in self.finders.values():
            proxies.extend(finder.find(level_type=level_type, use_cache=use_cache))
        return proxies
