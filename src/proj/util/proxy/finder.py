"""Scrape public proxy lists (e.g. Zdaye) into ``ProxySet`` instances."""

import time
import random
from curl_cffi import requests
from typing import Literal

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup , Tag

from src.proj.core import Silence
from src.proj.log import Logger
from src.proj.env import MACHINE
from .core import ProxySet

class BaseProxiesFinder(ABC):
    """Auto discover HTTP proxies from public proxy list."""
    @abstractmethod
    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        raise NotImplementedError

    def find(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Find proxies from free proxy list."""
        try:
            return self.find_candidates(level_type=level_type).set_source(self.__class__.__name__)
        except Exception as e:
            Logger.alert1(f"[!] Error occurred while finding proxies through {self}: {e}")
            return ProxySet()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
    
class ZDAYEFinder(BaseProxiesFinder):
    """Get proxies from Zdaye API"""
    MAIN_PAGE = "https://www.zdaye.com/"
    API_URL = "http://www.zdopen.com/FreeProxy/Get/"
    APP_ID = MACHINE.secret['accounts']['zdaye']['app_id']
    AKEY   = MACHINE.secret['accounts']['zdaye']['akey']
    INTERVAL = 1.2
    last_request_time = 0

    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Get free proxies from Zdaye API"""
        candidates = []
        for protocol in ["https" , "socks5" , "https" , "https"]:
            candidates += self._get_zdaye_proxies(None, protocol, level_type , silent = True)
        return ProxySet(candidates)

    @classmethod
    def _zdaye_api_url(cls, count: int | None = 100 , protocol_type: Literal["any" , "socks4" , "socks5" , "http", "https"] | str = "http" , level_type: Literal["any" , "anonymous"] = "anonymous") -> str:
        """URL of Zdaye API"""
        kwargs = {
            "app_id": cls.APP_ID,
            "akey": cls.AKEY,
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
                Logger.stdout("[*] Getting proxies from Zdaye API...")
                response = requests.get(url, headers=headers, timeout=15)
                cls.last_request_time = time.time()
                if response.status_code != 200:
                    Logger.alert1(f"[!] API request failed, status code: {response.status_code}")
                    return []
                
                data = response.json()
                code = data.get('code')
                
                if int(code) != 10001:
                    Logger.alert1(f"[!] API returned error: {data.get('msg', 'unknown error')} (code: {code})")
                    return []
                
                proxy_list = data.get('data', {}).get('proxy_list', [])
                proxies = []
                for item in proxy_list:
                    if isinstance(item, dict):
                        proxy_str = "{}://{}:{}".format(item.get('protocol'), item.get('ip'), item.get('port'))
                        proxies.append(proxy_str)
                    elif isinstance(item, str):
                        proxies.append(item)
                
                Logger.stdout(f"[+] Successfully got {len(proxies)} proxies")
                return proxies
                
            except requests.exceptions.RequestException as e:
                Logger.alert1(f"[!] Error occurred while requesting API: {e}")
                return []
            except ValueError as e:
                Logger.alert1(f"[!] Failed to parse JSON response: {e}")
                return []

class FreeProxyListNetFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    MAIN_PAGE = "https://free-proxy-list.net/"
    LIST_URLS = ["https://free-proxy-list.net/zh-cn/us-proxy.html"]
    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in self.LIST_URLS:
            r = requests.get(url)
            r.raise_for_status()
            candidates = self._parse_proxy_table(r.text , level_type)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , level_type: Literal["any" , "anonymous"] = "anonymous") -> list[str]:
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

    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in self.LIST_URLS:
            r = requests.get(url)
            r.raise_for_status()
            candidates += self._parse_proxy_table(r.text , level_type)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , level_type: Literal["any" , "anonymous"] = "anonymous") -> list[str]:
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

    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for protocol in ['https' , 'socks5']:
            url = self.FORMAT_URL.format(protocol = protocol , anonymity_level = 4 if level_type == "anonymous" else '')
            r = requests.get(url)
            r.raise_for_status()
            candidates += self._parse_proxy_table(r.text , protocol)
        return ProxySet(candidates)

    @classmethod
    def _parse_proxy_table(cls , html: str , protocol: Literal["http" , "socks4" , "socks5" , "https"] | str = "https") -> list[str]:
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

    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Fetch proxy candidates from free proxy list."""
        candidates = []
        for url in self.LIST_URLS:
            resp = requests.get(url, timeout=10.0)
            lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
            candidates += random.sample(lines , min(100 , len(lines) // 2))
        return ProxySet(candidates)

class FreeProxyFinder:
    FINDERS_TYPE = {
        'zdaye': ZDAYEFinder,
        'fplcc':  FreeProxyListCCFinder,
        'fpw': FreeProxyWorldFinder,
    }
    def __init__(self):
        self.finders : dict[str, BaseProxiesFinder] = {name: finder_type() for name, finder_type in self.FINDERS_TYPE.items()}

    def __len__(self) -> int:
        return len(self.finders)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)} finders)"

    def find(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> ProxySet:
        """Get free proxies from Zdaye API"""
        proxies = ProxySet()
        for finder in self.finders.values():
            proxies.extend(finder.find(level_type=level_type))
        return proxies
