import time
from curl_cffi import requests
from typing import Literal

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup , Tag

from src.proj.abc import Silence
from src.proj.log import Logger
from src.proj.util.http import http_session

class BaseProxiesFinder(ABC):
    """Auto discover HTTP proxies from public proxy list."""
    @abstractmethod
    def find_candidates(self , max_count: int = 100) -> list[str]:
        """Fetch proxy candidates from free proxy list."""
        raise NotImplementedError
    
class ZDAYEProxiesFinder(BaseProxiesFinder):
    ZDAYE_APPID = "202603230720009329"     
    ZDAYE_AKEY  = "e8a0f7acf306edea"   
    INTERVAL = 1.2
    last_request_time = 0

    @classmethod
    def _zdaye_api_url(cls, count: int | None = 100 , protocol_type: Literal["any" , "socks4" , "socks5" , "http", "https"] | str = "http" , level_type: Literal["any" , "anonymous"] = "anonymous") -> str:
        """URL of Zdaye API"""
        addr = "http://www.zdopen.com/FreeProxy/Get/"
        kwargs = {
            "app_id": cls.ZDAYE_APPID,
            "akey": cls.ZDAYE_AKEY,
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
        url = f"{addr}?" + "&".join([f"{k}={v}" for k, v in kwargs.items()])
        return url

    @classmethod
    def _get_zdaye_proxies(cls, count: int | None = None , protocol_type: Literal["any" , "socks4" , "socks5" , "http", "https"] | str = "http" , level_type: Literal["any" , "anonymous"] = "anonymous" , silent: bool = False) -> list[str]:
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
                    Logger.stderr(f"[!] API request failed, status code: {response.status_code}")
                    return []
                
                data = response.json()
                code = data.get('code')
                
                if int(code) != 10001:
                    Logger.stderr(f"[!] API returned error: {data.get('msg', 'unknown error')} (code: {code})")
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
                Logger.stderr(f"[!] Error occurred while requesting API: {e}")
                return []
            except ValueError as e:
                Logger.stderr(f"[!] Failed to parse JSON response: {e}")
                return []

    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous") -> list[str]:
        """Get free proxies from Zdaye API"""
        proxies = []
        for protocol in ["https" , "socks5" , "https" , "http" , "https"]:
            proxies += self._get_zdaye_proxies(None, protocol, level_type , silent = True)
        proxies = list(set(proxies))
        return proxies

class FPLProxiesFinder(BaseProxiesFinder):
    """Auto discover HTTP proxies from public proxy list."""
    def find_candidates(self , max_count: int = 100) -> list[str]:
        """Fetch proxy candidates from free proxy list."""
        with http_session(timeout=(10.0, 15.0)) as client:
            r = client.get("https://free-proxy-list.net/zh-cn/us-proxy.html")
            r.raise_for_status()
        return self._parse_proxy_table(r.text)[:max_count]

    @classmethod
    def _parse_proxy_table(cls , html: str) -> list[str]:
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
            https_ok = cols[6].get_text(strip=True).lower() == "yes"
            if not ip or not port or not https_ok:
                continue
            out.append(f"http://{ip}:{port}")
        return out

class FreeProxyFinder(ZDAYEProxiesFinder):
    def find_candidates(self , level_type: Literal["any" , "anonymous"] = "anonymous" , num_rounds: int = 1) -> list[str]:
        """Get free proxies from Zdaye API"""
        proxies = []
        for protocol in ["https" , "socks5" , "http" , "https"] * num_rounds:
            proxies += self._get_zdaye_proxies(None, protocol, level_type , silent = True)
        proxies = list(set(proxies))
        return proxies
