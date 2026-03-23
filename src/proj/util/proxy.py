import threading , json , requests , httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable , Literal
from datetime import datetime , timedelta

from abc import ABC, abstractmethod
from bs4 import BeautifulSoup , Tag

from src.proj.abc import Silence
from src.proj.env import PATH
from src.proj.log import Logger
from src.proj.calendar import BJTZ

class ProxyGetter:
    @classmethod
    def get_working_proxies(cls , url: str , force_refresh: bool = False) -> list[str]:
        """Get working proxies"""
        return ZDAYEProxiesPool.get_working_proxies(url , force_refresh = force_refresh)

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def http_client(
    *,
    proxy: str | None = None,
    trust_env: bool | None = None,
    **kwargs: object,
) -> httpx.Client:
    # Disable keep-alive to reduce "Server disconnected" type half-open connections
    limits = httpx.Limits(max_keepalive_connections=0, max_connections=20)
    kw: dict = {
        "headers": {"User-Agent": CHROME_UA},
        "timeout": httpx.Timeout(120.0, connect=30.0),
        "follow_redirects": True,
        "limits": limits,
    }
    if proxy:
        kw["proxy"] = proxy
    if trust_env is not None:
        kw["trust_env"] = trust_env
    kw.update(kwargs)
    return httpx.Client(**kw)

class ProxiesCache:
    DEFAULT_CACHE_TTL_SEC = 300.0
    _cached_proxies: dict[str, tuple[datetime, list[str]]] = {}
    _proxies_file = PATH.local_machine.joinpath('proxies' , 'proxies.json')
    _locks = {
        'cache': threading.Lock(),
        'file': threading.Lock(),
    }
    _refresh_time : dict[str, timedelta] = {
        'cache': timedelta(seconds=300),
        'file': timedelta(days=1),
    }

    _default_time = datetime.strptime('1900-01-01 00:00:00', "%Y-%m-%d %H:%M:%S").replace(tzinfo=BJTZ)

    @classmethod
    def load_file(cls , verify_url: str) -> tuple[datetime, list[str]]:
        """Load proxies from proxies.json file"""
        update_time = cls._default_time
        candidates = []
        if cls._proxies_file.exists():
            with open(cls._proxies_file, 'r') as f:
                proxies = json.load(f)
                if verify_url in proxies:
                    candidates = proxies[verify_url] if isinstance(proxies[verify_url], list) else proxies[verify_url]['candidates']
                    if isinstance(proxies[verify_url], dict) and 'update_time' in proxies[verify_url]:
                        update_time = datetime.strptime(proxies[verify_url]['update_time'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=BJTZ)     
        return update_time, candidates

    @classmethod
    def update_file(cls , verify_url: str, verified: list[str]) -> None:
        """Update working proxies in cache file"""
        update_time = datetime.now(tz=BJTZ).strftime("%Y-%m-%d %H:%M:%S")
        with cls._locks['file']:
            proxies = PATH.read_json(cls._proxies_file)
            proxies[verify_url] = {'update_time': update_time,'candidates': verified}
            PATH.dump_json(proxies, cls._proxies_file, overwrite=True)

    @classmethod
    def get_cached_proxies(cls , verify_url: str , force_refresh: bool = False) -> tuple[bool, list[str]]:
        """Get cached proxies from cache , return is if_obsolete and proxies"""
        if force_refresh:
            return True , []
        
        now = datetime.now(tz=BJTZ)
        with cls._locks['cache']:
            cached_time, cached_candidates = cls._cached_proxies.get(verify_url, (cls._default_time, []))
            if (now - cached_time) < cls._refresh_time['cache']:
                return False , cached_candidates
        
        with cls._locks['file']:
            update_time, filed_candidates = cls.load_file(verify_url)
            cached_candidates = list(set(cached_candidates + filed_candidates))
            if (now - update_time) < cls._refresh_time['file']:
                return False , cached_candidates
        
        return True , cached_candidates

    @classmethod
    def cache_proxies(cls , verify_url: str, verified: list[str]) -> None:
        """Cache proxies"""
        with cls._locks['cache']:
            cls._cached_proxies[verify_url] = (datetime.now(tz=BJTZ), verified)
        ProxiesCache.update_file(verify_url, verified)

class BaseProxiesPool(ABC):
    """Auto discover HTTP proxies from public proxy list."""
    VERIFY_URL = "https://www.szse.cn"
    DEFAULT_CACHE_TTL_SEC = 300.0
    TIMEOUT = 5
    MAX_WORKERS = 24

    cache_lock = threading.Lock()
    cached_proxies: dict[str, list[str]] = {}
    cache_monotonic_at: dict[str, float] = {}

    CACHE_file = PATH.local_machine.joinpath('proxies' , 'proxies.json')
    file_lock = threading.Lock()

    @abstractmethod
    def fetch_proxy_candidates(self , max_count: int = 100) -> list[str]:
        """Fetch proxy candidates from free proxy list."""
        raise NotImplementedError

    @classmethod
    def verify_proxy(cls , proxy_url: str, verify_url: str = VERIFY_URL, * , timeout_sec: float = TIMEOUT) -> bool:
        """Whether the proxy can access the test URL (GET, non-4xx/5xx is considered available)."""
        try:
            with http_client(
                proxy=proxy_url,
                trust_env=False,
                timeout=httpx.Timeout(timeout_sec, connect=min(5.0, timeout_sec)),
            ) as client:
                r = client.get(verify_url)
                return r.status_code < 400
        except Exception:
            return False

    @classmethod
    def filter_verified_proxies(cls , candidates: Iterable[str], verify_url: str = VERIFY_URL, * ,
        verify_timeout: float = 8.0,
        verify_workers: int = 24,
    ) -> list[str]:
        """parallel verify, take the first ``max_keep`` passed proxies in the original order of the candidate list."""
        cands = list(dict.fromkeys(candidates))

        passed: set[str] = set()
        workers = max(1, min(verify_workers, len(cands)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(cls.verify_proxy, p , verify_url=verify_url, timeout_sec=verify_timeout): p
                for p in cands
            }
            for fut in as_completed(fut_map):
                if fut.result():
                    passed.add(fut_map[fut])
        ordered = [p for p in cands if p in passed]
        return ordered

    @classmethod
    def get_working_proxies(
        cls , verify_url: str = VERIFY_URL, * , max_count: int = 100, min_count: int = 10, verify_timeout: float = 8.0, verify_workers: int = 24,
        force_refresh: bool = False) -> list[str]:
        """
        return the list of available proxy URLs; with in-process short-term cache to avoid hitting the free proxy site for each failed task.
        if force_refresh is True, ignore the expired cache.
        """
        if_obsolete, cached_candidates = ProxiesCache.get_cached_proxies(verify_url, force_refresh)
        verified = cls.filter_verified_proxies(cached_candidates, verify_url=verify_url, verify_timeout=verify_timeout, verify_workers=verify_workers)
        
        if if_obsolete and len(verified) < min_count:
            proxies_pool = cls()
            new_candidates = proxies_pool.fetch_proxy_candidates(max_count=max_count)
            new_candidates = [c for c in new_candidates if c not in cached_candidates]
            new_verified = cls.filter_verified_proxies(
                new_candidates, verify_url=verify_url,
                verify_timeout=verify_timeout,
                verify_workers=verify_workers,
            )
            verified = list(set(verified + new_verified))

        ProxiesCache.cache_proxies(verify_url, verified)
        return verified[:max_count]

class ZDAYEProxiesPool(BaseProxiesPool):
    ZDAYE_APPID = "202603230720009329"     
    ZDAYE_AKEY  = "e8a0f7acf306edea"   

    @classmethod
    def _zdaye_api_url(cls, count: int = 10 , type: Literal["http", "https"] = "http") -> str:
        """URL of Zdaye API"""
        protocol_type = 4 if type == "https" else 1
        addr = "http://www.zdopen.com/FreeProxy/Get/?"
        url = "{}app_id={}&akey={}&count={}&dalu=1&protocol_type={}&return_type=3".format(
            addr, cls.ZDAYE_APPID, cls.ZDAYE_AKEY, count, protocol_type)
        return url

    @classmethod
    def _get_zdaye_proxies(cls, count: int = 10 , type: Literal["http", "https"] = "http" , silent: bool = False) -> list[str]:
        """Get proxies from Zdaye API"""
        with Silence(silent):
            url = cls._zdaye_api_url(count, type)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            try:
                Logger.stdout("[*] Getting proxies from Zdaye API...")
                response = requests.get(url, headers=headers, timeout=15)
                
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

    def fetch_proxy_candidates(self , max_count: int = 100 , type: Literal["http", "https"] = "https") -> list[str]:
        """Get free proxies from Zdaye API"""
        return self._get_zdaye_proxies(max_count, type , silent = True)

class FPLProxiesPool(BaseProxiesPool):
    """Auto discover HTTP proxies from public proxy list."""
    def fetch_proxy_candidates(self , max_count: int = 100) -> list[str]:
        """Fetch proxy candidates from free proxy list."""
        headers = {"User-Agent": CHROME_UA}
        with httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(15.0, connect=min(10.0, 15.0)),
        ) as client:
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
