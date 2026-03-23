import httpx 
import random
from pprint import pformat
from datetime import datetime
from typing import Any , Generator
from urllib.parse import urlencode
from abc import ABC, abstractmethod
from dataclasses import dataclass , field

from threading import Lock

from src.proj import Logger
from src.proj.util import ProxyGetter
from .util import parse_jsonp , CHROME_UA , Announcement , http_client , call_with_retry , range_dates , AnnouncementExporter

EXCHANGES : list[str] = ['sse', 'szse', 'bse']

@dataclass
class FetcherTask:
    exchange: str
    start: int
    end: int
    redownload: bool = False

    proxies : list[str | None] = field(default_factory=lambda: [None])

    @property
    def exporter(self) -> AnnouncementExporter:
        return AnnouncementExporter(self.exchange)

    @property
    def title(self) -> str:
        return f"{self.exchange} {self.start}~{self.end}"

    @property
    def should_be_skipped(self) -> bool:
        return self.exporter.should_skip_download(self.start, self.end, redownload=self.redownload)

    def _fetch(self, px: str | None, te: bool | None , attempts: int = 1) -> list[Announcement] | None:
        with http_client(proxy=px, trust_env=te) as client:
            fetcher = AnnoucementFetcher.exchange_fetcher(self.exchange, client)
            suffix = f" [{px}]" if px else ""
            return call_with_retry(lambda: fetcher.fetch_date(self.start, self.end), label=f"{self.title}{suffix}", attempts=attempts)

    def _client_options(self, use_proxies: bool = True, proxy_only : bool = True) -> Generator[tuple[str | None, bool | None], None, None]:
        if use_proxies:
            proxies = ExchangeFetcherStates.get_proxies(self.exchange)
            random.shuffle(proxies)
            for px in proxies:
                yield px, False
        if not proxy_only:
            yield None, None

    def run(self, *, proxy: str | None = None, trust_env: bool | None = None, auto_discover_proxy: bool = False , indent : int = 1 , vb_level : Any = 3) -> list[Announcement] | None:
        """Fetch by natural day and exchange; try public proxy list when direct connection (and optional fixed proxy) still fails and ``auto_discover_proxy`` is enabled."""
        if not ExchangeFetcherStates.get_state(self.exchange):
            return

        if self.should_be_skipped:
            Logger.skipping(f"{self.title}, already have historical data (use redownload to force re-download)" , indent = indent, vb_level = vb_level)
            return

        result = None
        i = -1
        for proxy, trust_env in self._client_options(use_proxies=auto_discover_proxy, proxy_only=True):
            i += 1
            if not ExchangeFetcherStates.get_proxy_state(self.exchange , proxy):
                continue
            result = self._fetch(proxy, trust_env)
            if result is not None:
                ExchangeFetcherStates.set_proxy_state(self.exchange , proxy, True)
                break
            ExchangeFetcherStates.add_proxy_fail_counts(self.exchange , proxy , indent = indent , vb_level = vb_level - 2)
            if auto_discover_proxy and proxy is None and i == 0:
                Logger.alert1(f"{self.title} failed to connect, will try to discover public proxy list" , indent = indent, vb_level = vb_level)
        
        if result is not None:
            self.exporter.save_data(result, self.start, self.end)
        else:
            if auto_discover_proxy and i >= 1:
                ExchangeFetcherStates.set_state(self.exchange, False)
        return result

    def run_with_proxies(self, *, indent : int = 1 , vb_level : Any =2) -> None:
        self.run(auto_discover_proxy=True, indent = indent, vb_level = vb_level)

    @classmethod
    def tasks_flat(cls , start: int, end: int, step: int = 1, redownload: bool = False) -> list['FetcherTask']:
        tasks = []
        for start, end in range_dates(start, end, step):
            for exchange in EXCHANGES:
                tasks.append(FetcherTask(exchange, start, end , redownload))
        return tasks

    @classmethod
    def tasks_groups(cls , start: int, end: int, step: int = 1, redownload: bool = False , * , 
                     max_groups: int = 100, min_tasks_per_group: int = 3) -> list[list['FetcherTask']]:
        """partition the tasks into num_groups groups"""
        tasks = cls.tasks_flat(start, end, step, redownload)
        n = min(max(1, max_groups), len(tasks) // min_tasks_per_group)
        buckets: list[list[FetcherTask]] = [[] for _ in range(n)]
        for i, task in enumerate(tasks):
            buckets[i % n].append(task)
        return [bucket for bucket in buckets if bucket]

class ExchangeFetcherStates:
    _proxies : dict[str, list[str]] = {exchange: [] for exchange in EXCHANGES}
    _states : dict[str, bool] = {exchange: True for exchange in EXCHANGES}
    _proxy_states: dict[str, dict[None | str , bool]] = {exchange: {None: False} for exchange in EXCHANGES}
    _proxy_fail_counts: dict[str, dict[None | str , int]] = {exchange:{None:0} for exchange in EXCHANGES}
    _proxy_refresh_counts: dict[str, int] = {exchange: 0 for exchange in EXCHANGES}

    _locks : dict[str, Lock] = {'states': Lock(), 'proxy_states': Lock(), 'proxy_fail_counts': Lock(), 'proxies': Lock() , 'proxy_refresh_counts': Lock()}

    MAX_CONNECTION_FAILURES = 2
    MAX_PROXIES_FETCH_COUNTS = 3
    verify_urls = {
        "sse": "https://www.sse.com.cn", 
        "szse": "https://www.szse.cn", 
        "bse": "https://www.bse.cn"
    }

    @classmethod
    def get_proxies(cls , exchange: str) -> list[str]:
        with cls._locks['proxies']:
            if not cls._proxies[exchange]:
                proxies = ProxyGetter.get_working_proxies(cls.verify_urls[exchange])
                cls._proxies[exchange] = proxies
            return cls._proxies[exchange]

    @classmethod
    def init_all_proxies(cls) -> None:
        for exchange in EXCHANGES:
            cls.get_proxies(exchange)

    @classmethod
    def list_all_proxies(cls) -> str:
        return pformat(cls._proxies)

    @classmethod
    def get_states(cls) -> dict[str, bool]:
        with cls._locks['states']:
            return {exchange: state for exchange, state in cls._states.items()}

    @classmethod
    def get_state(cls , exchange: str) -> bool:
        with cls._locks['states']:
            return cls._states[exchange]

    @classmethod
    def set_state(cls , exchange: str , state: bool , indent : int = 1 , vb_level : Any = 0):
        with cls._locks['states']:
            if state is False:
                with cls._locks['proxy_refresh_counts']:
                    if cls._proxy_refresh_counts[exchange] >= cls.MAX_PROXIES_FETCH_COUNTS:
                        with cls._locks['proxies']:
                            cls._proxies[exchange].clear()
                        cls.get_proxies(exchange)
                        cls._states[exchange] = state
                        cls._proxy_refresh_counts[exchange] += 1
                        Logger.alert2(f"Exchange {exchange} may have failed too many times, try to refresh the proxy list" , indent = indent , vb_level = vb_level)
                    else:
                        Logger.alert2(f"Exchange {exchange} may have failed too many times, this task will not try this exchange anymore" , indent = indent , vb_level = vb_level)
            else:
                cls._states[exchange] = state
        
    @classmethod
    def get_proxy_state(cls , exchange: str , proxy: str | None) -> bool:
        with cls._locks['proxy_states']:
            return cls._proxy_states[exchange].get(proxy, True)

    @classmethod
    def set_proxy_state(cls , exchange: str , proxy: str | None , state: bool , indent : int = 1 , vb_level : Any = 0):
        with cls._locks['proxy_states']:
            cls._proxy_states[exchange][proxy] = state
        if state:
            cls.set_state(exchange, True , indent = indent , vb_level = vb_level)
        else:
            Logger.alert1(f"Exchange {exchange} with proxy {proxy} Aborted, exceed the limit of connection failures" , indent = indent , vb_level = vb_level)

    @classmethod
    def add_proxy_fail_counts(cls , exchange: str , proxy: str | None , indent : int = 1 , vb_level : Any = 0):
        with cls._locks['proxy_fail_counts']:
            cls._proxy_fail_counts[exchange][proxy] = cls._proxy_fail_counts[exchange].get(proxy, 0) + 1
            if cls._proxy_fail_counts[exchange][proxy] >= cls.MAX_CONNECTION_FAILURES:
                cls.set_proxy_state(exchange, proxy, False , indent = indent , vb_level = vb_level)
                
    @classmethod
    def reset_states(cls) -> None:
        """When a new task starts, clear the disabled and failure counts for each exchange."""
        with cls._locks['proxies']:
            cls._proxies = {exchange: [] for exchange in EXCHANGES}
        with cls._locks['states']:
            cls._states = {exchange: True for exchange in EXCHANGES}
        with cls._locks['proxy_states']:
            cls._proxy_states = {exchange: {None: True} for exchange in EXCHANGES}
        with cls._locks['proxy_fail_counts']:
            cls._proxy_fail_counts = {exchange: {None: 0} for exchange in EXCHANGES}

class AnnoucementFetcher(ABC):
    FETCH_KWARGS : list[dict[str, Any]] = [{}]

    def __init__(self, client: httpx.Client):
        self.client = client

    def fetch_date(self , start: int , end: int | None = None) -> list[Announcement]:
        if end is None:
            end = start
        out: list[Announcement] = []
        seen: set[tuple[str, str, str, int]] = set()
        for kwargs in self.FETCH_KWARGS:
            announcements = self.fetch_announcements(start, end, **kwargs)
            for ann in announcements:
                if ann and ann.key not in seen:
                    out.append(ann)
                seen.add(ann.key)
        return out

    @abstractmethod
    def fetch_announcements(self, start : int, end: int) -> list[Announcement]:
        raise NotImplementedError

    @classmethod
    def exchange_fetcher(cls, exchange: str , client: httpx.Client):
        if exchange.lower() == "sse":
            return SSEAnnFetcher(client)
        elif exchange.lower() == "szse":
            return SZSEAnnFetcher(client)
        elif exchange.lower() == "bse":
            return BSEAnnFetcher(client)
        else:
            raise ValueError(f"Invalid exchange: {exchange}")


class SSEAnnFetcher(AnnoucementFetcher):
    REFERER = "https://www.sse.com.cn/"
    JSONP_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
    FETCH_KWARGS : list[dict[str, Any]] = [{"stock_type": "1"}, {"stock_type": "8"}]

    def fetch_announcements(self, start : int, end: int, *, stock_type: str = "1", page_size: int = 50) -> list[Announcement]:
        """Shanghai Stock Exchange: query.sse.com.cn JSONP, fields contain SECURITY_CODE / SECURITY_NAME / TITLE / BULLETIN_TYPE_DESC / SSEDATE / URL."""
        start_s = datetime.strptime(str(start), "%Y%m%d").date().isoformat()
        end_s = datetime.strptime(str(end), "%Y%m%d").date().isoformat()
        all_rows: list[Announcement] = []
        page_no = 1

        while True:
            params = {
                "jsonCallBack": "jsonp",
                "isPagination": "true",
                "pageHelp.pageSize": page_size,
                "pageHelp.pageNo": page_no,
                "pageHelp.beginPage": page_no,
                "pageHelp.cacheSize": 1,
                "START_DATE": start_s,
                "END_DATE": end_s,
                "SECURITY_CODE": "",
                "TITLE": "",
                "stockType": stock_type,
            }
            r = self.client.get(self.JSONP_URL,params=params,headers={"Referer": self.REFERER})
            r.raise_for_status()
            payload = parse_jsonp(r.text)
            if not isinstance(payload, dict):
                break
            ph = payload.get("pageHelp") or {}
            data = ph.get("data") or []
            rows = self._parse_groups(data)
            all_rows.extend(rows)
            page_count = int(ph.get("pageCount") or 0)
            if page_no >= page_count or page_count == 0:
                break
            page_no += 1

        return all_rows

    @classmethod
    def _parse_groups(cls , groups: list[Any]) -> list[Announcement]:
        rows: list[Announcement] = []
        for g in groups:
            if isinstance(g, list):
                rows.extend([Announcement.from_sse(item) for item in g if isinstance(item, dict)])
            elif isinstance(g, dict):
                rows.append(Announcement.from_sse(g))
        return rows

class SZSEAnnFetcher(AnnoucementFetcher):
    REFERER = "https://www.szse.cn/disclosure/listed/notice/index.html"
    ANN_LIST = "https://www.szse.cn/api/disc/announcement/annList"

    def fetch_announcements(self, start : int, end: int, *, page_size: int = 50) -> list[Announcement]:
        """Shenzhen Stock Exchange: POST annList (POST JSON, paging by seDate interval."""
        start_s = datetime.strptime(str(start), "%Y%m%d").date().isoformat()
        end_s = datetime.strptime(str(end), "%Y%m%d").date().isoformat()
        body_base: dict[str, Any] = {
            "pageSize": page_size,
            "stock": [],
            "channelCode": ["listedNotice_disc"],
            "seDate": [start_s, end_s],
        }
        out: list[Announcement] = []
        page_num = 1
        headers = {"Referer": self.REFERER, "Content-Type": "application/json"}

        while True:
            body = {**body_base, "pageNum": page_num}
            url = f"{self.ANN_LIST}?random={random.random()}"
            r = self.client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            chunk = data.get("data") or []
            if not chunk:
                break
            out.extend([Announcement.from_szse(item) for item in chunk if isinstance(item, dict)])
            total = int(data.get("announceCount") or 0)
            if page_num * page_size >= total:
                break
            page_num += 1

        return out

class BSEAnnFetcher(AnnoucementFetcher):
    REFERER = "https://www.bse.cn/disclosure/announcement.html"
    ANNOUNCE_URL = "https://www.bse.cn/disclosureInfoController/companyAnnouncement.do"

    def fetch_announcements(self, start : int, end: int) -> list[Announcement]:
        """Beijing Stock Exchange: POST companyAnnouncement.do (POST JSON, paging by page."""
        # warmup
        r = self.client.get(self.REFERER, headers={"User-Agent": CHROME_UA})
        r.raise_for_status()
        headers = {
            "Referer": self.REFERER,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": CHROME_UA,
        }
        all_rows: list[Announcement] = []
        page = 0

        while True:
            body = self._query_body(page, start, end)
            r = self.client.post(self.ANNOUNCE_URL, content=body, headers=headers)
            r.raise_for_status()
            payload = parse_jsonp(r.text)
            if not isinstance(payload, list) or not payload:
                break
            block = payload[0]
            if not isinstance(block, dict):
                break
            li = block.get("listInfo") or {}
            content = li.get("content") or []
            if isinstance(content, list):
                all_rows.extend([Announcement.from_bse(item) for item in content if isinstance(item, dict)])
            total_pages = int(li.get("totalPages") or 0)
            if page + 1 >= total_pages or total_pages == 0:
                break
            page += 1

        return all_rows

    @classmethod
    def _query_body(cls, page: int, start: int, end: int) -> str:
        """Same as the form encoding on the official website (disclosureType[]=5 ...)."""
        need_fields = [
            "companyCd",
            "companyName",
            "disclosureTitle",
            "disclosurePostTitle",
            "destFilePath",
            "publishDate",
            "xxfcbj",
            "destFilePath",
            "fileExt",
            "xxzrlx",
            "disclosureType",
            "disclosureSubType",
        ]
        pairs: list[tuple[str, str]] = [
            ("disclosureType[]", "5"),
            ("disclosureSubtype[]", ""),
            ("page", str(page)),
            ("companyCd", ""),
            ("isNewThree", "1"),
            ("startTime", datetime.strptime(str(start), "%Y%m%d").date().isoformat()),
            ("endTime", datetime.strptime(str(end), "%Y%m%d").date().isoformat()),
            ("keyword", ""),
            ("xxfcbj[]", "2"),
            ("sortfield", "xxssdq"),
            ("sorttype", "asc"),
        ]
        for f in need_fields:
            pairs.append(("needFields[]", f))
        return urlencode(pairs, doseq=True)
