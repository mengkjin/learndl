import random
from curl_cffi import requests
from datetime import datetime
from typing import Any, Callable , Literal
from urllib.parse import urlencode
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.proj import Logger
from src.proj.util import ProxyAPI
from src.proj.util.proxy.caller import ProxyCaller
from src.proj.util.http import http_session , CHROME_UA , temporary_timeout_expand
from src.proj.util.error_handler import ErrorHandler
from .util import parse_jsonp , Announcement , range_dates , AnnouncementExporter

EXCHANGES : list[str] = ['sse', 'szse', 'bse']
EXCHANGE_URLS = {
    "sse": "https://www.sse.com.cn", 
    "szse": "https://www.szse.cn", 
    "bse": "https://www.bse.cn"
}
URL_KEYS = tuple(sorted(EXCHANGE_URLS.values()))

@dataclass
class FetcherTask:
    exchange: str
    start: int
    end: int
    redownload: bool = False

    @property
    def exporter(self) -> AnnouncementExporter:
        return AnnouncementExporter(self.exchange)

    @property
    def url(self) -> str:
        return EXCHANGE_URLS[self.exchange]

    @property
    def title(self) -> str:
        return f"{self.exchange} {self.start}~{self.end}"

    @property
    def should_be_skipped(self) -> bool:
        return self.exporter.should_skip_download(self.start, self.end, redownload=self.redownload)

    @property
    def proxy_pool(self):
        if not hasattr(self, '_proxy_pool'):
            self._proxy_pool = ProxyAPI.get_proxy_pool(URL_KEYS)
        return self._proxy_pool

    def fetch_date(self, proxy: str | None) -> list[Announcement] | Exception:
        with http_session(proxy=proxy , trust_env = proxy is None) as client:
            fetcher = AnnoucementFetcher.exchange_fetcher(self.exchange, client)
            return fetcher.fetch_date(self.start, self.end)

    def fetch_date_with_error_handler(self, proxy: str | None):
        fetch_date = ErrorHandler(self.fetch_date, handle_types=['http', 'all'], label=f"{self.title}" + (f"[{proxy}]" if proxy else ""))
        return fetch_date(proxy)

    def claw(self , proxy: str | None) -> bool | Exception:
        if self.should_be_skipped:
            return True
        result = self.fetch_date_with_error_handler(proxy)
        value = result.unwrap(error='return')
        if isinstance(value, Exception):
            return value
        elif isinstance(value, list):
            self.exporter.save_data(value, self.start, self.end)
            return True
        else:
            return False

    def to_proxy_caller(self , pool) -> ProxyCaller:
        return ProxyCaller(self.claw, self.url, pool = pool)

    def run(self, pool = None, *, max_proxies_try: int = 3, indent : int = 1 , vb_level : Any = 3 , error : Literal['raise' , 'return'] = 'return') -> bool | Exception:
        """Fetch by natural day and exchange; try public proxy list when direct connection (and optional fixed proxy) still fails and ``auto_discover_proxy`` is enabled."""
        result = False
        if self.should_be_skipped:
            Logger.skipping(f"{self.title}, already have historical data (use redownload to force re-download)" , indent = indent, vb_level = vb_level)
            return result

        if pool is None:
            pool = ProxyAPI.get_proxy_pool(self.url)
            
        if pool.proxies[self.url].valid_count > 0:
            for _ in range(max_proxies_try):
                proxy = pool.acquire(url=self.url)
                if proxy is None:
                    break
                result = self.claw(proxy.url)
                pool.release(proxy, result is not None)
                if result is not None:
                    break
        if result is None:
            result = self.claw(None)

        if error == 'raise' and isinstance(result, Exception):
            raise result
        return result

    @classmethod
    def tasks_flat(cls , start: int, end: int, step: int = 1, redownload: bool = False) -> list['FetcherTask']:
        tasks = []
        for start, end in range_dates(start, end, step):
            for exchange in EXCHANGES:
                task = FetcherTask(exchange, start, end , redownload)
                if not task.should_be_skipped:
                    tasks.append(task)
        return tasks

class AnnoucementFetcher(ABC):
    exchange: Literal["sse", "szse", "bse"]
    FETCH_KWARGS : list[dict[str, Any]] = [{}]

    def __init__(self, client: requests.Session):
        self.client = client

    def fetch_date(self , start: int , end: int | None = None) -> list[Announcement] | TimeoutError:
        if end is None:
            end = start
        out: list[Announcement] = []
        seen: set[tuple[str, str, str, int]] = set()
        for kwargs in self.FETCH_KWARGS:
            try:
                announcements = self.fetch_announcements(start, end, **kwargs)
            except TimeoutError as e:
                Logger.error(f"{self.__class__.__name__} at {start}~{end} TimeoutError: {e}")
                return e
            for ann in announcements:
                if ann and ann.key not in seen:
                    out.append(ann)
                seen.add(ann.key)
        return out

    def request_with_timeouterror(self, func: Callable[..., requests.Response], *args, **kwargs) -> requests.Response:
        try:
            r = func(*args, **kwargs)
            r.raise_for_status()
        except TimeoutError:
            with temporary_timeout_expand(self.client):
                r = func(*args, **kwargs)
                r.raise_for_status()
        return r

    @abstractmethod
    def fetch_announcements(self, start : int, end: int) -> list[Announcement]:
        raise NotImplementedError

    @classmethod
    def exchange_fetcher(cls, exchange: str , client: requests.Session):
        if exchange.lower() == "sse":
            return SSEAnnFetcher(client)
        elif exchange.lower() == "szse":
            return SZSEAnnFetcher(client)
        elif exchange.lower() == "bse":
            return BSEAnnFetcher(client)
        else:
            raise ValueError(f"Invalid exchange: {exchange}")

class SSEAnnFetcher(AnnoucementFetcher):
    exchange = "sse"
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
            r = self.request_with_timeouterror(self.client.get, self.JSONP_URL, params=params, headers={"Referer": self.REFERER})
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
    exchange = "szse"
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
            r = self.request_with_timeouterror(self.client.post, url, json=body, headers=headers)
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
    exchange = "bse"
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
            r = self.request_with_timeouterror(self.client.post, self.ANNOUNCE_URL, data=body, headers=headers)
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
