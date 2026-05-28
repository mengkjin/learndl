"""
HTTP fetch tasks for exchange disclosure announcements.

Defines ``FetcherTask`` (abstract base) and concrete subclasses for the
Shanghai (SSE) and Shenzhen (SZSE) stock exchanges.  Each task builds the
HTTP request URL and parses the response JSON into a list of announcement
metadata records.
"""
from __future__ import annotations

import asyncio
import random
from curl_cffi import requests
from datetime import datetime
from functools import cached_property
from typing import Any , Literal
from urllib.parse import urlencode
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.proj import BaseClass
from src.proj.util import ProxyAPI
from src.proj.util.proxy import ProxyCaller
from src.proj.util.web import (
    http_session,
    async_http_session,
    CHROME_UA,
    request_with_timeouterror,
    request_with_timeouterror_async,
)
from src.proj.util.error_handler import ErrorHandler
from .util import parse_jsonp , Announcement , range_dates , AnnouncementExporter , CrawlerLogger

EXCHANGES : list[str] = ['sse', 'szse', 'bse']
EXCHANGE_URLS = {
    "sse": "https://www.sse.com.cn", 
    "szse": "https://www.szse.cn", 
    "bse": "https://www.bse.cn"
}
URL_KEYS = tuple(sorted(EXCHANGE_URLS.values()))

class FetchCancelledError(asyncio.CancelledError):
    """Raised when a task is cancelled after winner result is confirmed."""

@dataclass
class FetchedAnnouncementBatch:
    task_key: str
    exchange: str
    start: int
    end: int
    proxy: str | None
    payload: list[Announcement] = field(default_factory=list)
    ok: bool = False
    error: Exception | None = None

class FetcherTask(BaseClass.BoundLogger):
    def __init__(self, exchange: str, start: int, end: int, redownload: bool = False, * , vb_level: int = 2, indent: int = 1):
        self.set_vb(vb_level , indent)
        self.exchange = exchange
        self.start = start
        self.end = end
        self.redownload = redownload

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exchange={self.exchange}, start={self.start}, end={self.end}, redownload={self.redownload})"

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

    @cached_property
    def proxy_pool(self):
        return ProxyAPI.get_proxy_pool(URL_KEYS)

    def fetch_date(self, proxy: str | None) -> list[Announcement] | Exception:
        fetcher = AnnoucementFetcher.exchange_fetcher(self.exchange, proxy)
        return fetcher.fetch_date(self.start, self.end , title=f'[fetch date] {self.title} proxy={proxy}')

    async def fetch_date_async(self, proxy: str | None) -> list[Announcement] | Exception:
        fetcher = AsyncAnnoucementFetcher.exchange_fetcher(self.exchange, proxy)
        return await fetcher.fetch_date(self.start, self.end, title=f'[fetch date async] {self.title} proxy={proxy}')

    def fetch_date_with_error_handler(self, proxy: str | None):
        fetch_date = ErrorHandler(
            self.fetch_date, handle_types = ['http', 'all'],
            label=f"{self.title}" + (f"[{proxy}]" if proxy else ""))
        return fetch_date(proxy)

    def fetch_payload(self, proxy: str | None) -> list[Announcement] | Exception:
        result = self.fetch_date_with_error_handler(proxy)
        return result.unwrap(error='return')

    async def fetch_payload_async(self, proxy: str | None, *, attempt_id: str | None = None) -> list[Announcement] | Exception:
        try:
            CrawlerLogger.stdout(f"[async-fetch] start {self.title} proxy={proxy} attempt={attempt_id}")
            payload = await self.fetch_date_async(proxy)
            CrawlerLogger.stdout(f"[async-fetch] done {self.title} proxy={proxy} attempt={attempt_id} rows={len(payload) if isinstance(payload, list) else 'NA'}")
            return payload
        except Exception as e:
            CrawlerLogger.alert(f"[async-fetch] failed {self.title} proxy={proxy} attempt={attempt_id} error={e!s}")
            return e

    def persist_payload(self, payload: list[Announcement]) -> bool:
        self.exporter.save_data(payload, self.start, self.end)
        return True

    def crawl_and_persist(self, proxy: str | None) -> bool | Exception:
        if self.should_be_skipped:
            return True
        self.logger.stdout(f"Crawling {self.title} with proxy {proxy}")
        value = self.fetch_payload(proxy)
        if isinstance(value, Exception):
            return value
        if isinstance(value, list):
            self.persist_payload(value)
            return True
        return False

    def crawl(self , proxy: str | None) -> bool | Exception:
        return self.crawl_and_persist(proxy)

    async def crawl_async(self, proxy: str | None) -> bool | Exception:
        if self.should_be_skipped:
            return True
        self.logger.stdout(f"Crawling {self.title} with proxy {proxy}")
        value = await self.fetch_payload_async(proxy)
        if isinstance(value, Exception):
            return value
        if isinstance(value, list):
            self.persist_payload(value)
            return True
        return False

    def to_proxy_caller(self , pool) -> ProxyCaller:
        return ProxyCaller(self.crawl, self.url, pool = pool)

    def run(self, pool = None, *, max_proxies_try: int = 3, error : Literal['raise' , 'return'] = 'return') -> bool | Exception:
        """
        Fetch by natural day and exchange; try public proxy list when direct connection (and optional fixed proxy) 
        still fails and ``auto_discover_proxy`` is enabled.
        """
        result = False
        if self.should_be_skipped:
            self.logger.skipping(f"{self.title}, already have historical data (use redownload to force re-download)" , vb = 1)
            return result

        if pool is None:
            pool = ProxyAPI.get_proxy_pool(self.url)
            
        if pool.proxies[self.url].valid_count > 0:
            for _ in range(max_proxies_try):
                proxy = pool.acquire(url=self.url)
                if proxy is None:
                    break
                result = self.crawl(proxy.url)
                pool.release(proxy, result is not None)
                if result is not None:
                    break
        if result is None:
            result = self.crawl(None)

        if error == 'raise' and isinstance(result, Exception):
            raise result
        return result

    @classmethod
    def tasks_flat(cls , start: int, end: int, step: int = 1, redownload: bool = False, force_update: int = 0) -> list[FetcherTask]:
        tasks : list[FetcherTask] = []
        ranges = range_dates(start, end, step)
        for i , (s, e) in enumerate(ranges):
            for exchange in EXCHANGES:
                task = FetcherTask(exchange, s, e , redownload)
                if not task.should_be_skipped or (i >= (len(ranges) - force_update)):
                    tasks.append(task)
        if (len(tasks) >= 100):
            cls.logger.error(f"Too many tasks in {start}~{end}, be cautious!")
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            cls.logger.stdout(f"Total Announcement Crawling Tasks: {len(tasks)} at {min_date}~{max_date} for {len(EXCHANGES)} exchanges")
        return tasks

class AnnoucementFetcher(ABC, BaseClass.BoundLogger):
    exchange: Literal["sse", "szse", "bse"]
    FETCH_KWARGS : list[dict[str, Any]] = [{}]

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    def init_session(self) -> requests.Session:
        self.session = http_session(proxy=self.proxy , trust_env = self.proxy is None)
        return self.session

    @cached_property
    def session(self) -> requests.Session:
        return http_session(proxy=self.proxy , trust_env = self.proxy is None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.proxy})"

    def fetch_date(
        self , start: int , end: int | None = None , * , title : str = ''
    ) -> list[Announcement] | TimeoutError | requests.exceptions.Timeout:
        if end is None:
            end = start
        out: list[Announcement] = []
        seen: set[tuple[str, str, str, int]] = set()
        with self.init_session():
            for kwargs in self.FETCH_KWARGS:
                try:
                    announcements = self.fetch_announcements(start, end, **kwargs, title=title)
                except (TimeoutError , requests.exceptions.Timeout) as e:
                    self.logger.error(f"{self.__class__.__name__} at {start}~{end} TimeoutError: {e}")
                    return e
                for ann in announcements:
                    if ann and ann.key not in seen:
                        out.append(ann)
                    seen.add(ann.key)
        return out

    @abstractmethod
    def fetch_announcements(self, start : int, end: int, *, title : str = '') -> list[Announcement]:
        raise NotImplementedError

    @classmethod
    def exchange_fetcher(cls, exchange: str , proxy: str | None = None):
        if exchange.lower() == "sse":
            return SSEAnnFetcher(proxy)
        elif exchange.lower() == "szse":
            return SZSEAnnFetcher(proxy)
        elif exchange.lower() == "bse":
            return BSEAnnFetcher(proxy)
        else:
            raise ValueError(f"Invalid exchange: {exchange}")

class AsyncAnnoucementFetcher(ABC, BaseClass.BoundLogger):
    exchange: Literal["sse", "szse", "bse"]
    FETCH_KWARGS : list[dict[str, Any]] = [{}]

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    def init_session(self) -> requests.AsyncSession:
        self.session = async_http_session(proxy=self.proxy , trust_env=self.proxy is None)
        return self.session

    @cached_property
    def session(self) -> requests.AsyncSession:
        return async_http_session(proxy=self.proxy , trust_env=self.proxy is None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.proxy})"

    async def fetch_date(
        self, start: int, end: int | None = None, *, title: str = ''
    ) -> list[Announcement] | TimeoutError | requests.exceptions.Timeout:
        if end is None:
            end = start
        out: list[Announcement] = []
        seen: set[tuple[str, str, str, int]] = set()
        session = self.init_session()
        try:
            for kwargs in self.FETCH_KWARGS:
                announcements = await self.fetch_announcements(start, end, **kwargs, title=title)
                for ann in announcements:
                    if ann and ann.key not in seen:
                        out.append(ann)
                    seen.add(ann.key)
        finally:
            await session.close()
        return out

    @abstractmethod
    async def fetch_announcements(self, start: int, end: int, *, title: str = '') -> list[Announcement]:
        raise NotImplementedError

    @classmethod
    def exchange_fetcher(cls, exchange: str, proxy: str | None = None):
        if exchange.lower() == "sse":
            return AsyncSSEAnnFetcher(proxy)
        elif exchange.lower() == "szse":
            return AsyncSZSEAnnFetcher(proxy)
        elif exchange.lower() == "bse":
            return AsyncBSEAnnFetcher(proxy)
        raise ValueError(f"Invalid exchange: {exchange}")

class SSEAnnFetcher(AnnoucementFetcher):
    exchange = "sse"
    REFERER = "https://www.sse.com.cn/"
    JSONP_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
    FETCH_KWARGS : list[dict[str, Any]] = [{"stock_type": "1"}, {"stock_type": "8"}]

    def fetch_announcements(self, start : int, end: int, *, stock_type: str = "1", page_size: int = 50 , title : str = '') -> list[Announcement]:
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
            r = request_with_timeouterror(self.session ,'get' , self.JSONP_URL, params=params, headers={"Referer": self.REFERER} , expansion=1, title=title)
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

    def fetch_announcements(self, start : int, end: int, *, page_size: int = 50 , title : str = '') -> list[Announcement]:
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
            r = request_with_timeouterror(self.session , 'post' , url, json=body, headers=headers , expansion=1, title=title)
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

    def fetch_announcements(self, start : int, end: int, *, title : str = '') -> list[Announcement]:
        """Beijing Stock Exchange: POST companyAnnouncement.do (POST JSON, paging by page."""
        # warmup
        r = self.session.get(self.REFERER, headers={"User-Agent": CHROME_UA})
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
            r = request_with_timeouterror(self.session , 'post' , self.ANNOUNCE_URL, data=body, headers=headers , expansion=1, title=title)
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

class AsyncSSEAnnFetcher(AsyncAnnoucementFetcher):
    exchange = "sse"
    REFERER = SSEAnnFetcher.REFERER
    JSONP_URL = SSEAnnFetcher.JSONP_URL
    FETCH_KWARGS: list[dict[str, Any]] = SSEAnnFetcher.FETCH_KWARGS

    async def fetch_announcements(self, start: int, end: int, *, stock_type: str = "1", page_size: int = 50, title: str = '') -> list[Announcement]:
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
            r = await request_with_timeouterror_async(
                self.session, 'get', self.JSONP_URL, params=params,
                headers={"Referer": self.REFERER}, expansion=1, title=title
            )
            payload = parse_jsonp(r.text)
            if not isinstance(payload, dict):
                break
            ph = payload.get("pageHelp") or {}
            data = ph.get("data") or []
            all_rows.extend(SSEAnnFetcher._parse_groups(data))
            page_count = int(ph.get("pageCount") or 0)
            if page_no >= page_count or page_count == 0:
                break
            page_no += 1
        return all_rows

class AsyncSZSEAnnFetcher(AsyncAnnoucementFetcher):
    exchange = "szse"
    REFERER = SZSEAnnFetcher.REFERER
    ANN_LIST = SZSEAnnFetcher.ANN_LIST

    async def fetch_announcements(self, start: int, end: int, *, page_size: int = 50, title: str = '') -> list[Announcement]:
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
            r = await request_with_timeouterror_async(
                self.session, 'post', url, json=body,
                headers=headers, expansion=1, title=title
            )
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

class AsyncBSEAnnFetcher(AsyncAnnoucementFetcher):
    exchange = "bse"
    REFERER = BSEAnnFetcher.REFERER
    ANNOUNCE_URL = BSEAnnFetcher.ANNOUNCE_URL

    async def fetch_announcements(self, start: int, end: int, *, title: str = '') -> list[Announcement]:
        r = await self.session.get(self.REFERER, headers={"User-Agent": CHROME_UA})
        r.raise_for_status()
        headers = {
            "Referer": self.REFERER,
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": CHROME_UA,
        }
        all_rows: list[Announcement] = []
        page = 0
        while True:
            body = BSEAnnFetcher._query_body(page, start, end)
            r = await request_with_timeouterror_async(
                self.session, 'post', self.ANNOUNCE_URL, data=body,
                headers=headers, expansion=1, title=title
            )
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
