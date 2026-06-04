"""
HTTP fetch tasks for exchange disclosure announcements.

Defines ``FetcherTask`` (abstract base) and concrete subclasses for the
Shanghai (SSE) and Shenzhen (SZSE) stock exchanges.  Each task builds the
HTTP request URL and parses the response JSON into a list of announcement
metadata records.
"""
from __future__ import annotations

from functools import cached_property
from typing import Literal

from src.proj import BaseClass
from src.proj.util.web.proxy import ProxyAPI , ProxyCaller
from src.proj.util.functional.handler import ErrorHandler

from .sync_fetcher import AnnoucementFetcher
from .async_fetcher import AsyncAnnoucementFetcher
from .util import Announcement , range_dates , AnnouncementExporter , CrawlerLogger
from . import const

class FetcherTask(BaseClass.BoundLogger):
    def __init__(self, exchange: const.ExchangeType, start: int, end: int, redownload: bool = False, * , indent: int = 1 , vb_level: int = 2, **kwargs):
        super().__init__(vb_level=vb_level, indent=indent, **kwargs)
        self.exchange : const.ExchangeType = exchange
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
        return const.EXCHANGE_URLS[self.exchange]

    @property
    def title(self) -> str:
        return f"{self.exchange} {self.start}~{self.end}"

    @property
    def should_be_skipped(self) -> bool:
        return self.exporter.should_skip_download(self.start, self.end, redownload=self.redownload)

    @cached_property
    def proxy_pool(self):
        return ProxyAPI.get_proxy_pool(const.URL_KEYS)

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
            for exchange in const.EXCHANGES:
                task = FetcherTask(exchange, s, e , redownload)
                if not task.should_be_skipped or (i >= (len(ranges) - force_update)):
                    tasks.append(task)
        if (len(tasks) >= 100):
            cls.logger.error(f"Too many tasks in {start}~{end}, be cautious!")
        if tasks:
            min_date = min(task.start for task in tasks)
            max_date = max(task.end for task in tasks)
            cls.logger.stdout(f"Total Announcement Crawling Tasks: {len(tasks)} at {min_date}~{max_date} for {len(const.EXCHANGES)} exchanges")
        return tasks
