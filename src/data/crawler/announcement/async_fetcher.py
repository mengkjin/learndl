"""
HTTP fetch tasks for exchange disclosure announcements.

Defines ``FetcherTask`` (abstract base) and concrete subclasses for the
Shanghai (SSE) and Shenzhen (SZSE) stock exchanges.  Each task builds the
HTTP request URL and parses the response JSON into a list of announcement
metadata records.
"""
from __future__ import annotations

import random
from curl_cffi import requests
from datetime import datetime
from functools import cached_property
from typing import Any
from abc import ABC, abstractmethod

from src.proj import Base
from src.proj.util.web.request import (
    async_http_session,
    CHROME_UA,
    request_with_timeouterror_async,
)
from . import const
from .util import Announcement , parse_jsonp , sse_parse_groups , bse_query_body

class AsyncAnnoucementFetcher(ABC, Base.BoundLogger):
    exchange : const.ExchangeType
    FETCH_KWARGS : tuple[dict[str, Any],...] = ({},)

    def __init__(self, proxy: str | None = None , * , indent: int = 1 , vb_level: int = 2 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
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
        if exchange.lower() == const.sse:
            return AsyncSSEAnnFetcher(proxy)
        elif exchange.lower() == const.szse:
            return AsyncSZSEAnnFetcher(proxy)
        elif exchange.lower() == const.bse:
            return AsyncBSEAnnFetcher(proxy)
        raise ValueError(f"Invalid exchange: {exchange}")

class AsyncSSEAnnFetcher(AsyncAnnoucementFetcher):
    exchange = const.sse
    REFERER = const.SSE_REFERER
    JSONP_URL = const.SSE_JSONP_URL
    FETCH_KWARGS: tuple[dict[str, Any],...] = const.SSE_FETCH_KWARGS

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
                headers={"Referer": self.REFERER}, expansion=1, 
                title=title, indent = self.indent + 1, vb_level = self.vb_level + 1
            )
            payload = parse_jsonp(r.text)
            if not isinstance(payload, dict):
                break
            ph = payload.get("pageHelp") or {}
            data = ph.get("data") or []
            all_rows.extend(sse_parse_groups(data))
            page_count = int(ph.get("pageCount") or 0)
            if page_no >= page_count or page_count == 0:
                break
            page_no += 1
        return all_rows

class AsyncSZSEAnnFetcher(AsyncAnnoucementFetcher):
    exchange = const.szse
    REFERER = const.SZSE_REFERER
    ANN_LIST = const.SZSE_ANN_LIST

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
                headers=headers, expansion=1, 
                title=title, indent = self.indent + 1, vb_level = self.vb_level + 1
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
    exchange = const.bse
    REFERER = const.BSE_REFERER
    ANNOUNCE_URL = const.BSE_ANNOUNCE_URL

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
            body = bse_query_body(page, start, end)
            r = await request_with_timeouterror_async(
                self.session, 'post', self.ANNOUNCE_URL, data=body,
                headers=headers, expansion=1, 
                title=title, indent = self.indent + 1, vb_level = self.vb_level + 1
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
