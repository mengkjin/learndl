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

from src.proj import BaseClass
from src.proj.util.web import (
    http_session,
    CHROME_UA,
    request_with_timeouterror,
)
from .util import parse_jsonp , Announcement , sse_parse_groups , bse_query_body
from . import const

class AnnoucementFetcher(ABC, BaseClass.BoundLogger):
    exchange: const.ExchangeType
    FETCH_KWARGS : tuple[dict[str, Any],...] = ({},)

    def __init__(self, proxy: str | None = None , * , indent: int = 1 , vb_level: int = 2 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
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
        if exchange.lower() == const.sse:
            return SSEAnnFetcher(proxy)
        elif exchange.lower() == const.szse:
            return SZSEAnnFetcher(proxy)
        elif exchange.lower() == const.bse:
            return BSEAnnFetcher(proxy)
        else:
            raise ValueError(f"Invalid exchange: {exchange}")

class SSEAnnFetcher(AnnoucementFetcher):
    exchange = const.sse
    REFERER = const.SSE_REFERER
    JSONP_URL = const.SSE_JSONP_URL
    FETCH_KWARGS = const.SSE_FETCH_KWARGS

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
            r = request_with_timeouterror(
                self.session ,'get' , self.JSONP_URL, 
                params=params, headers={"Referer": self.REFERER} , expansion=1, 
                title=title , indent = self.indent + 1)
            payload = parse_jsonp(r.text)
            if not isinstance(payload, dict):
                break
            ph = payload.get("pageHelp") or {}
            data = ph.get("data") or []
            rows = sse_parse_groups(data)
            all_rows.extend(rows)
            page_count = int(ph.get("pageCount") or 0)
            if page_no >= page_count or page_count == 0:
                break
            page_no += 1

        return all_rows

class SZSEAnnFetcher(AnnoucementFetcher):
    exchange = const.szse
    REFERER = const.SZSE_REFERER
    ANN_LIST = const.SZSE_ANN_LIST

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
            r = request_with_timeouterror(
                self.session , 'post' , url, 
                json=body, headers=headers , 
                expansion=1, title=title , indent = self.indent + 1)
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
    exchange = const.bse
    REFERER = const.BSE_REFERER
    ANNOUNCE_URL = const.BSE_ANNOUNCE_URL

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
            body = bse_query_body(page, start, end)
            r = request_with_timeouterror(
                self.session , 'post' , self.ANNOUNCE_URL, 
                data=body, headers=headers , expansion=1, 
                title=title , indent = self.indent + 1)
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