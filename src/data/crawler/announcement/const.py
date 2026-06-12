"""
Constants for the announcement crawler
"""

from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo
from src.proj import Base

__all__ = [
    'ExchangeType' , 'BJTZ' , 'SSE_REFERER' , 
    'SSE_JSONP_URL' , 'SSE_FETCH_KWARGS' , 'SZSE_REFERER' , 
    'SZSE_ANN_LIST' , 'BSE_REFERER' , 'BSE_ANNOUNCE_URL']

class ExchangeType(Base.StrEnum):
    SSE = "sse"
    SZSE = "szse"
    BSE = "bse"

    @classmethod
    def all_urls(cls) -> tuple[str,...]:
        return tuple(exchange.exchange_url for exchange in cls)

    @property
    def exchange_url(self) -> str:
        return {
            ExchangeType.SSE: "https://www.sse.com.cn",
            ExchangeType.SZSE: "https://www.szse.cn",
            ExchangeType.BSE: "https://www.bse.cn",
        }[self]

BJTZ = ZoneInfo("Asia/Shanghai")

SSE_REFERER = "https://www.sse.com.cn/"
SSE_JSONP_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
SSE_FETCH_KWARGS : tuple[dict[str, Any],...] = ({"stock_type": "1"}, {"stock_type": "8"})

SZSE_REFERER = "https://www.szse.cn/disclosure/listed/notice/index.html"
SZSE_ANN_LIST = "https://www.szse.cn/api/disc/announcement/annList"

BSE_REFERER = "https://www.bse.cn/disclosure/announcement.html"
BSE_ANNOUNCE_URL = "https://www.bse.cn/disclosureInfoController/companyAnnouncement.do"
