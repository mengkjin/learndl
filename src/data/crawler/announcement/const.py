from __future__ import annotations
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

class ExchangeType(StrEnum):
    SSE = "sse"
    SZSE = "szse"
    BSE = "bse"

    @classmethod
    def values(cls) -> tuple[ExchangeType,...]:
        return tuple(cls)

    @classmethod
    def all_urls(cls) -> tuple[str,...]:
        return (
            ExchangeType.SSE.exchange_url(),
            ExchangeType.SZSE.exchange_url(),
            ExchangeType.BSE.exchange_url(),
        )

    @classmethod
    def str_values(cls) -> tuple[str,...]:
        return tuple(cls.value for cls in cls)

    def exchange_url(self) -> str:
        if self == ExchangeType.SSE:
            return "https://www.sse.com.cn"
        elif self == ExchangeType.SZSE:
            return "https://www.szse.cn"
        elif self == ExchangeType.BSE:
            return "https://www.bse.cn"
        else:
            raise ValueError(f"Invalid exchange type: {self}")

BJTZ = ZoneInfo("Asia/Shanghai")

SSE_REFERER = "https://www.sse.com.cn/"
SSE_JSONP_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
SSE_FETCH_KWARGS : tuple[dict[str, Any],...] = ({"stock_type": "1"}, {"stock_type": "8"})

SZSE_REFERER = "https://www.szse.cn/disclosure/listed/notice/index.html"
SZSE_ANN_LIST = "https://www.szse.cn/api/disc/announcement/annList"

BSE_REFERER = "https://www.bse.cn/disclosure/announcement.html"
BSE_ANNOUNCE_URL = "https://www.bse.cn/disclosureInfoController/companyAnnouncement.do"
