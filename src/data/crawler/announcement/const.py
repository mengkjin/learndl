from __future__ import annotations
from typing import Any, Literal , TypeAlias
from zoneinfo import ZoneInfo

ExchangeType : TypeAlias = Literal["sse", "szse", "bse"]

bse : ExchangeType = "bse"
szse : ExchangeType = "szse"
sse : ExchangeType = "sse"

BJTZ = ZoneInfo("Asia/Shanghai")
EXCHANGES : tuple[ExchangeType,...] = (sse, szse, bse)
EXCHANGE_URLS : dict[ExchangeType, str] = {
    sse: "https://www.sse.com.cn", 
    szse: "https://www.szse.cn", 
    bse: "https://www.bse.cn"
}
URL_KEYS : tuple[str,...] = tuple(sorted(EXCHANGE_URLS.values()))

SSE_REFERER = "https://www.sse.com.cn/"
SSE_JSONP_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletinNew.do"
SSE_FETCH_KWARGS : tuple[dict[str, Any],...] = ({"stock_type": "1"}, {"stock_type": "8"})

SZSE_REFERER = "https://www.szse.cn/disclosure/listed/notice/index.html"
SZSE_ANN_LIST = "https://www.szse.cn/api/disc/announcement/annList"

BSE_REFERER = "https://www.bse.cn/disclosure/announcement.html"
BSE_ANNOUNCE_URL = "https://www.bse.cn/disclosureInfoController/companyAnnouncement.do"
