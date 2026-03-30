import json
import re
import threading
import pandas as pd
from typing import Literal, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import TypeVar

from src.proj import CALENDAR , DB

BJTZ = ZoneInfo("Asia/Shanghai")

T = TypeVar("T")

def range_dates(start: int, end: int , step: int = 1) -> list[tuple[int , int]]:
    end = min(end, CALENDAR.update_to())
    d_start = start
    date_tuples: list[tuple[int , int]] = []
    while d_start <= end:
        d_end = CALENDAR.cd(d_start , step - 1)
        date_tuples.append((d_start , min(d_end , end)))
        d_start = CALENDAR.cd(d_end , 1)
    return date_tuples

def parse_jsonp(text: str) -> object:
    text = text.strip()
    m = re.match(r"^[a-zA-Z0-9_$]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
    if not m:
        raise ValueError("Response is not JSONP format")
    return json.loads(m.group(1))

@dataclass
class Announcement:
    exchange: Literal["SSE", "SZSE", "BSE"]
    sec_code: str
    sec_name: str
    source_id: str
    title: str
    category: str
    announce_date: int
    detail_path: str
    crawled_at: str
    secid: int
    date: int

    def __bool__(self) -> bool:
        return bool(self.sec_code)

    @property
    def key(self) -> tuple[str , str, str, int]:
        return (self.exchange, self.source_id, self.title, self.announce_date)
 
    @classmethod
    def from_sse(cls , raw : dict[str, Any]) -> 'Announcement':
        crawled_iso = datetime.now(tz=BJTZ).replace(microsecond=0).isoformat()
        pdf_base_url = "https://static.sse.com.cn/disclosure"
        rel_path = str(raw.get("URL") or "").strip()
        code = str(raw.get("SECURITY_CODE") or "").strip()
        name = str(raw.get("SECURITY_NAME") or "").strip()
        date = cls.to_date_int(str(raw.get("SSEDATE") or "").strip())
        source_id = str(raw.get("ORG_BULLETIN_ID") or raw.get("URL") or "").strip()
        title = str(raw.get("TITLE") or "").strip()
        category = str(raw.get("BULLETIN_TYPE_DESC") or "").strip()
        return cls(
            exchange="SSE",
            sec_code=f'{code:0>6s}.SH',
            sec_name=name,
            source_id=source_id,
            title=title,
            category=category,
            announce_date=date,
            detail_path=cls.to_absolute_url(rel_path , pdf_base_url),
            crawled_at=crawled_iso,
            secid= int(code),
            date=date,
        )

    @classmethod
    def from_szse(cls, raw: dict[str, Any]) -> 'Announcement':
        crawled_iso = datetime.now(tz=BJTZ).replace(microsecond=0).isoformat()
        pdf_base_url = "https://disc.static.szse.cn/download"
        codes = raw.get("secCode") or []
        names = raw.get("secName") or []
        code = str(codes[0] if isinstance(codes, list) and codes else "").strip()
        name = str(names[0] if isinstance(names, list) and names else "").strip()
        date = cls.to_date_int(str(raw.get("publishTime") or "").strip())
        title = str(raw.get("title") or "").strip()
        rel_path = str(raw.get("attachPath") or "").strip()
        source_id = str(raw.get("annId") or raw.get("id") or "").strip()
        return cls(
            exchange="SZSE",
            sec_code=f'{code:0>6s}.SZ' ,
            sec_name=name,
            title=title,
            category="",
            announce_date=date,
            detail_path=cls.to_absolute_url(rel_path , pdf_base_url),
            source_id=source_id,
            crawled_at=crawled_iso,
            secid = int(code),
            date=date,
        )

    @classmethod
    def from_bse(cls, raw: dict[str, Any]) -> 'Announcement':
        crawled_iso = datetime.now(tz=BJTZ).replace(microsecond=0).isoformat()
        pdf_base_url = "https://www.bse.cn"
        title = (str(raw.get("disclosureTitle") or "") + str(raw.get("disclosurePostTitle") or "")).strip()
        m = re.match(r"^\[([^\]]+)\]", title)
        if m:
            category = m.group(1).strip()
        else:
            category = str(raw.get("disclosureType") or "").strip()
        date = cls.to_date_int(str(raw.get("publishDate") or "").strip())
        rel_path = str(raw.get("destFilePath") or "").strip()
        code = str(raw.get("companyCd") or "").strip()
        name = str(raw.get("companyName") or "").strip()
        source_id = f'{code}|{date}|{title}'
        return cls(
            exchange="BSE",
            sec_code=f'{code:0>6s}.BJ' ,
            sec_name=name,
            title=title,
            category=category,
            announce_date=date,
            detail_path=cls.to_absolute_url(rel_path , pdf_base_url),
            source_id=source_id,
            crawled_at=crawled_iso,
            secid = int(code),
            date=date,
        )

    @classmethod
    def to_date_int(cls, value: str) -> int:
        """Normalize ``YYYY-MM-DD``、``YYYYMMDD`` or string with time to ``%Y%m%d``."""
        v = (value or "").strip()
        if not v:
            return -1
        if len(v) >= 10 and v[4] == "-" and v[7] == "-":
            return int(f"{v[:4]}{v[5:7]}{v[8:10]}")
        digits = "".join(c for c in v if c.isdigit())
        if len(digits) >= 8:
            return int(digits[:8])
        return -1

    @classmethod
    def to_absolute_url(cls, rel: str, base: str = '') -> str:
        """Concatenate the internal path returned by each exchange to a directly accessible https URL."""
        p = (rel or "").strip()
        if not p:
            return ""
        if p.startswith("http://") or p.startswith("https://"):
            return p
        if not base:
            return p
        if not p.startswith("/"):
            p = "/" + p
        return base + p

    @classmethod
    def df_columns(cls) -> list[str]:
        return [
            "exchange",
            "sec_code",
            "sec_name",
            "title",
            "category",
            "announce_date",
            "detail_path",
            "source_id",
            "crawled_at",
            "secid",
            "date",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "sec_code": self.sec_code,
            "sec_name": self.sec_name,
            "title": self.title,
            "category": self.category,
            "announce_date": self.announce_date,
            "detail_path": self.detail_path,
            "source_id": self.source_id,
            "crawled_at": self.crawled_at,
            "secid": self.secid,
            "date": self.date,
        }

class PathLock:
    locks_guard = threading.Lock()
    locks: dict[str, threading.Lock] = {}

    @classmethod
    def get(cls, path: Path) -> threading.Lock:
        key = str(path.resolve())
        with cls.locks_guard:
            if key not in cls.locks:
                cls.locks[key] = threading.Lock()
            return cls.locks[key]

class AnnouncementExporter:
    DEDUPE_SUBSET = ["exchange", "source_id", "announce_date", "title"]
    _instances : dict[str, 'AnnouncementExporter'] = {}

    def __new__(cls, exchange: str):
        assert exchange.lower() in ['sse' , 'szse' , 'bse'] , f'{exchange} is not in [sse, szse, bse]'
        if exchange.lower() not in cls._instances:
            cls._instances[exchange.lower()] = super().__new__(cls)
            cls._instances[exchange.lower()].exchange = exchange.lower()
        return cls._instances[exchange.lower()]

    @property
    def exchange(self) -> str:
        return self._exchange

    @exchange.setter
    def exchange(self, value: str):
        self._exchange = value

    def export_path(self, date: int) -> Path:
        """``{yyyymmdd}.{SSE|SZSE|BSE}.feather``"""
        return DB.path('crawler' , f'announcement_{self.exchange}' , date)

    def save_data(self , data : pd.DataFrame | list[Announcement] | Announcement, start: int, end: int) -> None:
        """Save single exchange, single day data, and merge duplicates with existing file."""
        if isinstance(data, Announcement):
            data = pd.DataFrame([data.to_dict()])
        elif isinstance(data, list):
            data = pd.DataFrame([ann.to_dict() for ann in data if ann])

        df = pd.DataFrame(data)
        if df.empty:
            df = pd.DataFrame(columns=Announcement.df_columns())
        ex_codes = df["exchange"].unique()
        if len(ex_codes) > 1:
            raise ValueError("save_exchange_day only accepts rows with a single exchange")
        for date in CALENDAR.range(start, end , 'cd'):
            df_date = df.query("date == @date").reset_index(drop=True)
            path = self.export_path(date)
            with PathLock.get(path):
                old = DB.load_df(path)
                if not old.empty:
                    df_date = pd.concat([old, df_date], ignore_index=True).drop_duplicates(
                        subset=self.DEDUPE_SUBSET, keep="last"
                    )
                DB.save_df(df_date, path , empty_ok = True)

    def should_skip_download(self, start: int, end: int, *, redownload: bool = False,) -> bool:
        """If file exists and date is earlier than "today" (Shanghai), skip; ``redownload`` is true never skip."""
        if redownload:
            return False
        if end >= CALENDAR.update_to():
            return False
        end = min(end, CALENDAR.update_to())
        dates = CALENDAR.range(start, end , 'cd')
        exists = [path.exists() for path in [self.export_path(date) for date in dates]]
        return all(exists)