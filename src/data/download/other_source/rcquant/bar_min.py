"""
RiceQuant (rqdatac) minute-bar downloader.

Downloads 1-minute OHLCV bars for equities (CS), ETFs, futures, and convertible
bonds via the ``rqdatac`` Python API.

Daily update swallows quota exhaustion (warning only) so later pipeline steps
keep running. Historical sec backfill (``backfill_sec_min``) stops immediately
on quota so leftover quota is not wasted.

Start date behaviour (daily update):
- Equity bars: enabled from 2024-11-01 on non-HFM machines; disabled on HFM machines.
- ETF/future/CB bars: enabled from 2023-06-01 on updatable machines; disabled elsewhere.

Historical sec backfill (``backfill_sec_min``) walks from the day before 20241101
back to 20110101, independent of ``src_start_date``.
"""
from __future__ import annotations
import rqdatac
import pandas as pd
import numpy as np
import warnings

from datetime import datetime
from typing import Any , Literal , TypeAlias
from collections.abc import Sequence

from src.proj import MACHINE , CALENDAR , Dates , DB , Base , Save , Load , Logger
from src.proj.util.catcher import IOCatcher
from src.data.util import secid_adjust , trade_min_reform

from .initializer import RQInitializer , MinDataType , RQ_PATH , QuotaExceeded , is_quota_exceeded

__all__ = ['RcquantMinBarDownloader']

RcquantFileType : TypeAlias = Literal['secdf' , 'min']
SEC_BACKFILL_FLOOR = 20110101

def src_start_date(data_type : MinDataType) -> int:
    never = 20401231
    if data_type == 'sec':
        return never if MACHINE.belong_to_hfm else 20241101
    elif not MACHINE.updatable:
        return never
    else:
        assert data_type in ['etf' , 'fut' , 'cb'] , f'unsupported data type: {data_type}'
        return 20230601

def src_key(data_type : MinDataType , x_min : int = 1) -> str:
    if data_type == 'sec':
        prefix = ''
    else:
        prefix = f'{data_type}_'
    if x_min == 1:
        return f'{prefix}min'
    else:
        return f'{prefix}{x_min}min'

def load_list(date : int , data_type : MinDataType) -> pd.DataFrame | None:
    path = RQ_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    if path.exists(): 
        return Load.df(path)
    return None

def write_list(df : pd.DataFrame , date : int , data_type : MinDataType) -> None:
    path = RQ_PATH.joinpath(f'{data_type}list').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    Save.df(df , path , vb_level = 'max' , prefix = f'RcQuant {data_type} list {date}')

def load_min(date : int , data_type : MinDataType) -> pd.DataFrame | None:
    path = RQ_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    if path.exists(): 
        return Load.df(path)
    return None

def write_min(df : pd.DataFrame , date : int , data_type : MinDataType) -> None:
    path = RQ_PATH.joinpath(f'{data_type}min').joinpath(f'{date}.feather')
    path.parent.mkdir(exist_ok=True , parents=True)
    Save.df(df , path , vb_level = 'max' , prefix = f'RcQuant {data_type} min {date}')

def rcquant_past_dates(data_type : MinDataType , file_type : RcquantFileType) -> Dates:
    path = RQ_PATH.joinpath(f'{data_type}min') if file_type == 'min' else RQ_PATH.joinpath(f'{data_type}df')
    past_files = [p for p in path.iterdir()]
    past_dates = sorted([int(p.name.split('.')[-2][-8:]) for p in past_files])
    return Dates(past_dates)
    
def stored_dates(data_type : MinDataType , x_min : int = 1) -> Dates:
    assert x_min in [1 , 5 , 10 , 15 , 30 , 60] , f'only support 1min , 5min , 10min , 15min , 30min , 60min : {x_min}'
    if x_min != 1:
        assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
    return DB.dates('trade_ts' , src_key(data_type , x_min) , use_alt = False)

def last_date(data_type : MinDataType , offset : int = 0 , x_min : int = 1) -> int:
    dates = stored_dates(data_type , x_min)
    last_dt = max(dates) if len(dates) > 0 else 19970101
    return CALENDAR.cd(last_dt , offset)

def target_dates(
    data_type : MinDataType , start : int , end : int | None = None , * , overwrite : bool = False
) -> Dates:
    start = max(start , src_start_date(data_type))
    start , end = CALENDAR.update_schedule(start , end , key = 'rcquant_min')
    dates = Dates(start , end)
    if not overwrite:
        dates = dates.diff(stored_dates(data_type , 1))
    return dates

def x_mins_target_dates(
    data_type : MinDataType , start : int , end : int | None = None , * , overwrite : bool = False
) -> Dates:
    dates = Dates()
    if data_type != 'sec': 
        return dates
    end = CALENDAR.update_to(key = 'rcquant_min') if end is None else end
    for x_min in [5 , 10 , 15 , 30 , 60]:
        source_dates = DB.dates('trade_ts' , src_key(data_type , 1))
        stored = DB.dates('trade_ts' , src_key(data_type , x_min))
        sliced = source_dates.slice(min(start , src_start_date(data_type)) , end)
        target = sliced if overwrite else sliced.diff(stored)
        dates += target
    return dates

def x_mins_to_update(date : int , data_type : MinDataType) -> list[int]:
    if data_type != 'sec': 
        return []
    x_mins : list[int]= []
    for x_min in [5 , 10 , 15 , 30 , 60]:
        path = DB.path('trade_ts' , src_key(data_type , x_min) , date)
        if not path.exists(): 
            x_mins.append(x_min)
    return x_mins

def backfill_sec_dates() -> Dates:
    """Trading days in [20110101, last td before daily src_start) missing from trade_ts/min."""
    ceiling = src_start_date(MinDataType.SEC)
    if ceiling >= 20400101:
        return Dates()
    upper = int(CALENDAR.td(ceiling , -1))
    if SEC_BACKFILL_FLOOR > upper:
        return Dates()
    return Dates(SEC_BACKFILL_FLOOR , upper).diff(stored_dates(MinDataType.SEC , 1))

def _as_yyyymmdd(complete_time : Any) -> int | None:
    if complete_time is None:
        return None
    if isinstance(complete_time , datetime):
        return int(complete_time.strftime('%Y%m%d'))
    digits = str(complete_time).strip()[:10].replace('-' , '')
    if len(digits) >= 8 and digits[:8].isdigit():
        return int(digits[:8])
    return None

def daily_update_succeeded_today() -> bool:
    """True if autorun daily_update marked success with complete_time on calendar today (BJ)."""
    from src.proj.util.script.task_record import TaskRecorder
    recorder = TaskRecorder('autorun' , 'daily_update')
    today = CALENDAR.today()
    for row in recorder.get_finished_tasks():
        success , complete_time = row[1] , row[2]
        if not success:
            continue
        if _as_yyyymmdd(complete_time) == today:
            return True
    return False

def _backfill_deadline_hit(started_at : datetime) -> bool:
    """Stop starting new dates after midnight, or in the last minute of the 23:xx window."""
    now = CALENDAR.now(bj_tz = True)
    if now.date() > started_at.date():
        return True
    return now.hour == 23 and now.minute >= 59

def rq_get_price(code_list : np.ndarray , date : int , * , raise_on_quota : bool = False) -> Any:
    """``rqdatac.get_price`` for one session.

    Daily path (``raise_on_quota=False``): quota is a warning, return whatever
    payload we got (possibly ``None``). Backfill raises ``QuotaExceeded``.
    """
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always')
        with IOCatcher() as catcher:
            try:
                data = rqdatac.get_price(
                    code_list , start_date = str(date) , end_date = str(date) ,
                    frequency = '1m' , expect_df = True ,
                )
            except QuotaExceeded as e:
                if raise_on_quota:
                    raise
                Logger.warning(f'RcQuant get_price quota exceeded: {e}')
                return None
            except Exception as e:
                if is_quota_exceeded(e):
                    if raise_on_quota:
                        raise QuotaExceeded(str(e)) from e
                    Logger.warning(f'RcQuant get_price quota exceeded: {e}')
                    return None
                raise
        blob = '\n'.join(filter(None , (
            catcher.contents['stdout'] ,
            catcher.contents['stderr'] ,
            *(str(w.message) for w in caught) ,
        )))
        if stdout := catcher.contents['stdout']:
            Logger.stdout(stdout)
        if stderr := catcher.contents['stderr']:
            Logger.error(stderr)
        for w in caught:
            Logger.warning(str(w.message))
        if is_quota_exceeded(blob):
            if raise_on_quota:
                raise QuotaExceeded(blob)
            Logger.warning(f'RcQuant get_price quota exceeded: {blob}')
    return data

def rcquant_instrument_list(date : int , data_type : MinDataType) -> pd.DataFrame:
    secdf = load_list(date , data_type)
    if secdf is not None: 
        return secdf
    if not RQInitializer.init(): 
        return pd.DataFrame()
    secdf = rqdatac.all_instruments(type=data_type.instrument, date=str(date))
    secdf = secdf.rename(columns = {'order_book_id':'code'})
    if 'status' in secdf.columns:
        secdf['is_active'] = secdf['status'] == 'Active'
    else:
        secdf['is_active'] = True
    write_list(secdf , date , data_type)
    return secdf

def rcquant_trading_dates(start : int , end : int) -> Dates:
    if not RQInitializer.init(): 
        return Dates()
    return Dates([int(td.strftime('%Y%m%d')) for td in rqdatac.get_trading_dates(start, end, market='cn')])

class RcquantMinBarDownloader(Base.BasicUpdater):
    UPDATE_ALIAS = 'download'
    ACCEPTABLE_UPDATE_TYPES = (Base.UpdateType.UPDATE, )

    @classmethod
    def parse_update_input(cls , *args , **kwargs) -> dict[str , Any]:
        return super().parse_update_input(*args , key='rcquant_min' , **kwargs)

    @classmethod
    def proceed_update(
        cls , start : int , end : int , * ,
        first_n : int = -1 ,
        data_types : Sequence[MinDataType | str] | None = None ,
        overwrite : bool = False ,
        **kwargs
    ) -> Base.UpdateFlag:
        updater = cls(indent = cls.logger.indent + 1 , vb_level = cls.logger.vb_level + 1)
        return updater.download_since_last_data(
            start = start , end = end , first_n = first_n ,
            data_types = data_types , overwrite = overwrite ,
        )

    def download_since_last_data(
        self , start : int , end : int , first_n : int = -1 ,
        data_types : Sequence[MinDataType | str] | None = None ,
        overwrite : bool = False ,
    ) -> Base.UpdateFlag:
        flags = Base.UpdateFlagList()
        selected = list(MinDataType) if data_types is None else [MinDataType(dt) for dt in data_types]
        for data_type in selected:
            try:
                flags += self.download(start , end , data_type , first_n , overwrite = overwrite)
            except Exception as e:
                self.logger.error(f'RcQuant {data_type} minbar failed: {e}')
                flags += Base.UpdateFlag.FAILED
                continue
        return flags.summarize()

    def download(
        self , start : int , end : int , data_type : MinDataType | None = None ,
        first_n : int = -1 , * , overwrite : bool = False ,
    ) -> Base.UpdateFlag:
        assert data_type is not None , f'data_type is required'
        flags = Base.UpdateFlagList()
        dates = target_dates(data_type , start , end , overwrite = overwrite)
        if dates.empty: 
            self.logger.skipping(f'RcQuant {data_type} bar min is up to date')
            flags += Base.UpdateFlag.SKIPPED
        else:
            marks : list[bool] = []
            for dt in dates:
                mark = self.rcquant_bar_min(dt , data_type , first_n)
                if not mark: 
                    self.logger.alert1(f'Download RcQuant {data_type} bar min {dt} failed')
                marks.append(mark)
            if all(marks):
                self.logger.success(f'Download RcQuant {data_type} bar min at {dates}')
                flags += Base.UpdateFlag.SUCCESS
            else:
                flags += Base.UpdateFlag.FAILED

        dates = x_mins_target_dates(data_type , start , end , overwrite = overwrite)
        if dates.empty: 
            flags += Base.UpdateFlag.SKIPPED
        else:
            for dt in dates:
                for x_min in x_mins_to_update(dt , data_type = data_type):
                    min_df = DB.load('trade_ts' , src_key(data_type) , dt)
                    assert data_type == 'sec' , f'only sec support {x_min}min : {data_type}'
                    x_min_df = trade_min_reform(min_df , x_min , 1)
                    DB.save(x_min_df , 'trade_ts' , src_key(data_type , x_min) , dt , indent = self.indent + 1 , vb_level = self.vb_level + 1)
            self.logger.success(f'Transform RcQuant {data_type} X-min bars at {dates}')
            flags += Base.UpdateFlag.SUCCESS
        return flags.summarize()

    def rcquant_bar_min(
        self , date : int , data_type : MinDataType , first_n : int = -1 ,
        * , raise_on_quota : bool = False ,
    ) -> bool:    
        def code_map(x : str):
            if data_type != 'sec': 
                return x
            x = x.split('.')[0]
            if x[:1] in ['3', '0']:
                y = x+'.SZ'
            elif x[:1] in ['6']:
                y = x+'.SH'
            else:
                y = x
            return y

        if (sec_min := load_min(date , data_type)) is not None: 
            df = self.rcquant_min_to_normal_min(sec_min , data_type)
            DB.save(df , 'trade_ts' , src_key(data_type) , date = date , indent = self.indent + 1, vb_level = self.vb_level + 1)
            return True

        if not RQInitializer.init(raise_on_quota = raise_on_quota): 
            return False

        instrument_list = rcquant_instrument_list(date , data_type = data_type)
        instrument_list = instrument_list.loc[instrument_list['is_active']]
        if first_n > 0: 
            instrument_list = instrument_list.iloc[:first_n]
        code_list = instrument_list['code'].to_numpy(str)
        data = rq_get_price(code_list , date , raise_on_quota = raise_on_quota)
        if isinstance(data , pd.DataFrame) and not data.empty:
            data = data.reset_index().rename(columns = {'total_turnover':'amount', 'order_book_id':'code'}).assign(date = date)
            data['code'] = data['code'].map(code_map)
            data['time'] = data['datetime'].map(lambda x: getattr(x, 'strftime')('%H%M%S')).str.slice(0,4)
            data['date'] = data['datetime'].map(lambda x: getattr(x, 'strftime')('%Y%m%d'))

            write_min(data , date , data_type)

            df = self.rcquant_min_to_normal_min(data , data_type)
            DB.save(df , 'trade_ts' , src_key(data_type) , date = date , indent = self.indent + 1 , vb_level = self.vb_level + 1)
            return True
        else:
            return False

    def rcquant_min_to_normal_min(self , df : pd.DataFrame , data_type : MinDataType) -> pd.DataFrame:
        if data_type != 'sec': 
            return df
        df = df.copy()
        df.loc[:,['open','high','low','close','volume','amount']] = df.loc[:,['open','high','low','close','volume','amount']].astype(float)
        df = secid_adjust(df , ['code'] , drop_old=True)
        df['minute'] = ((df['time'].str.slice(0,2).astype(int) - 9.5) * 60 + df['time'].str.slice(2,4).astype(int)).astype(int) - 1
        df.loc[df['minute'] >= 120 , 'minute'] -= 90
        df['vwap'] = df['amount'] / df['volume'].where(df['volume'] > 0 , np.nan)
        df['vwap'] = df['vwap'].where(df['vwap'].notna() , df['open'])
        df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap','num_trades']].sort_values(['secid','minute']).reset_index(drop = True)
        return df

    @classmethod
    def backfill_sec_min(cls , * , force : bool = False , first_n : int = -1 , **kwargs) -> Base.UpdateFlag:
        """Fill missing equity 1-min bars from the day before daily start back to 20110101.

        Runs newest-missing first. Stops immediately on quota exhaustion or at Beijing
        midnight (no new date after 23:59). ``force`` skips the 23:xx window and the
        daily_update gate; quota and midnight still apply.
        """
        updater = cls(indent = cls.logger.indent + 1 , vb_level = cls.logger.vb_level + 1)
        return updater._backfill_sec_min(force = force , first_n = first_n)

    def _backfill_sec_min(self , * , force : bool , first_n : int) -> Base.UpdateFlag:
        started_at = CALENDAR.now(bj_tz = True)
        data_type = MinDataType.SEC
        filled : list[int] = []

        def conclude_filled(extra : str | None = None , level : str = 'info') -> None:
            if filled:
                dates_msg = f'RcQuant sec min updated dates ({len(filled)}): {",".join(str(d) for d in filled)}'
            else:
                dates_msg = 'RcQuant sec min updated dates: none'
            msg = f'{extra} | {dates_msg}' if extra else dates_msg
            self.logger.conclude(msg , level = level)

        if not MACHINE.updatable or MACHINE.belong_to_hfm:
            msg = f'RcQuant sec backfill skipped: machine {MACHINE.name} is not eligible'
            self.logger.skipping(msg)
            conclude_filled(msg)
            return Base.UpdateFlag.SKIPPED

        if not force and started_at.hour <= 21:
            msg = f'RcQuant sec backfill skipped: BJ hour is {started_at.hour}, window is 21:00-23:59'
            self.logger.skipping(msg)
            conclude_filled(msg)
            return Base.UpdateFlag.SKIPPED

        if not force and not daily_update_succeeded_today():
            msg = 'RcQuant sec backfill skipped: daily_update has not succeeded today'
            self.logger.skipping(msg)
            conclude_filled(msg)
            return Base.UpdateFlag.SKIPPED

        dates = backfill_sec_dates()
        if dates.empty:
            msg = f'RcQuant sec backfill is complete through {SEC_BACKFILL_FLOOR}'
            self.logger.skipping(msg)
            conclude_filled(msg)
            return Base.UpdateFlag.SKIPPED

        self.logger.info(f'RcQuant sec backfill {len(dates)} missing dates, newest first, floor={SEC_BACKFILL_FLOOR}')
        n_fail = 0
        for dt in reversed(dates):
            if _backfill_deadline_hit(started_at):
                msg = f'RcQuant sec backfill hit midnight deadline after {len(filled)} dates'
                self.logger.alert1(msg)
                conclude_filled(msg , level = 'warning')
                return Base.UpdateFlag.SUCCESS if filled else Base.UpdateFlag.SKIPPED
            try:
                mark = self.rcquant_bar_min(dt , data_type , first_n , raise_on_quota = True)
            except QuotaExceeded as e:
                msg = f'RcQuant quota exceeded after {len(filled)} dates at {dt}: {e}'
                self.logger.alert1(msg)
                conclude_filled(msg , level = 'warning')
                return Base.UpdateFlag.SUCCESS if filled else Base.UpdateFlag.SKIPPED
            if not mark:
                self.logger.alert1(f'Backfill RcQuant sec bar min {dt} failed')
                n_fail += 1
                continue
            for x_min in x_mins_to_update(dt , data_type = data_type):
                min_df = DB.load('trade_ts' , src_key(data_type) , dt)
                x_min_df = trade_min_reform(min_df , x_min , 1)
                DB.save(
                    x_min_df , 'trade_ts' , src_key(data_type , x_min) , dt ,
                    indent = self.indent + 1 , vb_level = self.vb_level + 1 ,
                )
            filled.append(dt)
            self.logger.success(
                f'Backfill RcQuant sec bar min {dt} ({len(filled)} done, {len(dates) - len(filled) - n_fail} remaining)'
            )

        if n_fail:
            msg = f'RcQuant sec backfill finished with {len(filled)} ok, {n_fail} failed'
            conclude_filled(msg , level = 'warning')
            return Base.UpdateFlag.FAILED
        msg = f'RcQuant sec backfill filled {len(filled)} dates through {SEC_BACKFILL_FLOOR}'
        self.logger.success(msg)
        conclude_filled(msg)
        return Base.UpdateFlag.SUCCESS