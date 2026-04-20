"""
Singleton providing access to static stock listing metadata, name-change history,
and Tushare industry classification for the Chinese A-share universe.

The ``INFO`` singleton is instantiated at import time and loaded from the
``information_ts`` database.  It is the authoritative source for:
- Listing / delisting dates (``description`` table)
- ST / suspended status (``change_name`` table)
- Tushare level-2 industry classification (``industry`` table)
"""
import numpy as np
import pandas as pd
from typing import Any

from src.proj import MACHINE , CALENDAR , TradeDate , DB , singleton

@singleton
class InfoDataAccess:
    """
    Date-aware interface to static Chinese A-share stock reference data.

    Loaded once at startup via the ``@singleton`` decorator; imported via the
    module-level ``INFO`` alias.

    Attributes
    ----------
    desc : pd.DataFrame
        Listing descriptions: ``secid``, ``list_dt``, ``delist_dt``,
        ``sec_name``, ``exchange_name``.
    cname : pd.DataFrame
        Name-change history sorted by ``(secid, ann_date, start_date)``,
        used to detect ST and suspension events.
    indus_dict : pd.DataFrame
        Mapping from Tushare ``l2_name`` to internal ``indus`` codes.
    indus_data : pd.DataFrame
        Industry membership history: ``secid``, ``in_date``, ``indus``.
    """
    def __init__(self) -> None:
        self.desc = DB.load('information_ts' , 'description') 
        self.desc['list_dt'] = np.maximum(self.desc['list_dt'] , CALENDAR.calendar_start())

        self.cname = DB.load('information_ts' , 'change_name') 
        self.cname = self.cname.query('secid >= 0').sort_values(['secid','ann_date','start_date']).rename(columns={'ann_date':'ann_dt'})

        self.indus_dict = pd.DataFrame(MACHINE.config.get('constant/data/industry/tushare'))
        self.indus_data = DB.load('information_ts' , 'industry') 

        self.indus_data['indus'] = self.indus_dict.loc[self.indus_data['l2_name'],'indus'].values
        self.indus_data = self.indus_data.sort_values(['secid','in_date'])

    def get_desc(self , date : int | TradeDate | None = None , set_index : bool = True , listed = True , exchange = ['SZSE', 'SSE', 'BSE']):
        """
        Return the listing description table, optionally filtered to securities
        that were listed on ``date``.

        Parameters
        ----------
        date : int | TradeDate | None
            If given, keep only rows where ``list_dt <= date < delist_dt``.
        set_index : bool
            If True (default) return with ``secid`` as the index.
        listed : bool
            Drop rows where ``list_dt <= 0`` (unlisted stubs).
        exchange : list[str]
            Keep only these exchange codes.
        """
        desc = self.desc
        if date is not None: 
            desc = desc.loc[(desc['list_dt'] <= int(date)) & (desc['delist_dt'] > int(date))]
        if listed: 
            desc = desc.query('list_dt > 0')
        if exchange: 
            desc = desc.query('exchange_name.isin(@exchange)')
        if set_index: 
            desc = desc.set_index('secid')
        return desc

    def get_secid(self , date : int | None = None) -> np.ndarray:
        """Return a sorted unique array of integer ``secid`` values listed on ``date``."""
        return np.unique(self.get_desc(date , set_index=False)['secid'].to_numpy(int))
    
    def get_st(self , date : int | TradeDate | None = None , reason = ['终止上市', '暂停上市' , 'ST', '*ST']):
        """
        Return securities flagged with an abnormal status (ST, suspended, or delisted).

        Filters ``cname`` to rows whose ``change_reason`` is in ``reason``.
        When ``date`` is provided, only the most recent status *as of* ``date``
        is returned (latest ``start_date <= date``), one row per secid.

        Returns a DataFrame with columns: ``secid``, ``entry_dt``, ``remove_dt``, ``ann_dt``.
        """
        new_cname = self.cname[self.cname['change_reason'].isin(reason)]
        if date is not None: 
            new_cname = new_cname.query('start_date <= @date').copy().drop_duplicates('secid' , keep = 'last')
        return new_cname.loc[:,['secid','entry_dt','remove_dt','ann_dt']]
    
    def get_list_dt(self , date : int | TradeDate | None = None , offset = 0):
        """
        Return a secid-indexed DataFrame with a single ``list_dt`` column.

        ``offset`` trading-day offset is applied to each listing date so that
        callers can build look-ahead-free masks (e.g. ``offset=21`` gives a
        21-day post-IPO exclusion window).
        """
        desc = self.get_desc(date)
        if offset != 0: 
            desc['list_dt'] = CALENDAR.td_array(desc['list_dt'] , offset)
        return desc.loc[:,['list_dt']].reset_index().drop_duplicates(subset='secid').set_index('secid')
    
    def get_abnormal(self , date : int | TradeDate | None = None , reason = ['终止上市', '暂停上市' , 'ST', '*ST', ]):
        """
        Return the name-change records matching ``reason``.

        Similar to ``get_st`` but returns the full ``cname`` subset without
        deduplication, giving a history of all status events.
        """
        if date is None: 
            new_cname = self.cname.copy()
        else:
            date = int(date)
            new_cname = self.cname.query('start_date <= @date').copy().drop_duplicates('secid' , keep = 'last')
        new_cname = new_cname[new_cname['change_reason'].isin(reason)]
        return new_cname
    
    def get_indus(self , date : int | TradeDate | None = None):
        """
        Return the most recent Tushare L2 industry classification for each secid.

        Point-in-time: only records with ``in_date <= date`` are considered.
        Returns a secid-indexed DataFrame with a single ``indus`` column.
        """
        if date is None: 
            df = self.indus_data.copy()
        else:
            df = self.indus_data[self.indus_data['in_date'] <= int(date)]
        df = df.groupby('secid')[['indus']].last()
        return df

    def add_indus(self , df : pd.DataFrame , date : int | TradeDate | None = None , na_industry_as : Any = None):
        """
        Join the industry classification onto ``df`` via a left join on ``secid``.

        Parameters
        ----------
        df : pd.DataFrame
            Frame with a ``secid`` column or index.
        date : int | TradeDate | None
            Point-in-time date for ``get_indus``.
        na_industry_as : scalar, optional
            Fill value for secids without an industry assignment.
        """
        if df.empty:
            return df
        df = df.join(self.get_indus(date) , on = 'secid' , how = 'left')
        if na_industry_as is not None: 
            df['indus'] = df['indus'].fillna(na_industry_as)
        return df

    def get_listed_mask(self , df : pd.DataFrame , list_dt_offset = 21 , reference_date : int | TradeDate | None = None):
        """
        Build a NaN mask array (shape same as ``df``) that is NaN before listing.

        Returns a 2-D numpy array of the same shape as ``df`` where cells that
        precede a stock's listing date (plus ``list_dt_offset`` trading days) are
        NaN and all other cells are 0.  Add this to a value DataFrame to blank out
        pre-listing observations.
        """
        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(df.columns.values).fillna(99991231).astype(int).reset_index()['list_dt']
        df_date = pd.DataFrame(np.tile(df.index.to_numpy()[:, None], (1, df.shape[1])), index=df.index, columns=df.columns)
        return np.where(list_dt > df_date , np.nan , 0)
    
    def mask_list_dt(self , df : pd.DataFrame , mask : bool = True , list_dt_offset : int = 21 , reference_date : int | None = None):
        """
        Add NaN to cells in ``df`` that fall before the stock's listing date.

        Supports both pivoted (date index × secid columns) and long-format
        (MultiIndex ``[date, secid]``) DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            Values frame.  Shape must match the described formats.
        mask : bool
            If False the function is a no-op.
        list_dt_offset : int
            Trading-day grace period after listing (default: 21 days).
        reference_date : int | None
            Date used to look up listing dates.  Defaults to ``df.index.max()``.
        """
        if not mask: 
            return df
        pivoted = df.columns.name == 'secid' and df.index.name == 'date'
        date_values  = df.index.to_numpy()   if pivoted else df.index.get_level_values('date').to_numpy()
        secid_values = df.columns.to_numpy() if pivoted else df.index.get_level_values('secid').to_numpy()

        if reference_date is None: 
            reference_date = date_values.max()

        list_dt = self.get_list_dt(date = reference_date , offset = list_dt_offset).\
            reindex(secid_values).fillna(99991231).astype(int).values
        if pivoted: 
            list_dt = list_dt.T
        
        df_date = pd.DataFrame(np.tile(date_values[:, None], (1, df.shape[1])))
        list_dt_mask = pd.DataFrame(np.where(df_date < list_dt , np.nan , 0) , index = df.index , columns = df.columns)

        return df + list_dt_mask

    def secname(self , secid : np.ndarray | list[int]):
        """Return a numpy array of security names (``sec_name``) for the given secids."""
        secid = np.array(secid) if isinstance(secid , list) else secid
        return self.desc.drop_duplicates('secid').set_index('secid').reindex(secid)['sec_name'].to_numpy()
    
INFO = InfoDataAccess()