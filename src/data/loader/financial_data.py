"""
Financial statement data access singletons for the Chinese A-share universe.

Provides point-in-time (announcement-date aware) access to quarterly financial
statements and computed transformations (TTM, QoQ, YoY).

Singletons exported
-------------------
- ``INDI`` : ``IndicatorDataAccess`` — per-share and ratio indicators
- ``BS``   : ``BalanceSheetAccess`` — balance sheet data
- ``CF``   : ``CashFlowAccess`` — cash-flow statement data
- ``IS``   : ``IncomeStatementAccess`` — income statement data
- ``FINA`` : ``FinancialDataAccess`` — dividend, disclosure, forecast tables

``FinData`` provides an expression-based DSL for composing multi-statement
financial ratios (e.g. ``'is@revenue@ttm / bs@total_assets@acc'``).
"""
import re
import numpy as np
import pandas as pd

from typing import Any , Literal

from src.proj import CALENDAR , DB , singleton

from .access import DateDataAccess

ANN_DATA_COLS : list[str] = ['f_ann_date' , 'ann_date']

class FDataAccess(DateDataAccess):
    """
    Abstract base for point-in-time financial statement data access.

    Subclasses set ``SINGLE_TYPE`` to one of the ``DB_KEYS`` entries and
    optionally override ``FROZEN_QTR_METHOD``, ``DEFAULT_*_METHOD``, and
    ``ANN_DATA_COL``.

    Key transformations
    -------------------
    ``acc``   : Cumulative (YTD) values — raw from the database.
    ``qtr``   : Single-quarter values via differencing (``'diff'``) or exact reporting.
    ``ttm``   : Trailing-twelve-months sum / mean over 4 consecutive quarters.
    ``qoq``   : Quarter-on-quarter growth rate.
    ``yoy``   : Year-on-year growth rate.

    All ``*_hist`` methods return a MultiIndex ``(secid, end_date)`` DataFrame
    with the ``lastn`` most recent observations.
    All ``*_latest`` methods return a per-secid ``pd.Series``.
    """
    MAX_LEN = 40
    DATE_KEY = 'end_date'
    DB_SRC = 'financial_ts'
    DB_KEYS = {
        'income' : 'income' ,
        'cashflow' : 'cashflow' ,
        'balance' : 'balance' ,
        'dividend' : 'dividend' ,
        'disclosure' : 'disclosure' ,
        'express' : 'express' ,
        'forecast' : 'forecast' ,
        'mainbz' : 'mainbz' ,
        'indicator' : 'indicator'
    }
    ANN_DATA_COL : str = 'ann_date'
    SINGLE_TYPE : str | Any = None
    FROZEN_QTR_METHOD  : Literal['diff' , 'exact'] | None = None
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    DEFAULT_QOQ_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'qtr'
    DEFAULT_YOY_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'ttm'

    def data_loader(self , date , data_type):
        """Load a single-date slice for ``data_type`` without NA column checking."""
        return DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , check_na_cols=False)

    def get_ann_dt(self , date , latest_n = 1 , within_days = 365):
        """
        Return the announcement date(s) for the most recent quarters up to ``date``.

        Loads trailing quarter-end dates, applies point-in-time filtering
        (``ann_date <= date``), optionally restricts to announcements within
        ``within_days`` calendar days, and returns the ``latest_n`` most
        recent per-secid rows indexed by ``(secid, end_date)``.
        """
        assert self.SINGLE_TYPE , 'SINGLE_TYPE must be set'
        ann_dt = self.gets(CALENDAR.qe_trailing(date , latest_n + 5 , 0) , self.SINGLE_TYPE , ['secid' , self.ANN_DATA_COL]).\
            rename(columns = {self.ANN_DATA_COL : 'ann_date'}).dropna(subset = ['ann_date']).reset_index(drop = False).\
            astype({'end_date':int , 'ann_date':int}).sort_values(['end_date' , 'ann_date']).drop_duplicates(['secid' , 'end_date'])
        #income_ann_dt = [self.get(qtr_end , 'income' , ['secid' , 'ann_date']) for qtr_end in self.qtr_ends(date , latest_n + 5 , 0)]
        #ann_dt : pd.DataFrame = pd.concat(income_ann_dt)
        ann_dt['td_backward'] = CALENDAR.td_array(ann_dt['ann_date'] , backward = True)
        ann_dt['td_forward']  = CALENDAR.td_array(ann_dt['ann_date'] , backward = False)
        ann_dt = ann_dt.loc[ann_dt['td_forward'] <= date , :]
        if within_days > 0:
            ann_dt = ann_dt[ann_dt['td_backward'] >= CALENDAR.cd(date , -within_days)]
        ann_dt = ann_dt.query('secid >= 0').sort_values(['secid' , 'td_backward']).set_index(['secid','end_date'])
        if latest_n <= 0:
            return ann_dt
        else:
            grp = ann_dt.reset_index(drop=False).groupby('secid')
            ann_dt = grp.last() if latest_n == 1 else grp.tail(latest_n).groupby('secid').first()
            return ann_dt.set_index(['end_date'] , append=True)

    def get_ann_calendar(self , date , after_days = 7 , within_days = 365):
        """
        Build a boolean mask indicating which (secid, date) pairs had an announcement
        in the ``after_days`` days following the given date.

        Returns a DataFrame indexed by ``(secid, date)`` with a single bool column
        ``anndt``, useful for building event-study style datasets.
        """
        assert after_days > 0 , f'after_days must be greater than 0 , got {after_days}'
        dates = CALENDAR.cd_trailing(date , within_days)
        ann_dt = self.get_ann_dt(date , 0 , within_days).reset_index('end_date' , drop=True).assign(count = 1)
        v = ann_dt.pivot_table(index = 'ann_date' , columns = 'secid' , values = 'count').reindex(dates).fillna(0)

        ann_calendar = pd.DataFrame(0, index=v.index, columns=v.columns)
        for i in range(after_days):
            ann_calendar += v.shift(i).fillna(0)
        ann_calendar = ann_calendar.melt(ignore_index=False).reset_index().rename(columns = {'ann_date':'date','value':'anndt'})
        return ann_calendar[ann_calendar['anndt'] > 0].set_index(['secid','date']).sort_index().astype(bool)

    @staticmethod
    def index_interpolate(df : pd.DataFrame , year_only : bool = False):
        """
        Fill in missing quarter-end dates in a ``(secid, end_date)`` MultiIndex DataFrame.

        Expands each secid's date range from its min to max quarter-end to a
        full Cartesian grid of quarter-ends (or year-ends when ``year_only=True``),
        then reindexes to introduce NaN for any missing quarters.
        """
        old_index = df.dropna().index.to_frame(index=False)
        min_end_date : pd.Series | Any = old_index.groupby('secid')['end_date'].min()
        max_end_date : pd.Series | Any = old_index.groupby('secid')['end_date'].max()
        index_range = min_end_date.rename('min').to_frame().join(max_end_date.rename('max'))
        full_index = pd.MultiIndex.from_product([old_index['secid'].unique().tolist() , 
                                                 CALENDAR.qe_within(index_range['min'].min(), index_range['max'].max(), year_only = year_only).tolist()],
                                                 names = ['secid' , 'end_date'])
        new_index = full_index.to_frame(index=False).join(index_range , on = ['secid'] , how='left')
        new_index = new_index[(new_index['end_date'] >= new_index['min']) & (new_index['end_date'] <= new_index['max'])].set_index(['secid' , 'end_date'])
        return df.reindex(new_index.index)

    @staticmethod
    def value_interpolate(df : pd.DataFrame , method : Literal['diff' , 'exact'] , year_only : bool = False):
        """
        Fill in missing intermediate quarters using proportional interpolation.

        ``method='diff'`` (for cumulative flow statements):
            Missing Q1 is estimated from Q2/2, Q3/3, or Q4/4.
            Missing Q2 is interpolated from Q1 and Q3.
            Missing Q3 is interpolated from Q2 and Q4.
            Missing Q4 falls back to Q3.
        ``method='exact'`` (for balance sheet or exact reporting):
            Uses groupby forward-fill within each secid.
        """
        if method == 'diff' and not year_only:
            raw_index = df.index
            quarter = raw_index.get_level_values('end_date') % 10000 // 300
            df = df.assign(year = raw_index.get_level_values('end_date') // 10000).set_index('year' , append=True).reset_index('end_date',drop=True)
            dfq = [df.loc[quarter == i + 1] for i in range(4)]

            dfq[0] = dfq[0].fillna(dfq[1] / 2).fillna(dfq[2] / 3).fillna(dfq[3] / 4)
            dfq[1] = dfq[1].fillna((dfq[0] + dfq[2]) / 2).fillna(dfq[2] * 2 / 3).fillna(dfq[3] / 2).fillna(dfq[0])
            dfq[2] = dfq[2].fillna((dfq[1] + dfq[3]) / 2).fillna(dfq[3] * 3 / 4).fillna(dfq[1])
            dfq[3] = dfq[3].fillna(dfq[2])

            df = pd.concat([sub_df.assign(quarter = i + 1) for i , sub_df in enumerate(dfq)]).reset_index('year')
            df['end_date'] = df['year'] * 10000 + df['quarter'].replace({1 : 331 , 2 : 630 , 3 : 930 , 4 : 1231})
            df = df.set_index('end_date' , append=True).drop(columns=['year' , 'quarter']).reindex(raw_index)
        elif method == 'exact':
            df = df.groupby('secid').ffill()
        return df

    def _fin_hist_data_transform(self , df : pd.DataFrame , val : str , benchmark_df : pd.DataFrame | None = None ,
                                 lastn = 1 , pivot = False , ffill = False):
        """
        Post-process a historical financial DataFrame: align to benchmark, ffill, tail, pivot.

        Aligns to ``benchmark_df`` if provided (masks positions where benchmark is NaN),
        optionally forward-fills within each secid, keeps only the last ``lastn`` rows,
        replaces inf values with NaN, and optionally pivots to wide format.
        """
        if benchmark_df is not None:
            df = df.reindex(benchmark_df.index).where(~benchmark_df.isna() , np.nan)
        if ffill: 
            df = df.groupby('secid').ffill()
        df = df.groupby('secid').tail(lastn).replace([np.inf , -np.inf] , np.nan)
        if pivot: 
            df = df.pivot_table(val , 'end_date' , 'secid').sort_index()
        return df
    
    def _get_qtr_method(self , qtr_method : Literal['diff' , 'exact'] | None = None):
        """Resolve the quarter method: use FROZEN_QTR_METHOD if set, else fall back to arg or DEFAULT."""
        if self.FROZEN_QTR_METHOD: 
            qtr_method = self.FROZEN_QTR_METHOD
        elif qtr_method is None: 
            qtr_method = self.DEFAULT_QTR_METHOD
        return qtr_method

    def _get_data_acc_hist(self , data_type : str , val : str , date : int , lastn = 1 ,
                           pivot = False , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None ,
                           year_only = False):
        """
        Load historical cumulative (YTD) values for ``val`` from ``data_type``.

        Applies point-in-time filtering (``ann_date <= date``), deduplicates on
        ``(secid, end_date)`` keeping the last update, then interpolates missing
        quarter-end slots and their values.
        Returns a MultiIndex ``(secid, end_date)`` DataFrame with the ``lastn`` tails.
        """
        assert data_type in ['income' , 'cashflow' , 'balance' , 'indicator'] , \
            f'invalid data_type: {data_type} , must be one of ' + str(['income' , 'cashflow' , 'balance' , 'indicator'])
        qtr_method = self._get_qtr_method(qtr_method)

        dates = CALENDAR.qe_trailing(date , n_past = lastn + 4 , year_only = year_only)
        field = ['secid' , self.ANN_DATA_COL , 'update_flag' , val]
        
        df_acc = self.gets(dates , data_type , field , rename_date_key = 'end_date').\
            rename(columns = {self.ANN_DATA_COL : 'ann_date'}).reset_index(drop = False).dropna(subset = ['ann_date' , 'end_date']).\
            astype({'end_date':int , 'ann_date':int})
        df_acc = df_acc.query('ann_date <= @date and end_date in @dates and secid >= 0')
        df_acc = df_acc.sort_values('update_flag').drop_duplicates(['secid' , 'end_date'] , keep = 'last')\
            [['secid','end_date',val]].set_index(['secid' , 'end_date']).sort_index()
        df_acc = self.index_interpolate(df_acc , year_only = year_only)
        df_acc = self.value_interpolate(df_acc , method = qtr_method , year_only = year_only)
        return self._fin_hist_data_transform(df_acc , val , None , lastn , pivot , ffill)
    
    def _get_data_qtr_hist(self , data_type : str , val : str , date : int , lastn = 1 ,
                           pivot = False , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None):
        """
        Compute single-quarter values by differencing consecutive cumulative quarters.

        For ``qtr_method='diff'``: subtracts each quarter-start (year-start stub at 0)
        from the cumulative to obtain single-quarter values.
        For ``qtr_method='exact'``: returns the raw accumulated values unchanged
        (delegates to ``_get_data_acc_hist``).
        Aligned to the cumulative index via ``_fin_hist_data_transform``.
        """
        qtr_method = self._get_qtr_method(qtr_method)
        if qtr_method == 'exact':
            return self._get_data_acc_hist(data_type , val , date , lastn , pivot , ffill)

        df_acc = self._get_data_acc_hist(data_type , val , date , lastn + 4 , pivot = False , qtr_method = qtr_method)
        secid = df_acc.index.get_level_values('secid').unique()
        ystarts = np.unique(df_acc.index.get_level_values('end_date').unique() // 10000 * 10000).tolist()
        df_ystarts = pd.DataFrame({val:0} , index = pd.MultiIndex.from_product([secid , ystarts] , names=['secid' , 'end_date']))
        df_qtr = pd.concat([df_acc , df_ystarts]).sort_index().groupby('secid').ffill().fillna(0).groupby('secid').diff()
        return self._fin_hist_data_transform(df_qtr , val , df_acc , lastn , pivot , ffill)

    def _get_data_ttm_hist(self , data_type : str , val : str , date : int , lastn = 1 ,
                           pivot = False , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None):
        """
        Compute trailing-twelve-months values.

        For ``qtr_method='diff'``: rolling 4-quarter sum.
        For ``qtr_method='exact'``: rolling 5-quarter mean.
        Handles a pandas groupby-rolling multi-index duplication bug by
        resetting the extra level when present.
        """
        qtr_method = self._get_qtr_method(qtr_method)
        df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 8 , pivot = False , ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_ttm = df_qtr.reindex(full_index).fillna(0)
        grp = df_ttm.groupby('secid')
        if qtr_method == 'diff': 
            df_ttm = grp.rolling(4).sum()
        elif qtr_method == 'exact': 
            df_ttm = grp.rolling(5).mean()
        if len(df_ttm.index.names) > 2 and df_ttm.index.names[0] == 'secid' and df_ttm.index.names[1] == 'secid':
            df_ttm = df_ttm.reset_index(level=0, drop=True)
        assert isinstance(df_ttm , pd.DataFrame) , f'df_ttm must be a DataFrame , got {type(df_ttm)}'
        return self._fin_hist_data_transform(df_ttm , val , df_qtr , lastn , pivot , ffill)
    
    def _get_data_qoq_hist(self , data_type : str , val : str , date : int , lastn = 1 ,
                           pivot = False , ffill = False ,
                           qtr_method : Literal['diff' , 'exact'] | None = None ,
                           qoq_method : Literal['ttm' , 'acc' , 'qtr'] | None = None):
        """
        Compute quarter-on-quarter growth rate: ``(current - prev) / |prev|``.

        The base series is determined by ``qoq_method`` (qtr, ttm, or acc).
        The previous quarter is obtained via a 1-quarter lag within each secid.
        """
        if qoq_method is None: 
            qoq_method = self.DEFAULT_QOQ_METHOD
        if qoq_method == 'qtr':
            df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 2 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        elif qoq_method == 'ttm':
            df_qtr = self._get_data_ttm_hist(data_type , val , date , lastn + 2 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_qoq = df_qtr.reindex(full_index)
        df_qoq_base = df_qoq.groupby('secid').shift(1)
        df_qoq = (df_qoq - df_qoq_base) / df_qoq_base.abs()
        return self._fin_hist_data_transform(df_qoq , val , df_qtr , lastn , pivot , ffill)
    
    def _get_data_yoy_hist(self , data_type : str , val : str , date : int , lastn = 1 ,
                           pivot = False , ffill = False ,
                           qtr_method : Literal['diff' , 'exact'] | None = None ,
                           yoy_method : Literal['ttm' , 'acc' , 'qtr'] | None = None):
        """
        Compute year-on-year growth rate: ``(current - prev_year) / |prev_year|``.

        The base series is determined by ``yoy_method`` (ttm, acc, or qtr).
        The same-quarter prior year is obtained via a 4-quarter lag within each secid.
        """
        if yoy_method is None: 
            yoy_method = self.DEFAULT_YOY_METHOD
        if yoy_method == 'ttm':
            df_qtr = self._get_data_ttm_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        elif yoy_method == 'acc':
            df_qtr = self._get_data_acc_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False)
        elif yoy_method == 'qtr':
            df_qtr = self._get_data_qtr_hist(data_type , val , date , lastn + 5 , pivot = False , 
                                             ffill = False , qtr_method = qtr_method)
        full_index = pd.MultiIndex.from_product([df_qtr.index.get_level_values('secid').unique() ,
                                                 df_qtr.index.get_level_values('end_date').unique()])
        df_yoy = df_qtr.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(4)
        df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()
        return self._fin_hist_data_transform(df_yoy , val , df_qtr , lastn , pivot , ffill)
    
    def _fin_latest_data_transform(self , df : pd.DataFrame , val : str):
        """Extract the last non-NaN value per secid from a ``(secid, end_date)`` DataFrame."""
        return df.dropna().groupby('secid').last()[val]

    def _get_data_acc_latest(self , data_type : str , val : str , date : int):
        """Return the latest cumulative value per secid as of ``date``."""
        df_acc = self._get_data_acc_hist(data_type , val , date , 3)
        return self._fin_latest_data_transform(df_acc , val)

    def _get_data_qtr_latest(self , data_type : str , val : str , date : int ,
                             qtr_method : Literal['diff' , 'exact'] | None = None):
        """Return the latest single-quarter value per secid as of ``date``."""
        df_qtr = self._get_data_qtr_hist(data_type , val , date , 3 , ffill = False , qtr_method = qtr_method)
        return self._fin_latest_data_transform(df_qtr , val)

    def _get_data_ttm_latest(self , data_type : str , val : str , date : int ,
                             qtr_method : Literal['diff' , 'exact'] | None = None):
        """Return the latest TTM value per secid as of ``date``."""
        df_ttm = self._get_data_ttm_hist(data_type , val , date , 3 , ffill = False , qtr_method = qtr_method)
        return self._fin_latest_data_transform(df_ttm , val)

    def _get_data_qoq_latest(self , data_type : str , val : str , date : int ,
                             qtr_method : Literal['diff' , 'exact'] | None = None ,
                             qoq_method : Literal['qtr' , 'ttm'] | None = None):
        """Return the latest QoQ growth rate per secid as of ``date``."""
        df_qoq = self._get_data_qoq_hist(data_type , val , date , 3 ,
                                         qtr_method = qtr_method , qoq_method = qoq_method)
        return self._fin_latest_data_transform(df_qoq , val)

    def _get_data_yoy_latest(self , data_type : str , val : str , date : int ,
                             qtr_method : Literal['diff' , 'exact'] | None = None ,
                             yoy_method : Literal['ttm' , 'acc'] | None = None):
        """Return the latest YoY growth rate per secid as of ``date``."""
        df_yoy = self._get_data_yoy_hist(data_type , val , date , 3 , qtr_method = qtr_method , yoy_method = yoy_method)
        return self._fin_latest_data_transform(df_yoy , val)
    
    @property
    def fields(self):
        """Return the column names available in ``SINGLE_TYPE``."""
        assert self.SINGLE_TYPE is not None , 'SINGLE_TYPE must be set'
        if self.collections[self.SINGLE_TYPE]: 
            return self.collections[self.SINGLE_TYPE].columns()
        else:
            return self.get(20231231 , self.SINGLE_TYPE).columns.values

    def acc(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , year_only = False):
        """Return historical cumulative (YTD) values for ``val``."""
        return self._get_data_acc_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , year_only = year_only)

    def qtr(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False):
        """Return historical single-quarter values for ``val``."""
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill)

    def ttm(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False):
        """Return historical trailing-twelve-months values for ``val``."""
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill)

    def qoq(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False ,
            qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        """Return historical quarter-on-quarter growth for ``val``."""
        return self._get_data_qoq_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , qoq_method)

    def yoy(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False ,
            qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        """Return historical year-on-year growth for ``val``."""
        return self._get_data_yoy_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , yoy_method)

    def acc_latest(self, val: str, date: int):
        """Return the latest cumulative value per secid."""
        return self._get_data_acc_latest(self.SINGLE_TYPE , val, date)

    def qtr_latest(self, val: str, date: int):
        """Return the latest single-quarter value per secid."""
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date)

    def ttm_latest(self, val: str, date: int):
        """Return the latest TTM value per secid."""
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date)

    def qoq_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        """Return the latest QoQ growth rate per secid."""
        return self._get_data_qoq_latest(self.SINGLE_TYPE , val, date , qtr_method , qoq_method)

    def yoy_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        """Return the latest YoY growth rate per secid."""
        return self._get_data_yoy_latest(self.SINGLE_TYPE , val, date , qtr_method , yoy_method)
    
@singleton
class IndicatorDataAccess(FDataAccess):
    """
    Per-share and ratio indicators (EPS, gross margin, FCFE, ROIC, etc.) from
    ``financial_ts/indicator``.  Exported as ``INDI``.
    """
    SINGLE_TYPE = 'indicator'
    FROZEN_QTR_METHOD  : Literal['diff' , 'exact'] | None = None
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'

    def acc(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None , year_only = False):
        return self._get_data_acc_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , year_only)
    
    def qtr(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method)

    def ttm(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_ttm_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method)
    
    def qoq(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , qoq_method)
    
    def yoy(self , val : str , date : int , lastn = 1 , pivot = False , ffill = False , 
            qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_hist(self.SINGLE_TYPE , val , date , lastn , pivot , ffill , qtr_method , yoy_method)
    
    def qtr_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_qtr_latest(self.SINGLE_TYPE , val, date , qtr_method)
    
    def ttm_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None):
        return self._get_data_ttm_latest(self.SINGLE_TYPE , val, date , qtr_method)
    
    def qoq_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , qoq_method : Literal['qtr' , 'ttm'] | None = None):
        return self._get_data_qoq_latest(self.SINGLE_TYPE , val, date , qtr_method , qoq_method)
    
    def yoy_latest(self, val: str, date: int , qtr_method : Literal['diff' , 'exact'] | None = None , yoy_method : Literal['ttm' , 'acc'] | None = None):
        return self._get_data_yoy_latest(self.SINGLE_TYPE , val, date , qtr_method , yoy_method)

@singleton
class BalanceSheetAccess(FDataAccess):
    """
    Balance sheet data from ``financial_ts/balance``.  Uses ``'exact'`` quarter
    method (balance sheet items are point-in-time, not cumulative).  Exported as ``BS``.
    """
    SINGLE_TYPE = 'balance'
    FROZEN_QTR_METHOD = 'exact'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'exact'
    DEFAULT_YOY_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'acc'
    DEFAULT_QOQ_METHOD : Literal['ttm' , 'acc' , 'qtr'] = 'acc'
    ANN_DATA_COL : str = 'f_ann_date'

@singleton
class CashFlowAccess(FDataAccess):
    """
    Cash-flow statement data from ``financial_ts/cashflow``.  Uses ``'diff'``
    quarter method (cumulative YTD flows).  Exported as ``CF``.
    """
    SINGLE_TYPE = 'cashflow'
    FROZEN_QTR_METHOD = 'diff'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    ANN_DATA_COL : str = 'f_ann_date'

@singleton
class IncomeStatementAccess(FDataAccess):
    """
    Income statement data from ``financial_ts/income``.  Uses ``'diff'``
    quarter method (cumulative YTD income items).  Exported as ``IS``.
    """
    SINGLE_TYPE = 'income'
    FROZEN_QTR_METHOD = 'diff'
    DEFAULT_QTR_METHOD : Literal['diff' , 'exact'] = 'diff'
    ANN_DATA_COL : str = 'f_ann_date'
  
@singleton
class FinancialDataAccess(FDataAccess):
    """
    Supplementary financial tables (dividend, disclosure schedule, earnings
    express, earnings forecast, main business breakdown) from ``financial_ts``.
    Exported as ``FINA``.  Does not have a ``SINGLE_TYPE``.
    """
    DB_KEYS = {
        'dividend' : 'dividend' ,
        'disclosure' : 'disclosure' ,
        'express' : 'express' ,
        'forecast' : 'forecast' ,
        'mainbz' : 'mainbz'
    }

INDI = IndicatorDataAccess()
BS   = BalanceSheetAccess()
CF   = CashFlowAccess()
IS   = IncomeStatementAccess()
FINA = FinancialDataAccess()

class FinData:
    """
    Expression-based financial data evaluator.

    Parses a ``@``-delimited DSL string like
    ``'is@revenue@ttm / bs@total_assets@acc'`` and evaluates it by dispatching
    each token to the corresponding ``IS``, ``CF``, ``INDI``, or ``BS`` singleton
    and combining via Python ``eval``.

    Token format: ``<statement>@<field>@<transform>``
    - ``statement`` : ``is`` / ``cf`` / ``indi`` / ``bs``
    - ``field``     : column name in the corresponding table
    - ``transform`` : ``ttm`` / ``qtr`` / ``acc`` / ``yoy`` / ``qoq``

    A set of short-form aliases (``ta``, ``sales``, ``npro``, etc.) are expanded
    by ``reserved_numerator`` before parsing.
    """
    def __init__(self , expression : str , **kwargs):
        """
        Parameters
        ----------
        expression : str
            DSL expression string (e.g. ``'is@revenue@ttm / bs@total_assets@acc'``).
        **kwargs
            Default keyword arguments forwarded to the underlying accessor calls
            (e.g. ``qtr_method='exact'``).
        """
        self.expression = expression
        self.kwargs = kwargs

    @classmethod
    def reserved_numerator(cls , expr : str , kwargs):
        """
        Expand short-form signal aliases to their full ``statement@field`` form.

        Also injects the correct ``qtr_method`` for certain indicators
        (e.g. ``'gp'`` → ``qtr_method='diff'``; ``'tangible_asset'`` → ``'exact'``).
        """
        first_signal = expr.split('@')[0]
        reserved_signals = {
            'ta'        : 'bs@total_assets' ,
            'equ'       : 'bs@total_hldr_eqy_exc_min_int' ,
            'liab'      : 'bs@total_liab' ,

            'sales'     : 'is@revenue' ,
            'oper_np'   : 'is@operate_profit' ,
            'total_np'  : 'is@total_profit' ,
            'npro'      : 'is@n_income_attr_p' ,
            'ebit'      : 'is@ebit' ,
            'ebitda'    : 'is@ebitda' ,
            'tax'       : 'is@income_tax' ,
            
            'ncfo'      : 'cf@n_cashflow_act' ,
            'ncfi'      : 'cf@n_cashflow_inv_act' ,
            'ncff'      : 'cf@n_cash_flows_fnc_act' ,
            'incfo'     : 'cf@c_inf_fr_operate_a' ,
            'incfi'     : 'cf@stot_inflows_inv_act' ,
            'incff'     : 'cf@stot_cash_in_fnc_act' ,
            'outcfo'    : 'cf@st_cash_out_act' ,
            'outcfi'    : 'cf@stot_out_inv_act' ,
            'outcff'    : 'cf@stot_cashout_fnc_act' ,


            'dedt'      : 'indi@profit_dedt' ,
            'eps'       : 'indi@eps' ,
            'gp'        : 'indi@gross_margin' ,
            'fcfe'      : 'indi@fcfe' ,
            'roic'      : 'indi@roic' ,

            'tangible_asset' : 'indi@tangible_asset' ,
        }
        if first_signal not in reserved_signals:
            ...
        else:
            expr = expr.replace(f'{first_signal}@' , f'{reserved_signals[first_signal]}@')
            if first_signal in ['gp' , 'dedt' , 'fcfe' , 'eps' , 'roic']:
                kwargs['qtr_method'] = 'diff'
            elif first_signal in ['tangible_asset']:
                kwargs['qtr_method'] = 'exact'
        return expr , kwargs
    
    def parse_and_eval(self , date : int , category : Literal['latest' , 'hist'] , **kwargs):
        """
        Parse the expression, evaluate each token via the financial singletons,
        and combine results using Python ``eval``.

        Each ``@``-delimited token is replaced by ``dfs[i]`` in the expression
        string before evaluation.  The result is a ``pd.Series`` (for ``'latest'``)
        or ``pd.DataFrame`` (for ``'hist'``).
        """
        pattern = re.compile(r'([a-zA-Z!@#$_@0-9]+)')
        items = pattern.findall(self.expression)
        repls = {item:f'dfs[{i}]' for i , item in enumerate(items)}
        reprs = {}
        dfs : list[pd.Series | pd.DataFrame] = []
        for item in items:
            expr , kwgs = self.reserved_numerator(item , self.kwargs | kwargs)
            fstatement , fval , ftype = expr.split('@')
            reprs[item] = fval
            assert fstatement in ['is' , 'cf' , 'indi' , 'bs'] , fstatement
            assert ftype in ['ttm' , 'qtr' , 'acc' , 'yoy' , 'qoq'] , ftype
            df = self._f_func(fstatement , ftype , category)(fval , date , **kwgs)
            if isinstance(df , pd.DataFrame) and df.shape[1] == 1 and isinstance(df.columns[0] , str): 
                df = df.iloc[:,0]
            dfs.append(df)

        python_expression = pattern.sub(lambda x:repls[x.group()], self.expression)
        naming_expression = pattern.sub(lambda x:reprs[x.group()], self.expression)

        try:
            result = eval(python_expression)
            if isinstance(result , pd.Series): 
                result.name = naming_expression
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
        
        return result
    
    @classmethod
    def _f_func(cls , fstatement : Literal['is' , 'cf' , 'indi' , 'bs'] | str ,
                ftype : Literal['ttm' , 'qtr' , 'acc' , 'yoy' , 'qoq'] | str ,
                category : Literal['latest' , 'hist']):
        """
        Resolve the callable method from the correct singleton.

        Maps ``(fstatement, ftype, category)`` to the appropriate
        ``<singleton>.<ftype>_latest`` or ``<singleton>.<ftype>`` method.
        """
        if fstatement == 'is': 
            f_source = IS
        elif fstatement == 'cf': 
            f_source = CF
        elif fstatement == 'indi': 
            f_source = INDI
        elif fstatement == 'bs': 
            f_source = BS
        else: 
            raise ValueError(f'invalid statement: {fstatement}')
        if category == 'latest':
            return getattr(f_source , f'{ftype}_latest')
        elif category == 'hist':
            return getattr(f_source , f'{ftype}')
        else: 
            raise ValueError(f'invalid category: {category}')

    def get_latest(self , date : int , new_name : str | None = None) -> pd.Series:
        """
        Evaluate the expression and return the latest value per secid as a Series.

        Optionally renames the Series to ``new_name``.
        """
        val = self.parse_and_eval(date , 'latest')
        if isinstance(val , pd.DataFrame): 
            val = val.iloc[:,0]
        assert isinstance(val , pd.Series) , f'invalid latest value: {val}'
        if new_name is not None: 
            val.name = new_name
        return val
    
    def get_hist(self , date : int , lastn : int = 1 , new_name : str | None = None) -> pd.DataFrame:
        """
        Evaluate the expression and return the trailing ``lastn`` quarterly observations
        per secid as a MultiIndex ``(secid, end_date)`` DataFrame.
        """
        val = self.parse_and_eval(date , 'hist' , lastn = lastn)
        if isinstance(val , pd.Series):
            val = val.to_frame()
        assert isinstance(val , pd.DataFrame) , f'invalid hist value: {val}'
        if new_name is not None: 
            val.columns = [new_name]
        return val