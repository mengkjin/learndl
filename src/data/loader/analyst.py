"""
Sell-side analyst report data access singleton.

Provides access to analyst earnings estimates (EPS, NP, ROE, etc.),
consensus target prices, and report-level data from the ``analyst_ts`` database.
Exported as the ``ANALYST`` singleton.
"""
import numpy as np
import pandas as pd

from typing import Literal

from src.proj import CALENDAR , DB , singleton

from .access import DateDataAccess

@singleton
class AnalystDataAccess(DateDataAccess):
    """
    Singleton data access object for sell-side analyst reports.

    The cache stores daily report snapshots (``report`` key).  High-level
    methods aggregate reports over a trailing window and compute
    exponentially decay-weighted consensus estimates.
    """
    MAX_LEN = 2000
    DB_SRC = 'analyst_ts'
    DB_KEYS = {'report' : 'report'}

    def data_loader(self , date , data_type):
        """Load a single-date slice for ``data_type`` from the database."""
        df = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 'never')
        return df

    def get_report(self , date , field = None):
        """Return the analyst report DataFrame for a single ``date``."""
        return self.get(date , 'report' , field)

    def get_trailing_reports(self , date : int , n_month : int = 3 , lag_month : int = 0, latest = False , **filter_kwargs):
        """
        Collect analyst reports published in the trailing ``n_month`` months.

        Parameters
        ----------
        date : int
            Reference date (yyyyMMdd).
        n_month : int
            Trailing window size in calendar months.
        lag_month : int
            Shift the window back by this many months (for point-in-time testing).
        latest : bool
            If True, keep only the most recent report per (secid, org_name, quarter).
        **filter_kwargs
            Additional column=value filters applied to each daily frame.
        """
        d0 = CALENDAR.cd(date , -30 * (n_month + lag_month))
        d1 = CALENDAR.cd(date , -30 * lag_month) 
        dates = CALENDAR.range(d0 , d1 , 'cd')
        reports : list[pd.DataFrame] = []
        for date in dates:
            df = self.get_report(date)
            if df.empty: 
                continue
            for key , value in filter_kwargs.items():
                df = df.query(f'{key} == @value')
            reports.append(df)
        df = pd.concat([rep for rep in reports if not rep.empty]).astype({'report_date':int}).reset_index(drop = True)
        if latest:
            df = df.sort_values(['secid' , 'report_date']).groupby(['secid' , 'org_name' , 'quarter']).last().reset_index(drop = False)
        return df
    
    @staticmethod
    def weighted_val(df : pd.DataFrame , end : int , col : str , half_life : int = 180):
        """
        Compute an exponentially decay-weighted mean of ``col`` within each secid.

        Weights decay with a half-life of ``half_life`` calendar days relative
        to the reference date ``end``.
        """
        df = df.assign(_w = np.exp(-np.log(2) * CALENDAR.cd_diff_array(end , df['report_date']) / half_life))
        return df.groupby('secid').apply(lambda x,**kwg:(x[col] * x['_w']).sum() / x['_w'].sum() , include_groups = False)

    @staticmethod
    def val_multiplier(val : str):
        """Return the unit scaling factor for a valuation metric (1e4 for flow metrics, else 1)."""
        return 1e4 if val in ['sales' , 'op' , 'np' , 'tp'] else 1

    def get_val_est(self , date : int , year : int , val : Literal['sales' , 'np' , 'tp' , 'op' , 'eps' , 'roe'] ,
                    n_month : int = 12 , lag_month : int = 0):
        """
        Return the consensus estimate for ``val`` for fiscal year ``year``.

        Aggregates trailing reports via ``weighted_val`` with exponential decay.
        ``val`` may be ``'sales'``, ``'np'``, ``'tp'``, ``'op'``, ``'eps'``, or ``'roe'``.
        Flow metrics (sales/op/np/tp) are scaled to CNY by ``val_multiplier``.
        """
        date = CALENDAR.cd(date , -30 * lag_month)
        col = {'sales' : 'op_rt' , 'np' : 'np' , 'tp' : 'tp' , 'op' : 'op_pr' , 'eps' : 'eps' , 'roe' : 'roe'}[val]
        multiplier = self.val_multiplier(val)
        df = self.get_trailing_reports(date , n_month , latest = True , quarter = f'{year}Q4')
        est = self.weighted_val(df , date , col) * multiplier
        return est
    
    def get_val_ftm(self , date : int , val : Literal['sales' , 'np' , 'tp' , 'op' , 'eps' , 'roe'] , n_month : int = 12 , lag_month : int = 0):
        """
        Return the forward-twelve-months (FTM) estimate for ``val``.

        Interpolates between the current-year and next-year annual estimates
        using the number of remaining months in the current fiscal year as weights.
        Missing estimates are forward-filled from the other year.
        """
        date = CALENDAR.cd(date , -30 * lag_month)
        month = date // 100 % 100
        year = date // 10000
        val0 = self.get_val_est(date , year , val , n_month)
        val1 = self.get_val_est(date , year + 1 , val , n_month)
        ftm = pd.concat([val0 , val1] , axis = 1).rename(columns = {0 : 'val0' , 1 : 'val1'})
        ftm['val0'] = ftm['val0'].fillna(ftm['val1'])
        ftm['val1'] = ftm['val1'].fillna(ftm['val0'])
        return (12 - month) * ftm['val0'] + month * ftm['val1']
    
    def target_price(self , date : int , n_month : int = 12 , lag_month : int = 0):
        """
        Return the simple-average consensus target price per secid.

        Computes the mean of each analyst's midpoint (average of max_price and
        min_price), then averages across analysts for each stock.
        """
        date = CALENDAR.cd(date , -30 * lag_month)
        df = self.get_trailing_reports(date , n_month , latest = True)
        df = df.query('max_price.notna() | min_price.notna()').copy()
        df['target_price'] = df.loc[:,['max_price' , 'min_price']].mean(axis = 1)
        v = df.groupby('secid')['target_price'].mean()
        return v
        
ANALYST = AnalystDataAccess()