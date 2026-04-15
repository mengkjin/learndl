"""
Custom daily microstructure risk exposure data access singleton.

Provides per-stock daily risk feature exposures computed by
``DailyRiskUpdater`` and stored in the ``exposure/daily_risk`` database.
Exported as the ``EXPO`` singleton.
"""
import pandas as pd

from typing import Literal

from src.proj import TradeDate , DB , singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class ExposureAccess(DateDataAccess):
    """
    Singleton data access object for custom daily microstructure risk exposures.

    Available fields in ``daily_risk``
    -----------------------------------
    ``true_range``, ``turnover``, ``large_buy_pdev``, ``small_buy_pct``,
    ``sqrt_avg_size``, ``open_close_pct``, ``ret_volatility``, ``ret_skewness``
    """
    MAX_LEN = 300
    DB_SRC = 'exposure'
    DB_KEYS = {'daily_risk' : 'daily_risk'}
    
    def data_loader(self , date , data_type):
        """Load a single-date slice; filters to listed securities via ``INFO.get_secid``."""
        df : pd.DataFrame = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 'never' , use_alt = True)
        if not df.empty:
            df = df[df['secid'].isin(INFO.get_secid(date))]
        return df

    def get_daily_risk(self , date):
        """Return the raw daily risk exposure DataFrame for a single ``date``."""
        return self.get(date , 'daily_risk')

    def get_risks(
        self , start : int | TradeDate , end : int | TradeDate ,
        field : Literal['true_range' , 'turnover' , 'large_buy_pdev' , 'small_buy_pct' ,
        'sqrt_avg_size' , 'open_close_pct' , 'ret_volatility' , 'ret_skewness'] | str | list , prev = False ,
        mask = False , pivot = False , **kwargs
    ) -> pd.DataFrame:
        """
        Return microstructure risk exposure data for ``[start, end]``.

        Optionally applies listing-date masking and pivots to wide format.
        The underlying cache is truncated after each call (``drop_old=True``).
        """
        qte = self.get_specific_data(start , end , 'daily_risk' , field = field , prev = prev , 
                                     mask = mask , pivot = False , drop_old = True)
        
        if pivot:
            qte = qte.pivot_table(field , 'date' , 'secid')
        return qte
        
EXPO= ExposureAccess()