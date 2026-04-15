"""
Risk model data access singleton for the Tushare CNE5 factor model.

Provides factor exposures, residual returns, specific risk, covariance matrices,
and model coefficients from the ``models`` database.
Exported as the ``RISK`` singleton.
"""
from src.proj import TradeDate , DB , singleton
from .access import DateDataAccess

@singleton
class RiskModelAccess(DateDataAccess):
    """
    Singleton data access object for the Tushare CNE5 risk model outputs.

    DB_KEYS map
    -----------
    ``res``   : ``tushare_cne5_res``   — per-stock residual (alpha) returns
    ``exp``   : ``tushare_cne5_exp``   — per-stock factor exposures
    ``spec``  : ``tushare_cne5_spec``  — per-stock specific variance
    ``cov``   : ``tushare_cne5_cov``   — factor covariance matrix
    ``coef``  : ``tushare_cne5_coef``  — regression coefficients
    """
    MAX_LEN = 2000
    DB_SRC = 'models'
    DB_KEYS = {
        'res' : 'tushare_cne5_res' , 
        'exp' : 'tushare_cne5_exp' , 
        'spec' : 'tushare_cne5_spec' , 
        'cov' : 'tushare_cne5_cov' , 
        'coef' : 'tushare_cne5_coef'
    }
    
    def data_loader(self , date , data_type):
        """Load a single-date slice for ``data_type`` from the risk model database."""
        df = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 'never')
        return df

    def db_loads_callback(self , *args , **kwargs):
        """No-op: bulk preload is not supported for risk model data."""
        return

    def get_res(self , date , field = None):
        """Return residual return data for a single ``date``."""
        return self.get(date , 'res' , field)

    def get_exp(self , date , field = None):
        """Return factor exposure data for a single ``date``."""
        return self.get(date , 'exp' , field)

    def get_exret(self , start : int | TradeDate , end : int | TradeDate ,
                  mask = True , pivot = True , drop_old = False):
        """
        Return per-stock residual (excess) returns from the risk model.

        The ``resid`` column is extracted from the ``res`` data type.
        Applies listing-date masking when ``mask=True`` and returns a
        (date × secid) wide frame when ``pivot=True``.
        """
        return self.get_specific_data(start , end , 'res' , 'resid' , prev = False , 
                                      mask = mask , pivot = pivot , drop_old = drop_old)

RISK = RiskModelAccess()