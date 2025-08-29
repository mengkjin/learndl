from src.proj import PATH
from src.basic import CALENDAR , TradeDate , DB
from src.func.singleton import singleton

from .access import DateDataAccess
    
@singleton
class RiskModelAccess(DateDataAccess):
    MAX_LEN = 2000
    DATA_TYPE_LIST = ['res' , 'exp' , 'spec' , 'cov' , 'coef']  
    
    def data_loader(self , date , data_type):
        if data_type in self.DATA_TYPE_LIST: 
            df = DB.db_load('models' , f'tushare_cne5_{data_type}' , date , verbose = False)
        else:
            raise KeyError(data_type)
        # if df is not None: df = df.reset_index().assign(date = date)
        return df

    def get_res(self , date , field = None):
        return self.get(date , 'res' , field)
    
    def get_exp(self , date , field = None):
        return self.get(date , 'exp' , field)
    
    def get_exret(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                  mask = True , pivot = True , drop_old = False):
        return self.get_specific_data(start_dt , end_dt , 'res' , 'resid' , prev = False , 
                                      mask = mask , pivot = pivot , drop_old = drop_old)

RISK = RiskModelAccess()