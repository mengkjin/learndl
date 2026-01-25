from src.proj import TradeDate , DB 
from src.proj.func import singleton
from .access import DateDataAccess
    
@singleton
class RiskModelAccess(DateDataAccess):
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
        df = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 99)
        return df

    def db_loads_callback(self , *args , **kwargs):
        return

    def get_res(self , date , field = None):
        return self.get(date , 'res' , field)
    
    def get_exp(self , date , field = None):
        return self.get(date , 'exp' , field)
    
    def get_exret(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                  mask = True , pivot = True , drop_old = False):
        return self.get_specific_data(start_dt , end_dt , 'res' , 'resid' , prev = False , 
                                      mask = mask , pivot = pivot , drop_old = drop_old)

RISK = RiskModelAccess()