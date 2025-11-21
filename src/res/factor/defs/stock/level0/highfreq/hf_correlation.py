import polars as pl

from src.data import DATAVENDOR
from src.res.factor.calculator import HfCorrelationFactor


__all__ = [
    'inday_mkt_beta' , 'inday_mkt_corr' , 'inday_ret_autocorr' , 
    'inday_vol_ret1_corr' , 'inday_vol_vwap_corr' , 'inday_vol_autocorr' ,
    'inday_mkt_beta_std' , 'inday_mkt_corr_std' , 'inday_ret_autocorr_std' , 
    'inday_vol_ret1_corr_std' , 'inday_vol_vwap_corr_std' , 'inday_vol_autocorr_std'
]

def eval_corr_expression(expression : str):
    '''expression : val1@lag1,val2@lag2,beta'''
    default_expression = {
        'beta' : 'ret,mkt,True' ,
        'corr' : 'ret,mkt' ,
        'vol_vwap' : 'volume,vwap' ,
        'vol_ret1' : 'volume,ret@1' ,
        'vol_autocorr' : 'volume,volume@1' ,
        'ret_autocorr' : 'ret,ret@1' ,
    }
    expression = default_expression.get(expression , expression)
    comps = expression.split(',')
    assert len(comps) in [2 , 3] , f'expression must be in the format of val1@lag1,val2@lag2(,beta) , got {expression}'
    vl1 = comps[0].split('@')
    vl2 = comps[1].split('@')
    beta = bool(eval(comps[2])) if len(comps) > 2 else False
    return {'val1' : vl1[0] , 'lag1' : int(vl1[1]) if len(vl1) > 1 else 0 , 
            'val2' : vl2[0] , 'lag2' : int(vl2[1]) if len(vl2) > 1 else 0 , 
            'beta' : beta}

def inday_corr_avg(date : int , n_day : int , expression : str):
    dates = DATAVENDOR.CALENDAR.td_trailing(date , n_day)
    dfs = [DATAVENDOR.MKLINE.get_inday_corr(date , **eval_corr_expression(expression)) for date in dates]
    df = pl.concat(dfs).group_by('secid').agg(pl.col('value').mean().alias('value')).\
        to_pandas().set_index('secid')['value'].sort_index()
    return df

def inday_corr_std(date : int , n_day : int , expression : str):
    dates = DATAVENDOR.CALENDAR.td_trailing(date , n_day)
    dfs = [DATAVENDOR.MKLINE.get_inday_corr(date , **eval_corr_expression(expression)) for date in dates]
    df = pl.concat(dfs).group_by('secid').agg(pl.col('value').std().alias('value')).\
        to_pandas().set_index('secid')['value'].sort_index()
    return df

class inday_mkt_beta(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内beta均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'beta')
    
class inday_mkt_corr(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内量价相关性均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'corr')
    
class inday_ret_autocorr(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内收益率自相关系数均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'ret_autocorr')
    
class inday_vol_ret1_corr(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内波动价相关性均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'vol_ret1')
    
class inday_vol_vwap_corr(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内波动价相关性均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'vol_vwap')
    
class inday_vol_autocorr(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内成交量自相关系数均值'

    def calc_factor(self, date: int):
        return inday_corr_avg(date , 20 , 'vol_autocorr')
    
class inday_mkt_beta_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内beta标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'beta')
    
class inday_mkt_corr_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内量价相关性标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'corr')
    
class inday_ret_autocorr_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内收益率自相关系数标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'ret_autocorr')
    
class inday_vol_ret1_corr_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内波动价相关性标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'vol_ret1')
    
class inday_vol_vwap_corr_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内波动价相关性标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'vol_vwap')
    
class inday_vol_autocorr_std(HfCorrelationFactor):
    init_date = 20110101
    description = '20日日内成交量自相关系数标准差'

    def calc_factor(self, date: int):
        return inday_corr_std(date , 20 , 'vol_autocorr')
    
