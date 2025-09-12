from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'eps_yoy_zscore' , 'sales_yoy_zscore' , 'gp_yoy_zscore' , 'npro_yoy_zscore' , 
    'dedt_yoy_zscore' , 'tax_yoy_zscore' , 'roe_yoy_zscore' , 
    'gp_margin_yoy_zscore' , 'oper_margin_yoy_zscore' , 'net_margin_yoy_zscore' ,
    'cfo_yoy_zscore'
]

def get_yoy_zscore(expression : str , date : int , n_last : int = 20 , **kwargs):
    df = DATAVENDOR.get_fin_hist(expression , date , n_last , pivot = False , **kwargs).iloc[:,0]
    grp = df.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

class eps_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM每股收益行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('eps@ttm' , date)

class sales_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('sales@ttm' , date)
    
class gp_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('gp@ttm' , date)
    
class npro_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('npro@ttm' , date)
    
class dedt_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM扣非归母净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('dedt@ttm' , date)
    
class tax_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('tax@ttm' , date)

class roe_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净资产收益率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('npro@ttm / equ@ttm' , date)
    
class gp_margin_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('gp@ttm / sales@ttm' , date)

class oper_margin_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM营业利润率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('oper_np@ttm / sales@ttm' , date)

class net_margin_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM净利率行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('npro@ttm / sales@ttm' , date)
    
class cfo_yoy_zscore(StockFactorCalculator):
    init_date = 20110101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM经营活动现金流行业内分位数之差'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('ncfo@ttm' , date)

