from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'ta_yoy' , 'ta_qoq' , 
    'equ_yoy' , 'equ_qoq' , 
    'liab_yoy' , 'liab_qoq' , 
    'gp_qoq' , 'gp_yoy' , 'gp_ttm_yoy' , 
    'gp_margin_yoy' , 'gp_margin_ttm_yoy' ,
    'dedt_equ_yoy' , 'dedt_equ_ttm_yoy' , 
    'dedt_qoq' , 'dedt_yoy' , 'dedt_ttm_yoy' , 
    'oper_margin_yoy' , 'oper_margin_ttm_yoy' , 
    'npro_qoq' , 'npro_yoy' , 'npro_ttm_yoy' , 
    'roe_yoy' , 'roe_ttm_yoy' , 
    'roa_yoy' , 'roa_ttm_yoy' , 
    'net_margin_yoy' , 'net_margin_ttm_yoy' , 
    'sales_qoq' , 'sales_yoy' , 'sales_ttm_yoy' , 
    'assetsturn_yoy' , 'assetsturn_ttm_yoy' , 
    'tax_qoq' , 'tax_yoy' , 'tax_ttm_yoy' , 
    'cfo_qoq' , 'cfo_yoy' , 'cfo_ttm_yoy'
]

def get_yoy_latest(expression : str , date : int):
    return DATAVENDOR.get_fin_yoy(expression , date , 4).dropna().groupby('secid').last().iloc[:,0]

def get_qoq_latest(expression : str , date : int):
    return DATAVENDOR.get_fin_qoq(expression , date , 4).dropna().groupby('secid').last().iloc[:,0]

class ta_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '总资产同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('ta@qtr' , date)
    
class ta_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '总资产环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('ta@qtr' , date)
    
class equ_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '净资产同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('equ@qtr' , date)
    
class equ_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '净资产环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('equ@qtr' , date)
    
class liab_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '总负债同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('liab@qtr' , date)

class liab_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '总负债环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('liab@qtr' , date)

class gp_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度毛利润环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('gp@qtr' , date)

class gp_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度毛利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp@qtr' , date)

class gp_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM毛利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp@ttm' , date)
    
class gp_margin_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度毛利润/营业收入同比变化值'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp@qtr / sales@qtr' , date)
    
class gp_margin_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM毛利润/营业收入同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp@ttm / sales@ttm' , date)
    
class dedt_equ_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度扣非归母净利润/净资产同比变化值'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('dedt@qtr/equ@qtr' , date)
    
class dedt_equ_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM扣非归母净利润/净资产同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('dedt@ttm/equ@ttm' , date)

class dedt_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度扣非归母净利润环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('dedt@qtr' , date)

class dedt_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度扣非归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('dedt@qtr' , date)
    
class dedt_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM扣非归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('dedt@ttm' , date)
    
class oper_margin_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度营业利润率/营业收入同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('oper_np@qtr / sales@qtr' , date)

class oper_margin_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM营业利润率/营业收入同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('oper_np@ttm / sales@ttm' , date)

class npro_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度归母净利润环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('npro@qtr' , date)
    
class npro_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@qtr' , date)

class npro_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@ttm' , date)

class roe_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度净资产收益率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@qtr / equ@qtr' , date)
    
class roe_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM净资产收益率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@ttm / equ@ttm' , date)

class roa_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度总资产收益率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@qtr / ta@qtr' , date)
    
class roa_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM总资产收益率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@ttm / ta@ttm' , date)

class net_margin_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度净利率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@qtr / sales@qtr' , date)

class net_margin_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM净利率同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('npro@ttm / sales@ttm' , date)

class sales_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度营业收入环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('sales@qtr' , date)
    
class sales_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度营业收入同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('sales@qtr' , date)

class sales_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM营业收入同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('sales@ttm' , date)

class assetsturn_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度营业收入/总资产(周转率)同比变化值'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('sales@qtr / ta@qtr' , date)

class assetsturn_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM营业收入/总资产(周转率)同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('sales@ttm / ta@ttm' , date)

class tax_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度所得税环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('tax@qtr' , date)
class tax_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM所得税同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('tax@qtr' , date)
    
class tax_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM所得税同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('tax@ttm' , date) 
    
class cfo_qoq(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度经营活动现金流环比变化率'
    
    def calc_factor(self, date: int):
        return get_qoq_latest('ncfo@qtr' , date)
    
class cfo_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = '单季度经营活动现金流同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('ncfo@qtr' , date)
    
class cfo_ttm_yoy(StockFactorCalculator):
    init_date = 20110101
    category1 = 'growth'
    description = 'TTM经营活动现金流同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('ncfo@ttm' , date)

