from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'assetcur_asset' , 'liab_ta' , 'liabcur_liab' , 
    'ratio_cash' , 'ratio_current' , 'ratio_quick' ,
    'assetsturn_qtr' , 'assetsturn_ttm' ,
    'ta_equ' , 
    'npro_tp_qtr' , 'npro_tp_ttm' ,
    'oper_total_qtr' , 'dedt_npro_qtr' ,
    'cfo_cf_qtr' , 'net_cfo_ratio'
]

class assetcur_asset(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '流动资产/总资产'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('ca_to_assets' , date) / 100
    
class liab_equ(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '产权比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('debt_to_eqt' , date)
    
class liab_ta(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '资产负债率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('debt_to_assets' , date) / 100
    
class liabcur_liab(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '流动负债/总负债'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('currentdebt_to_debt' , date) / 100
    
class ratio_cash(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '现金比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('cash_ratio' , date) / 100
    
class ratio_current(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '流动比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('current_ratio' , date) / 100
    
class ratio_quick(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '速动比率'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('quick_ratio' , date) / 100

class assetsturn_qtr(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '单季度资产周转率'
    
    def calc_factor(self, date: int):
        sales = DATAVENDOR.IS.qtr_latest('revenue' , date)
        ta    = DATAVENDOR.BS.qtr_latest('total_assets' , date)
        return sales / ta
    
class assetsturn_ttm(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = 'TTM资产周转率'
    
    def calc_factor(self, date: int):
        sales = DATAVENDOR.IS.ttm_latest('revenue' , date)
        ta    = DATAVENDOR.BS.ttm_latest('total_assets' , date)
        return sales / ta
class ta_equ(StockFactorCalculator):
    init_date = 20110101
    category1 = 'quality'
    description = '权益乘数'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.INDI.acc_latest('assets_to_eqt' , date) / 100
    
class npro_tp_qtr(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = '单季度归母净利润/利润总额'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@qtr / total_np@qtr' , date)

class npro_tp_ttm(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = 'TTM归母净利润/利润总额'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@ttm / total_np@ttm' , date)

class oper_total_qtr(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = '单季度营业利润/营业收入(营业利润率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('oper_np@qtr / total_np@qtr' , date)

class dedt_npro_qtr(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = '单季度净利润/营业收入(净利润率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('npro@qtr / sales@qtr' , date)
    
class cfo_cf_qtr(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = '单季度经营活动现金流/营业收入(经营活动现金流率)'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('incfo@qtr / (incfo@qtr + incff@qtr + incfi@qtr)' , date)
    
class net_cfo_ratio(StockFactorCalculator):
    init_date = 20110101
    category1 = 'earning'
    description = '单季度经营活动净额占比'
    
    def calc_factor(self, date: int):
        return DATAVENDOR.get_fin_latest('ncfo@qtr / incfo@qtr' , date)

