"""
Style factors for risk level0
"""
from __future__ import annotations
from src.res.factor.calculator.factor_calc import StyleFactor

__all__ = ['risk_lncap' , 'risk_beta']

class risk_lncap(StyleFactor):
    description = '风险因子: 市值(对数)'
    load_db_col = 'size'

    def calc_factor(self , date : int):
        return self.load_factor(date)

class risk_beta(StyleFactor):
    description = '风险因子: Beta'
    load_db_col = 'beta'

    def calc_factor(self , date : int):
        return self.load_factor(date)