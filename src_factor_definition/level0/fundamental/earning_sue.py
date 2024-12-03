import pandas as pd
import numpy as np

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

class sue_gp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润'
    
    def calc_factor(self, date: int):
        ...

class sue_gp_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外毛利润-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class sue_npro(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润'
    
    def calc_factor(self, date: int):
        ...

class sue_npro_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外归母净利润-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class sue_op(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润'
    
    def calc_factor(self, date: int):
        ...

class sue_op_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业利润-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class sue_tp(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额'
    
    def calc_factor(self, date: int):
        ...

class sue_tp_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外利润总额-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class sue_sales(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入'
    
    def calc_factor(self, date: int):
        ...

class sue_sales_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外营业收入-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class sue_tax(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税'
    
    def calc_factor(self, date: int):
        ...

class sue_tax_reg(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = '预期外所得税-带截距回归'
    
    def calc_factor(self, date: int):
        ...

class tax_equ(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'earning'
    description = 'TTM所得税/净资产'
    
    def calc_factor(self, date: int):
        ...