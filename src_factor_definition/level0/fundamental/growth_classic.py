import numpy as np
import pandas as pd
import statsmodels.api as sm
import polars as pl

from typing import Literal

from src.factor.calculator import StockFactorCalculator
from src.data import DATAVENDOR

def get_yoy_latest(numerator : str , date : int , yoy_method : Literal['ttm' , 'acc' , 'qtr'] | None = None , **kwargs):
    '''statement@field@fin_type'''
    numerator = f'{numerator}@yoy'
    return DATAVENDOR.get_fin_latest(numerator , date , yoy_method = yoy_method , **kwargs)

def get_compound_growth(numerator: str , date: int , n_year : int = 5 , **kwargs):
    '''cannot deal with < -100% growth compounding, use simple instead'''
    df = DATAVENDOR.get_fin_hist(f'{numerator}@ttm' , date , 4*n_year + 1 , pivot = False).iloc[:,0].reset_index('end_date',drop=False)
    df = pd.concat([df.groupby('secid').first() , df.groupby('secid').last()], axis=0)
    val = df.columns[-1]
    
    df['qtrs'] = (df['end_date'] // 10000) * 4 + df['end_date'] % 10000 // 300 
    df = df.set_index('end_date',append=True).sort_index()

    # df = (df.groupby('secid')[val].pct_change() + 1) ** (4 / df.groupby('secid')['qtrs'].diff()) - 1
    df = df.groupby('secid')[val].pct_change() * 4 / df.groupby('secid')['qtrs'].diff()
    return df.groupby('secid').last()

def get_reg_growth(numerator: str , date: int , n_year : int = 5 , **kwargs):
    def _std_beta(args) -> pl.Series:
        y = args[0].to_numpy()[::-4][::-1]
        x = np.arange(1, len(y) + 1)
        try:
            v = sm.OLS(y, sm.add_constant(x)).fit().params[1] / y.mean()
            return pl.Series([v], dtype=pl.Float64)
        except Exception as e:
            return pl.Series([np.nan], dtype=pl.Float64)
    
    y_var = DATAVENDOR.get_fin_hist(f'{numerator}@ttm' , date , n_year * 4 + 1 , pivot = False ,**kwargs).iloc[:,0]
    y_name = str(y_var.name)
    df = pl.from_pandas(y_var.to_frame() , include_index=True)
    df = df.with_columns([
        (pl.col(y_name) / pl.col(y_name).mean().over('secid')).alias(y_name),
    ]).with_columns(
        pl.when(pl.col(y_name).is_infinite()).then(0).otherwise(pl.col(y_name)).alias(y_name),
    ).drop_nulls()

    df = df.sort(['secid','end_date']).group_by('secid', maintain_order=True).\
        agg(pl.map_groups(exprs=[y_name], function=_std_beta)).to_pandas().set_index('secid').iloc[:,0]
    return df

def get_yoy_zscore(numerator : str , date : int , n_last : int = 20 , **kwargs):
    df = DATAVENDOR.get_fin_hist(f'{numerator}@yoy' , date , n_last , pivot = False , **kwargs).iloc[:,0]
    grp = df.groupby('secid')
    return (grp.last() - grp.mean()) / grp.std()

def calc_yoy(data : pd.Series):
    full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                             data.index.get_level_values('end_date').unique()])
    df_yoy = data.reindex(full_index)
    df_yoy_base = df_yoy.groupby('secid').shift(4)
    df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()

    df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
    return df_yoy

def calc_qoq(data : pd.Series):
    full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                             data.index.get_level_values('end_date').unique()])
    df_yoy = data.reindex(full_index)
    df_yoy_base = df_yoy.groupby('secid').shift(1)
    df_yoy = (df_yoy - df_yoy_base) / df_yoy_base.abs()

    df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
    return df_yoy

def calc_yoy_latest(data : pd.Series):
    return calc_yoy(data).dropna().groupby('secid').last()

def calc_qoq_latest(data : pd.Series):
    return calc_qoq(data).dropna().groupby('secid').last()

def calc_trend(data : pd.Series):
    def _trend(args) -> pl.Series:
        y = args[0].to_numpy()
        x = np.arange(1, len(y) + 1)
        try:
            v = sm.OLS(y, sm.add_constant(x)).fit().params[1] / y.mean()
            return pl.Series([v], dtype=pl.Float64)
        except Exception as e:
            return pl.Series([np.nan], dtype=pl.Float64)
    if not data.name: data = data.rename('data')
    y_name = str(data.name)
    df = pl.from_pandas(data.to_frame() , include_index=True)
    df = df.with_columns(
        pl.when(pl.col(y_name).is_infinite()).then(0).otherwise(pl.col(y_name)).alias(y_name),
    ).drop_nulls()

    df = df.sort(['secid','end_date']).group_by('secid', maintain_order=True).\
        agg(pl.map_groups(exprs=[y_name], function=_trend)).to_pandas().set_index('secid').iloc[:,0]
    return df

class ta_yoy(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总资产同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('ta' , date)

class ta_gro5y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-总资产'
    
    def calc_factor(self, date: int):
        return get_compound_growth('ta' , date)

class equ_yoy(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净资产同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('equ' , date)

class equ_gro5y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-净资产'
    
    def calc_factor(self, date: int):
        return get_compound_growth('equ' , date)

class sales_gro5y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-营业收入'
    
    def calc_factor(self, date: int):
        return get_compound_growth('sales' , date)

class npro_gro5y(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '5年年度增长率-归母净利润'
    
    def calc_factor(self, date: int):
        return get_compound_growth('npro' , date)

class gp_yoy_zscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'gp单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        return get_yoy_zscore('gp' , date , qtr_method = 'diff' , yoy_method = 'qtr')

class gp_qtr_yoy(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '单季度毛利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp' , date , qtr_method = 'diff' , yoy_method = 'qtr')

class gp_ttm_yoy(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'TTM毛利润同比变化率'
    
    def calc_factor(self, date: int):
        return get_yoy_latest('gp' , date , qtr_method = 'diff' , yoy_method = 'ttm')

class gp_ta_qtr_qoq(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润/总资产环比变化值'
    
    def calc_factor(self, date: int):
        gp_ta_qtr = (DATAVENDOR.get_fin_hist('gp@qtr' , date , 20 , qtr_method = 'diff').iloc[:,0] / 
                     DATAVENDOR.get_fin_hist('ta@qtr' , date , 20).iloc[:,0])
        return calc_qoq_latest(gp_ta_qtr)
class gp_ta_qtr_yoy(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润/总资产同比变化值'
    
    def calc_factor(self, date: int):
        gp_ta_qtr = (DATAVENDOR.get_fin_hist('gp@qtr' , date , 20 , qtr_method = 'diff').iloc[:,0] / 
                     DATAVENDOR.get_fin_hist('ta@qtr' , date , 20).iloc[:,0])
        return calc_yoy_latest(gp_ta_qtr)

class gp_ttm_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润TTM行业内分位数之差'
    
    def calc_factor(self, date: int):
        df = DATAVENDOR.get_fin_hist('gp@ttm' , date , 5 , qtr_method = 'diff')
        df = DATAVENDOR.INFO.add_indus(df , date , 'unknown')

        df['gross_margin'] = df.groupby(['end_date' , 'indus'])['gross_margin'].rank(pct=True)
        df = df.drop(columns = ['indus'])
        return (df - df.groupby('secid').shift(4)).dropna().groupby('secid').last().iloc[:,0]

class gp_ta_qoq_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润/总资产环比变化趋势'
    
    def calc_factor(self, date: int):
        gp_ta_qtr = (DATAVENDOR.get_fin_hist('gp@qtr' , date , 20 , qtr_method = 'diff').iloc[:,0] / 
                     DATAVENDOR.get_fin_hist('ta@qtr' , date , 20).iloc[:,0])
        qoq = calc_qoq(gp_ta_qtr)
        return calc_trend(qoq)

class gp_ta_yoy_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '毛利润/总资产同比变化趋势'
    
    def calc_factor(self, date: int):
        gp_ta_qtr = (DATAVENDOR.get_fin_hist('gp@qtr' , date , 20 , qtr_method = 'diff').iloc[:,0] / 
                     DATAVENDOR.get_fin_hist('ta@qtr' , date , 20).iloc[:,0])
        yoy = calc_yoy(gp_ta_qtr)
        return calc_trend(yoy)

class liab_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '总负债同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '扣非归母净利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_dedu_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_dedu_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...

class npro_dedu_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '扣非归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_op_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_op_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_op单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class npro_op_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_op_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_op_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...

class npro_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '归母净利润同比变化率'
    
    def calc_factor(self, date: int):
        ...

class npro_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_q_ta同比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '净利润行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_ta环比变化值'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额加速度'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'npro_tp单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额同比变化率'
    
    def calc_factor(self, date: int):
        ...


class npro_tp_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '利润总额行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class npro_trend(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '业绩趋势因子'
    
    def calc_factor(self, date: int):
        ...


class op_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'op_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class op_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'op_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...


class sales_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入加速度'
    
    def calc_factor(self, date: int):
        ...


class sales_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sales单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...


class sales_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入同比变化率'
    
    def calc_factor(self, date: int):
        ...


class sales_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入同比变化率'
    
    def calc_factor(self, date: int):
        ...


class sales_q_ta_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sale_q_ta同比变化值'
    
    def calc_factor(self, date: int):
        ...


class sales_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '营业收入行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class sales_ta_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'sale_ta环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_acce(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税加速度'
    
    def calc_factor(self, date: int):
        ...


class tax_czscore(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax单季度同比增速的标准化得分'
    
    def calc_factor(self, date: int):
        ...

class tax_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税同比变化率'
    
    def calc_factor(self, date: int):
        ...


class tax_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tax_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...


class tax_q_pctq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税同比变化率'
    
    def calc_factor(self, date: int):
        ...

class tax_rank_delta(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = '所得税行业内分位数之差'
    
    def calc_factor(self, date: int):
        ...


class tp_equ_chgq1(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tp_equ环比变化值'
    
    def calc_factor(self, date: int):
        ...


class tp_q_equ_chgq4(StockFactorCalculator):
    init_date = 20070101
    category0 = 'fundamental'
    category1 = 'growth'
    description = 'tp_q_equ同比变化值'
    
    def calc_factor(self, date: int):
        ...
