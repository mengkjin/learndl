import pandas as pd
import numpy as np
import polars as pl

from typing import Any , Literal

from src.basic import PATH , CALENDAR
from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator

from src.data.util import DFCollection

RawHoldings = DFCollection(40 , 'end_date')
ActiveFundHoldings = DFCollection(10 , 'end_date')
ActiveTopHoldings = DFCollection(10 , 'end_date')
FundInfo = PATH.db_load('information_ts' , 'mutual_fund_info').\
    loc[:,['fund_id' , 'name' , 'fund_type' , 'list_date' , 'delist_date' , 'invest_type' , 'type' , 'market']]

__all__ = [
    'holding_act_weight' , 'holding_median' , 'holding_num' , 'holding_mv' , 'holding_rel_weight' , 'holding_rel_median' , 
    'holding_top_median' , 'holding_top_num' , 'holding_top_mv' , 'holding_top_addnum' , 'holding_top_addmv'
]

def full_holding_date(date):
    year = date // 10000
    month = date // 100 % 100

    if month < 3:
        return (year - 1) * 10000 + 630
    elif month < 9:
        return (year - 1) * 10000 + 1231
    else:
        return year * 10000 + 630

def top_holding_date(date):
    year = date // 10000
    month = date // 100 % 100
    if month < 2:
        return (year - 1) * 10000 + 930
    elif month < 5:
        return (year - 1) * 10000 + 1231
    elif month < 8:
        return year * 10000 + 331
    elif month < 11:
        return year * 10000 + 630
    else:
        return year * 10000 + 930
    
def one_quater_ago(date):
    return CALENDAR.cd(date , -90)

def get_holding(qtr : int):
    if qtr in RawHoldings:
        return RawHoldings.get(qtr)
    df = PATH.db_load('holding_ts' , 'mutual_fund' , qtr)
    df = df.groupby(['end_date','fund_id','secid','symbol']).sum().reset_index()
    df = df.sort_values(by=['fund_id', 'mkv'], ascending=[True, False])
    df['rank'] = df.groupby('fund_id').cumcount()
    RawHoldings.add(qtr , df)
    return df

def filter_holding(df : pd.DataFrame , ann_date : int | None = None , active = False , top_only = False , ashare_only = True):
    if ann_date:
        df = df[df['ann_date'] <= ann_date]
    if active:
        df = df[df['fund_id'].isin(get_fund_info(ann_date , active)['fund_id'])]
    if top_only:
        df = df[df['rank'] < 10]
    if ashare_only:
        df = df[df['secid'] >= 0]
    return df

def get_fund_info(date : int | None = None , active = True):
    finfo : pd.DataFrame = FundInfo
    if date:
        finfo = finfo[(finfo['list_date'] <= date) & (finfo['delist_date'] > date)]
    if active:
        fund_type_in = ['混合型','股票型']
        invest_type_out = ['增强指数型', '被动指数型']
        finfo = finfo[
            ~finfo['name'].str.contains('FOF') &
            ~finfo['name'].str.contains('量化') &
            ~finfo['name'].str.contains('增强') &
            finfo['fund_type'].isin(fund_type_in) &
            ~finfo['invest_type'].isin(invest_type_out)
        ]
    return finfo

def get_active_fund_holding(date : int):
    if date in ActiveFundHoldings:
        return ActiveFundHoldings.get(date).reset_index(drop = True)
    full_date = full_holding_date(date)
    top_date = top_holding_date(date)
    full_port = filter_holding(get_holding(full_date) , date , active = True , top_only = False , ashare_only = True).\
        reset_index(drop = True).loc[:,['fund_id' , 'secid' , 'mkv','amount','stk_mkv_ratio','stk_float_ratio']]
    if top_date > full_date:
        top_port  = filter_holding(get_holding(top_date)  , date , active = True , top_only = True , ashare_only = True).\
            reset_index(drop = True).loc[:,['fund_id' , 'secid' , 'mkv','amount','stk_mkv_ratio','stk_float_ratio']]
        full_port = pd.concat([top_port , full_port] , axis = 0).drop_duplicates(['fund_id' , 'secid'])
    ActiveFundHoldings.add(date , full_port)
    return full_port

def get_active_top_holding(date : int):
    if date in ActiveTopHoldings:
        return ActiveTopHoldings.get(date).reset_index(drop = True)
    top_date = top_holding_date(date)
    top_port  = filter_holding(get_holding(top_date)  , date , active = True , top_only = True , ashare_only = True).\
        reset_index(drop = True).loc[:,['fund_id' , 'secid' , 'mkv','amount','stk_mkv_ratio','stk_float_ratio']]
    ActiveTopHoldings.add(date , top_port)
    return top_port

def get_mkt_port(date : int) -> pd.Series:
    mv = DATAVENDOR.TRADE.get_val(date).loc[:,['secid' , 'circ_mv']]
    mkt_port = mv.groupby('secid')['circ_mv'].sum()
    mkt_port = mkt_port / mkt_port.sum()
    return mkt_port

class holding_act_weight(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金持股市值占比-市场市值占比'

    def calc_factor(self, date: int):
        mkt_port = get_mkt_port(date)
        full_port = get_active_fund_holding(date)
        fund_port = full_port.groupby('secid')['mkv'].sum() / full_port['mkv'].sum()
        return (fund_port - mkt_port).dropna()
    
class holding_median(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金内个股权重中位数'

    def calc_factor(self, date: int):
        full_port = get_active_fund_holding(date)
        return full_port.groupby('secid')['stk_mkv_ratio'].median() / 100
    
class holding_mv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股市值'

    def calc_factor(self, date: int):
        full_port = get_active_fund_holding(date)
        return full_port.groupby('secid')['mkv'].sum() / 10**8
    
class holding_num(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股基金数量'

    def calc_factor(self, date: int):
        full_port = get_active_fund_holding(date)
        return full_port.groupby('secid')['fund_id'].count()
    
class holding_mv_ratio(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '持有个股股数占股本比例'

    def calc_factor(self, date: int):
        full_port = get_active_fund_holding(date)
        return full_port.groupby('secid')['stk_float_ratio'].sum()
    
class holding_rel_weight(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金持股市值占比/市场市值占比'

    def calc_factor(self, date: int):
        mkt_port = get_mkt_port(date)
        full_port = get_active_fund_holding(date)
        fund_port = full_port.groupby('secid')['mkv'].sum() / full_port['mkv'].sum()
        return (fund_port / mkt_port).dropna()
    
class holding_rel_median(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '基金内个股主动权重中位数'

    def calc_factor(self, date: int):
        full_port = get_active_fund_holding(date)
        mkt_port = get_mkt_port(date).rename('mkt_port') * 100
        full_port = full_port.set_index('secid').join(mkt_port)
        full_port['stk_mkv_ratio'] = full_port['stk_mkv_ratio'] / full_port['mkt_port']
        return full_port.groupby('secid')['stk_mkv_ratio'].median().sort_index()
    
class holding_top_median(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '重仓股基金个股权重中位数'

    def calc_factor(self, date: int):
        top_port = get_active_top_holding(date)
        return top_port.groupby('secid')['stk_mkv_ratio'].median() / 100
    
class holding_top_mv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '重仓股持有个股市值'

    def calc_factor(self, date: int):
        top_port = get_active_top_holding(date)
        return top_port.groupby('secid')['mkv'].sum() / 10**8
 
class holding_top_num(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '重仓股持有个股基金数量'

    def calc_factor(self, date: int):
        top_port = get_active_top_holding(date)
        return top_port.groupby('secid')['fund_id'].count()
    
class holding_top_addnum(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '重仓股持有个股基金数量增量'

    def calc_factor(self, date: int):
        top_port = get_active_top_holding(date)
        top_port_0 = get_active_top_holding(one_quater_ago(date))
        return top_port.groupby('secid')['fund_id'].count() - top_port_0.groupby('secid')['fund_id'].count()
    
class holding_top_addmv(StockFactorCalculator):
    init_date = 20110101
    category0 = 'money_flow'
    category1 = 'holding'
    description = '重仓股持有个股市值增量'

    def calc_factor(self, date: int):
        top_port = get_active_top_holding(date)
        top_port_0 = get_active_top_holding(one_quater_ago(date))
        return top_port.groupby('secid')['mkv'].sum() / 10**8 - top_port_0.groupby('secid')['mkv'].sum() / 10**8
    