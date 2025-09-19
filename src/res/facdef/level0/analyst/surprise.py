import pandas as pd
import numpy as np
from typing import Any , Literal

from src.data import DATAVENDOR
from src.res.factor.calculator import StockFactorCalculator


__all__ = [
    'outperform_title' , 'outperform_titlepct' , 'outperform_npro' , 'outperform_sales'
]

def get_title_outperform(date : int , type : Literal['num' , 'pct']):
    target_quarter = f'{date // 10000}Q4'
    n_month = 12
    within_ann_days = 7
    start_date = DATAVENDOR.CALENDAR.cd(date , -30 * n_month)


    df = DATAVENDOR.ANALYST.get_trailing_reports(date , n_month + 6)
    df = df[(df['quarter'] == target_quarter) & (df['report_date'] >= start_date)].set_index(['secid','report_date']).sort_index()

    if within_ann_days is not None:
        assert within_ann_days > 0 , 'within_ann_days must be positive'
        ann_cal = DATAVENDOR.IS.get_ann_calendar(date , after_days = within_ann_days , within_days = n_month * 30 + 5).\
            rename_axis(index = {'date':'report_date'}).reindex(df.index)
        df = df[ann_cal['anndt'] > 0]

    title : pd.Series | Any = df['report_title']
    df['title_outperform'] = (title.str.contains('预期') & 
                              (title.str.contains('超') | 
                               title.str.contains('好于')))
    if type == 'num':
        df = df.groupby(['secid' , 'org_name']).last().groupby(['secid'])['title_outperform'].sum()
    elif type == 'pct':
        df = df.groupby(['secid' , 'org_name']).last().groupby(['secid'])['title_outperform'].mean()
    return df    

def get_profit_outperform(date : int , val : Literal['tp' , 'npro' , 'sales' , 'op']):
    rp_col = {'tp':'tp' , 'npro':'np' , 'sales':'op_rt' , 'op':'op_pr'}[val]
    is_col = {'tp':'total_np@acc' , 'npro':'npro@acc' , 'sales':'sales@acc' , 'op':'oper_np@acc'}[val]

    anndt = DATAVENDOR.IS.get_ann_dt(date , 0).groupby('secid').tail(2)
    profit = DATAVENDOR.get_fin_hist(is_col , date , 2, new_name='profit').reset_index()

    anndt['num'] = anndt.groupby('secid').cumcount().astype(str)
    anndt = anndt.pivot_table(index = 'secid' , columns = 'num' , values = 'ann_date').dropna().astype(int)
    anndt.columns = ['anndt_1' , 'anndt_2']

    profit['num'] = profit.groupby('secid').cumcount().astype(str)

    end_date = profit.pivot_table(index = 'secid' , columns = 'num' , values = 'end_date').dropna().astype(int)
    qtr = end_date % 10000 // 300
    qtr.columns = ['qtr_1' , 'qtr_2']
    qtr['target_quarter'] = ((end_date // 10000).iloc[:,-1].astype(str) + 'Q4')

    profit = profit.pivot_table(index = 'secid' , columns = 'num' , values = 'profit')
    profit.columns = ['profit_1' , 'profit_2']

    df = DATAVENDOR.ANALYST.get_trailing_reports(date , 12)

    df = df.set_index(['secid']).merge(anndt , on = 'secid').merge(qtr , on = 'secid').merge(profit , on = 'secid')
    df = df[(df['report_date'] >= df['anndt_1']) & (df['report_date'] < df['anndt_2']) & (df['quarter'] == df['target_quarter'])]

    df['est_profit'] = (df[rp_col] * 1e4 - np.where(df['qtr_1'] < 4 , df['profit_1'], 0)) / (4 - np.where(df['qtr_1'] < 4 , df['qtr_1'], 0))
    df['ann_profit'] = df['profit_2'] - np.where(df['qtr_1'] < 4 , df['profit_1'], 0)

    df = df.loc[:,['report_date' , 'org_name' , 'quarter' , rp_col , 'qtr_1' , 'qtr_2' , 'profit_1' , 'profit_2' , 'est_profit' , 'ann_profit']]
    df['outperform_pct'] = (df['ann_profit'] - df['est_profit']) / df[rp_col].abs() / 1e4

    df = df.sort_values(['secid' , 'report_date']).groupby(['secid' , 'org_name']).last()
    return df.groupby('secid')['outperform_pct'].mean()


class outperform_title(StockFactorCalculator):
    init_date = 20110101
    category1 = 'surprise'
    description = '研报标题超预期个数'

    def calc_factor(self, date: int):
        return get_title_outperform(date , type = 'num')
    
class outperform_titlepct(StockFactorCalculator):
    init_date = 20110101
    category1 = 'surprise'
    description = '研报标题超预期比例'

    def calc_factor(self, date: int):
        return get_title_outperform(date , type = 'pct')
    
class outperform_npro(StockFactorCalculator):
    init_date = 20110101
    category1 = 'surprise'
    description = '单季度净利润超预期幅度'

    def calc_factor(self, date: int):
        return get_profit_outperform(date , val = 'npro')
    
class outperform_sales(StockFactorCalculator):
    init_date = 20110101
    category1 = 'surprise'
    description = '单季度营业收入超预期幅度'

    def calc_factor(self, date: int):
        return get_profit_outperform(date , val = 'sales')