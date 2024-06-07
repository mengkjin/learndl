import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Literal , Optional

from src.data import DataBlock
from .data_util import DATAVENDOR
from .port_util import Benchmark
from .perf_util import factor_val_breakdown , factor_mask , calc_ic , calc_grp_perf
from . import plot_util as PlotUtil

def calc_decay_ic(factor_val : DataBlock | pd.DataFrame, nday : int = 10 , lag_init : int = 2 , lag_num : int = 5 ,
                  benchmark : Optional[Benchmark] = None , ic_type : Literal['pearson' , 'spearman'] = 'pearson' , 
                  ret_type : Literal['close' , 'vwap'] = 'close'):
    '''
    nday : days of future return
    lag_init : starting lag of most recent future return , usually 1 or 2
    lag_num  : how many lagging period to calculate ic
    benchmark : within some benchmark
    '''
    decay_pnl_df = []
    factor_val = factor_mask(factor_val , benchmark)
    for lag in range(lag_num):
        decay_pnl = calc_ic(factor_val, nday , lag_init + lag * nday , ic_type = ic_type , ret_type = ret_type)
        decay_pnl_df.append(pd.DataFrame({'lag_type':f'lag{lag}','ic_mean':decay_pnl.mean()}))
    decay_pnl_df = pd.concat(decay_pnl_df, axis=0)
    return decay_pnl_df.reset_index().rename(columns={'index':'factor_name'})

def calc_decay_grp_perf(factor_val : DataBlock | pd.DataFrame, nday : int = 10 , lag_init : int = 2 , group_num : int = 10 ,
                        lag_num : int = 5 , benchmark : Optional[Benchmark] = None , ret_type : Literal['close' , 'vwap'] = 'close'):
    decay_grp_perf = []
    factor_val = factor_mask(factor_val , benchmark)
    for lag in range(lag_num):
        grp_perf = calc_grp_perf(factor_val , nday , lag_init + lag * nday , group_num , ret_type = ret_type)
        decay_grp_perf.append(pd.DataFrame({'lag_type':f'lag{lag}',**grp_perf}))
    
    decay_grp_perf = pd.concat(decay_grp_perf, axis=0).reset_index()

    n_periods  = 243 / nday
    group_cols = ['factor_name', 'group', 'lag_type']
    decay_grp_ret_mean = decay_grp_perf.groupby(group_cols,observed=False)['group_ret'].mean() * n_periods
    deacy_grp_ret_std  = decay_grp_perf.groupby(group_cols,observed=False)['group_ret'].std() * np.sqrt(n_periods)
    decay_grp_ret_ir   = decay_grp_ret_mean / deacy_grp_ret_std
    rtn = pd.concat([decay_grp_ret_mean.rename('decay_grp_ret'),decay_grp_ret_ir.rename('decay_grp_ir')], axis=1, sort=True)
    rtn = pd.DataFrame(rtn.rename_axis('stats_name', axis='columns').stack() , columns=['stats_value']).reset_index(drop=False)
    return rtn

def calc_style_corr(factor_val : DataBlock | pd.DataFrame):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    risk_style = DATAVENDOR.risk_style_exp(secid , date).to_dataframe()

    style_list , factor_list = risk_style.columns.tolist() , factor_val.columns.tolist()
    factor_style = pd.merge(factor_val, risk_style, on=['secid', 'date'], how='inner')

    factor_style_corr = factor_style.groupby('date' , observed=True).apply(
        lambda x: x.corr(method='spearman').loc[factor_list, style_list])
    factor_style_corr.index.rename(['date','factor_name'], inplace=True)
    factor_style_corr.reset_index(drop=False, inplace=True)
    return factor_style_corr

def calc_distribution(factor_val : DataBlock | pd.DataFrame , sampling_date_num : int = 20 , hist_bins : int = 50):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    date_chosen = date[::int(np.ceil(len(date) / sampling_date_num))]
    factor_list = factor_val.columns.tolist()
    factor_val_date = factor_val.reset_index(drop = False).set_index('date')
    rtn = []
    for factor_name in factor_list:
        hist_dict = {}
        for date in date_chosen:
            factor_sample = factor_val_date.loc[date, [factor_name]].copy()
            cnts, bins = np.histogram(factor_sample, bins=hist_bins, density=False)
            hist_dict[date] = (cnts, bins)
        hist_df = pd.DataFrame(hist_dict, index=['hist_cnts', 'hist_bins']).T
        hist_df['factor_name'] = factor_name
        rtn.append(hist_df)
    rtn = pd.concat(rtn, axis=0)
    rtn.index.rename('date', inplace=True)
    rtn = rtn.reset_index(drop=False).loc[:,['date' , 'factor_name', 'hist_cnts', 'hist_bins']]
    return rtn

def _calc_factor_qtile_by_day(factor : pd.DataFrame , scaling : bool = True):
    if scaling: factor = (factor - factor.mean()) / factor.std()
    rtn = pd.concat([factor.quantile(q / 100).rename(f'{q}%') for q in (5,25,50,75,95)], axis=1, sort=True)
    return rtn

def calc_factor_qtile(factor_val : pd.DataFrame , scaling : bool = True):
    factor_val , _ , _ = factor_val_breakdown(factor_val)
    rtn = factor_val.groupby(['date']).apply(_calc_factor_qtile_by_day , scaling = scaling).\
        reset_index(drop=False).rename(columns={'level_1':'factor_name'})
    return rtn

if __name__ == '__main__':
    factor_val = DATAVENDOR.random_factor().to_dataframe()
    benchmark  = Benchmark('csi300')

    g = calc_grp_perf(factor_val)
    fig = PlotUtil.plot_grp_perf(g)

    a = calc_decay_ic(factor_val)
    fig= PlotUtil.plot_decay_ic(a)

    a = calc_decay_grp_perf(factor_val)
    fig = PlotUtil.plot_decay_grp_perf(a , 'ret')
    fig = PlotUtil.plot_decay_grp_perf(a , 'ir')

    a = calc_distribution(factor_val)
    fig = PlotUtil.plot_distribution(a)

    a = calc_factor_qtile(factor_val)
    fig = PlotUtil.plot_factor_qtile(a)