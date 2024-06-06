import torch
import numpy as np
import pandas as pd

from src.data import DataBlock
from typing import Literal , Optional

from .data_util import DATAVENDOR
from .port_util import Benchmark

factor_val = DATAVENDOR.random_factor().to_dataframe()
benchmark  = Benchmark('csi300')

def factor_val_breakdown(factor_val : DataBlock | pd.DataFrame):
    if isinstance(factor_val , DataBlock):
        secid , date = factor_val.secid , factor_val.date
        factor_val = factor_val.to_dataframe()
    else:
        secid = factor_val.index.get_level_values('secid').unique().values
        date  = factor_val.index.get_level_values('date').unique().values
    return factor_val , secid , date

def calc_ic(factor_val : DataBlock | pd.DataFrame , nday : int = 10 , lag : int = 2 , 
            ic_type  : Literal['pearson' , 'spearman'] = 'pearson' ,
            ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    factor_ret = DATAVENDOR.nday_fut_ret(secid , date , nday , lag , ret_type = ret_type).to_dataframe()
    factor_list = factor_val.columns.tolist()
    factor_val = factor_val.join(factor_ret)
    ic = factor_val.groupby(by=['date'], as_index=True).apply(
        lambda x: x[factor_list].corrwith(x['ret'], method=ic_type))
    return ic

def calc_grp_avg(x : pd.DataFrame , x_column : list, y_name : str = 'ret', group_num : int = 10):
    y = pd.DataFrame(x[y_name], index=x.index, columns=[y_name])
    rtn = list()
    for x_name in x_column:
        y['group'] = pd.qcut(x[x_name], group_num, labels=[f'group{i}' for i in range(1, group_num + 1)])
        grp_avg_ret = y.groupby('group' , observed = True)[y_name].mean().rename(x_name)
        rtn.append(grp_avg_ret)
    rtn = pd.concat(rtn, axis=1, sort=True)
    return rtn

def calc_grp_perf(factor_val : DataBlock | pd.DataFrame, nday : int = 10 , lag : int = 2 , group_num : int = 10 ,
                  ret_type : Literal['close' , 'vwap'] = 'close'):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    factor_ret = DATAVENDOR.nday_fut_ret(secid , date , nday , lag , ret_type = ret_type).to_dataframe()
    factor_list = factor_val.columns.tolist()
    factor_ret = factor_ret.merge(factor_val , on = ['secid','date'])
    df = factor_ret.groupby('date').apply(calc_grp_avg , x_column = factor_list , y_name = 'ret' , group_num = group_num) # type: ignore
    return df.rename_axis('factor_name', axis='columns').stack().rename('group_ret').reset_index().sort_values(['date','group'])

def factor_mask(factor_val : DataBlock | pd.DataFrame , benchmark : Optional[Benchmark] = None):
    factor_val , secid , date = factor_val_breakdown(factor_val)
    if benchmark is None or benchmark.name is None: return factor_val
    factor_list = factor_val.columns.to_list()
    univ = benchmark.universe(secid , date).to_dataframe()
    factor_val = factor_val.join(univ)
    factor_val.loc[~factor_val['universe'] , factor_list] = np.nan
    del factor_val['universe']
    return factor_val.dropna()