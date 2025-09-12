import pandas as pd
import numpy as np

from typing import Any , Literal

from src.data import DATAVENDOR

def eval_cum_ret(ret : pd.Series | pd.DataFrame | np.ndarray , how : Literal['exp' , 'lin'] = 'lin' , 
            groupby : str | list[str] | None = None):
    assert not isinstance(ret , np.ndarray) or groupby is None , 'groupby is not supported for numpy array'
    if isinstance(ret , np.ndarray): 
        ret = pd.Series(ret)
    ret = ret.fillna(0)
    if groupby is None:
        if how == 'lin':
            return ret.cumsum()
        else:
            return (ret + 1).cumprod() - 1
    else:
        if how == 'lin':
            return ret.groupby(groupby , observed=True).cumsum()
        else:
            return (ret + 1).groupby(groupby , observed=True).cumprod() - 1

def eval_cum_peak(ret : pd.Series | pd.DataFrame | np.ndarray , how : Literal['exp' , 'lin'] = 'lin' ,
                  groupby : str | list[str] | None = None):
    cum_ret = eval_cum_ret(ret , how , groupby)
    return cum_ret.cummax() if groupby is None else cum_ret.groupby(groupby , observed=True).cummax()

def eval_drawdown(ret : pd.DataFrame | pd.Series | np.ndarray , how : Literal['exp' , 'lin'] = 'lin' , 
                  groupby : str | list[str] | None = None):
    cum_ret = eval_cum_ret(ret , how , groupby)
    cum_peak = cum_ret.cummax() if groupby is None else cum_ret.groupby(groupby , observed=True).cummax()
    if how == 'lin':
        return cum_ret - cum_peak
    else:
        return (cum_ret + 1) / (cum_peak + 1) - 1

def eval_drawdown_start(ret : pd.DataFrame | pd.Series | np.ndarray , how : Literal['exp' , 'lin'] = 'lin' , 
                        groupby : str | list[str] | None = None):
    if groupby is None:
        cum = eval_cum_ret(ret , how , groupby)
        return cum.expanding().apply(lambda x: x.argmax(), raw=True).astype(int)
    else:
        assert not isinstance(ret , np.ndarray) , 'groupby is not supported for numpy array'
        return ret.groupby(groupby , observed=True).apply(eval_drawdown_start , how = how)

def eval_max_drawdown(ret : pd.Series | np.ndarray | pd.DataFrame , how : Literal['exp' , 'lin'] = 'lin'):
    dd , st = eval_drawdown(ret , how) , eval_drawdown_start(ret , how)
    assert isinstance(dd , pd.Series) and isinstance(st , pd.Series) , 'dd and st must be pd.Series'
    mdd = -dd.min()
    idx_ed = int(dd.argmin())
    idx_st = int(st.iloc[idx_ed])

    return mdd , idx_st , idx_ed

def eval_pf_stats(grp : pd.DataFrame , mdd_period = True , **kwargs):
    period_len = abs(DATAVENDOR.CALENDAR.cd_diff(grp['start'].min() , grp['end'].max())) + 1
    period_n   = len(grp)

    with np.errstate(divide = 'ignore' , invalid = 'ignore'):
        pf_ret = np.prod(grp['pf'] + 1) - 1.
        bm_ret = np.prod(grp['bm'] + 1) - 1.
        excess = (pf_ret - bm_ret)
        ex_ann = np.power(np.prod(1 + grp['excess']) , 365 / period_len) - 1
        # pf_mdd = eval_max_drawdown(grp['pf'] , 'exp')
        ex_mdd , ex_mdd_st , ex_mdd_ed = eval_max_drawdown(grp['excess'] , 'lin')
        te     = np.std(grp['excess']) * np.sqrt(365 * period_n / period_len)
        ex_ir  = ex_ann / te
        ex_calmar = ex_ann / ex_mdd
        turn   = np.sum(grp['turn'])
        rslt = pd.DataFrame({'pf':pf_ret , 'bm':bm_ret , 'excess' : excess , 'annualized' : ex_ann , 'mdd' : ex_mdd , 
                             'te' : te , 'ir' : ex_ir , 'calmar' : ex_calmar , 'turnover' : turn} , index = pd.Index([0]))
    if mdd_period:
        rslt['mdd_period'] = ['{}-{}'.format(grp['end'].iloc[ex_mdd_st] , grp['end'].iloc[ex_mdd_ed])]
    return rslt.assign(**kwargs)

def eval_uncovered_max_drawdown(dd : pd.Series | Any , groupby : str | list[str] | None = None):
    if groupby is None:
        umd = dd * 0.
        umd.iloc[0] = dd.iloc[0]
        for i , d in enumerate(dd[1:] , 1):
            if d < 0:
                umd.iloc[i] = min(umd.iloc[i - 1] , d)
    else:
        umd = dd.groupby(groupby , observed=True , group_keys = False).apply(eval_uncovered_max_drawdown)
    return umd

def eval_detailed_drawdown(pf : pd.Series , groupby : str | list[str] | None = None):
    '''
    based on the pf , calculate the detailed drawdown , including:
    cum_ret , peak , drawdown , uncovered_max_drawdown , recover_ratio
    '''
    df = pf.to_frame('pf')
    df['cum_ret'] = eval_cum_ret(df['pf'] , how = 'exp' , groupby = groupby)
    df['peak'] = eval_cum_peak(df['pf'] , how = 'exp' , groupby = groupby)
    df['drawdown'] = (df['cum_ret'] + 1) / (df['peak'] + 1) - 1
    df['uncovered_max_drawdown'] = eval_uncovered_max_drawdown(df['drawdown'] , groupby = groupby)
    df['recover_ratio'] = 1 - (df['drawdown'] / df['uncovered_max_drawdown']).fillna(0)
    return df
