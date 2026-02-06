import pandas as pd
import numpy as np

from src.proj.calendar import CALENDAR

def _get_ret_df(*input : pd.DataFrame | pd.Series | np.ndarray):
    """
    input: from input transform to return df with date and ret
    input can be:
    - pd.DataFrame with date and ret
    - tuple of (date , ret)
    """

    if len(input) == 2:
        d , r = input
        assert len(d) == len(r) , 'dates and ret must have the same length'
        return pd.DataFrame({'date' : d , 'ret' : r})
    elif len(input) == 1:
        if isinstance(input[0] , pd.DataFrame):
            if 'ret' in input[0].columns:
                return input[0].loc[:,['date' , 'ret']]
            elif len(input[0].columns) == 2:
                ret_col = input[0].columns.drop('date').item()
                return input[0].loc[:,['date' , ret_col]].rename(columns = {ret_col : 'ret'})
            else:
                raise ValueError('ret column must be present in the DataFrame or the DataFrame must have only two columns')
        else:
            d , r = input[0]
            assert len(d) == len(r) , 'dates and ret must have the same length'
            return pd.DataFrame({'date' : d , 'ret' : r})
    else:
        raise ValueError('input must be a pd.DataFrame or a tuple of (date , ret)')

def _period_ret(r : pd.Series):
    """calculate the period return"""
    r = r + 1
    return np.prod(r) - 1

def eval_basic_stats(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    start_date = ret['date'].min()
    end_date = ret['date'].max()
    total_days = CALENDAR.cd_diff(start_date , end_date) + 1
    total_return = _period_ret(ret['ret'])
    annualized_return = (1 + total_return) ** (365 / total_days) - 1
    datas = [
        ('StartDate' , str(start_date)),
        ('EndDate' , str(end_date)),
        ('Total' , total_return),
        ('Annualized' , annualized_return),
    ]
    return pd.DataFrame(datas , columns = ['feature' , 'ret']).set_index('feature')['ret']

def eval_year_ret(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    df = ret.assign(feature = ret['date'] // 10000).groupby('feature')['ret'].apply(_period_ret).reset_index(drop = False)
    df['feature'] = 'Y' + df['feature'].astype(str)
    return df.set_index('feature')['ret']

def eval_recent_ret(*input : pd.DataFrame | pd.Series | np.ndarray , end_date : int = -1 , periods : list[str] = ['w' , 'm' , 'q' , 'y']) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    
    if end_date <= 0:
        if end_date == 0:
            end_date = -1
        end_date = ret['date'].nlargest(-end_date).min()

    period_names = {'w' : 'Last Week' , 'm' : 'Last Month' , 'q' : 'Last Quarter' , 'y' : 'Last Year'}
    datas = []
    for period in periods:
        start_dt , end_dt = CALENDAR.cd_start_end(end_date , 1 , period) # noqa
        datas.append((period_names[period] , _period_ret(ret.query('date <= @end_dt and date >= @start_dt')['ret'])))
    
    df = pd.DataFrame(datas , columns = ['feature' , 'ret'])
    return df.set_index('feature')['ret']

def eval_period_ret(*input : pd.DataFrame | pd.Series | np.ndarray , end_date : int = -1) -> pd.Series:
    basic_stats = eval_basic_stats(*input)
    year_ret = eval_year_ret(*input)
    recent_ret = eval_recent_ret(*input , end_date = end_date)
    return pd.concat([basic_stats , year_ret , recent_ret])
        
def eval_period_ret_multi(inputs : dict | pd.DataFrame , end_date : int = -1) -> pd.DataFrame:
    """
    get the period return of multiple inputs
    inputs can be a dict of (name , (date , ret)) or a pd.DataFrame with date and ret columns
    example:
        dates = CALENDAR.td_within(20240101 , 20251231)
        ret1 = np.random.randn(len(dates)) / 100
        ret2 = np.random.randn(len(dates)) / 100
        df0 = eval_period_ret_multi({'ret1' : (dates , ret1) , 'ret2' : (dates , ret2)})
        df1 = eval_period_ret_multi(pd.DataFrame({'date' : dates , 'ret1' : ret1 , 'ret2' : ret2}))
    """
    if isinstance(inputs , pd.DataFrame):
        assert 'date' in inputs.columns , f'inputs must contain date and ret columns , got {inputs.columns}'
        names = inputs.columns.drop(['date'])
        dfs = {name : eval_period_ret(inputs.loc[:,['date' , name]] , end_date = end_date) for name in names}
    else:
        dfs = {name : eval_period_ret(inputs[name] , end_date = end_date) for name in inputs}
    return pd.concat(dfs , axis = 1)