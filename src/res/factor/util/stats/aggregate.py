import pandas as pd
import numpy as np

from src.proj import CALENDAR

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

def eval_ret_stats(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    start = ret['date'].min()
    end = ret['date'].max()
    total_days = CALENDAR.cd_diff(start , end) + 1
    total_return = _period_ret(ret['ret'])
    annualized_return = (1 + total_return) ** (365 / total_days) - 1
    datas = [
        ('Start' , str(start)),
        ('End' , str(end)),
        ('Total' , f'{total_return * 100:.2f}%'),
        ('Annual' , f'{annualized_return * 100:.2f}%'),
    ]
    return pd.DataFrame(datas , columns = ['period' , 'ret']).set_index('period')['ret']

def eval_year_ret(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    df = ret.assign(period = ret['date'] // 10000).groupby('period')['ret'].apply(_period_ret).reset_index(drop = False)
    df['period'] = 'Y' + df['period'].astype(str)
    df['ret'] = df['ret'].apply(lambda x : f'{x * 100:.2f}%')
    return df.set_index('period')['ret']

def eval_recent_ret(*input : pd.DataFrame | pd.Series | np.ndarray , end : int = -1) -> pd.Series:
    ret = _get_ret_df(*input)
    if ret.empty:
        return pd.Series()
    
    if end <= 0:
        if end == 0:
            end = -1
        end = ret['date'].nlargest(-end).min()

    period_names = {'d' : 'Last Day' , 'w' : 'Last Week' , 'm' : 'Last Month' , 'q' : 'Last Qtr' , 'y' : 'Last Year'}
    datas = []
    for period , name in period_names.items():
        start , end = CALENDAR.cd_start_end(end , 1 , period) # noqa
        datas.append((name , _period_ret(ret.query('date <= @end and date >= @start')['ret'])))
    
    df = pd.DataFrame(datas , columns = ['period' , 'ret'])
    df['ret'] = df['ret'].apply(lambda x : f'{x * 100:.2f}%')
    return df.set_index('period')['ret']

def eval_period_ret(*input : pd.DataFrame | pd.Series | np.ndarray , end : int = -1) -> pd.Series:
    basic_stats = eval_ret_stats(*input)
    year_ret = eval_year_ret(*input)
    recent_ret = eval_recent_ret(*input , end = end)
    return pd.concat([basic_stats , year_ret , recent_ret])

def _get_ic_df(*input : pd.DataFrame | pd.Series | np.ndarray):
    """
    input: from input transform to return df with date and ret
    input can be:
    - pd.DataFrame with date and ret
    - tuple of (date , ret)
    """

    if len(input) == 2:
        d , ic = input
        assert len(d) == len(ic) , 'dates and ic must have the same length'
        return pd.DataFrame({'date' : d , 'ic' : ic})
    elif len(input) == 1:
        if isinstance(input[0] , pd.DataFrame):
            if 'ic' in input[0].columns:
                return input[0].loc[:,['date' , 'ic']]
            elif len(input[0].columns) == 2:
                ic_col = input[0].columns.drop('date').item()
                return input[0].loc[:,['date' , ic_col]].rename(columns = {ic_col : 'ic'})
            else:
                raise ValueError('ic column must be present in the DataFrame or the DataFrame must have only two columns')
        else:
            d , ic = input[0]
            assert len(d) == len(ic) , 'dates and ic must have the same length'
            return pd.DataFrame({'date' : d , 'ic' : ic})
    else:
        raise ValueError('input must be a pd.DataFrame or a tuple of (date , ic)')

def eval_ic_stats(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ic = _get_ic_df(*input)
    if ic.empty:
        return pd.Series()
    start = ic['date'].min()
    end = ic['date'].max()
    datas = [
        ('Start' , str(start)),
        ('End' , str(end)),
        ('Avg' , f'{ic['ic'].mean() * 100:.2f}%'),
        ('Sum' , f'{ic['ic'].sum():.2f}'),
        ('Std' , f'{ic['ic'].std() * 100:.2f}%'),
        ('T' , f'{((ic['ic'].mean() / ic['ic'].std()) * (len(ic['date'].unique())**0.5)):.2f}'),
        ('IR' , f'{((ic['ic'].mean() / ic['ic'].std()) * ((240 / 10)**0.5)):.2f}'),
    ]
    return pd.DataFrame(datas , columns = ['period' , 'ic']).set_index('period')['ic']

def eval_year_ic(*input : pd.DataFrame | pd.Series | np.ndarray) -> pd.Series:
    ic = _get_ic_df(*input)
    if ic.empty:
        return pd.Series()
    df = ic.assign(period = ic['date'] // 10000).groupby('period')['ic'].mean().reset_index(drop = False)
    df['period'] = 'Y' + df['period'].astype(str)
    df['ic'] = df['ic'].apply(lambda x : f'{x * 100:.2f}%')
    return df.set_index('period')['ic']

def eval_recent_ic(*input : pd.DataFrame | pd.Series | np.ndarray , end : int = -1) -> pd.Series:
    ic = _get_ic_df(*input)
    if ic.empty:
        return pd.Series()
    
    if end <= 0:
        if end == 0:
            end = -1
        end = ic['date'].nlargest(-end).min()

    period_names = {'d' : 'Last Day' , 'w' : 'Last Week' , 'm' : 'Last Month' , 'q' : 'Last Qtr' , 'y' : 'Last Year'}
    datas = []
    for period , name in period_names.items():
        start , end = CALENDAR.cd_start_end(end , 1 , period) # noqa
        datas.append((name , ic.query('date <= @end and date >= @start')['ic'].mean()))
    
    df = pd.DataFrame(datas , columns = ['period' , 'ic'])
    df['ic'] = df['ic'].apply(lambda x : f'{x * 100:.2f}%')
    return df.set_index('period')['ic']

def eval_period_ic(*input : pd.DataFrame | pd.Series | np.ndarray , end : int = -1) -> pd.Series:
    basic_stats = eval_ic_stats(*input)
    year_ic = eval_year_ic(*input)
    recent_ic = eval_recent_ic(*input , end = end)
    return pd.concat([basic_stats , year_ic , recent_ic])
        
def eval_period_ret_multi(inputs : dict | pd.DataFrame , end : int = -1) -> pd.DataFrame:
    """
    get the period return of multiple inputs
    inputs can be a dict of (name , (date , ret)) or a pd.DataFrame with date and ret columns
    example:
        dates = CALENDAR.range(20240101 , 20251231 , 'td')
        ret1 = np.random.randn(len(dates)) / 100
        ret2 = np.random.randn(len(dates)) / 100
        df0 = eval_period_ret_multi({'ret1' : (dates , ret1) , 'ret2' : (dates , ret2)})
        df1 = eval_period_ret_multi(pd.DataFrame({'date' : dates , 'ret1' : ret1 , 'ret2' : ret2}))
    """
    if isinstance(inputs , pd.DataFrame):
        assert 'date' in inputs.columns , f'inputs must contain date and ret columns , got {inputs.columns}'
        names = inputs.columns.drop(['date'])
        dfs = {name : eval_period_ret(inputs.loc[:,['date' , name]] , end = end) for name in names}
    else:
        dfs = {name : eval_period_ret(inputs[name] , end = end) for name in inputs}
    return pd.concat(dfs , axis = 1)

def eval_period_ic_multi(inputs : dict | pd.DataFrame , end : int = -1) -> pd.DataFrame:
    """
    get the period ic of multiple inputs
    inputs can be a dict of (name , (date , ic)) or a pd.DataFrame with date and ic columns
    example:
        dates = CALENDAR.range(20240101 , 20251231 , 'td')
        ic1 = np.random.randn(len(dates)) / 100
        ic2 = np.random.randn(len(dates)) / 100
        df0 = eval_period_ic_multi({'ic1' : (dates , ic1) , 'ic2' : (dates , ic2)})
        df1 = eval_period_ic_multi(pd.DataFrame({'date' : dates , 'ic1' : ic1 , 'ic2' : ic2}))
    """
    if isinstance(inputs , pd.DataFrame):
        assert 'date' in inputs.columns , f'inputs must contain date and ic columns , got {inputs.columns}'
        names = inputs.columns.drop(['date'])
        dfs = {name : eval_period_ic(inputs.loc[:,['date' , name]] , end = end) for name in names}
    else:
        dfs = {name : eval_period_ic(inputs[name] , end = end) for name in inputs}
    df = pd.concat(dfs , axis = 1)
    index0 = ['Start' , 'End' , 'Avg' , 'Sum' , 'Std' , 'T' , 'IR']
    index2 = ['Last Day' , 'Last Week' , 'Last Month' , 'Last Qtr' , 'Last Year']
    index1 = sorted(set(df.index.tolist()) - set(index0) - set(index2))
    return df.reindex(index = index0 + index1 + index2)