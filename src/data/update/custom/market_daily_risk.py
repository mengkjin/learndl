import pandas as pd
import numpy as np

from typing import Literal , Any
from src.proj import Logger , CALENDAR , DB

from src.data.update.custom.basic import BasicCustomUpdater

class MarketDailyRiskUpdater(BasicCustomUpdater):
    START_DATE = max(20100101 , DB.min_date('trade_ts' , '5min' , use_alt=True))
    DB_SRC = 'market_daily'
    DB_KEY = 'risk'
    
    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : int = 1):
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all custom index is supported , but beware of the performance for {cls.__name__}!')
            stored_dates = np.array([])
        elif update_type == 'update':
            stored_df = DB.load_df(DB.path(cls.DB_SRC , cls.DB_KEY))
            stored_dates = np.array([], dtype = int) if stored_df.empty else stored_df.reset_index()['date'].to_numpy(int)
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            stored_df = DB.load_df(DB.path(cls.DB_SRC , cls.DB_KEY))
            stored_dates = np.array([], dtype = int) if stored_df.empty else stored_df.reset_index()['date'].to_numpy(int)
            stored_dates = CALENDAR.slice(stored_dates , 0 , rollback_date - 1)
        else:
            raise ValueError(f'Invalid update type: {update_type}')
            
        end_date = min(CALENDAR.updated() , DB.max_date('trade_ts' , '5min' , use_alt=True))
        update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
        if len(update_dates) == 0:
            Logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date' , indent = indent , vb_level = vb_level)
            return

        new_dfs : list[pd.DataFrame] = []
        for date in update_dates:
            new_dfs.append(calc_market_daily_risk(date))
            Logger.stdout(f'Calculate market daily risk at {date}' , indent = indent + 1 , vb_level = vb_level + 2)

        cls.append_result(pd.concat(new_dfs) , indent = indent , vb_level = vb_level)

        Logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {CALENDAR.dates_str(update_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , indent : int = 2 , vb_level : int = 2):
        cls.append_result(calc_market_daily_risk(date) , indent = indent , vb_level = vb_level)

    @classmethod
    def append_result(cls , new_df : pd.DataFrame , indent : int = 1 , vb_level : int = 1):
        old_df = DB.load(cls.DB_SRC , cls.DB_KEY)
        df = pd.concat([old_df , new_df])
        if not df.empty:
            df = df.drop_duplicates('date' , keep = 'last').reset_index(drop = True).sort_values('date')
            DB.save(df , cls.DB_SRC , cls.DB_KEY , indent = indent , vb_level = vb_level)

def get_inputs(date : int) -> dict[str , pd.DataFrame]:
    inputs : dict[str , pd.DataFrame] = {
        'quote' : DB.load('trade_ts' , 'day' , date) ,
        'val' : DB.load('trade_ts' , 'day_val' , date) ,
        'moneyflow' : DB.load('trade_ts' , 'day_moneyflow' , date),
        'min' : DB.load('trade_ts' , '5min' , date , use_alt=True)
    }
    for name , df in inputs.items():
        if not df.empty:
            inputs[name] = df.set_index('secid')
    return inputs

def fillinf(series : pd.Series , fill_value : Any = 0) -> pd.Series:
    return series.where(np.isfinite(series) , fill_value)

def calc_market_daily_risk(date : int):
    inputs = get_inputs(date)
    funcs = [
        market_day_true_range , 
        market_day_turnover ,
        market_day_largebuy_price_deviation , 
        market_day_smallbuy_percentage , 
        market_day_sqrt_avg_size , 
        market_day_open_close_percentage , 
        market_day_5min_ret_volatility , 
        market_day_5min_ret_skewness
    ]
    result = pd.DataFrame({func.__name__ : func(**inputs) for func in funcs} , index = pd.Index([date] , name = 'date')).reset_index()
    return result

def market_day_true_range(quote : pd.DataFrame , val : pd.DataFrame , **kwargs) -> float:
    tr = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['preclose']).abs() , (quote['low'] - quote['preclose']).abs()] , axis = 1).max(axis = 1)
    tr = fillinf((tr / quote['preclose']).rename('true_range') , 0)
    weight = (val['float_share'] * quote['preclose']).fillna(0)
    return tr.fillna(0).mul(weight).sum() / weight.sum()

def market_day_turnover(quote : pd.DataFrame , val : pd.DataFrame , **kwargs) -> float:
    turnover = quote['turn_fl'] / 100
    weight = (val['float_share'] * quote['preclose']).fillna(0)
    return turnover.mul(weight).sum() / weight.sum()

def market_day_largebuy_price_deviation(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> float:
    q = quote.join(moneyflow.loc[:,['buy_elg_amount' , 'buy_elg_vol' , 'buy_lg_amount' , 'buy_lg_vol']])
    q['lbp'] = (q['buy_elg_amount'] + q['buy_lg_amount']) / (q['buy_elg_vol'] + q['buy_lg_vol']) * 100
    q['large_buy_pdev'] = fillinf(abs(q['lbp'] - q['vwap']) / q['vwap'] , np.nan)
    weight = quote['amount'].fillna(0)
    return q['large_buy_pdev'].fillna(0).mul(weight).sum() / weight.sum()

def market_day_smallbuy_percentage(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> float:
    q = quote.join(moneyflow.loc[:,['buy_sm_amount']])
    q['small_buy_pct'] = fillinf((q['buy_sm_amount']) / q['amount'] * 10 , 0)
    weight = quote['amount'].fillna(0)
    return q['small_buy_pct'].fillna(0).mul(weight).sum() / weight.sum()

def market_day_sqrt_avg_size(quote : pd.DataFrame , min : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> float:
    if not min.empty and 'num_trades' in min.columns:
        num_trades = min.groupby('secid')['num_trades'].sum()
    else:
        avg_size = [5 , 20 , 100 , 500]
        mf_buy_size = moneyflow.loc[:,['buy_sm_amount' , 'buy_md_amount' , 'buy_lg_amount' , 'buy_elg_amount']]
        mf_buy_num = (mf_buy_size / avg_size).sum(axis = 1)

        mf_sell_size = moneyflow.loc[:,['sell_sm_amount' , 'sell_md_amount' , 'sell_lg_amount' , 'sell_elg_amount']]
        mf_sell_num = (mf_sell_size / avg_size).sum(axis = 1)

        num_trades = (mf_buy_num + mf_sell_num) / 2 * 10
        num_trades = num_trades.rename('num_trades')

    q = quote.join(num_trades)
    return (fillinf(q['amount'] , 0).sum() / fillinf(q['num_trades'] , 0).sum()) ** 0.5

def market_day_open_close_percentage(quote : pd.DataFrame , min : pd.DataFrame , **kwargs) -> float:
    ocamount = min.query('minute <= 5 or minute >= 42').groupby('secid')['amount'].sum().rename('open_close_amount')
    q = quote.join(ocamount)
    q['open_close_pct'] = fillinf(q['open_close_amount'] / q['amount'] / 1000 , 0)
    weight = quote['amount'].fillna(0)
    return q['open_close_pct'].fillna(0).mul(weight).sum() / weight.sum()

def market_day_5min_ret_volatility(quote : pd.DataFrame , min : pd.DataFrame , val : pd.DataFrame , **kwargs) -> float:
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    weight = (val['float_share'] * quote['preclose']).fillna(0).rename('weight')
    min_ret = min.join(weight).groupby('minute')[['ret' , 'weight']].apply(lambda x: x['ret'].mul(x['weight']).sum() / x['weight'].sum()).rename('ret').to_frame()
    ret_volatility = min_ret.query('minute >= 1')['ret'].std()
    return ret_volatility

def market_day_5min_ret_skewness(quote : pd.DataFrame , min : pd.DataFrame , val : pd.DataFrame , **kwargs) -> float | Any:
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    weight = (val['float_share'] * quote['preclose']).fillna(0).rename('weight')
    min_ret = min.join(weight).groupby('minute')[['ret' , 'weight']].apply(lambda x: x['ret'].mul(x['weight']).sum() / x['weight'].sum()).rename('ret').to_frame()
    ret_skewness = min_ret.query('minute >= 1')['ret'].skew()
    return ret_skewness
