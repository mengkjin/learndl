import pandas as pd

from src.basic import CALENDAR , DB

from .basic import BasicUpdater

class DailyRiskUpdater(BasicUpdater):
    START_DATE = 20110101
    DB_SRC = 'exposure'
    DB_KEY = 'daily_risk'

    @classmethod
    def update(cls):
        stored_dates = DB.dates(cls.DB_SRC , cls.DB_KEY)
        end_date = min(CALENDAR.updated() , DB.max_date('trade_ts' , '5min'))
        update_dates = CALENDAR.diffs(cls.START_DATE , end_date , stored_dates)
        for date in update_dates:
            cls.update_one(date)

    @classmethod
    def update_rollback(cls , rollback_date : int):
        CALENDAR.check_rollback_date(rollback_date)
        
        start_date = CALENDAR.td(rollback_date)
        end_date = min(CALENDAR.updated() , DB.max_date('trade_ts' , '5min'))
        update_dates = CALENDAR.td_within(start_dt = start_date , end_dt = end_date)
        for date in update_dates:
            cls.update_one(date)

    @classmethod
    def update_one(cls , date : int):
        DB.save(calc_daily_risk(date) , cls.DB_SRC , cls.DB_KEY , date , verbose = True)

def calc_daily_risk(date : int):
    inputs : dict[str , pd.DataFrame] = {
        'quote' : DB.load('trade_ts' , 'day' , date).set_index('secid') ,
        'moneyflow' : DB.load('trade_ts' , 'day_moneyflow' , date).set_index('secid') ,
        'min' : DB.load('trade_ts' , '5min' , date).set_index('secid')
    }
    funcs = [
        day_true_range , 
        day_turnover ,
        day_largebuy_price_deviation , 
        day_smallbuy_percentage , 
        day_sqrt_avg_size , 
        day_open_close_percentage , 
        day_5min_ret_volatility , 
        day_5min_ret_skewness
    ]
    results = {func.__name__ : func(**inputs) for func in funcs}
    result = pd.concat([df.reindex(inputs['quote'].index) for df in results.values()] , axis = 1)
    return result.reset_index()

def day_true_range(quote : pd.DataFrame , **kwargs):
    tr = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['preclose']).abs() , (quote['low'] - quote['preclose']).abs()] , axis = 1).max(axis = 1)
    tr = (tr / quote['preclose']).rename('true_range')
    return tr

def day_turnover(quote : pd.DataFrame , **kwargs):
    turnover = quote['turn_fl'] / 100
    return turnover.rename('turnover')

def day_largebuy_price_deviation(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs):
    q = quote.join(moneyflow.loc[:,['buy_elg_amount' , 'buy_elg_vol' , 'buy_lg_amount' , 'buy_lg_vol']])
    q['lbp'] = (q['buy_elg_amount'] + q['buy_lg_amount']) / (q['buy_elg_vol'] + q['buy_lg_vol']) * 100
    q['large_buy_pdev'] = abs(q['lbp'] - q['vwap']) / q['vwap']
    return q['large_buy_pdev']

def day_smallbuy_percentage(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs):
    q = quote.join(moneyflow.loc[:,['buy_sm_amount']])
    q['small_buy_pct'] = (q['buy_sm_amount']) / q['amount'] * 10
    return q['small_buy_pct']

def day_sqrt_avg_size(quote : pd.DataFrame , min : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs):
    if not min.empty and 'num_trades' in min.columns:
        num_trades = min.groupby('secid')['num_trades'].sum()
    else:
        avg_size = [5 , 20 , 100 , 500]
        mf_buy_size = moneyflow.loc[:,['buy_sm_amount' , 'buy_md_amount' , 'buy_lg_amount' , 'buy_elg_amount']]
        mf_buy_num = (mf_buy_size / avg_size).sum(axis = 1)

        mf_sell_size = moneyflow.loc[:,['sell_sm_amount' , 'sell_md_amount' , 'sell_lg_amount' , 'sell_elg_amount']]
        mf_sell_num = (mf_sell_size / avg_size).sum(axis = 1)

        num_trades = (mf_buy_num + mf_sell_num) / 2 * 10

    q = quote.join(num_trades)
    q['sqrt_avg_size'] = (q['amount'] / q['num_trades']).pow(0.5)
    return q['sqrt_avg_size']

def day_open_close_percentage(quote : pd.DataFrame , min : pd.DataFrame , **kwargs):
    ocamount = min.query('minute <= 5 or minute >= 42').groupby('secid')['amount'].sum().rename('open_close_amount')
    q = quote.join(ocamount)
    q['open_close_pct'] = q['open_close_amount'] / q['amount'] / 1000
    return q['open_close_pct']

def day_5min_ret_volatility(quote : pd.DataFrame , min : pd.DataFrame , **kwargs):
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    ret_volatility = min.query('minute >= 1').groupby('secid')['ret'].std().rename('ret_volatility')
    return quote.join(ret_volatility)['ret_volatility']

def day_5min_ret_skewness(quote : pd.DataFrame , min : pd.DataFrame , **kwargs):
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    ret_skewness = min.query('minute >= 1').groupby('secid')['ret'].skew().rename('ret_skewness')
    return quote.join(ret_skewness)['ret_skewness']
