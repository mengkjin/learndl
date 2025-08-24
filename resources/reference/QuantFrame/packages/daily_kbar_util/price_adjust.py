import numpy as np
import pandas as pd


def calc_adj_factor(data):
    prices = data.set_index(['CalcDate', 'Code']).unstack()
    close_price = prices['close_price'].fillna(method='ffill')
    prev_close = prices['prev_close'].fillna(close_price)
    adj_factor = prev_close.to_numpy()[1:] / close_price.to_numpy()[:-1]
    adj_factor = np.vstack((adj_factor, np.ones((1, adj_factor.shape[1]))))
    adj_factor = np.cumproduct(adj_factor[::-1], axis=0)[::-1]
    adj_factor = pd.DataFrame(adj_factor, index=prev_close.index, columns=prev_close.columns)
    adj_factor = adj_factor.stack().rename('adj_factor').reset_index(drop=False)
    rtn = pd.merge(data, adj_factor, how='left', on=['CalcDate', 'Code'])
    rtn['adj_factor'].fillna(1.0, inplace=True)
    return rtn