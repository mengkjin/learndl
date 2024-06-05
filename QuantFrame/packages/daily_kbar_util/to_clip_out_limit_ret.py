import numpy as np
import pandas as pd


def clip_stk_ret(ret_data):
    assert pd.Index(['CalcDate', 'Code', 'ret']).difference(ret_data.columns).empty
    data = ret_data[["CalcDate", "Code", 'ret']].copy()
    limit_20_flg = (data['Code'].str[:3] == '688') | ((data['Code'].str[0] == '3') & (data['CalcDate'] >= '2020-06-15'))
    limit_10_flg = ~limit_20_flg
    #
    data.loc[limit_20_flg, 'ret'] = data.loc[limit_20_flg, 'ret'].clip(lower=-0.2, upper=0.2)
    data.loc[limit_10_flg, 'ret'] = data.loc[limit_10_flg, 'ret'].clip(lower=-0.1, upper=0.1)
    return data