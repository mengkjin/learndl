# 导入tushare
import tushare as ts
from typing import Any

import numpy as np
import pandas as pd

pro = ts.pro_api('2026c96ef5fa7fc3241c96baafd638c585284c7fefaa00b93ef0a62c')

def tscode_to_secid(df : pd.DataFrame , retain = False):
    '''switch old symbol into secid'''
    if 'ts_code' not in df.columns.values: return df
    replace_dict = {'T00018' : '600018'}
    df['secid'] = df['ts_code'].astype(str).str.slice(0, 6).replace(replace_dict)
    df['secid'] = df['secid'].where(df['secid'].str.isdigit() , '-1').astype(int)
    if not retain: del df['ts_code']
    return df

def get_basic_calendar():
    columns = {
        'cal_date' : 'calendar' ,
        'is_open':'trade' ,
    }
    fields : str | Any = list(columns.keys())
    df = pro.trade_cal(fields = fields , exchange='SSE').rename(columns=columns)
    df = df.sort_values('calendar').reset_index(drop = True)
    return df

def get_basic_description():
    columns = {
        'ts_code' : 'ts_code' ,
        'name':'sec_name' ,
        'exchange':	'exchange_name'	,
        'list_date' : 'list_dt' ,
        'delist_date' : 'delist_dt'
    }
    fields : str | Any = list(columns.keys())
    df = pd.concat([
        pro.stock_basic(fields = fields , list_status = 'L') ,
        pro.stock_basic(fields = fields , list_status = 'D') ,
        pro.stock_basic(fields = fields , list_status = 'P')
    ]).rename(columns=columns)

    df = tscode_to_secid(df , retain=True)
    df['list_dt'] = df['list_dt'].fillna(-1).astype(int)
    df['delist_dt'] = df['delist_dt'].fillna(99991231).astype(int)

    return df

def get_basic_industry():
    limit = 2000
    args_is_new = ['Y' , 'N']
    dfs = []
    for is_new in args_is_new:
        offset = 0
        while True:
            df = pro.index_member_all(is_new = is_new , limit = limit , offset = offset)
            if len(df) == 0: break
            dfs.append(df)
            offset += limit

    df = pd.concat(dfs)
    df = tscode_to_secid(df)
    df['out_date'] = df['out_date'].fillna(99991231)
    return df