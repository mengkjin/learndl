import zipfile
import numpy as np
import pandas as pd

from typing import Literal
from pathlib import Path

from src.data.util import trade_min_fillna
from src.basic import PATH , IS_SERVER

download_path = PATH.miscel.joinpath('JSMinute')

def get_js_min(date : int):
    renamer = {
        'ticker'    : 'secid'  ,
        'secoffset' : 'minute' ,
        'openprice' : 'open' ,
        'highprice' : 'high' , 
        'lowprice'  : 'low' , 
        'closeprice': 'close' , 
        'value'     : 'amount' , 
        'volume'    : 'volume' , 
        'vwap'      : 'vwap' , 
    }
    df = extract_js_min(date)
    df = add_sec_type(df).rename(columns=renamer)
    df['minute'] = (df['minute'] / 60).astype(int)
    df['minute'] = (df['minute'] - 90) * (df['minute'] <= 240) + (df['minute'] - 180) * (df['minute'] > 240)
    return df

def extract_js_min(date):
    target_path = download_path.joinpath(f'min/min.{date}.feather')
    target_path.parent.mkdir(parents=True , exist_ok=True)
    df = None
    if target_path.exists(): 
        try:
            df = pd.read_feather(target_path)
        except Exception as e:
            print(e)
            target_path.unlink()
    if df is None:
        zip_file_path = download_path.joinpath(f'min.{date}.zip')
        txt_file_path = f'equity_pricemin{date}.txt'

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: 
            zip_ref.extract(txt_file_path , download_path)
            df = pd.read_csv(download_path.joinpath(txt_file_path) , sep = '\t' , low_memory=False)
            download_path.joinpath(txt_file_path).unlink()
    try:
        df['ticker'] = df['ticker'].astype(int)
    except Exception as e:
        print(e)
        df = df.loc[df['ticker'].str.isdigit()]
        df['ticker'] = df['ticker'].astype(int)
    df.to_feather(target_path)
    return df

def add_sec_type(df : pd.DataFrame):
    SZ_types : dict[tuple[int,int],str] = {
        (0 , 29) : 'A' ,
        (30 , 39) : 'option' ,
        (100 , 149) : 'bond' ,
        (150 , 188) : 'fund' ,
        (158 , 159) : 'etf' ,
        (189 , 199) : 'bond' ,
        (200 , 299) : 'B' ,
        (300 , 349) : 'A' ,
        (399 , 399) : 'index' ,
        (900 , 999) : 'index' ,
    }

    SH_types : dict[tuple[int,int],str] = {
        (0 , 8) : 'index' ,
        (9 , 299) : 'bond' ,
        (310 , 399) : 'other' ,
        (500 , 599) : 'fund' ,
        (580 , 582) : 'option' ,
        (510 , 518) : 'etf' ,
        (560 , 579) : 'etf' ,
        (588 , 589) : 'etf' ,
        (600 , 699) : 'A' ,
        (700 , 899) : 'other' ,
        (900 , 999) : 'B' ,
    }

    df_sec = pd.DataFrame({
        'ticker' : df['ticker'] ,
        'exchangecd' : df['exchangecd'] ,
        'shortnm' : df['shortnm'] ,
    }).drop_duplicates()
    df_sec['range'] = df_sec['ticker'] // 1000
    df_sec['sec_type'] = 'notspecified'
    sz_sec = df_sec.loc[df_sec['exchangecd'] == 'XSHE']
    sh_sec = df_sec.loc[df_sec['exchangecd'] == 'XSHG']

    # sz
    for (start , end) , sec_type in SZ_types.items():
        sz_sec.loc[(sz_sec['range'] >= start) & (sz_sec['range'] <= end) , 'sec_type'] = sec_type
    sz_sec.loc[(sz_sec['sec_type'] == 'bond') & (sz_sec['shortnm'].str.contains('转')) , 'sec_type'] = 'convertible'


    #sh
    for (start , end) , sec_type in SH_types.items():
        sh_sec.loc[(sh_sec['range'] >= start) & (sh_sec['range'] <= end) , 'sec_type'] = sec_type
    sh_sec.loc[(sh_sec['sec_type'] == 'bond') & (sh_sec['shortnm'].str.contains('转')) , 'sec_type'] = 'convertible'

    df_sec = pd.concat([sz_sec , sh_sec]).loc[:,['ticker' , 'exchangecd' , 'sec_type']]
    df = df.merge(df_sec , on = ['ticker' , 'exchangecd'])
    return df

def filter_sec(df : pd.DataFrame , sec_type : Literal['sec' , 'etf' , 'cb'] | str):
    sec_type_map = {'sec' : 'A' , 'etf' : 'etf' , 'cb' : 'convertible'}
    return df.loc[df['sec_type'] == sec_type_map[sec_type]]

def transform_sec(df : pd.DataFrame):
    
    with np.errstate(invalid='ignore' , divide = 'ignore'): 
        df = df.sort_values(['secid','minute'])
        df = df.loc[:,['secid','minute','open','high','low','close','amount','volume','vwap']]
        df = trade_min_fillna(df)

    return df

def perform_extraction_and_transform(date : int):
    df = get_js_min(date)
    for sec_type in ['sec' , 'etf' , 'cb']:
        sec_df = filter_sec(df , sec_type)
        if sec_type == 'sec':
            sec_df = transform_sec(df)
        src_key = f'min' if sec_type == 'sec' else f'{sec_type}_min'
        PATH.db_save(sec_df , 'trade_js' , src_key , date , verbose = True)
    
def main():
    if not IS_SERVER: return
    target_dates = np.array([int(p.name.split('.')[-2][-8:]) for p in download_path.iterdir() if not p.is_dir()])
    stored_dates_sec = PATH.db_dates('trade_js' , 'min')
    stored_dates_etf = PATH.db_dates('trade_js' , 'etf_min')
    stored_dates_cb  = PATH.db_dates('trade_js' , 'cb_min')
    dates = target_dates[np.isin(target_dates , stored_dates_sec) & 
                         np.isin(target_dates , stored_dates_etf) & 
                         np.isin(target_dates , stored_dates_cb)]
    dates.sort()
    for date in dates:
        perform_extraction_and_transform(date)

if __name__ == '__main__':
     main()