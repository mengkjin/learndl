# http://baostock.com/baostock/index.php/首页

import os
import baostock as bs
import numpy as np
import pandas as pd  

from ...basic import PATH



BAO_PATH = f'{PATH.main}/Baostock'
os.makedirs(BAO_PATH , exist_ok=True)
os.makedirs(f'{BAO_PATH}/tmp' , exist_ok=True)

def baostock_5m_bar(date : int):
    # date = 20240704
    final_path = f'{BAO_PATH}/5m_bar_{date}.feather'
    if not os.path.exists(final_path):
        downloaded = [d for d in os.listdir(f'{BAO_PATH}/tmp/') if d.startswith(str(date))]
        downloaded_codes = [d.split('@')[1] for d in downloaded]

        date_str = f'{date // 10000}-{(date // 100) % 100}-{date % 100}'
        bs.login()  
        stock_rs = bs.query_all_stock(date_str)  
        stock_df = stock_rs.get_data()  
        codes = np.setdiff1d(stock_df['code'] , downloaded_codes)

        dfs = [pd.read_feather(f'{BAO_PATH}/tmp/{d}') for d in downloaded]
        for i , code in enumerate(codes):
            rs = bs.query_history_k_data_plus(code, 'date,time,open,high,low,close,volume,amount,adjustflag',
                                            start_date=date_str,end_date=date_str,frequency='5', adjustflag='3')
            data_list = rs.get_data()
            result = pd.DataFrame(data_list,columns=rs.fields)
            result['code'] = code
            result.to_feather(f'{BAO_PATH}/tmp/{date}@{code}')
            dfs.append(result)

            print(f'{i}/{len(codes)} {code}...' , end = '\r')

        bs.logout()  
        df_all = pd.concat(dfs)
        df_all.to_feather(f'{BAO_PATH}/5m_bar_{date}.feather')

        downloaded = [d for d in os.listdir(f'{BAO_PATH}/tmp') if d.startswith(str(date))]
        [os.remove(f'{BAO_PATH}/tmp/{d}') for d in downloaded]
