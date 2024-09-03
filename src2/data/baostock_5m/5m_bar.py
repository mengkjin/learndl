# http://baostock.com/baostock/index.php/首页

import baostock as bs
import numpy as np
import pandas as pd  

from ...basic import PATH

BAO_PATH = PATH.main.joinpath('Baostock')
BAO_PATH.mkdir(exist_ok=True)
BAO_PATH.joinpath('tmp').mkdir(exist_ok=True)

def baostock_5m_bar(date : int):
    # date = 20240704
    final_path = BAO_PATH.joinpath(f'5m_bar_{date}.feather')
    if not final_path.exists():
        downloaded = [d.name for d in BAO_PATH.joinpath('tmp').iterdir() if d.name.startswith(str(date))]
        downloaded_codes = [d.split('@')[1] for d in downloaded]

        date_str = f'{date // 10000}-{(date // 100) % 100}-{date % 100}'
        bs.login()  
        stock_rs = bs.query_all_stock(date_str)  
        stock_df = stock_rs.get_data()  
        codes = np.setdiff1d(stock_df['code'] , downloaded_codes)

        dfs = [pd.read_feather(BAO_PATH.joinpath('tmp' , 'd')) for d in downloaded]
        for i , code in enumerate(codes):
            rs = bs.query_history_k_data_plus(code, 'date,time,open,high,low,close,volume,amount,adjustflag',
                                            start_date=date_str,end_date=date_str,frequency='5', adjustflag='3')
            data_list = rs.get_data()
            result = pd.DataFrame(data_list,columns=rs.fields)
            result['code'] = code
            result.to_feather(BAO_PATH.joinpath('tmp' , f'{date}@{code}'))
            dfs.append(result)

            print(f'{i}/{len(codes)} {code}...' , end = '\r')

        bs.logout()  
        df_all = pd.concat(dfs)
        df_all.to_feather(BAO_PATH.joinpath(f'5m_bar_{date}.feather'))

        [d.unlink() for d in BAO_PATH.joinpath('tmp').iterdir() if d.name.startswith(str(date))]
