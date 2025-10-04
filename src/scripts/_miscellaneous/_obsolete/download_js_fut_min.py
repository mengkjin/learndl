from src.basic import DB , CALENDAR
import argparse
import jsdata , datetime # type: ignore 
import pandas as pd
from typing import Any

def download_jsdata(date : int , limit = 5000 , api : Any = None):
    offset = 0
    dfs = []
    while True:
        df = api.future_min(**{
        "freq": "1min",
        "start_time": f"{datetime.datetime.strptime(str(date),'%Y%m%d').strftime('%Y-%m-%d')} 00:00:00",
        "end_time": f"{datetime.datetime.strptime(str(date),'%Y%m%d').strftime('%Y-%m-%d')} 23:59:59",
        "limit": limit ,
        "offset" : offset})
        offset += limit
        dfs.append(df)
        if not isinstance(df , pd.DataFrame) or len(df) < limit: 
            break
    df = pd.concat(dfs).reset_index(drop=True)
    df = df.rename(columns={'code':'ts_code'})
    df['trade_time'] = df['trade_time'].str.replace('T',' ')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start' , type=int , default=20190101)
    parser.add_argument('--end' , type=int , default=20191231)
    args = parser.parse_args()
    target_dates = CALENDAR.cd_within(args.start , args.end)
    stored_dates = DB.dates('trade_js' , 'fut_min')
    dates = CALENDAR.diffs(target_dates , stored_dates)
    api = jsdata.get_api()
    for date in dates:
        df = download_jsdata(date , api = api)
        DB.save(df , 'trade_js' , 'fut_min' , date , verbose = True)