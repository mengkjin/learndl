import pandas as pd
from .wind_conn import get_wind_conn


DAILYKBAR_WINDFLD_MAP = {
    'open_price': 'S_DQ_OPEN',
    'close_price': 'S_DQ_CLOSE',
    'high_price': 'S_DQ_HIGH',
    'low_price': 'S_DQ_LOW',
    'prev_close': 'S_DQ_PRECLOSE',
    'vwap': 'S_DQ_AVGPRICE',
    'amount': 'S_DQ_AMOUNT',
    'volume': 'S_DQ_VOLUME',
    'turnover': 'TURNOVER_D'
}

TABLE_MAP = {
    'AShareEodPrices': ['S_DQ_OPEN', 'S_DQ_CLOSE', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_PRECLOSE', 'S_DQ_AVGPRICE', 'S_DQ_AMOUNT', 'S_DQ_VOLUME'],
    'AShareYield': ['TURNOVER_D']
}


def load_daily_bar_by_wind(start_date, end_date, flds_):
    assert len(flds_) == len(set(flds_))
    table_flds = {}
    for tb, col_list in TABLE_MAP.items():
        table_flds[tb] = list()
        for f in flds_:
            wf = DAILYKBAR_WINDFLD_MAP[f]
            if wf in col_list:
                table_flds[tb].append((f, wf))
    conn = get_wind_conn()
    sql = "select TRADE_DT, S_INFO_WINDCODE from AShareEodPrices where TRADE_DT between '{0}' and '{1}' " \
          "order by TRADE_DT, S_INFO_WINDCODE".format(start_date.replace('-', ''), end_date.replace('-', ''))
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={'TRADE_DT': 'CalcDate', 'S_INFO_WINDCODE': 'Code'}, errors="raise", inplace=True)
    for tb, query_flds in table_flds.items():
        if query_flds:
            sql = "select TRADE_DT, S_INFO_WINDCODE, {0} from {1} " \
                  "where TRADE_DT between '{2}' and '{3}' " \
                  "order by TRADE_DT, S_INFO_WINDCODE".format(','.join([fld[1] for fld in query_flds]), tb,
                start_date.replace('-', ''), end_date.replace('-', ''))
            data = pd.read_sql(sql, conn)
            data.columns = data.columns.str.upper()
            data.rename(columns={'TRADE_DT': 'CalcDate', 'S_INFO_WINDCODE': 'Code'}, errors="raise", inplace=True)
            data.rename(columns={fld[1]: fld[0] for fld in query_flds}, errors="raise", inplace=True)
            df = pd.merge(df, data, how='left', on=['CalcDate', 'Code'])
    #
    assert df.shape[1] == 2 + len(flds_)
    df['CalcDate'] = df['CalcDate'].str[:4] + '-' + df['CalcDate'].str[4:6] + '-' + df['CalcDate'].str[6:]
    if 'amount' in df.columns:
        df['amount'] = df['amount'] * 1000.0
    if 'turnover' in df.columns:
        df['turnover'] = df['turnover'] / 100.0
    if 'volume' in df.columns:
        df['volume'] = df['volume'] * 100.0
    return df