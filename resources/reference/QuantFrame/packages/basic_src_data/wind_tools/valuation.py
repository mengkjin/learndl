import pandas as pd
from .wind_conn import get_wind_conn
from events_system.calendar_util import CALENDAR_UTIL


WINDFLD_MAP = {
    "total_value": 'S_VAL_MV', 'float_value': 'S_DQ_MV', 'book_value': 'NET_ASSETS_TODAY',
    'free_float_share': 'FREE_SHARES_TODAY', 'total_share': 'TOT_SHR_TODAY', 'float_share': 'FLOAT_A_SHR_TODAY',
}


def load_valuation_by_wind(start_date, end_date_, query_flds_):
    data_start_date = CALENDAR_UTIL.get_last_trading_dates([start_date], inc_self_if_is_trdday=True)[0]
    assert len(query_flds_) == len(set(query_flds_))
    wind_flds = [WINDFLD_MAP.get(f) for f in query_flds_]
    assert None not in wind_flds
    conn = get_wind_conn()
    sql = "select S_INFO_WINDCODE, TRADE_DT, {0} from AShareEODDerivativeIndicator " \
          "where TRADE_DT between '{1}' and '{2}' " \
          "order by TRADE_DT, S_INFO_WINDCODE".format(",".join(wind_flds),
                                                      data_start_date.replace('-', ''), end_date_.replace('-', ''))
    df = pd.read_sql(sql, conn)
    df.columns = df.columns.str.upper()
    df.rename(columns={"S_INFO_WINDCODE": "Code", "TRADE_DT": "CalcDate"}, inplace=True)
    df.rename(columns=dict(zip(wind_flds, query_flds_)), inplace=True)
    df.drop_duplicates(subset=['Code', 'CalcDate'], keep='last', inplace=True)
    df = df[['CalcDate', 'Code'] + query_flds_]
    df = df[df['Code'].str[0] != 'A']
    if 'total_value' in query_flds_:
        df['total_value'] = df['total_value'] * 10000.0
    if 'float_value' in query_flds_:
        df['float_value'] = df['float_value'] * 10000.0
    if 'free_float_share' in query_flds_:
        df['free_float_share'] = df['free_float_share'] * 10000.0
    if 'total_share' in query_flds_:
        df['total_share'] = df['total_share'] * 10000.0
    if 'float_share' in query_flds_:
        df['float_share'] = df['float_share'] * 10000.0
    df['CalcDate'] = df['CalcDate'].str[:4] + '-' + df['CalcDate'].str[4:6] + '-' + df['CalcDate'].str[6:]
    #
    #sql = "select S_INFO_WINDCODE from AShareDescription " \
    #      "where S_INFO_LISTDATE is not null and left(S_INFO_WINDCODE,1) != 'A' order by S_INFO_WINDCODE"
    sql = "select S_INFO_WINDCODE from AShareDescription " \
          "where S_INFO_LISTDATE is not null and substr(S_INFO_WINDCODE,1,1) != 'A' order by S_INFO_WINDCODE"
    listed_codes = pd.read_sql(sql, conn)
    listed_codes.columns = listed_codes.columns.str.upper()
    df = df[df['Code'].isin(listed_codes['S_INFO_WINDCODE'])]
    df.reset_index(drop=True, inplace=True)
    #
    assert df['CalcDate'].drop_duplicates().tolist() == CALENDAR_UTIL.get_ranged_dates(data_start_date, end_date_)
    df = df.set_index(['CalcDate', 'Code']).unstack().fillna(method='ffill')
    rtn = df.loc[start_date: end_date_].stack().reset_index()
    return rtn