import pandas as pd
import numpy as np
from events_system.calendar_util import CALENDAR_UTIL
from .wind_conn import get_wind_conn


def get_index_members_from_winddf(start_date_, end_date_, index_name_):
    # assert index_name_ in ('000300.SH', '000905.SH', '000852.SH', '932000.CSI', '000906.SH', '931865.CSI')
    sql = "select S_CON_WINDCODE, S_CON_INDATE, S_CON_OUTDATE from AIndexMembers " \
          "where S_INFO_WINDCODE = '{0}' and S_CON_INDATE <= '{1}' and (S_CON_OUTDATE is null or S_CON_OUTDATE >= '{2}') order by S_CON_INDATE".format(
        index_name_, end_date_.replace('-', ''), start_date_.replace('-', ''))
    conn = get_wind_conn()
    all_index_mem_data = pd.read_sql(sql, conn)
    all_index_mem_data.columns = all_index_mem_data.columns.str.upper()
    all_index_mem_data.rename(
        columns={'S_CON_WINDCODE': 'member_code', 'S_CON_INDATE': 'indate', 'S_CON_OUTDATE': 'outdate'}, errors="raise", inplace=True)
    assert not all_index_mem_data['member_code'].isnull().any() and not all_index_mem_data['indate'].isnull().any()
    rtn = None
    if not all_index_mem_data.empty:
        all_index_mem_data['indate'] = all_index_mem_data['indate'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
        all_index_mem_data['outdate'] = all_index_mem_data['outdate'].apply(
            lambda x: CALENDAR_UTIL.get_latest_n_dates(x[:4] + '-' + x[4:6] + '-' + x[6:], 1)[0] if x is not None else end_date_)
        all_index_mem_data['outdate'] = all_index_mem_data['outdate'].apply(lambda x: x if x <= end_date_ else end_date_)
        all_index_mem_data['indate'] = all_index_mem_data['indate'].apply(lambda x: x if x >= start_date_ else start_date_)
        assert (all_index_mem_data['indate'] <= all_index_mem_data['outdate']).all()
        all_index_mem_data.sort_values(by=['member_code'], inplace=True)
        all_index_mem_data['dates'] = all_index_mem_data.apply(lambda x: CALENDAR_UTIL.get_ranged_trading_dates(x['indate'], x['outdate']), axis=1)
        rtn = all_index_mem_data[['member_code', 'dates']].explode(column='dates')
        rtn.drop_duplicates(subset=['member_code', 'dates'], keep='first', inplace=True)
        rtn.sort_values(by=['dates', 'member_code'], inplace=True)
        rtn = rtn[['dates', 'member_code']]
        rtn.rename(columns={'member_code': 'Code', 'dates': 'Date'}, inplace=True)
        rtn.reset_index(drop=True, inplace=True)
    return rtn


def get_index_member_weight_by_cumprod(scd, ecd, index_name_):
    start_date_ = scd
    end_date_ = ecd
    wind_start_date = start_date_.replace('-', '')
    wind_end_date = end_date_.replace('-', '')
    sql = "select distinct TRADE_DT from AIndexHS300FreeWeight where S_INFO_WINDCODE = '{0}' and TRADE_DT <= '{1}' order by TRADE_DT".format(index_name_, wind_start_date)
    conn = get_wind_conn()
    bwd_dates = pd.read_sql(sql, conn)
    bwd_dates.columns = bwd_dates.columns.str.upper()
    if bwd_dates.empty:
        index_start_date = wind_start_date
    else:
        index_start_date = bwd_dates["TRADE_DT"].iloc[-1]
    # 清洗指数权重表
    if index_name_ == '000906.SH':
        index_start_date = max(index_start_date, '20130329')
    elif index_name_ == '931865.CSI':
        index_start_date = max(index_start_date, '20190329')
    #
    sql = "select TRADE_DT, S_CON_WINDCODE, I_WEIGHT from AIndexHS300FreeWeight where S_INFO_WINDCODE = '{0}' and TRADE_DT " \
          "between '{1}' and '{2}' order by TRADE_DT, S_CON_WINDCODE".format(
        index_name_, index_start_date, wind_end_date)
    free_index_weight_data = pd.read_sql(sql, conn)
    free_index_weight_data.columns = free_index_weight_data.columns.str.upper()
    if free_index_weight_data.empty:
        rtn = pd.DataFrame(columns=['CalcDate', 'Code', index_name_])
        pass
    else:
        assert free_index_weight_data.groupby(["TRADE_DT"])["I_WEIGHT"].sum().between(99.6, 100.3, inclusive="both").all()
        free_index_weight_data.rename(columns={"S_CON_WINDCODE": "S_INFO_WINDCODE"}, inplace=True)
        free_index_weight_data = free_index_weight_data.set_index(keys=['TRADE_DT', 'S_INFO_WINDCODE'])['I_WEIGHT'].unstack()
        free_index_weight_data.fillna(0.0, inplace=True)
        sql = "select TRADE_DT, S_INFO_WINDCODE, S_DQ_PCTCHANGE / 100.0 PCT_CHANGE from AShareEODPrices " \
              "where TRADE_DT between '{0}' and '{1}' order by TRADE_DT, S_INFO_WINDCODE".format(
            free_index_weight_data.index[0], wind_end_date)
        trading_data = pd.read_sql(sql, conn)
        trading_data.columns = trading_data.columns.str.upper()
        trading_data = trading_data[trading_data['S_INFO_WINDCODE'].isin(free_index_weight_data.columns)]
        trading_data = trading_data.set_index(keys=['TRADE_DT', 'S_INFO_WINDCODE'])['PCT_CHANGE'].unstack()
        if not trading_data.columns.equals(free_index_weight_data.columns):
            print("  warning::wind_tools>>index>>")
        free_index_weight_data = free_index_weight_data[trading_data.columns].copy()
        assert free_index_weight_data.index.isin(trading_data.index).all()
        weight_adj_factor = (trading_data + 1.0).cumprod()
        weight_reb_factor = weight_adj_factor.loc[free_index_weight_data.index, :].reindex(weight_adj_factor.index).fillna(method='ffill')
        weight_adj_factor = weight_reb_factor / weight_adj_factor
        free_index_weight_data = free_index_weight_data.reindex(weight_adj_factor.index).fillna(method='ffill')
        free_index_weight_data = free_index_weight_data * weight_adj_factor
        free_index_weight_data = free_index_weight_data.div(free_index_weight_data.sum(axis=1), axis=0)
        rtn = free_index_weight_data[free_index_weight_data.index.to_series().between(wind_start_date, wind_end_date)].stack().rename('weight').reset_index(drop=False)
        rtn = rtn[rtn['weight'] > 0.0].copy()
        rtn.rename(columns={'TRADE_DT': 'CalcDate', 'S_INFO_WINDCODE': 'Code', 'weight': index_name_}, inplace=True)
        rtn['CalcDate'] = rtn['CalcDate'].str[:4] + '-' + rtn['CalcDate'].str[4:6] + '-' + rtn['CalcDate'].str[6:]
    return rtn


def get_index_level_by_wind(scd, ecd, index_name):
    wind_start_date = scd.replace('-', '')
    wind_end_date = ecd.replace('-', '')
    #
    index_suffix = index_name.split(".")[1]
    if index_suffix == "WI":
        db_name = "AIndexWindIndustriesEOD"
    elif index_suffix in ("SZ", "SH", "CSI", "CNI"):
        db_name = "AIndexEODPrices"
    elif index_suffix in ("CS", ):
        db_name = "CBIndexEODPrices"
    else:
        assert False
    #
    sql = "select TRADE_DT, S_INFO_WINDCODE, S_DQ_CLOSE, S_DQ_OPEN, " \
          "S_DQ_HIGH, S_DQ_LOW, S_DQ_PRECLOSE, S_DQ_VOLUME,"\
          "S_DQ_AMOUNT from {0} " \
          "where TRADE_DT between '{1}' and '{2}' and S_INFO_WINDCODE = '{3}' order by TRADE_DT".format(
        db_name, wind_start_date, wind_end_date, index_name)
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename(
        columns={'TRADE_DT': 'CalcDate', 'S_INFO_WINDCODE': 'Code', 'S_DQ_CLOSE': 'close_level', 'S_DQ_OPEN': 'open_level',
                 'S_DQ_HIGH': 'high_level', 'S_DQ_LOW': 'low_level', 'S_DQ_PRECLOSE': 'preclose_level',
                 'S_DQ_VOLUME': 'volume', 'S_DQ_AMOUNT': 'amount'}, errors="raise", inplace=True)
    rtn["amount"] = rtn["amount"] * 1000.0
    rtn["volume"] = rtn["volume"] * 100.0
    rtn['CalcDate'] = rtn['CalcDate'].str[:4] + '-' + rtn['CalcDate'].str[4:6] + '-' + rtn['CalcDate'].str[6:]
    return rtn


def get_common_index_member_weight(scd, ecd):
    assert scd >= '2005-04-08'
    index_nms = ["000300.SH", "000905.SH", "000852.SH", "932000.CSI"]
    wind_start_date = scd.replace('-', '')
    wind_end_date = ecd.replace('-', '')
    sql = "select distinct TRADE_DT, S_INFO_WINDCODE from AIndexHS300FreeWeight "\
          "where TRADE_DT <= '{0}' and S_INFO_WINDCODE in ('{1}') "\
          "order by TRADE_DT".format(wind_start_date, "','".join(index_nms))
    conn = get_wind_conn()
    bwd_dates = pd.read_sql(sql, conn)
    bwd_dates.columns = bwd_dates.columns.str.upper()
    bwd_dates.rename(
        columns={'TRADE_DT': 'TradeDate', 'S_INFO_WINDCODE': 'index_code'}, errors="raise", inplace=True)
    index_start_date = bwd_dates[["TradeDate", "index_code"]].drop_duplicates(subset=['index_code'], keep="last")["TradeDate"].min()
    sql = "select TRADE_DT, S_INFO_WINDCODE, S_CON_WINDCODE, I_WEIGHT from AIndexHS300FreeWeight "\
          "where TRADE_DT between '{0}' and '{1}' and S_INFO_WINDCODE in ('{2}')"\
          "order by TRADE_DT, S_CON_WINDCODE".format(index_start_date, wind_end_date, "','".join(index_nms))
    free_index_weight_data = pd.read_sql(sql, conn)
    free_index_weight_data.columns = free_index_weight_data.columns.str.upper()
    free_index_weight_data.rename(
        columns={'TRADE_DT': 'TradeDate', 'S_INFO_WINDCODE': 'index_code', 'S_CON_WINDCODE': 'Code'}, errors="raise", inplace=True)
    # 清洗指数权重表
    free_index_weight_data = free_index_weight_data.loc[~((free_index_weight_data['index_code'] == '000300.SH') & (free_index_weight_data['TradeDate'] < '20101231'))]
    free_index_weight_data = free_index_weight_data.loc[~((free_index_weight_data['index_code'] == '000905.SH') & (free_index_weight_data['TradeDate'] < '20130329'))]
    free_index_weight_data = free_index_weight_data.loc[~((free_index_weight_data['index_code'] == '000852.SH') & (free_index_weight_data['TradeDate'] < '20150529'))]
    free_index_weight_data = free_index_weight_data.loc[~((free_index_weight_data['index_code'] == '932000.CSI') & (free_index_weight_data['TradeDate'] < '20230831'))]
    #
    assert (free_index_weight_data.groupby(["TradeDate", "index_code"])["I_WEIGHT"].sum() >= 99).all()
    #
    sql = "select TRADE_DT, S_INFO_WINDCODE, S_DQ_PCTCHANGE / 100.0 PCT_CHANGE from AShareEODPrices " \
          "where TRADE_DT between '{0}' and '{1}' order by TRADE_DT, S_INFO_WINDCODE".format(
        index_start_date, wind_end_date)
    all_trading_data = pd.read_sql(sql, conn)
    all_trading_data.columns = all_trading_data.columns.str.upper()
    all_trading_data.rename(
        columns={'TRADE_DT': 'TradeDate', 'S_INFO_WINDCODE': 'Code', 'PCT_CHANGE': 'pct_change'}, errors="raise", inplace=True)
    all_trading_data = all_trading_data.set_index(keys=['TradeDate', 'Code'])['pct_change'].unstack()
    #
    rtn = list()
    for index_nm in free_index_weight_data["index_code"].unique().tolist():
        index_weight = free_index_weight_data.query("index_code == '{0}'".format(index_nm)).set_index(
            keys=['TradeDate', 'Code'])['I_WEIGHT'].unstack()
        index_weight.fillna(0.0, inplace=True)
        trading_data = all_trading_data.T.reindex(index_weight.columns).T
        trading_data.fillna(0.0, inplace=True)
        assert index_weight.index.isin(trading_data.index).all()
        weight_adj_factor = (trading_data + 1.0).cumprod()
        weight_reb_factor = weight_adj_factor.loc[index_weight.index, :].reindex(weight_adj_factor.index).fillna(method='ffill')
        weight_adj_factor = weight_reb_factor / weight_adj_factor
        index_weight = index_weight.reindex(weight_adj_factor.index).fillna(method='ffill')
        index_weight = index_weight * weight_adj_factor
        index_weight.index.rename("CalcDate", inplace=True)
        rtn.append(index_weight.stack().rename(index_nm))
    rtn = pd.concat(rtn, axis=1, sort=True)
    rtn.mask(rtn < 1e-8, other=np.nan, inplace=True)
    rtn.dropna(axis=0, how="all", inplace=True)
    #
    conflict_flg = (rtn.count(axis=1) > 1)
    if conflict_flg.any():
        conflict_data = rtn[conflict_flg].copy()
        conflict_data.columns.rename("index_code", inplace=True)
        conflict_data = conflict_data.stack().rename("weight").reset_index(drop=False)
        conflict_data["CalcDate_alias"] = conflict_data["CalcDate"].astype(int)
        free_index_weight_data["TradeDate_alias"] = free_index_weight_data["TradeDate"].astype(int)
        conflict_data = pd.merge_asof(conflict_data, free_index_weight_data[["TradeDate_alias", "TradeDate", "Code", "index_code"]],
                                      left_on="CalcDate_alias", right_on="TradeDate_alias",
                                      by=["Code", "index_code"], direction="backward", allow_exact_matches=True)
        assert not conflict_data.duplicated(["CalcDate", "Code", "TradeDate"]).any()
        conflict_data.sort_values(["CalcDate", "Code", "TradeDate"], ascending=True, inplace=True)
        conflict_data["weight"].mask(conflict_data.duplicated(subset=["CalcDate", "Code"], keep="last"), other=np.nan, inplace=True)
        conflict_data = conflict_data.set_index(["CalcDate", "Code", "index_code"])["weight"].unstack()
        rtn = pd.concat((rtn[~conflict_flg], conflict_data), axis=0).sort_index()
    rtn = rtn / rtn.groupby(["CalcDate"]).sum()
    rtn[list(set(index_nms) - set(rtn.columns))] = 0.0
    rtn.fillna(0.0, inplace=True)
    rtn.reset_index(drop=False, inplace=True)
    rtn['CalcDate'] = rtn['CalcDate'].str[:4] + '-' + rtn['CalcDate'].str[4:6] + '-' + rtn['CalcDate'].str[6:]
    rtn = rtn.loc[rtn["CalcDate"].between(scd, ecd, inclusive="both"), ["CalcDate", "Code"] + index_nms].copy()
    return rtn