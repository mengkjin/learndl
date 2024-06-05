import pandas as pd
from daily_bar.api import load_daily_bar_data
from basic_src_data.wind_tools.wind_conn import get_wind_conn


def get_index_member_weight_by_cumprod(root_path, scd, ecd, index_name):
    start_date_ = scd
    end_date_ = ecd
    wind_start_date = start_date_.replace('-', '')
    wind_end_date = end_date_.replace('-', '')
    sql = "select distinct TRADE_DT from AIndexHS300FreeWeight where S_INFO_WINDCODE = '{0}' and TRADE_DT <= '{1}' order by TRADE_DT".format(index_name, wind_start_date)
    conn = get_wind_conn()
    bwd_dates = pd.read_sql(sql, conn)
    if bwd_dates.empty:
        index_start_date = wind_start_date
    else:
        index_start_date = bwd_dates["TRADE_DT"].iloc[-1]
    sql = "select TRADE_DT, S_CON_WINDCODE, I_WEIGHT I_WEIGHT from AIndexHS300FreeWeight where S_INFO_WINDCODE = '{0}' and TRADE_DT " \
          "between '{1}' and '{2}' order by TRADE_DT, S_CON_WINDCODE".format(
        index_name, index_start_date, wind_end_date)
    free_index_weight_data = pd.read_sql(sql, conn)
    if free_index_weight_data.empty:
        rtn = pd.DataFrame(columns=['CalcDate', 'Code', index_name])
        pass
    else:
        assert free_index_weight_data.groupby(["TRADE_DT"])["I_WEIGHT"].sum().between(99.6, 100.3, inclusive="both").all()
        free_index_weight_data.rename(columns={"S_CON_WINDCODE": "S_INFO_WINDCODE"}, inplace=True)
        free_index_weight_data = free_index_weight_data.set_index(keys=['TRADE_DT', 'S_INFO_WINDCODE'])['I_WEIGHT'].unstack()
        free_index_weight_data.fillna(0.0, inplace=True)
        #
        dbar_sd = free_index_weight_data.index[0]
        dbar_sd = "-".join([dbar_sd[:4], dbar_sd[4:6], dbar_sd[6:8]])
        trading_data = load_daily_bar_data(root_path, "basic", dbar_sd, ecd)[["CalcDate", "Code", "ret"]].copy()
        trading_data["CalcDate"] = trading_data["CalcDate"].str.replace("-", "")
        trading_data.rename(columns={"CalcDate": "TRADE_DT", "Code": "S_INFO_WINDCODE"}, errors="raise", inplace=True)
        trading_data = trading_data[trading_data['S_INFO_WINDCODE'].isin(free_index_weight_data.columns)]
        trading_data = trading_data.set_index(keys=['TRADE_DT', 'S_INFO_WINDCODE'])['ret'].unstack()
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
        rtn.rename(columns={'TRADE_DT': 'CalcDate', 'S_INFO_WINDCODE': 'Code', 'weight': index_name}, inplace=True)
        rtn['CalcDate'] = rtn['CalcDate'].str[:4] + '-' + rtn['CalcDate'].str[4:6] + '-' + rtn['CalcDate'].str[6:]
    return rtn
