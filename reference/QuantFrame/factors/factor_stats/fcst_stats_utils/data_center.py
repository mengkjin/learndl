from daily_bar.api import load_daily_bar_data
from events_system.calendar_util import CALENDAR_UTIL
import numpy as np
import pandas as pd
from stk_index_utils.api import load_index_level


INDEX_LIST = ['000300.SH', '000905.SH', '000852.SH']


def prepare_ret_data(root_path, scd, ecd, price_type):
    daily_bar = load_daily_bar_data(root_path, "basic", scd, ecd)
    if price_type == "close":
        rtn = daily_bar.set_index(["CalcDate", "Code"])["log_ret"].unstack().fillna(0.0)
    elif price_type == "vwap":
        rtn = daily_bar.set_index(["CalcDate", "Code"])["vwp_log_ret"].unstack().fillna(0.0)
    else:
        assert False, "  error:>>price type {0} is unknown!".format(price_type)
    return rtn


def prepare_index_ret_data(root_path, scd, ecd):
    rtn = []
    for bm_index_nm in INDEX_LIST:
        index_close = load_index_level(
            root_path, CALENDAR_UTIL.get_latest_n_trading_dates(scd, 2)[0], ecd, "publish:" + bm_index_nm)
        index_close = index_close.set_index(["CalcDate"])["close_level"]
        index_ret = np.log(index_close / index_close.shift(1))
        index_ret = index_ret[scd:].to_frame(name=bm_index_nm)
        rtn.append(index_ret)
    rtn = pd.concat(rtn, axis=1, sort=True)
    rtn["000000.SH"] = 0.0
    return rtn


class DataVendor:
    def __init__(self):
        self.stock_data = pd.DataFrame()
        self.index_data = pd.DataFrame()
        self.root_path = None

    def init_environments(self, root_path):
        self.root_path = root_path

    def _update_stock_data(self, scd, ecd, ret_type, price_type):
        assert ret_type == 'prd_ret'
        date_range = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
        calc_date_exist = self.stock_data.index.tolist()
        date_sub = list(set(date_range).difference(set(calc_date_exist)))
        rets_added = prepare_ret_data(self.root_path, min(date_sub), max(date_sub), price_type)
        self.stock_data = pd.concat((self.stock_data, rets_added.loc[date_sub]), axis=0).sort_index()
        if self.stock_data.columns.name is None:
            self.stock_data.columns.name = "Code"

    def _update_index_data(self, scd, ecd):
        date_range = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
        calc_date_exist = self.index_data.index.tolist()
        date_sub = list(set(date_range).difference(set(calc_date_exist)))
        rets_added = prepare_index_ret_data(self.root_path, min(date_sub), max(date_sub))
        self.index_data = pd.concat((self.index_data, rets_added.loc[date_sub]), axis=0).sort_index()
        self.index_data.index.name = "CalcDate"

    def query_period_data(self, y_info, ret_type, bm_index_nm, price_type):
        assert ret_type == 'prd_ret'
        assert y_info.columns.equals(pd.Index(["Code", "y_start", "y_end"]))
        syd, eyd = y_info["y_start"].min(), y_info["y_end"].max()
        trd_dates = CALENDAR_UTIL.get_ranged_trading_dates(syd, eyd)
        assert set(y_info["y_start"]).issubset(trd_dates) and set(y_info["y_end"]).issubset(trd_dates)
        if not set(trd_dates).issubset(self.stock_data.index):
            self._update_stock_data(syd, eyd, ret_type, price_type)
        if not set(trd_dates).issubset(self.index_data.index):
            self._update_index_data(syd, eyd)
        stock_cum_ret = self.stock_data[sorted(list(set(self.stock_data.columns) & set(y_info["Code"].unique())))].cumsum(axis=0).stack().\
            rename("stk_cum_ret").reset_index(drop=False)
        stk_y = pd.merge(
            y_info,
            stock_cum_ret.rename(columns={"CalcDate": "y_start", "stk_cum_ret": "stk_start_cum_ret"}, errors="raise"),
            on=["y_start", "Code"], how="left")
        stk_y = pd.merge(
            stk_y,
            stock_cum_ret.rename(columns={"CalcDate": "y_end", "stk_cum_ret": "stk_end_cum_ret"}, errors="raise"),
            on=["y_end", "Code"], how="left")
        stk_y["stk_prd_ret"] = np.exp(stk_y["stk_end_cum_ret"] - stk_y["stk_start_cum_ret"]) - 1
        #
        if bm_index_nm == "999999.SH":
            stk_y = pd.merge(stk_y, y_info, how="inner", on=["Code", "y_start", "y_end"], sort=True)
            stk_mean_ret = stk_y.groupby(["y_start", "y_end"])["stk_prd_ret"].mean().rename("idx_prd_ret").reset_index(drop=False)
            rtn = pd.merge(stk_y, stk_mean_ret, on=["y_start", "y_end"], how="left")
        elif bm_index_nm == "888888.SH":
            from stk_ret_tools.market_ret import calc_sqrt_fv_wt_market_ret
            market_ret = calc_sqrt_fv_wt_market_ret(self.root_path, syd, eyd)
            market_cum_ret = np.log(market_ret + 1).cumsum()
            market_cum_ret = market_cum_ret.to_frame("index_cum_ret").reset_index(drop=False)
            rtn = pd.merge(
                stk_y,
                market_cum_ret.rename(columns={"CalcDate": "y_start", "index_cum_ret": "idx_start_cum_ret"},
                                      errors="raise"),
                on=["y_start"], how="left")
            rtn = pd.merge(
                rtn,
                market_cum_ret.rename(columns={"CalcDate": "y_end", "index_cum_ret": "idx_end_cum_ret"}, errors="raise"),
                on=["y_end"], how="left")
            rtn["idx_prd_ret"] = np.exp(rtn["idx_end_cum_ret"] - rtn["idx_start_cum_ret"]) - 1
            rtn.drop(columns=["idx_start_cum_ret", "idx_end_cum_ret"], inplace=True)
        elif bm_index_nm in INDEX_LIST + ["000000.SH"]:
            index_cum_ret = self.index_data[bm_index_nm].cumsum(axis=0).rename("index_cum_ret").reset_index(drop=False)
            rtn = pd.merge(
                stk_y,
                index_cum_ret.rename(columns={"CalcDate": "y_start", "index_cum_ret": "idx_start_cum_ret"}, errors="raise"),
                on=["y_start"], how="left")
            rtn = pd.merge(
                rtn,
                index_cum_ret.rename(columns={"CalcDate": "y_end", "index_cum_ret": "idx_end_cum_ret"}, errors="raise"),
                on=["y_end"], how="left")
            rtn["idx_prd_ret"] = np.exp(rtn["idx_end_cum_ret"] - rtn["idx_start_cum_ret"]) - 1
            rtn.drop(columns=["idx_start_cum_ret", "idx_end_cum_ret"], inplace=True)
        else:
            assert False, "  error:>>data_center>>{0} is unknown!".format(bm_index_nm)
        rtn["y"] = rtn["stk_prd_ret"] - rtn["idx_prd_ret"]
        rtn = pd.merge(rtn, y_info, how="inner", on=["Code", "y_start", "y_end"], sort=True)
        rtn = rtn[['Code', 'y_start', 'y_end', "y"]].copy()
        return rtn


DATAVENDOR = DataVendor()