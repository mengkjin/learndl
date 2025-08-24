import pandas as pd


def calc_derived_data(daily_bar_data, fld_list, keep_origin=True):
    rtn = daily_bar_data.copy()
    if "hlv" in fld_list:
        rtn["hlv"] = rtn["high_price"] / rtn["low_price"] - 1.0
    if "cor" in fld_list:
        rtn["cor"] = rtn["close_price"] / rtn["open_price"] - 1.0
    if "opr" in fld_list:
        rtn["opr"] = rtn["open_price"] / rtn["prev_close"] - 1.0
    if not keep_origin:
        rtn = daily_bar_data[["CalcDate", "Code"] + fld_list].copy()
    return rtn