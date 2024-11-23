import os
import pandas as pd
from basic_src_data.wind_tools.basic import get_stock_description, get_stock_st_info, DISTANT_DATE
from .config import get_file_path


def _calc_description_data():
    EARLY_KICK_WINSIZE = 60
    df = get_stock_description()
    df["early_delist_date"] = DISTANT_DATE
    delist_flg = df["delist_date"] < DISTANT_DATE
    df.loc[delist_flg, "early_delist_date"] = (pd.to_datetime(df.loc[delist_flg, "delist_date"]) - pd.Timedelta(days=EARLY_KICK_WINSIZE)).dt.strftime("%Y-%m-%d")
    df = df[["Code", "list_date", "delist_date", "early_delist_date", "stock_name", "list_board", "comp_id"]].sort_values(["Code"])
    return df


def _save_data(df, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    df.to_csv(file_path, index=False, encoding="gbk")


def gen_stk_basic_info_data(root_path, category):
    if category == "description":
        df = _calc_description_data()
    elif category == "st":
        df = get_stock_st_info()
    else:
        assert False
    file_path = get_file_path(root_path, category)
    _save_data(df, file_path)
