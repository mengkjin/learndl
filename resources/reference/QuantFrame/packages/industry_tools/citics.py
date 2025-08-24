import pandas as pd
import os


DICT_FILE = os.path.join(os.path.dirname(__file__), "citics_industry_dict.csv")


def from_ch_to_en(ch_nm_list_):
    MAP = pd.read_csv(DICT_FILE, encoding='gbk', index_col=0).to_dict()["EN_NM"]
    rtn = list()
    for ch_nm in ch_nm_list_:
        rtn.append(
            MAP.get(ch_nm)
        )
    return rtn


def from_en_to_id(en_nm_list_):
    df = pd.read_csv(DICT_FILE, encoding='gbk')
    df.sort_values(by=["EN_NM"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["id"] = df.index
    a = pd.merge(
        pd.DataFrame(en_nm_list_, columns=["EN_NM"]),
        df,
        how="left",
        on=["EN_NM"]
    )
    rtn = a["id"].astype(int).values
    return rtn


def from_ch_to_id(ch_nm_list_):
    en_nm = from_ch_to_en(ch_nm_list_)
    rtn = from_en_to_id(en_nm)
    return rtn