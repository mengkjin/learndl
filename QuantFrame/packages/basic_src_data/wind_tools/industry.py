import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from .wind_conn import get_wind_conn


REMOTE_DATE = "2999-12-31"


def get_industry_member_from_winddf(scd, ecd, ind_type, level_list=(1, 2, 3)):
    assert isinstance(level_list, tuple) and all(isinstance(lv, int) for lv in level_list)
    if ind_type == "citics":
        code_prefix = "b1"
        sql = "select distinct S_INFO_WINDCODE CODE, CITICS_IND_CODE IND_CODE, ENTRY_DT ENTRY_DT, REMOVE_DT REMOVE_DT " \
              "from AShareIndustriesClassCITICS where ENTRY_DT <= '{1}' and (REMOVE_DT >= '{0}' or REMOVE_DT is null) " \
              "order by ENTRY_DT, S_INFO_WINDCODE".format(scd.replace('-', ''), ecd.replace('-', ''))
        conn = get_wind_conn()
        all_data = pd.read_sql(sql, conn)
    elif ind_type == "sw":
        code_prefix = "76"
        sql = "select distinct S_INFO_WINDCODE CODE, SW_IND_CODE IND_CODE, ENTRY_DT ENTRY_DT, REMOVE_DT REMOVE_DT " \
              "from AShareSWNIndustriesClass where ENTRY_DT <= '{1}' and (REMOVE_DT >= '{0}' or REMOVE_DT is null) " \
              "order by ENTRY_DT, S_INFO_WINDCODE".format(scd.replace('-', ''), ecd.replace('-', ''))
        conn = get_wind_conn()
        all_data = pd.read_sql(sql, conn)
    elif ind_type == "cj":
        code_prefix = "b2"
        sql = "select distinct S_INFO_WINDCODE CODE, SEC_IND_CODE IND_CODE, ENTRY_DT ENTRY_DT, REMOVE_DT REMOVE_DT " \
              "from AShareCJZQIndustriesClass where ENTRY_DT <= '{1}' and (REMOVE_DT >= '{0}' or REMOVE_DT is null) " \
              "order by ENTRY_DT, S_INFO_WINDCODE".format(scd.replace('-', ''), ecd.replace('-', ''))
        conn = get_wind_conn()
        all_data = pd.read_sql(sql, conn)
    else:
        assert False
    all_data.columns = all_data.columns.str.upper()
    all_data.rename(columns={"CODE": "Code", "IND_CODE": "ind_code", "ENTRY_DT": "entry_dt", "REMOVE_DT": "remove_dt"},
                    errors="raise", inplace=True)
    #sql = "select distinct INDUSTRIESCODE IND_CODE, INDUSTRIESNAME IND_NM, LEVELNUM LEVEL_NUM " \
    #      "from AShareIndustriesCode where left(INDUSTRIESCODE, 2) = '{0}'".format(code_prefix)
    sql = "select distinct INDUSTRIESCODE IND_CODE, INDUSTRIESNAME IND_NM, LEVELNUM LEVEL_NUM " \
          "from AShareIndustriesCode where substr(INDUSTRIESCODE, 1, 2) = '{0}'".format(code_prefix)
    conn = get_wind_conn()
    ind_code_nm_df = pd.read_sql(sql, conn)
    ind_code_nm_df.columns = ind_code_nm_df.columns.str.upper()
    ind_code_nm_df.rename(columns={"IND_CODE": "ind_code", "IND_NM": "ind_nm", "LEVEL_NUM": "level_num"},
                          errors="raise", inplace=True)
    #
    all_data = all_data[all_data["Code"].str[-2:].isin(["SH", "SZ"])].copy()
    all_data['entry_dt'] = all_data['entry_dt'].str[:4] + "-" + all_data['entry_dt'].str[4:6] + "-" + all_data['entry_dt'].str[6:8]
    all_data['remove_dt'].fillna(REMOTE_DATE.replace("-", ""), inplace=True)
    all_data['remove_dt'] = all_data['remove_dt'].str[:4] + "-" + all_data['remove_dt'].str[4:6] + "-" + all_data['remove_dt'].str[6:8]
    #
    for level in level_list:
        ind_level_code_nm = ind_code_nm_df[ind_code_nm_df["level_num"] == (level + 1)].copy()
        str_len = 2 + level * 2
        ind_level_code_nm["ind_code_level"] = ind_level_code_nm["ind_code"].str[:str_len]
        all_data["ind_code_level"] = all_data["ind_code"].str[:str_len]
        all_data = pd.merge(all_data, ind_level_code_nm[["ind_code_level", "ind_nm"]], how="left",
                            on=["ind_code_level"]).drop(columns=["ind_code_level"]).\
            rename(columns={"ind_nm": "{0}_{1}".format(ind_type, str(level))}, errors="raise")
    all_data.drop(columns=["ind_code"], inplace=True)
    assert all_data.notna().all().all()
    #
    all_data["CalcDate"] = all_data.apply(
        lambda x: CALENDAR_UTIL.get_ranged_dates(max(scd, x["entry_dt"]), min(ecd, x["remove_dt"])), axis=1)
    rtn = all_data.drop(columns=["entry_dt", "remove_dt"]).explode(column="CalcDate")
    assert not rtn[["CalcDate", "Code"]].duplicated().any(), \
        "  error::>>industry>>there may be stock whose new entry_dt less than backup remove_dt"
    rtn = rtn[pd.Index(["CalcDate", "Code"]).append(rtn.columns.drop(["CalcDate", "Code"]))].sort_values(
        ["CalcDate", "Code"]).reset_index(drop=True)
    return rtn