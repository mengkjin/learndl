import pandas as pd
from events_system.calendar_util import CALENDAR_UTIL
from basic_src_data.wind_tools.industry import get_industry_member_from_winddf
from industry_tools.citics import from_ch_to_en


def gen_2002_citics_industry():
    data = get_industry_member_from_winddf('2003-01-01', '2003-01-01', 'citics', level_list=(1, 2, 3))
    data = data[data['Code'].str[0] != 'T'].copy()
    date_list = CALENDAR_UTIL.get_ranged_dates('2002-01-01', '2002-12-31')
    data['CalcDate'] = [date_list] * len(data)
    data = data.explode('CalcDate')
    rtn = data[['CalcDate', 'Code', 'citics_1', 'citics_2', 'citics_3']].sort_values(['CalcDate', 'Code'])
    return rtn


def gen_non_2002_citics_industry(scd, ecd):
    assert (scd <= ecd) and (scd >= '2003-01-01')
    rtn = get_industry_member_from_winddf(scd, ecd, 'citics', level_list=(1, 2, 3))
    return rtn


def _patch_industry_data(data):
    assert data["CalcDate"].is_monotonic_increasing
    scd, ecd = data["CalcDate"].iloc[0], data["CalcDate"].iloc[-1]
    patch_info = [
        ("601598.SH", "2019-01-18", "2019-03-13", "CommunicationsAndTransportation", "航运港口", "航运"),
        ("601868.SH", "2021-09-28", "2021-10-17", "BuildingDecoration", "建筑施工", "专业工程及其他"),
        ("600087.SH", "1997-06-12", "2014-04-12", "CommunicationsAndTransportation", "航运港口", "航运"),
    ]
    patch_info_chosen = [info for info in patch_info if not (ecd < info[1] or scd > info[2])]
    if patch_info_chosen:
        patch_data = pd.DataFrame(patch_info_chosen, columns=["Code", "start_date", "end_date", "citics_1", "citics_2", "citics_3"])
        patch_data["CalcDate"] = patch_data.apply(
            lambda x: CALENDAR_UTIL.get_ranged_dates(max(x["start_date"], scd), min(x["end_date"], ecd)), axis=1)
        patch_data = patch_data[["CalcDate", "Code", "citics_1", "citics_2", "citics_3"]].explode(column="CalcDate")
        rtn = pd.concat((data, patch_data), axis=0).sort_values(["CalcDate", "Code"]).drop_duplicates(
            ["CalcDate", "Code"], keep="first").reset_index(drop=True)
    else:
        rtn = data
    return rtn


def gen_citics_data(scd, ecd):
    assert scd >= '2002-01-01'
    rtn = list()
    if scd <= '2002-12-31':
        assert scd == '2002-01-01' and ecd >= '2003-01-01'
        data_2002 = gen_2002_citics_industry()
        rtn.append(data_2002)
    wind_scd = max(scd, '2003-01-01')
    wind_ecd = ecd
    data_non_2002 = gen_non_2002_citics_industry(wind_scd, wind_ecd)
    rtn.append(data_non_2002)
    rtn = pd.concat(rtn, axis=0)
    #
    ind_nm_df = rtn[["citics_1"]].drop_duplicates()
    ind_nm_df["citics_1_en"] = from_ch_to_en(ind_nm_df["citics_1"])
    rtn = pd.merge(rtn, ind_nm_df, how="left", on=["citics_1"]).drop(columns=["citics_1"]).rename(
        columns={"citics_1_en": "citics_1"}, errors="raise")
    rtn = rtn[['CalcDate', 'Code', 'citics_1', 'citics_2', 'citics_3']].\
        sort_values(["CalcDate", "Code"]).reset_index(drop=True)
    rtn = _patch_industry_data(rtn)
    return rtn
