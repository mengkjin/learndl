import pandas as pd
from basic_src_data.wind_tools.comp_info import load_ashare_comp_id, load_comp_property_data
from basic_src_data.wind_tools.basic import get_listed_ashare_codes_from_winddf


def calc_company_property(scd, ecd):
    ashare_description = load_ashare_comp_id()
    comp_introduction = load_comp_property_data()
    comp_property = pd.merge(ashare_description, comp_introduction, on=["comp_id"])
    comp_property["Property_State"] = (comp_property["comp_property"].isin(["中央国有企业", "地方国有企业"])) * 1
    #
    listed_stocks = get_listed_ashare_codes_from_winddf(scd, ecd)
    listed_stocks = listed_stocks.rename(columns={"Date": "CalcDate"}, errors="raise")
    rtn = pd.merge(listed_stocks, comp_property[["Code", "Property_State"]],
                   how="inner", on=["Code"]).sort_values(["CalcDate", "Code"])
    rtn = rtn.set_index(["CalcDate", "Code"])["Property_State"]
    return rtn