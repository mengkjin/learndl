import pandas as pd
import os
from basic_src_data.wind_tools.basic import get_listed_ashare_codes_from_winddf


def get_insurance_stock(scd, ecd):
    insurance_stk = pd.read_csv(os.path.join(os.path.dirname(__file__), 'insurance_stk_list.csv'))
    data = get_listed_ashare_codes_from_winddf(scd, ecd)
    rtn = pd.merge(data, insurance_stk, how="inner", on=['Code'])
    rtn = rtn.rename(columns={"Date": "CalcDate"}, errors="raise").sort_values(["CalcDate", "Code"])
    return rtn