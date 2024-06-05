import pandas as pd
from .wind_conn import get_wind_conn


def load_comp_property_data():
    sql = "select COMP_ID, COMP_PROPERTY "\
          "from CompIntroduction where IS_LISTED=1 and COMP_PROPERTY is not null"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, con=conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename({"COMP_ID": "comp_id", "COMP_PROPERTY": "comp_property"}, axis=1, inplace=True)
    return rtn


def load_ashare_comp_id():
    sql = "select S_INFO_COMPCODE, S_INFO_WINDCODE from AShareDescription"
    conn = get_wind_conn()
    rtn = pd.read_sql(sql, con=conn)
    rtn.columns = rtn.columns.str.upper()
    rtn.rename({"S_INFO_WINDCODE": "Code", "S_INFO_COMPCODE": "comp_id"}, axis=1, inplace=True)
    return rtn
