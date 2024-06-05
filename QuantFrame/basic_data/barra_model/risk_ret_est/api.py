import pandas as pd
import os
from events_system.calendar_util import CALENDAR_UTIL
from crosec_mem.data_vendor import DataVendor
from .configs import get_risk_ret_path, get_special_ret_path


DATAVENDOR = dict()


def load_special_ret(root_path, barra_type, scd, ecd):
    path = get_special_ret_path(root_path, barra_type)
    if path not in DATAVENDOR:
        DATAVENDOR[path] = DataVendor(path, 'barra_special_risk>>{0}'.format(barra_type))
    rtn = DATAVENDOR[path].load_data(scd, ecd, 'trade')
    return rtn


def load_risk_ret(root_path, barra_type, scd, ecd):
    path = get_risk_ret_path(root_path, barra_type)
    file = os.path.join(path, 'risk_ret.csv')
    if os.path.exists(file):
        rtn = pd.read_csv(file, index_col=0)
    else:
        assert False, "risk_ret>>api>>load>>can not find risk ret from '{0}' to '{1}'".format(scd, ecd)
    #
    rtn = rtn.loc[scd: ecd].copy()
    trade_date_list = CALENDAR_UTIL.get_ranged_trading_dates(scd, ecd)
    assert rtn.index.tolist() == trade_date_list
    return rtn

def get_riskmodel_lastest_days(root_path, barra_type, days):
    path = get_risk_ret_path(root_path, barra_type)
    file = os.path.join(path, 'risk_ret.csv')
    if os.path.exists(file):
        rtn = pd.read_csv(file, index_col=0)
    else:
        assert False, "risk_ret>>api>>load>>can not find risk ret"
    #
    return rtn.index[-1*days], rtn.index[-1]