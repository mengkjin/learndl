from ashare_stock_id_sys.impl import *
import numpy as np


def get_sys_id(code_list_):
    assert isinstance(code_list_, (list, tuple, np.ndarray))
    rtn = ID_SYS.get_sys_id(code_list_)
    return rtn


def get_tickers(sys_id_list_):
    rtn = ID_SYS.get_tickers(sys_id_list_)
    return rtn


def get_all_sys_ids():
    rtn = ID_SYS.get_all_sys_ids()
    return rtn


def get_all_tickers():
    rtn = ID_SYS.get_all_tickers()
    return rtn