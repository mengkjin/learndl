from .configs import *
from events_system.calendar_util import CALENDAR_UTIL
import pandas as pd


DATA_VENDOR = dict()


def load_special_vol(root_path, barra_type, scd, ecd):
    path = os.path.join(get_model_path(root_path, barra_type), 'special_vol')
    start_y, end_y = scd[:4], ecd[:4]
    df = list()
    for y in range(int(start_y), int(end_y) + 1):
        file = os.path.join(path, str(y) + '.h5')
        if os.path.exists(file):
            y_data = pd.read_hdf(file, key='df')
            df.append(y_data)
    if df:
        df = pd.concat(df, axis=0)
        df = df[df['CalcDate'].between(scd, ecd, inclusive="both")].copy()
        df_dates = df['CalcDate'].unique().tolist()
        required_trading_dates = CALENDAR_UTIL.get_ranged_trading_dates(df_dates[0], df_dates[-1])
        assert required_trading_dates == df_dates
        rtn = df
    else:
        assert False, "  error::barra>>api>>load_special_vol>>can not find special vol " \
                      "from '{0}' to '{1}' for barra type:{2}.".format(scd, ecd, barra_type)
    return rtn


def load_risk_cov(root_path, barra_type, scd, ecd):
    path = os.path.join(get_model_path(root_path, barra_type), 'risk_cov')
    start_y, end_y = scd[:4], ecd[:4]
    df = list()
    for y in range(int(start_y), int(end_y) + 1):
        file = os.path.join(path, str(y) + '.h5')
        if os.path.exists(file):
            y_data = pd.read_hdf(file, key='df')
            df.append(y_data)
    if df:
        df = pd.concat(df, axis=0)
        df = df[df['CalcDate'].between(scd, ecd, inclusive="both")].copy()
        df_dates = df['CalcDate'].unique().tolist()
        required_dates = CALENDAR_UTIL.get_ranged_trading_dates(df_dates[0], df_dates[-1])
        assert required_dates == df_dates
        rtn = df.fillna(0.0)
    else:
        assert False, "  error::barra>>api>>load_risk_cov>>can not find risk cov " \
                      "from '{0}' to '{1}' for barra type:{2}.".format(scd, ecd, barra_type)
    return rtn