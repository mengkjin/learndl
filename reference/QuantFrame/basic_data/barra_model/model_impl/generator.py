import numpy as np
import pandas as pd
from barra_model.risk_ret_est.api import load_risk_ret, load_special_ret
import os
from .barra_cov import estimate_cov
from events_system.calendar_util import CALENDAR_UTIL


def get_model_path(root_path, barra_type):
    path = os.path.sep.join([root_path, 'barra_data', 'models', barra_type])
    return path


def _save_special_vol(root_path, data, barra_type):
    path = os.path.join(get_model_path(root_path, barra_type), 'special_vol')
    if not os.path.exists(path):
        os.makedirs(path)
    df = data
    df.sort_values(by=['CalcDate', 'Code'], inplace=True)
    years = df['CalcDate'].str[:4]
    for y in years.unique():
        file = os.path.join(path, y + '.h5')
        if os.path.exists(file):
            y_data = pd.read_hdf(file, key='df')
            y_df = pd.concat((y_data, df[years == y]), axis=0)
            y_df.drop_duplicates(subset=['CalcDate', 'Code'], keep='first', inplace=True)
            y_df.sort_values(by=['CalcDate', 'Code'], inplace=True)
        else:
            y_df = df[years == y].copy()
        assert y_df["CalcDate"].unique().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(
            y_df["CalcDate"].iloc[0], y_df["CalcDate"].iloc[-1])
        y_df.reset_index(drop=True, inplace=True)
        y_df.to_hdf(file, key='df')


def _save_risk_cov(root_path, data, barra_type):
    path = os.path.join(get_model_path(root_path, barra_type), 'risk_cov')
    if not os.path.exists(path):
        os.makedirs(path)
    df = data
    df.sort_values(by=['CalcDate', 'FactorName'], inplace=True)
    years = df['CalcDate'].str[:4]
    for y in years.unique():
        file = os.path.join(path, y + '.h5')
        if os.path.exists(file):
            y_data = pd.read_hdf(file, key='df')
            y_df = pd.concat((y_data, df[years == y]), axis=0)
            y_df.drop_duplicates(subset=['CalcDate', 'FactorName'], keep='first', inplace=True)
            y_df.sort_values(by=['CalcDate', 'FactorName'], inplace=True)
        else:
            y_df = df[years == y].copy()
        assert y_df["CalcDate"].unique().tolist() == CALENDAR_UTIL.get_ranged_trading_dates(
            y_df["CalcDate"].iloc[0], y_df["CalcDate"].iloc[-1])
        y_df.reset_index(drop=True, inplace=True)
        y_df.to_hdf(file, key='df')


def gen_risk_cov(root_path, barra_type, scd, ecd):
    rolling_winsize = 504
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=False, n=rolling_winsize)[0]
    risk_ret = load_risk_ret(root_path, barra_type, data_sd, ecd)
    if 'INDUSTRY.SynFinance' in risk_ret.columns:
        risk_ret["INDUSTRY.SynFinance"].fillna(0.0, inplace=True)
    risk_cov = estimate_cov(risk_ret)
    #
    risk_cov.reset_index(drop=False, inplace=True)
    risk_cov = risk_cov[risk_cov["CalcDate"].between(scd, ecd, inclusive="both")].copy()
    _save_risk_cov(root_path, risk_cov, barra_type)
    print("  status::barra_generator>>generator>>generate barra covariance data from {0} to {1}.".format(scd, ecd))


def gen_special_vol(root_path, barra_type, scd, ecd):
    rolling_winsize = 252
    min_roll_rate = 0.6
    data_sd = CALENDAR_UTIL.get_last_trading_dates([scd], inc_self_if_is_trdday=False, n=rolling_winsize)[0]
    special_ret = load_special_ret(root_path, barra_type, data_sd, ecd)
    special_ret = special_ret.set_index(["CalcDate", "Code"]).unstack()
    special_vol = special_ret.rolling(rolling_winsize, min_periods=int(min_roll_rate * rolling_winsize)).std().loc[scd: ecd].stack()
    special_vol = special_vol.reset_index(drop=False).rename(columns={"special_ret": "special_vol"}, errors="raise")
    #
    _save_special_vol(root_path, special_vol, barra_type)
    print("  status::barra_generator>>generator>>generate special vol data from {0} to {1}.".format(scd, ecd))
