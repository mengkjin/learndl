import numpy as np
import pandas as pd


def calc_density_info(factor_val_df, factor_sampling_num, hist_bins):
    date_list = sorted(factor_val_df["CalcDate"].unique())
    slice_interval = int(np.ceil(len(date_list) / factor_sampling_num))
    date_chosen = date_list[0: len(date_list): slice_interval]
    factor_name_list = factor_val_df.columns.drop(["CalcDate", "Code"]).tolist()
    rtn = []
    for factor_name in factor_name_list:
        hist_dict = {}
        for date in date_chosen:
            factor_sample = factor_val_df.loc[factor_val_df["CalcDate"] == date, factor_name].copy()
            cnts, bins = np.histogram(factor_sample, bins=hist_bins, density=False)
            hist_dict[date] = (cnts, bins)
        hist_df = pd.DataFrame(hist_dict, index=["hist_cnts", "hist_bins"]).T
        hist_df["factor_name"] = factor_name
        rtn.append(hist_df)
    rtn = pd.concat(rtn, axis=0)
    rtn.index.rename("CalcDate", inplace=True)
    rtn = rtn.reset_index(drop=False)
    rtn = rtn[["CalcDate", "factor_name", "hist_cnts", "hist_bins"]].copy()
    return rtn


def _calc_factor_qtile_by_day(factor):
    factor = (factor - factor.mean()) / factor.std()
    rtn = pd.concat((factor.quantile(0.05).rename("5%"), factor.quantile(0.25).rename("25%"),
                     factor.quantile(0.50).rename("50%"), factor.quantile(0.75).rename("75%"),
                     factor.quantile(0.95).rename("95%")), axis=1, sort=True)
    return rtn


def calc_factor_qtile(factor_val_df):
    factor_val_df = factor_val_df.set_index(["CalcDate", "Code"])
    factor_val_df.columns.rename("factor_name", inplace=True)
    rtn = factor_val_df.groupby(["CalcDate"]).apply(_calc_factor_qtile_by_day).reset_index(drop=False)
    return rtn


def _calc_factor_qtile_without_scaling_by_day(factor):
    rtn = pd.concat((factor.quantile(0.05).rename("5%"), factor.quantile(0.25).rename("25%"),
                     factor.quantile(0.50).rename("50%"), factor.quantile(0.75).rename("75%"),
                     factor.quantile(0.95).rename("95%")), axis=1, sort=True)
    return rtn


def calc_factor_qtile_without_scaling(factor_val_df):
    factor_val_df = factor_val_df.set_index(["CalcDate", "Code"])
    factor_val_df.columns.rename("factor_name", inplace=True)
    rtn = factor_val_df.groupby(["CalcDate"]).apply(_calc_factor_qtile_without_scaling_by_day).reset_index(drop=False)
    return rtn