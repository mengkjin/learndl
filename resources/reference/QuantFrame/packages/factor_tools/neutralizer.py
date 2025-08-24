import pandas as pd
import numpy as np
from factor_tools.general import onehotize


def neutralize(factor_df, w_col=None, numeric_x_cols=None, categorical_x_col=None, categorical_w_col=None,
               date_code_cols=None, prefix="", suffix="", irrel_cols=None, is_copy=True):
    assert w_col is None or isinstance(w_col, str)
    assert categorical_x_col is None or isinstance(categorical_x_col, str)
    assert categorical_w_col is None or isinstance(categorical_w_col, str)
    assert numeric_x_cols is None or isinstance(numeric_x_cols, list)
    assert irrel_cols is None or isinstance(irrel_cols, list)
    if date_code_cols is None:
        assert isinstance(factor_df.index, pd.MultiIndex) and len(factor_df.index.names) == 2
        date_col, code_col = factor_df.index.names
        factor_df = factor_df.copy()
    else:
        factor_df = factor_df.set_index(date_code_cols)
        date_col, code_col = date_code_cols
    assert factor_df.index.get_level_values(date_col).is_monotonic_increasing
    to_drop_flds = set()
    if irrel_cols is not None:
        to_drop_flds.update(irrel_cols)
    if w_col is not None:
        to_drop_flds.add(w_col)
    if categorical_w_col is not None:
        to_drop_flds.add(categorical_w_col)
    if numeric_x_cols is not None:
        to_drop_flds.update(numeric_x_cols)
    if categorical_x_col is not None:
        to_drop_flds.add(categorical_x_col)
    fld_list = factor_df.columns.drop(list(to_drop_flds))
    #
    if w_col is None:
        assert "sample_weight" not in factor_df.columns
        w_col = "sample_weight"
        factor_df[w_col] = 1.0
    if categorical_w_col is None:
        categorical_w_col = w_col
    #
    assert 'intercept' not in factor_df.columns
    factor_df['intercept'] = 1.0
    risk_flds = pd.Index(['intercept'])
    if categorical_x_col is not None:
        cate_data = onehotize(factor_df[[categorical_x_col]].dropna())
        cate_x_cols = categorical_x_col + '.' + cate_data.columns
        cate_data.rename(columns=dict(zip(cate_data.columns, cate_x_cols)), errors='raise', inplace=True)
        factor_df[cate_x_cols] = cate_data
        risk_flds = risk_flds.append(cate_x_cols)
    if numeric_x_cols is not None:
        risk_flds = risk_flds.append(pd.Index(numeric_x_cols))
    #
    beta_calc_df = factor_df.dropna(subset=risk_flds.append(pd.Index(list({w_col, categorical_w_col}))), how='any')
    if beta_calc_df[fld_list].isnull().any().any():
        print("  warning::neutralizer>>y has null value, may cause inaccuracy.")
    beta_data = beta_calc_df.groupby(by=[date_col], group_keys=True).apply(
        _rsknt, y_fld=fld_list, risk_flds=risk_flds, w_col=w_col,
        categorical_x_col=categorical_x_col, categorical_w_col=categorical_w_col)
    beta_data.columns = prefix + beta_data.columns + suffix
    beta_data = beta_data.unstack()
    coef = beta_data.fillna(0.0)
    #
    rsknt_factor = list()
    if irrel_cols is not None:
        rsknt_factor.append(factor_df[irrel_cols])
    for f_nm in fld_list:
        new_f_nm = prefix + f_nm + suffix
        res = factor_df[f_nm] - (factor_df[risk_flds] * coef[new_f_nm][risk_flds]).sum(axis=1, skipna=False)
        rsknt_factor.append(res.rename(new_f_nm))
    rsknt_factor = pd.concat(rsknt_factor, axis=1, sort=True)
    if date_code_cols is not None:
        rsknt_factor.reset_index(drop=False, inplace=True)
    return rsknt_factor, beta_data


def _risk_neut_without_categorical_risk(vals, risk_vals, weight_val):
    weight_val = weight_val / np.sum(weight_val)
    wt_risk = weight_val.reshape(-1, 1) * risk_vals
    design_mat = wt_risk.T.dot(risk_vals)
    assert (np.abs(design_mat.diagonal()) > 1.0e-8).all()
    design_mat_inv = np.linalg.inv(design_mat)
    xy = np.nanmean(
        wt_risk.T.reshape(wt_risk.T.shape + (1,)) * vals.reshape((1,) + vals.shape), axis=1
    ) * len(wt_risk)
    beta = np.dot(design_mat_inv, xy)
    return beta


def _risk_neut_with_categorical_risk(vals, non_cate_risk_array, cate_array, weight_val, cate_stk_weight):
    cate_weight = cate_stk_weight.dot(cate_array)
    cate_weight = cate_weight / cate_weight.sum()
    #
    weight_val = weight_val / weight_val.sum()
    X = np.c_[non_cate_risk_array, cate_array]
    #
    X_adj = X[:, :-1].copy()
    X_adj[:, -len(cate_weight) + 1:] = \
        X_adj[:, -len(cate_weight) + 1:] - (
                    (cate_weight[:-1] / cate_weight[-1]).reshape(-1, 1) * X[:, -1]).T
    wt_x_adj = weight_val.reshape(-1, 1) * X_adj
    design_mat = wt_x_adj.T.dot(X_adj)
    assert (np.abs(design_mat.diagonal()) > 1.0e-8).all()
    #
    design_mat_inv = np.linalg.inv(design_mat)
    xy = np.nanmean(
        wt_x_adj.T.reshape(wt_x_adj.T.shape + (1,)) * vals.reshape((1,) + vals.shape), axis=1
    ) * len(wt_x_adj)
    beta_adj_vals = np.dot(design_mat_inv, xy)
    #
    last_beta = - np.nanmean(beta_adj_vals[-len(cate_weight) + 1:] * cate_weight[:-1].reshape(-1, 1), axis=0,
                            keepdims=True) / cate_weight[-1] * (len(cate_weight) - 1)
    beta_vals = np.r_[beta_adj_vals, last_beta]
    return beta_vals
    
    
def _rsknt(x, y_fld, risk_flds, w_col, categorical_x_col, categorical_w_col):
    factor_vals = x[y_fld].to_numpy()
    risk_vals = x[risk_flds].to_numpy()
    weight_val = x[w_col].to_numpy()
    if categorical_x_col is not None:
        categorical_stk_weight = x[categorical_w_col].to_numpy()
        categorical_flg = risk_flds.str.contains(categorical_x_col + "\.")
        valid_cate_flg = (categorical_stk_weight.dot(risk_vals) > 1e-8) & categorical_flg
        #
        categorical_array = risk_vals[:, valid_cate_flg]
        beta = _risk_neut_with_categorical_risk(
            factor_vals, risk_vals[:, ~categorical_flg], categorical_array, weight_val, categorical_stk_weight)
        beta_vals = np.full(shape=(len(risk_flds), beta.shape[1]), fill_value=np.nan)
        beta_vals[valid_cate_flg, :] = beta[-categorical_array.shape[1]:, :]
        beta_vals[~categorical_flg, :] = beta[:-categorical_array.shape[1], :]
    else:
        beta_vals = _risk_neut_without_categorical_risk(factor_vals, risk_vals, weight_val)
    rtn = pd.DataFrame(beta_vals, index=risk_flds.rename("xs"), columns=y_fld)
    return rtn