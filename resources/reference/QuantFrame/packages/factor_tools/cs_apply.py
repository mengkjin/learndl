import pandas as pd
from barra_model.factor_impl.api import load_barra_data
from industry.api import load_industry_data
from common_ops.risk_neut import risk_neut_by_ind_style
from factor_tools.general import onehotize


def apply_rsknt(root_path_, factor_df, fld_list=None, keep_origin=False, barra_type='cne6',
                barra_list=("STYLE.Bp", "STYLE.Size"), prefix="rsknt_"):
    def _calc_by_barra(data_, y_fld_, barra_flds_, fm_prefix):
        factor_vals = data_[y_fld_].astype(float).to_numpy()
        barra_vals = data_[barra_flds_].astype(float).to_numpy()
        rsknt_fct_vals = risk_neut_by_ind_style(factor_vals, barra_vals)
        rsknt_fct_vals = pd.DataFrame(rsknt_fct_vals, index=data_.index, columns=['{0}{1}'.format(fm_prefix, f) for f in y_fld_])
        return rsknt_fct_vals
    if fld_list is None:
        fld_list = factor_df.columns.drop(['CalcDate', 'Code'])
        other_flds = pd.Index([])
    else:
        assert set(fld_list).issubset(factor_df.columns)
        fld_list = pd.Index(fld_list)
        other_flds = factor_df.columns.drop(['CalcDate', 'Code']).drop(fld_list)
    barra_data = load_barra_data(root_path_, barra_type, factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate'])
    ind_fld = barra_data.columns[barra_data.columns.str.contains('INDUSTRY.')].copy()
    style_flds = barra_data.columns[barra_data.columns.isin(barra_list)].copy()
    assert len(style_flds) == len(barra_list) == len(set(barra_list))
    barra_data = barra_data[pd.Index(["CalcDate", "Code"]).append(ind_fld).append(style_flds)]
    #
    factor_val_df = pd.merge(factor_df, barra_data, how='inner', on=['CalcDate', 'Code'], sort=True)
    factor_flds = factor_df.columns.drop(["CalcDate", "Code"])
    factor_val_df = pd.concat((factor_val_df[pd.Index(['CalcDate', 'Code']).append(style_flds).append(factor_flds)],
                              onehotize(factor_val_df[ind_fld])
                               ), axis=1)
    factor_val_df.set_index(['CalcDate', 'Code'], inplace=True)
    rsknt_factor = factor_val_df.groupby(by=['CalcDate'], group_keys=False).apply(_calc_by_barra, y_fld_=fld_list,
                                                                barra_flds_=factor_val_df.columns.drop(factor_flds),
                                                                fm_prefix=prefix)
    #
    if keep_origin:
        other_flds = other_flds.append(fld_list)
    else:
        pass
    rtn = pd.concat((factor_val_df[other_flds], rsknt_factor), axis=1, sort=True)
    rtn.reset_index(drop=False, inplace=True)
    return rtn


def apply_rsknt_style(
        root_path_, factor_df, fld_list=None, keep_origin=False, barra_type='cne6',
        barra_list=("STYLE.Bp", "STYLE.Size"), prefix="rsknt_"):
    def _calc_by_barra(data_, y_fld_, barra_flds_, fm_prefix):
        factor_vals = data_[y_fld_].astype(float).to_numpy()
        barra_vals = data_[barra_flds_].astype(float).to_numpy()
        rsknt_fct_vals = risk_neut_by_ind_style(factor_vals, barra_vals)
        rsknt_fct_vals = pd.DataFrame(rsknt_fct_vals, index=data_.index, columns=['{0}{1}'.format(fm_prefix, f) for f in y_fld_])
        return rsknt_fct_vals
    if fld_list is None:
        fld_list = factor_df.columns.drop(['CalcDate', 'Code'])
        other_flds = pd.Index([])
    else:
        assert set(fld_list).issubset(factor_df.columns)
        fld_list = pd.Index(fld_list)
        other_flds = factor_df.columns.drop(['CalcDate', 'Code']).drop(fld_list)
    #
    barra_data = load_barra_data(root_path_, barra_type, factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate'])
    style_flds = barra_data.columns[barra_data.columns.isin(barra_list)].copy()
    assert len(style_flds) == len(barra_list) == len(set(barra_list))
    barra_data = barra_data[pd.Index(["CalcDate", "Code"]).append(style_flds)]
    #
    factor_val_df = pd.merge(factor_df, barra_data, how='inner', on=['CalcDate', 'Code'], sort=True)
    factor_flds = factor_df.columns.drop(["CalcDate", "Code"])
    factor_val_df = factor_val_df[pd.Index(['CalcDate', 'Code']).append(style_flds).append(factor_flds)].copy()
    factor_val_df["COUNTRY"] = 1.0
    factor_val_df.set_index(['CalcDate', 'Code'], inplace=True)
    rsknt_factor = factor_val_df.groupby(by=['CalcDate']).apply(
        _calc_by_barra, y_fld_=fld_list, barra_flds_=factor_val_df.columns.drop(factor_flds), fm_prefix=prefix)
    #
    if keep_origin:
        other_flds = other_flds.append(fld_list)
    else:
        pass
    rtn = pd.concat((factor_val_df[other_flds], rsknt_factor), axis=1, sort=True)
    rtn.reset_index(drop=False, inplace=True)
    return rtn


def apply_rsknt_temp(root_path_, factor_df, fld_list=None, keep_origin=False, barra_type='cne6',
                     barra_list=("STYLE.Size", "STYLE.Bp"), prefix="rsknt_"):
    def risk_neut_by_ind_style_temp(vals, barra_array, neut_flds):
        import numpy as np
        design_mat = np.dot(barra_array.T, barra_array)
        flg = np.abs(design_mat.diagonal()) > 1.0e-8
        barra_array = barra_array[:, flg]
        design_mat = design_mat[flg, :][:, flg]
        design_mat_inv = np.linalg.inv(design_mat)
        neut_flds = neut_flds[flg]
        neut_cols = [i for i in range(len(neut_flds)) if neut_flds[i]]
        xy = np.nanmean(
            barra_array.T.reshape(barra_array.T.shape + (1,)) * vals.reshape((1,) + vals.shape), axis=1
        ) * len(barra_array)
        beta = np.dot(design_mat_inv, xy)
        res = vals - np.dot(barra_array[:, neut_cols], beta[neut_cols])
        return res

    def _calc_by_barra(data_, y_fld_, barra_flds_, fm_prefix):
        factor_vals = data_[y_fld_].astype(float).to_numpy()
        barra_vals = data_[barra_flds_].astype(float).to_numpy()
        neut_flds = pd.Series({barra_flds_[i]: True if "STYLE" not in barra_flds_[i] else False
                              for i in range(len(barra_flds_))})
        rsknt_fct_vals = risk_neut_by_ind_style_temp(factor_vals, barra_vals, neut_flds)
        rsknt_fct_vals = pd.DataFrame(rsknt_fct_vals, index=data_.index, columns=['{0}{1}'.format(fm_prefix, f) for f in y_fld_])
        return rsknt_fct_vals
    if fld_list is None:
        fld_list = factor_df.columns.drop(['CalcDate', 'Code'])
        other_flds = pd.Index([])
    else:
        assert set(fld_list).issubset(factor_df.columns)
        fld_list = pd.Index(fld_list)
        other_flds = factor_df.columns.drop(['CalcDate', 'Code']).drop(fld_list)
    barra_data = load_barra_data(root_path_, barra_type, factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate'])
    ind_fld = barra_data.columns[barra_data.columns.str.contains('INDUSTRY.')].copy()
    style_flds = barra_data.columns[barra_data.columns.isin(barra_list)].copy()
    assert len(style_flds) == len(barra_list) == len(set(barra_list))
    barra_data = barra_data[pd.Index(["CalcDate", "Code"]).append(ind_fld).append(style_flds)]
    #
    factor_val_df = pd.merge(factor_df, barra_data, how='inner', on=['CalcDate', 'Code'], sort=True)
    factor_flds = factor_df.columns.drop(["CalcDate", "Code"])
    factor_val_df = pd.concat((factor_val_df[pd.Index(['CalcDate', 'Code']).append(style_flds).append(factor_flds)],
                              onehotize(factor_val_df[ind_fld])
                               ), axis=1)
    factor_val_df.set_index(['CalcDate', 'Code'], inplace=True)
    rsknt_factor = factor_val_df.groupby(by=['CalcDate']).apply(_calc_by_barra, y_fld_=fld_list,
                                                                barra_flds_=factor_val_df.columns.drop(factor_flds),
                                                                fm_prefix=prefix)
    #
    if keep_origin:
        other_flds = other_flds.append(fld_list)
    else:
        pass
    rtn = pd.concat((factor_val_df[other_flds], rsknt_factor), axis=1, sort=True)
    rtn.reset_index(drop=False, inplace=True)
    return rtn


def apply_rsknt_by_extra(root_path_, factor_df, x_list, y_list, tag, keep_origin=True, barra_type='cne6',
                         barra_list=("STYLE.Bp", "STYLE.Size")):
    def _calc_by_barra(data_, y_fld_, barra_flds_):
        factor_vals = data_[y_fld_].astype(float).to_numpy()
        barra_vals = data_[barra_flds_.append(pd.Index(x_list))].astype(float).to_numpy()
        rsknt_fct_vals = risk_neut_by_ind_style(factor_vals, barra_vals)
        rsknt_fct_vals = pd.DataFrame(rsknt_fct_vals, index=data_.index, columns=['rsknt_{0}_{1}'.format(tag, f) for f in y_fld_])
        return rsknt_fct_vals
    other_flds = factor_df.columns.drop(['CalcDate', 'Code'] + y_list)
    barra_data = load_barra_data(root_path_, barra_type, factor_df.iloc[0]['CalcDate'],
                                                      factor_df.iloc[-1]['CalcDate'])
    ind_fld = barra_data.columns[barra_data.columns.str.contains('INDUSTRY.')].copy()
    style_flds = barra_data.columns[barra_data.columns.isin(barra_list)].copy()
    assert len(style_flds) == len(barra_list) == len(set(barra_list))
    barra_data = barra_data[pd.Index(["CalcDate", "Code"]).append(ind_fld).append(style_flds)]
    #
    factor_val_df = pd.merge(factor_df, barra_data, how='inner', on=['CalcDate', 'Code'], sort=True)
    factor_flds = factor_df.columns.drop(["CalcDate", "Code"])
    factor_val_df = pd.concat((factor_val_df[pd.Index(['CalcDate', 'Code']).append(style_flds).append(factor_flds)],
                              onehotize(factor_val_df[ind_fld])
                               ), axis=1)
    factor_val_df.set_index(['CalcDate', 'Code'], inplace=True)
    rsknt_factor = factor_val_df.groupby(by=['CalcDate']).apply(_calc_by_barra, y_fld_=y_list, barra_flds_=factor_val_df.columns.drop(factor_flds))
    #
    if keep_origin:
        other_flds = other_flds.append(pd.Index(y_list))
    else:
        pass
    rtn = pd.concat((factor_val_df[other_flds], rsknt_factor), axis=1, sort=True)
    rtn.reset_index(drop=False, inplace=True)
    return rtn


def apply_mktnt(root_path_, factor_df, fld_list, prefix="mktnt_"):
    other_list = factor_df.columns.drop(fld_list).tolist()
    factor_df = factor_df.join(
        factor_df.groupby('CalcDate')[fld_list].mean(), on='CalcDate', rsuffix='_avg')
    factor_df.set_index(other_list, inplace=True)
    rtn = pd.DataFrame(
        factor_df[fld_list].to_numpy() - factor_df[[f + '_avg' for f in fld_list]].to_numpy(),
        index=factor_df.index,
        columns=[prefix + f for f in fld_list])
    rtn.reset_index(drop=False, inplace=True)
    return rtn


def apply_indnt(root_path, industry_cate, factor_df, prefix='indnt_'):
    assert factor_df['CalcDate'].is_monotonic_increasing
    fld_list = list(factor_df.columns.drop(['CalcDate', 'Code']))
    start_calc_date, end_calc_date = factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate']
    industry_data = load_industry_data(root_path, start_calc_date, end_calc_date, industry_cate, as_sys_id=False)
    factor_val_df = pd.merge(factor_df, industry_data, how='inner', on=['CalcDate', 'Code'])
    mean_df = factor_val_df.groupby(by=['CalcDate', 'citics_1'], as_index=False)[fld_list].mean().rename(columns=dict(zip(fld_list, [f + '_mean' for f in fld_list])))
    factor_val_df = pd.merge(factor_val_df, mean_df, how='left', on=['CalcDate', 'citics_1'])
    factor_val_df[['indnt_' + f for f in fld_list]] = factor_val_df[fld_list].values - factor_val_df[[f + '_mean' for f in fld_list]].values
    rtn = factor_val_df[['CalcDate', 'Code'] + [prefix + f for f in fld_list]].copy()
    return rtn


def apply_indrk(root_path_, industry_cate_, factor_df_, prefix="indrk_", suffix=""):
    assert factor_df_['CalcDate'].is_monotonic_increasing
    fld_list = list(factor_df_.columns.drop(['CalcDate', 'Code']))
    start_calc_date, end_calc_date = factor_df_.iloc[0]['CalcDate'], factor_df_.iloc[-1]['CalcDate']
    industry_data = load_industry_data(root_path_, start_calc_date, end_calc_date, industry_cate_, as_sys_id=False)
    factor_val_df = pd.merge(factor_df_, industry_data, how='inner', on=['CalcDate', 'Code'])
    ranked_val = factor_val_df.drop(columns=["Code"]).groupby(by=['CalcDate', industry_cate_]).rank(
        method='dense', na_option='keep', pct=True)
    ranked_val.fillna(value=0.5, inplace=True)
    rtn = pd.concat((factor_val_df[['CalcDate', 'Code']], ranked_val), axis=1, sort=True)
    rtn.sort_values(by=['CalcDate', 'Code'], inplace=True)
    rtn.reset_index(drop=True, inplace=True)
    rtn.rename(columns=dict(zip(fld_list, [prefix + fld + suffix for fld in fld_list])), inplace=True)
    return rtn


def apply_rk(factor_df, prefix="rk_"):
    assert factor_df['CalcDate'].is_monotonic_increasing
    ranked_val = factor_df.set_index(['CalcDate', 'Code']).groupby(level='CalcDate').rank(method='dense', na_option='keep', pct=True)
    rtn = ranked_val.fillna(value=0.5).reset_index()
    fld_list = factor_df.columns.drop(['CalcDate', 'Code'])
    rtn.rename(columns=dict(zip(fld_list, [prefix + f for f in fld_list])), inplace=True)
    return rtn


def apply_op(df, to_apply_func, to_apply_flds=None):
    def apply_func(x, f, flds):
        val = f(x[flds].to_numpy())
        other_flds = x.columns.drop(flds)
        rslt = pd.concat((x[other_flds], pd.DataFrame(val, index=x.index, columns=flds)), axis=1, sort=True)
        return rslt
    if to_apply_flds is None:
        to_apply_flds = df.columns.drop(['CalcDate', 'Code'])
    rtn = df.groupby(by=['CalcDate'], group_keys=False, as_index=False).apply(apply_func, f=to_apply_func, flds=to_apply_flds)
    rtn.reset_index(drop=True, inplace=True)
    return rtn


def apply_qtile_standardize(root_path, industry_cate, factor_df):
    assert factor_df['CalcDate'].is_monotonic
    scd, ecd = factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate']
    industry_data = load_industry_data(root_path, scd, ecd, industry_cate, as_sys_id=False)
    factor_val_df = pd.merge(factor_df, industry_data, how='inner', on=['CalcDate', 'Code'])
    median = factor_val_df.drop(columns=['Code']).groupby(['CalcDate', 'citics_1'], as_index=False).median()
    quantiles = factor_val_df.drop(columns=['Code']).groupby(['CalcDate', 'citics_1'], as_index=False).quantile([0.25, 0.75]).unstack()
    quantiles = quantiles.swaplevel(axis='columns')
    q_diff = (quantiles[0.75].set_index(['CalcDate', 'citics_1']) - quantiles[0.25].set_index(['CalcDate', 'citics_1'])).reset_index()
    #
    median = pd.merge(factor_val_df[['CalcDate', 'Code', 'citics_1']], median, how='left', on=['CalcDate', 'citics_1'])
    q_diff = pd.merge(factor_val_df[['CalcDate', 'Code', 'citics_1']], q_diff, how='left', on=['CalcDate', 'citics_1'])

    rtn = (factor_val_df.set_index(['CalcDate', 'Code']).drop(columns=['citics_1']) - median.set_index(['CalcDate', 'Code']).drop(columns=['citics_1'])) / q_diff.set_index(['CalcDate', 'Code']).drop(columns=['citics_1'])
    return rtn