import pandas as pd


def normalize(factor_df, dm_mtd='mean', rescale_mtd='std', w_col=None,
              categorical_col=None, date_code_cols=None, prefix='', suffix='', irrel_cols=None):
    assert categorical_col is None or categorical_col in factor_df.columns
    assert irrel_cols is None or isinstance(irrel_cols, list)
    assert w_col is None or w_col in factor_df.columns
    if date_code_cols is None:
        assert isinstance(factor_df.index, pd.MultiIndex) and len(factor_df.index.names) == 2
        date_col, code_col = factor_df.index.names
        rtn = factor_df.copy()
    else:
        rtn = factor_df.set_index(date_code_cols)
        date_col, code_col = date_code_cols
    assert rtn.index.get_level_values(date_col).is_monotonic_increasing
    assert w_col is None, "TODO:to add w_col logic"
    assert categorical_col is None, "TODO:to add categorical logic. to consider, what if there is null in categorical_col?"
    fld_list = rtn.columns
    if irrel_cols is not None:
        fld_list = fld_list.drop(irrel_cols)
    if categorical_col is not None:
        fld_list = fld_list.drop(categorical_col)
    if w_col is not None:
        fld_list = fld_list.drop(w_col)
    rtn = _demean(rtn, fld_list, dm_mtd, date_col, categorical_col, w_col)
    rtn = _rescale(rtn, fld_list, rescale_mtd, date_col, categorical_col, w_col)
    if irrel_cols:
        rtn = rtn[list(fld_list) + irrel_cols].copy()
    else:
        rtn = rtn[fld_list].copy()
    if date_code_cols is not None:
        rtn.reset_index(drop=False, inplace=True)
    rtn.rename(columns=dict(zip(fld_list, prefix + fld_list + suffix)), inplace=True, errors='raise')
    return rtn


def _demean(factor_df, fld_list, demean_mtd, date_col, categorical_col, w_col):
    if demean_mtd == 'mean':
        center = factor_df.groupby(date_col)[fld_list].transform('mean').values
    elif demean_mtd == 'skip':
        center = 0.0
    else:
        assert False
    factor_df[fld_list] = factor_df[fld_list] - center
    return factor_df


def _rescale(factor_df, fld_list, rescale_mtd, date_col, categorical_col, w_col):
    if rescale_mtd == 'std':
        scales = factor_df.groupby(date_col)[fld_list].transform('std').values
    elif rescale_mtd == 'skip':
        scales = 1.0
    else:
        assert False
    factor_df[fld_list] = factor_df[fld_list] / scales
    return factor_df


# def apply_qtile_standardize(root_path, industry_cate, factor_df):
#     assert factor_df['CalcDate'].is_monotonic
#     scd, ecd = factor_df.iloc[0]['CalcDate'], factor_df.iloc[-1]['CalcDate']
#     industry_data = load_industry_data(root_path, scd, ecd, industry_cate)
#     factor_val_df = pd.merge(factor_df, industry_data, how='inner', on=['CalcDate', 'Code'], sort=True)
#     median = factor_val_df.drop(columns=['Code']).groupby(['CalcDate', 'citics_1'], as_index=False).median()
#     quantiles = factor_val_df.drop(columns=['Code']).groupby(['CalcDate', 'citics_1'], as_index=False).quantile([0.25, 0.75]).unstack()
#     quantiles = quantiles.swaplevel(axis='columns')
#     q_diff = (quantiles[0.75].set_index(['CalcDate', 'citics_1']) - quantiles[0.25].set_index(['CalcDate', 'citics_1'])).reset_index()
#     #
#     median = pd.merge(factor_val_df[['CalcDate', 'Code', 'citics_1']], median, how='left', on=['CalcDate', 'citics_1'])
#     q_diff = pd.merge(factor_val_df[['CalcDate', 'Code', 'citics_1']], q_diff, how='left', on=['CalcDate', 'citics_1'])
#
#     rtn = (factor_val_df.set_index(['CalcDate', 'Code']).drop(columns=['citics_1']) - median.set_index(['CalcDate', 'Code']).drop(columns=['citics_1'])) / q_diff.set_index(['CalcDate', 'Code']).drop(columns=['citics_1'])
#     return rtn