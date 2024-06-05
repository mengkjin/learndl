import pandas as pd
import numpy as np


def rank(factor_df, as_pct=True,
         categorical_col=None, date_code_cols=None, prefix='', suffix='', irrel_cols=None):
    assert (categorical_col is None) or (categorical_col in factor_df.columns)
    assert irrel_cols is None or isinstance(irrel_cols, list)
    if date_code_cols is None:
        assert isinstance(factor_df.index, pd.MultiIndex) and len(factor_df.index.names) == 2
        date_col, code_col = factor_df.index.names
        rtn = factor_df.copy()
    else:
        rtn = factor_df.set_index(date_code_cols)
        date_col, code_col = date_code_cols
    assert rtn.index.get_level_values(date_col).is_monotonic_increasing
    # logic
    assert as_pct, "TODO:not implemented for as_pct=False."
    fld_list = rtn.columns
    if irrel_cols is not None:
        fld_list = fld_list.drop(irrel_cols)
    to_grp_fld = [date_col]
    if categorical_col is not None:
        fld_list = fld_list.drop(categorical_col)
        to_grp_fld.append(categorical_col)
    ranked_val = rtn.groupby(by=to_grp_fld, dropna=True, group_keys=False)[fld_list].rank(method='dense', na_option='keep', pct=as_pct)
    assert ranked_val.index.names == rtn.index.names
    rtn[fld_list] = np.nan
    rtn[fld_list] = ranked_val
    #
    if irrel_cols is None:
        rtn = rtn[fld_list].copy()
    else:
        rtn = rtn[fld_list + irrel_cols].copy()
    if date_code_cols is not None:
        rtn.reset_index(drop=False, inplace=True)
    rtn.rename(columns=dict(zip(fld_list, prefix + fld_list + suffix)), inplace=True, errors='raise')
    return rtn


# def apply_indrk(root_path_, factor_df_, industry_cate_='citics_1', prefix='indrk_', suffix=''):
#     assert factor_df_['CalcDate'].is_monotonic
#     fld_list = list(factor_df_.columns.drop(['CalcDate', 'Code']))
#     start_calc_date, end_calc_date = factor_df_.iloc[0]['CalcDate'], factor_df_.iloc[-1]['CalcDate']
#     industry_data = load_industry_data(root_path_, start_calc_date, end_calc_date, industry_cate_)
#     factor_val_df = pd.merge(factor_df_, industry_data, how='inner', on=['CalcDate', 'Code'], sort=True).set_index(['CalcDate', 'Code', 'citics_1'])
#     ranked_val = factor_val_df.groupby(by=['CalcDate', industry_cate_]).rank(method='dense', na_option='keep',
#                                                                                 pct=True)
#     ranked_val.fillna(value=0.5, inplace=True)
#     rtn = ranked_val.reset_index().drop(columns=['citics_1'])
#     rtn.sort_values(by=['CalcDate', 'Code'], inplace=True)
#     rtn.reset_index(drop=True, inplace=True)
#     rtn.rename(columns=dict(zip(fld_list, [prefix + fld + suffix for fld in fld_list])), inplace=True)
#     return rtn
#
#
# def apply_rk(root_path, factor_df, prefix='rk_', ast_col='Code'):
#     assert factor_df['CalcDate'].is_monotonic
#     ranked_val = factor_df.set_index(['CalcDate', ast_col]).groupby(level='CalcDate').rank(method='dense', na_option='keep', pct=True)
#     rtn = ranked_val.fillna(value=0.5).reset_index()
#     fld_list = factor_df.columns.drop(['CalcDate', ast_col])
#     rtn.rename(columns=dict(zip(fld_list, [prefix + f for f in fld_list])), inplace=True)
#     return rtn