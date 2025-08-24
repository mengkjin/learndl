import pandas as pd


def winsorize(factor_df, method='qtile', both_sides=True, q_dist=1.75,
              categorical_col=None, date_code_cols=None, prefix='', suffix='', irrel_cols=None):
    assert (categorical_col is None) or (categorical_col in factor_df.columns)
    assert irrel_cols is None or isinstance(irrel_cols, list)
    assert categorical_col is None, "TODO:not implemented for categorical col."
    if date_code_cols is None:
        assert isinstance(factor_df.index, pd.MultiIndex) and len(factor_df.index.names) == 2
        date_col, code_col = factor_df.index.names
        rtn = factor_df.copy()
    else:
        rtn = factor_df.set_index(date_code_cols)
        date_col, code_col = date_code_cols
    assert rtn.index.get_level_values(date_col).is_monotonic_increasing
    #
    fld_list = rtn.columns
    if irrel_cols is not None:
        fld_list = fld_list.drop(irrel_cols)
    if categorical_col is not None:
        fld_list = fld_list.drop([categorical_col])
    #
    lower, upper = calc_bnds(rtn, fld_list, method, both_sides, q_dist)
    rtn[fld_list] = rtn[fld_list].clip(lower=lower, upper=upper)
    if irrel_cols is None:
        rtn = rtn[fld_list].copy()
    else:
        rtn = rtn[fld_list + irrel_cols].copy()
    if date_code_cols is not None:
        rtn.reset_index(drop=False, inplace=True)
    rtn.rename(columns=dict(zip(fld_list, prefix + fld_list + suffix)), inplace=True, errors='raise')
    return rtn


def calc_bnds(factor_df, fld_list, method, both_sides, q_dist):
    if method == 'qtile':
        if both_sides:
            quatiles = factor_df.groupby(['CalcDate'])[fld_list].quantile([0.25, 0.75]).swaplevel()
            q1 = quatiles.loc[0.25]
            q4 = quatiles.loc[0.75]
            IQR = q4 - q1
            lower = q1 - q_dist * IQR
            upper = q4 + q_dist * IQR
        else:
            quatiles = factor_df.groupby(['CalcDate'])[fld_list].quantile([0.25, 0.5, 0.75]).swaplevel()
            q1 = quatiles.loc[0.25]
            median = quatiles.loc[0.5]
            q4 = quatiles.loc[0.75]
            IQR_rht = (q4 - median) * 2
            IQR_lft = (median - q1) * 2
            lower = q1 - q_dist * IQR_lft
            upper = q4 + q_dist * IQR_rht
    else:
        raise NotImplementedError
    return lower, upper