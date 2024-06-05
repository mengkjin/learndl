import pandas as pd


def apply_qtile_shrink(factor_df):
    assert factor_df.index.is_monotonic_increasing
    quatiles = factor_df.groupby(['CalcDate']).quantile([0.25, 0.5, 0.75]).swaplevel()
    q1 = quatiles.loc[0.25]
    median = quatiles.loc[0.5]
    q3 = quatiles.loc[0.75]
    IQR = q3 - q1
    inner_min = q1 - IQR * 1.75
    inner_max = q3 + IQR * 1.75
    # rtn = factor_df.where(factor_df.notnull(), median)
    rtn = factor_df.where(factor_df.gt(inner_min), inner_min)
    rtn = rtn.where(rtn.lt(inner_max), inner_max)
    return rtn


def apply_qtile_side_by_side_shrink(factor_df, m=2.0):
    assert factor_df.index.is_monotonic_increasing
    quatiles = factor_df.groupby(['CalcDate']).quantile([0.25, 0.5, 0.75]).swaplevel()
    q1 = quatiles.loc[0.25]
    median = quatiles.loc[0.5]
    q3 = quatiles.loc[0.75]
    IQR1 = median - q1
    IQR3 = q3 - median
    inner_min = q1 - IQR1 * 1.75 * m
    inner_max = q3 + IQR3 * 1.75 * m
    rtn = factor_df.where(factor_df.notnull(), median)
    rtn = rtn.where(factor_df.gt(inner_min), inner_min)
    rtn = rtn.where(rtn.lt(inner_max), inner_max)
    return rtn


from sklearn.linear_model import LinearRegression


def fill_nan(factor_df):
    def _calc_by_barra(data_, y_flds, ind_nm):
        ind_data = pd.get_dummies(data_[ind_nm], prefix='ind')
        data = pd.concat((ind_data, data_), axis=1, sort=True)
        x_cols = ind_data.columns.append(pd.Index(['Bp', 'Size']))
        lr = LinearRegression(fit_intercept=False)
        for y_nm in y_flds:
            num_flg = data[y_nm].notnull()
            if not num_flg.all():
                nan_flg = ~num_flg
                lr.fit(
                    data.loc[num_flg, x_cols],
                    data.loc[num_flg, y_nm]
                )
                data.loc[nan_flg, y_nm] = lr.predict(data.loc[nan_flg, x_cols])
            else:
                print("  status>>barra_model>>no nan values found for factor {0} on date {1}.".format(y_nm, data_.index.get_level_values('CalcDate')[0]))
        data = data[['industry', 'Bp', 'Size'] + y_flds.tolist()].copy()
        return data
    ind_fld = 'industry'
    style_flds = ['Bp', 'Size']
    fld_list = factor_df.columns.drop([ind_fld]).drop(style_flds)
    rtn = factor_df.groupby(by=['CalcDate'], group_keys=False).apply(_calc_by_barra, y_flds=fld_list, ind_nm=ind_fld)
    assert rtn.notnull().all().all()
    return rtn


def apply_zscore(factor_df):
    assert factor_df.index.is_monotonic_increasing
    avg = factor_df.groupby('CalcDate').mean()
    std = factor_df.groupby('CalcDate').std()
    rtn = (factor_df - avg) / std
    return rtn