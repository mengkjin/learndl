import numpy as np
from sklearn.linear_model import LinearRegression


def risk_neut_by_ind_style_old(val, barra_array):
    assert val.shape[0] == barra_array.shape[0]
    X = barra_array
    Y = val
    lm = LinearRegression(fit_intercept=False).fit(X, Y)
    res = Y - lm.predict(X)
    return res


def risk_neut_by_ind_style(vals, barra_array):
    design_mat = np.dot(barra_array.T, barra_array)
    flg = np.abs(design_mat.diagonal()) > 1.0e-8
    barra_array = barra_array[:, flg]
    design_mat = design_mat[flg, :][:, flg]
    design_mat_inv = np.linalg.inv(design_mat)
    xy = np.nanmean(
        barra_array.T.reshape(barra_array.T.shape + (1,)) * vals.reshape((1,) + vals.shape), axis=1
    ) * len(barra_array)
    beta = np.dot(design_mat_inv, xy)
    res = vals - np.dot(barra_array, beta)
    return res