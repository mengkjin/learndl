import pandas as pd
import numpy as np


def create_industry_lin_con(ind_list, con_ind_list, lb, ub):
    assert isinstance(ind_list, list) and isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray) and isinstance(con_ind_list, list)
    assert len(lb) == len(ub) and (lb <= ub).all() and len(con_ind_list) == len(lb)
    #
    ind_df = pd.DataFrame(ind_list, columns=["industry"])
    ind_df = pd.get_dummies(ind_df, columns=["industry"], prefix="", prefix_sep="")
    A = ind_df[con_ind_list].values.T
    b = [np.array(["ra"] * len(ub)), lb.astype(float), ub.astype(float)]
    rtn = (A, b)
    return rtn


def create_industry_lin_con_by_dummy_input(ind_data, lb, ub):
    assert isinstance(ind_data, np.ndarray) and isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
    assert len(lb) == len(ub) and (lb <= ub).all()
    assert len(lb) == ind_data.shape[1]
    #
    A = ind_data.T
    b = [np.array(["ra"] * len(ub)), lb.astype(float), ub.astype(float)]
    rtn = (A, b)
    return rtn


def create_style_lin_con(style_data, lb, ub):
    assert isinstance(style_data, np.ndarray) and isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
    assert len(lb) == len(ub) and (lb <= ub).all()
    assert len(lb) == style_data.shape[1]
    #
    A = style_data.T
    b = [np.array(["ra"] * len(lb)), lb.astype(float), ub.astype(float)]
    rtn = (A, b)
    return rtn


def create_leverage_lin_con(leverage, stk_num):
    assert isinstance(leverage, float) and isinstance(stk_num, int)
    A = np.ones(shape=(1, stk_num))
    b = [np.array(["fx"]), np.array([leverage]), np.array([leverage])]
    rtn = (A, b)
    return rtn


def create_index_member_lower_con(is_member, lb):
    assert isinstance(is_member, np.ndarray) and isinstance(lb, float)
    A = is_member * 1.0
    A = A.reshape(1, -1)
    b = [np.array(["lo"]), np.array([lb]), np.array([1.0])]
    rtn = (A, b)
    return rtn


def create_member_lin_con(is_member, bnd_type, lb, ub):
    assert isinstance(is_member, np.ndarray) and isinstance(lb, float) and isinstance(ub, float)
    A = is_member * 1.0
    A = A.reshape(1, -1)
    b = [np.array([bnd_type]), np.array([lb]), np.array([ub])]
    rtn = (A, b)
    return rtn


def combine_lin_con(lin_con_list):
    assert isinstance(lin_con_list, list)
    A = np.vstack([lin_con[0] for lin_con in lin_con_list])
    b = [np.hstack([lin_con[1][0] for lin_con in lin_con_list]), np.hstack([lin_con[1][1] for lin_con in lin_con_list]),
         np.hstack([lin_con[1][2] for lin_con in lin_con_list])]
    return A, b


def create_turn_con(w0, to, rho):
    assert isinstance(w0, np.ndarray) and isinstance(to, float) and isinstance(rho, float)
    return w0, to, rho


def create_brcov_info(lmbd, F, C, S):
    assert isinstance(lmbd, float) and isinstance(F, np.ndarray) and isinstance(C, np.ndarray) and isinstance(S, np.ndarray)
    return lmbd, F, C, S


def create_nmcov_info(lmbd, Cov):
    assert isinstance(lmbd, float) and isinstance(Cov, np.ndarray)
    return lmbd, Cov


def create_bnd_con(lb, ub):
    assert isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray)
    assert len(lb) == len(ub) and (lb <= ub).all()
    rtn = [np.array(["ra"] * len(lb)), lb, ub]
    return rtn