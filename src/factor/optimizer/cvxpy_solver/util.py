import numpy as np


def parse_linear_condition(lin_con):
    coef_matrix = lin_con[0]
    bnd_key, bl, bu = lin_con[1][0], lin_con[1][1], lin_con[1][2]
    fx_bnds_loc = [i for i in range(len(bnd_key)) if bnd_key[i] == 'fx']
    eq_matrix = coef_matrix[fx_bnds_loc, :].copy()
    eq_bnd = bl[fx_bnds_loc]
    #
    up_bl_bu_loc = [i for i in range(len(bnd_key)) if bnd_key[i] in ('ra', 'up')]
    lo_bl_bu_loc = [i for i in range(len(bnd_key)) if bnd_key[i] in ('ra', 'lo')]
    ineq_matrix = np.vstack((coef_matrix[up_bl_bu_loc, :], -coef_matrix[lo_bl_bu_loc, :]))
    ineq_bnd = np.hstack((bu[up_bl_bu_loc], -bl[lo_bl_bu_loc]))
    return ineq_matrix, ineq_bnd, eq_matrix, eq_bnd


def parse_bound_conditions(bnd_con, bnd_inf):
    assert isinstance(bnd_inf, float) and bnd_inf > 0.0
    bnd_key, lb, ub = bnd_con
    lb[bnd_key == 'up'] = -bnd_inf
    ub[bnd_key == 'lo'] = bnd_inf
    return bnd_key, lb, ub