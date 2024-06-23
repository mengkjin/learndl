from cvxopt import spmatrix
import numpy as np


def parse_line_condition(lin_con):
    coef_matrix = lin_con[0]
    bl_bu_bnd_key, bl, bu = lin_con[1][0], lin_con[1][1], lin_con[1][2]
    ineq_bnd_loc = [i for i in range(len(bl_bu_bnd_key)) if bl_bu_bnd_key[i] != 'fx']
    A = coef_matrix[ineq_bnd_loc, :].copy()
    bl_bu_bnds = [bl_bu_bnd_key[ineq_bnd_loc], bl[ineq_bnd_loc], bu[ineq_bnd_loc]]
    #
    fx_bnd_loc = [i for i in range(len(bl_bu_bnd_key)) if bl_bu_bnd_key[i] == 'fx']
    G = coef_matrix[fx_bnd_loc, :].copy()
    h = bl[fx_bnd_loc]
    return A, bl_bu_bnds, G, h


def combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con):
    lb_bu_bnd_key, lb, ub = bnd_con[0], bnd_con[1], bnd_con[2]
    bl_bu_bnd_key, bl, bu = bl_bu_bnds[0], bl_bu_bnds[1], bl_bu_bnds[2]
    bnd_matrix = np.eye(len(lb))
    #
    fx_bnd_loc = [i for i in range(len(lb_bu_bnd_key)) if lb_bu_bnd_key[i] == 'fx']
    if fx_bnd_loc:
        eq_matrix = np.vstack((G, bnd_matrix[fx_bnd_loc, :]))
        eq_bnd = np.hstack((h, lb[fx_bnd_loc]))
    else:
        eq_matrix = G.copy()
        eq_bnd = h.copy()
    #
    up_bl_bu_loc = [i for i in range(len(bl_bu_bnd_key)) if bl_bu_bnd_key[i] in ('ra', 'up')]
    up_lb_ub_loc = [i for i in range(len(lb_bu_bnd_key)) if lb_bu_bnd_key[i] in ('ra', 'up')]
    ineq_matrix = np.vstack((A[up_bl_bu_loc, :], bnd_matrix[up_lb_ub_loc, :]))
    ineq_bnd = np.hstack((bu[up_bl_bu_loc], ub[up_lb_ub_loc]))
    #
    lo_lb_ub_loc = [i for i in range(len(lb_bu_bnd_key)) if lb_bu_bnd_key[i] in ('ra', 'lo')]
    lo_bl_bu_loc = [i for i in range(len(bl_bu_bnd_key)) if lb_bu_bnd_key[i] in ('ra', 'lo')]
    ineq_matrix = np.vstack((ineq_matrix, -A[lo_bl_bu_loc, :], -bnd_matrix[lo_lb_ub_loc, :]))
    ineq_bnd = np.hstack((ineq_bnd, -bl[lo_bl_bu_loc], -lb[lo_lb_ub_loc]))
    return ineq_matrix, ineq_bnd, eq_matrix, eq_bnd


def transfer_to_spmatrix(x):
    x_loc = np.nonzero(x)
    rtn = spmatrix(x[x_loc], x_loc[0], x_loc[1], size=x.shape)
    return rtn