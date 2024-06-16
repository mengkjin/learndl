from cvxopt import matrix, solvers
from .util import parse_line_condition, combine_lin_bnd_condition, transfer_to_spmatrix
import numpy as np
from scipy.linalg import block_diag


def _transform_parameter_with_brcov(u, C, S, F, wb, A, bl_bu_bnds, G, h, bnd_con):
    s_vol = np.sqrt(S)
    u_adj = u * (1.0 / s_vol)
    V = np.linalg.cholesky(C).T
    assert np.allclose(V.T.dot(V), C)
    K = V.dot(F) * (1.0 / s_vol)
    G_adj = G * (1.0 / s_vol)
    A_adj = A * (1.0 / s_vol)
    #
    h_adj = h - G.dot(wb)
    #
    bl_bu_bnd_key, bl, bu = bl_bu_bnds[0], bl_bu_bnds[1], bl_bu_bnds[2]
    bl_adj = np.array(bl) - A.dot(wb)
    bu_adj = bu - A.dot(wb)
    bl_bu_bnds_adj = (bl_bu_bnd_key, bl_adj, bu_adj)
    #
    lb_lu_bnd_key, lb, lu = bnd_con[0], bnd_con[1], bnd_con[2]
    lb_adj = (lb - wb) * s_vol
    lu_adj = (lu - wb) * s_vol
    lb_lu_bnds_adj = (lb_lu_bnd_key, lb_adj, lu_adj)
    return u_adj, K, A_adj, bl_bu_bnds_adj, G_adj, h_adj, lb_lu_bnds_adj


def _transform_parameter_with_nmcov(cov, wb, A, bl_bu_bnds, G, h, bnd_con):
    V = np.linalg.cholesky(cov).T
    #
    h_adj = h - G.dot(wb)
    #
    bl_bu_bnd_key, bl, bu = bl_bu_bnds[0], bl_bu_bnds[1], bl_bu_bnds[2]
    bl_adj = np.array(bl) - A.dot(wb)
    bu_adj = bu - A.dot(wb)
    bl_bu_bnds_adj = (bl_bu_bnd_key, bl_adj, bu_adj)
    #
    lb_lu_bnd_key, lb, lu = bnd_con[0], bnd_con[1], bnd_con[2]
    lb_adj = lb - wb
    lu_adj = lu - wb
    lb_lu_bnds_adj = (lb_lu_bnd_key, lb_adj, lu_adj)
    return V, bl_bu_bnds_adj, h_adj, lb_lu_bnds_adj


def _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, F, C, S = cov_info
    te_sq = te ** 2
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    u_adj, K, A_adj, bl_bu_bnds_adj, G_adj, h_adj, bnd_con_adj = \
        _transform_parameter_with_brcov(u, C, S, F, wb, A, bl_bu_bnds, G, h, bnd_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = combine_lin_bnd_condition(A_adj, bl_bu_bnds_adj, G_adj, h_adj, bnd_con_adj)
    #
    L, N = K.shape
    #
    q = matrix(np.hstack((- u_adj, np.zeros(L))))
    P = np.diag([lmbd] * N + [lmbd] * L)
    P = transfer_to_spmatrix(P)
    #
    cvx_A = np.vstack(
        (np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], L)))),
         np.hstack((K, -np.eye(L)))
         ))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(np.hstack((eq_bnd, np.zeros(L))))
    #
    cvx_G = np.vstack(
        (np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], L)))),
         np.zeros((1, N + L)),
         np.diag(np.array([1.0] * N + [1.0] * L))
         ))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, [np.sqrt(te_sq)], np.zeros(N + L))))
    dims = {'l': ineq_matrix.shape[0], 'q': [N + L + 1], 's': []}
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.coneqp(P=P, q=q, G=cvx_G, h=cvx_h, dims=dims, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    x = np.array(sol['x'])[:N, 0]
    w = np.array(x) * (1 / np.sqrt(S)) + wb
    return w, is_success, status


def _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, F, C, S = cov_info
    w0, to, rho = turn_con
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    u_adj, K, A_adj, bl_bu_bnds_adj, G_adj, h_adj, bnd_con_adj = \
        _transform_parameter_with_brcov(u, C, S, F, wb, A, bl_bu_bnds, G, h, bnd_con)
    x0 = (w0 - wb) * np.sqrt(S)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = combine_lin_bnd_condition(A_adj, bl_bu_bnds_adj, G_adj, h_adj, bnd_con_adj)
    #
    L, N = K.shape
    #
    q = matrix(np.hstack((- u_adj, rho * np.ones(N), np.zeros(L))))
    P = np.diag([lmbd] * N + [0.0] * N + [lmbd] * L)
    P = transfer_to_spmatrix(P)
    #
    cvx_A = np.vstack(
        (np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], N + L)))),
         np.hstack((K, np.zeros((L, N)), -np.eye(L)))
         ))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(np.hstack((eq_bnd, np.zeros(L))))
    #
    cvx_G = np.vstack(
        (np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], N + L)))),
         np.hstack((np.eye(N), -np.diag(np.sqrt(S)), np.zeros((N, L)))),
         np.hstack((-np.eye(N), -np.diag(np.sqrt(S)), np.zeros((N, L)))),
         np.hstack((np.zeros((N, N)), -np.eye(N), np.zeros((N, L)))),
         np.hstack(([0.0] * N, [1.0] * N, [0.0] * L)).reshape(1, -1),
         np.zeros((1, 2 * N + L)),
         np.diag(np.array([1.0] * N + [0.0] * N + [1.0] * L))
         ))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, x0, -x0, np.zeros(N), [to], [te], np.zeros(2 * N + L))))
    dims = {'l': ineq_matrix.shape[0] + N * 3 + 1, 'q': [2 * N + L + 1], 's': []}
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.coneqp(P=P, q=q, G=cvx_G, h=cvx_h, dims=dims, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    x = np.array(sol['x'])[:N, 0]
    w = np.array(x) * (1 / np.sqrt(S)) + wb
    return w, is_success, status


def _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, cov = cov_info
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    V, bl_bu_bnds_adj, h_adj, bnd_con_adj = _transform_parameter_with_nmcov(cov, wb, A, bl_bu_bnds, G, h, bnd_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = combine_lin_bnd_condition(A, bl_bu_bnds_adj, G, h_adj, bnd_con_adj)
    #
    N = u.shape[0]
    #
    q = matrix(- u)
    P = cov * lmbd
    P = transfer_to_spmatrix(P)
    #
    cvx_A = transfer_to_spmatrix(eq_matrix)
    cvx_b = matrix(eq_bnd)
    #
    cvx_G = np.vstack((ineq_matrix, np.zeros((1, N)), V))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, [te], np.zeros(V.shape[0]))))
    dims = {'l': ineq_matrix.shape[0], 'q': [V.shape[0] + 1], 's': []}
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.coneqp(P=P, q=q, G=cvx_G, h=cvx_h, dims=dims, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    x = np.array(sol['x'])[:N, 0]
    w = np.array(x) + wb
    return w, is_success, status


def _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, cov = cov_info
    w0, to, rho = turn_con
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    V, bl_bu_bnds_adj, h_adj, bnd_con_adj = _transform_parameter_with_nmcov(cov, wb, A, bl_bu_bnds, G, h, bnd_con)
    x0 = w0 - wb
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = combine_lin_bnd_condition(A, bl_bu_bnds_adj, G, h_adj, bnd_con_adj)
    #
    N = u.shape[0]
    #
    q = matrix(np.hstack((- u, rho * np.ones(N))))
    P = block_diag(lmbd * cov, np.zeros((N, N)))
    P = transfer_to_spmatrix(P)
    #
    cvx_A = transfer_to_spmatrix(np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], N)))))
    cvx_b = matrix(eq_bnd)
    #
    cvx_G = np.vstack(
        (np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], N)))),
         np.hstack((np.eye(N), -np.eye(N))),
         np.hstack((-np.eye(N), -np.eye(N))),
         np.hstack((np.zeros((N, N)), -np.eye(N))),
         np.hstack(([0.0] * N, [1.0] * N)).reshape(1, -1),
         np.zeros((1, 2 * N)),
         np.hstack((V, np.zeros((V.shape[0], N))))
         ))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, x0, -x0, np.zeros(N), [to], [te], np.zeros(V.shape[0]))))
    dims = {'l': ineq_matrix.shape[0] + N * 3 + 1, 'q': [V.shape[0] + 1], 's': []}
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.coneqp(P=P, q=q, G=cvx_G, h=cvx_h, dims=dims, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    x = np.array(sol['x'])[:N, 0]
    w = np.array(x) + wb
    return w, is_success, status


def solve(u, cov_info, wb, te, lin_con, bnd_con, turn_con, solver_params):
    if len(cov_info) == 2:
        if turn_con is None:
            rtn = _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params)
        else:
            rtn = _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params)
    elif len(cov_info) == 4:
        if turn_con is None:
            rtn = _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params)
        else:
            rtn = _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params)
    else:
        assert False, ' error::>>cvxopt>>socp>>cov_info is unknown.'
    return rtn