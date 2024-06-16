from cvxopt import matrix, solvers
import numpy as np
from .util import parse_line_condition, combine_lin_bnd_condition, transfer_to_spmatrix
from scipy.linalg import block_diag


def _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, F, C, S = cov_info
    u_adj = u.T + lmbd * (wb.dot(F.T).dot(C).dot(F) + wb * S)
    #
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    #
    L, N = F.shape
    #
    q = matrix(np.hstack((- u_adj, np.zeros(L))))
    P = block_diag(lmbd * np.diag(S), lmbd * C)
    P = transfer_to_spmatrix(P)
    #
    cvx_A = np.vstack(
        (np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], L)))),
         np.hstack((F, -np.eye(L)))
         ))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(np.hstack((eq_bnd, np.zeros(L))))
    #
    cvx_G = np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], L))))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(ineq_bnd)
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    #
    sol = solvers.qp(P=P, q=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, F, C, S = cov_info
    w0, to, rho = turn_con
    u_adj = u.T + lmbd * (wb.dot(F.T).dot(C).dot(F) + wb * S)
    #
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    #
    L, N = F.shape
    #
    q = matrix(np.hstack((- u_adj, rho * np.ones(N), np.zeros(L))))
    P = block_diag(lmbd * np.diag(S), np.zeros((N, N)), lmbd * C)
    P = transfer_to_spmatrix(P)
    #
    cvx_A = np.vstack(
        (np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], N + L)))),
         np.hstack((F, np.zeros((L, N)), -np.eye(L)))
         ))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(np.hstack((eq_bnd, np.zeros(L))))
    #
    cvx_G = np.vstack(
        (np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], N + L)))),
         np.hstack((np.eye(N), -np.eye(N), np.zeros((N, L)))),
         np.hstack((-np.eye(N), -np.eye(N), np.zeros((N, L)))),
         np.hstack((np.zeros((N, N)), -np.eye(N), np.zeros((N, L)))),
         np.hstack(([0.0] * N, [1.0] * N, [0.0] * L)).reshape(1, -1),
         ))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, w0, -w0, np.zeros(N), [to])))
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.qp(P=P, q=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, cov = cov_info
    u_adj = u.T + lmbd * wb.dot(cov)
    #
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    #
    N = u.shape[0]
    #
    q = matrix(- u_adj)
    P = transfer_to_spmatrix(cov * lmbd)
    #
    cvx_A = transfer_to_spmatrix(eq_matrix)
    cvx_b = matrix(eq_bnd)
    #
    cvx_G = transfer_to_spmatrix(ineq_matrix)
    cvx_h = matrix(ineq_bnd)
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    #
    sol = solvers.qp(P=P, q=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, cov = cov_info
    w0, to, rho = turn_con
    u_adj = u.T + lmbd * wb.dot(cov)
    #
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    #
    N = u.shape[0]
    #
    q = matrix(np.hstack((- u_adj, rho * np.ones(N))))
    P = block_diag(lmbd * cov, np.zeros((N, N)))
    P = transfer_to_spmatrix(P)
    #
    cvx_A = np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], N))))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(eq_bnd)
    #
    cvx_G = np.vstack(
        (np.hstack((ineq_matrix, np.zeros((ineq_matrix.shape[0], N)))),
         np.hstack((np.eye(N), -np.eye(N))),
         np.hstack((-np.eye(N), -np.eye(N))),
         np.hstack((np.zeros((N, N)), -np.eye(N))),
         np.hstack(([0.0] * N, [1.0] * N)).reshape(1, -1),
         ))
    cvx_G = transfer_to_spmatrix(cvx_G)
    cvx_h = matrix(np.hstack((ineq_bnd, w0, -w0, np.zeros(N), [to])))
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.qp(P=P, q=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def solve(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    if len(cov_info) == 2:
        if turn_con is None:
            rtn = _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params)
        else:
            rtn = _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params)
    elif len(cov_info) == 4:
        if turn_con is None:
            rtn = _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params)
        else:
            rtn = _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params)
    else:
        assert False, ' error::>>cvxopt>>quadprog>>cov_info is unknown.'
    return rtn
