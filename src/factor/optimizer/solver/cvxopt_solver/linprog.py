from cvxopt import matrix , solvers
from .util import parse_line_condition, combine_lin_bnd_condition, transfer_to_spmatrix
import numpy as np


def _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params):
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    N = u.shape[0]
    #
    q = matrix(-u)
    #
    cvx_A = transfer_to_spmatrix(eq_matrix)
    cvx_b = matrix(eq_bnd.reshape(-1, 1))
    #
    cvx_G = transfer_to_spmatrix(ineq_matrix)
    cvx_h = matrix(ineq_bnd)
    #
    for key, val in solver_params.items():
        solvers.options[key] = val
    sol = solvers.lp(c=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params):
    w0, to, rho = turn_con
    A, bl_bu_bnds, G, h = parse_line_condition(lin_con)
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = \
        combine_lin_bnd_condition(A, bl_bu_bnds, G, h, bnd_con)
    N = u.shape[0]
    #
    q = matrix(np.hstack((-u, rho * np.ones(N))))
    #
    cvx_A = np.hstack((eq_matrix, np.zeros((eq_matrix.shape[0], N))))
    cvx_A = transfer_to_spmatrix(cvx_A)
    cvx_b = matrix(eq_bnd.reshape(-1, 1))
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
    sol = solvers.lp(c=q, G=cvx_G, h=cvx_h, A=cvx_A, b=cvx_b)
    #
    status = sol['status']
    if status == 'optimal':
        is_success = True
    else:
        is_success = False
    w = np.array(sol['x'])[:N, 0]
    return w, is_success, status


def solve(u, lin_con, bnd_con, turn_con, solver_params):
    if turn_con is not None:
        rtn = _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params)
    else:
        rtn = _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params)
    return rtn