import numpy as np
import cvxpy as cp
from .util import parse_linear_condition, parse_bound_conditions


def _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, F, C, S = cov_info
    L, N = F.shape
    u_a = u.T + lmbd * (wb.dot(F.T).dot(C).dot(F) + wb * S)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N + L)
    obj = cp.Minimize((lmbd / 2.0) * cp.quad_form(x[N:], C) + (lmbd / 2.0) * cp.sum_squares(cp.multiply(np.sqrt(S), x[:N])) -
                      u_a.T @ x[:N])
    constraints = [
        x[:N] >= lb,
        x[:N] <= ub,
        F @ x[:N] - x[N:] == 0]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = x.value[:N]
    return w, is_success, status


def _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, F, C, S = cov_info
    L, N = F.shape
    u_a = u.T + lmbd * (wb.dot(F.T).dot(C).dot(F) + wb * S)
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    w0, to, rho = turn_con
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N * 2 + L)
    obj = cp.Minimize((lmbd / 2.0) * cp.quad_form(x[2 * N:], C) + (lmbd / 2.0) * cp.sum_squares(cp.multiply(np.sqrt(S), x[:N])) -
                      u_a.T @ x[:N] + rho * cp.sum(x[N: 2 * N]))
    constraints = [
        x[:N] >= lb,
        x[:N] <= ub,
        x[:N] - x[N: 2 * N] <= w0,
        -x[:N] - x[N: 2 * N] <= -w0,
        x[N: 2 * N] >= 0,
        cp.sum(x[N: 2 * N]) <= to,
        F @ x[:N] - x[2 * N:] == 0]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = x.value[:N]
    return w, is_success, status


def _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, cov = cov_info
    N = u.shape[0]
    u_a = u.T + lmbd * wb.dot(cov)
    V = np.linalg.cholesky(cov).T
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N)
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(V @ x) - u_a.T @ x)
    constraints = [
        x >= lb,
        x <= ub
    ]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = x.value[:N]
    return w, is_success, status


def _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    lmbd, cov = cov_info
    N = u.shape[0]
    u_a = u.T + lmbd * wb.dot(cov)
    V = np.linalg.cholesky(cov).T
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    w0, to, rho = turn_con
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N * 2)
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(V @ x[:N]) - u_a.T @ x[:N] + rho * cp.sum(x[N: 2 * N]))
    constraints = [
        x[:N] >= lb,
        x[:N] <= ub,
        x[:N] - x[N: 2 * N] <= w0,
        -x[:N] - x[N: 2 * N] <= -w0,
        x[N: 2 * N] >= 0,
        cp.sum(x[N: 2 * N]) <= to]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = x.value[:N]
    return w, is_success, status


WARNING_NOT_PRINTED = True


def solve(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params):
    global WARNING_NOT_PRINTED
    if WARNING_NOT_PRINTED:
        print('  warning::>>cvxpy>>identity cov matrix input may raise error after multiple runs '
              'due to a deep scipy bug.')
        WARNING_NOT_PRINTED = False
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
        assert False, ' error::>>cvxpy>>quadprog>>cov_info is unknown.'
    if rtn[2] == 'optimal_inaccurate':
        print('  warning::>>cvxpy>>quadprog>>the result is optimal but inaccurate, increase max_iters/max_iter may help.')
    return rtn


