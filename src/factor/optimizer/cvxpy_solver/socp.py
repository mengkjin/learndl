import numpy as np
import cvxpy as cp
from .util import parse_linear_condition, parse_bound_conditions


def _transform_parameter_with_brcov(u, C, S, F, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd):
    vol_S = np.sqrt(S)
    u_adj = u * (1.0 / vol_S)
    V = np.linalg.cholesky(C).T
    K = V.dot(F) * (1.0 / vol_S)
    #
    eq_matrix_adj = eq_matrix * (1.0 / vol_S)
    eq_bnd_adj = eq_bnd - eq_matrix.dot(wb)
    ineq_matrix_adj = ineq_matrix * (1.0 / vol_S)
    ineq_bnd_adj = ineq_bnd - ineq_matrix.dot(wb)
    #
    lb_adj = (bnd_con[1] - wb) * vol_S
    ub_adj = (bnd_con[2] - wb) * vol_S
    bnd_con = [bnd_con[0], lb_adj, ub_adj]
    return u_adj, K, bnd_con, ineq_matrix_adj, ineq_bnd_adj, eq_matrix_adj, eq_bnd_adj


def _transform_parameter_with_nmcov(cov, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd):
    V = np.linalg.cholesky(cov).T
    #
    eq_bnd_adj = eq_bnd - eq_matrix.dot(wb)
    ineq_bnd_adj = ineq_bnd - ineq_matrix.dot(wb)
    #
    lb_adj = bnd_con[1] - wb
    ub_adj = bnd_con[2] - wb
    bnd_con = [bnd_con[0], lb_adj, ub_adj]
    return V, bnd_con, ineq_bnd_adj, eq_bnd_adj


def _solve_without_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, F, C, S = cov_info
    te_sq = te ** 2
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    u_adj, K, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = _transform_parameter_with_brcov(
        u, C, S, F, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd)
    #
    L, N = K.shape
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N + L)
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(x[: N]) + (lmbd / 2.0) * cp.sum_squares(x[N:]) -
                      u_adj.T @ x[:N])
    constraints = [
        cp.sum_squares(x[: N]) + cp.sum_squares(x[N:]) <= te_sq,
        x[:N] >= lb,
        x[:N] <= ub,
        K @ x[:N] - x[N:] == 0]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = np.array(x.value[:N]) * (1 / np.sqrt(S)) + wb
    return w, is_success, status


def _solve_with_turn_with_brcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    w0, to, rho = turn_con
    te_sq = te ** 2
    lmbd, F, C, S = cov_info
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    u_adj, K, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = _transform_parameter_with_brcov(
        u, C, S, F, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd)
    vol_S = np.sqrt(S)
    x0 = (w0 - wb) * vol_S
    L, N = K.shape
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N * 2 + L)
    #
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(x[: N]) + (lmbd / 2.0) * cp.sum_squares(x[2 * N:]) -
                      u_adj.T @ x[:N] + rho * cp.sum(x[N: 2 * N]))
    constraints = [
        cp.sum_squares(x[: N]) + cp.sum_squares(x[2 * N:]) <= te_sq,
        x[:N] >= lb,
        x[:N] <= ub,
        x[:N] - cp.multiply(x[N: 2 * N], vol_S) <= x0,
        -x[:N] - cp.multiply(x[N: 2 * N], vol_S) <= -x0,
        x[N: 2 * N] >= 0,
        cp.sum(x[N: 2 * N]) <= to,
        K @ x[:N] - x[2 * N:] == 0]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = np.array(x.value[:N]) * (1.0 / vol_S) + wb
    return w, is_success, status


def _solve_without_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    lmbd, cov = cov_info
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    V, bnd_con, ineq_bnd, eq_bnd = _transform_parameter_with_nmcov(
        cov, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd)
    #
    N = u.shape[0]
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N)
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(V @ x) - u.T @ x)
    constraints = [
        cp.norm2(V @ x[:N]) <= te,
        x >= lb,
        x <= ub
    ]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == 'optimal' or status == 'optimal_inaccurate')
    w = np.array(x.value[:N]) + wb
    return w, is_success, status


def _solve_with_turn_with_nmcov(u, cov_info, wb, lin_con, bnd_con, turn_con, te, solver_params):
    w0, to, rho = turn_con
    lmbd, cov = cov_info
    #
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    V, bnd_con, ineq_bnd, eq_bnd = _transform_parameter_with_nmcov(
        cov, wb, bnd_con, ineq_matrix, ineq_bnd, eq_matrix, eq_bnd)
    x0 = w0 - wb
    N = u.shape[0]
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N * 2)
    #
    obj = cp.Minimize((lmbd / 2.0) * cp.sum_squares(V @ x[:N]) - u.T @ x[:N] + rho * cp.sum(x[N: 2 * N]))
    constraints = [
        cp.norm2(V @ x[:N]) <= te,
        x[:N] >= lb,
        x[:N] <= ub,
        x[:N] - x[N: 2 * N] <= x0,
        -x[:N] - x[N: 2 * N] <= -x0,
        x[N: 2 * N] >= 0.0,
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
    w = np.array(x.value[:N]) + wb
    return w, is_success, status


WARNING_NOT_PRINTED = True


def solve(u, cov_info, wb, te, lin_con, bnd_con, turn_con, solver_params):
    global WARNING_NOT_PRINTED
    if WARNING_NOT_PRINTED:
        print('  warning::>>cvxpy>>identity cov matrix input may raise error after multiple runs '
              'due to a deep scipy bug.')
        WARNING_NOT_PRINTED = False
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
        assert False, ' error::>>cvxpy>>socp>>cov_info is unknown.'
    if rtn[2] == 'optimal_inaccurate':
        print('  warning::>>cvxpy>>socp>>the result is optimal but inaccurate, increase max_iters/max_iter may help.')
    return rtn
