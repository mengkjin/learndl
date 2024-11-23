import numpy as np
import cvxpy as cp
from .util import parse_linear_condition, parse_bound_conditions


def _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params):
    N = u.shape[0]
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N)
    obj = cp.Minimize(-u.T @ x)
    constraints = [
        x >= lb,
        x <= ub,
    ]
    if ineq_matrix.size != 0:
        constraints.append(ineq_matrix @ x[:N] <= ineq_bnd)
    if eq_matrix.size != 0:
        constraints.append(eq_matrix @ x[:N] == eq_bnd)
    #
    prob = cp.Problem(obj, constraints)
    prob.solve(**solver_params)
    status = prob.status
    is_success = (status == "optimal" or status == "optimal_inaccurate")
    w = np.array(x.value[:N])
    return w, is_success, status


def _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params):
    N = u.shape[0]
    bnd_key, lb, ub = parse_bound_conditions(bnd_con, solver_params.pop('bnd_inf'))
    ineq_matrix, ineq_bnd, eq_matrix, eq_bnd = parse_linear_condition(lin_con)
    w0, to, rho = turn_con
    #
    # Define and solve the CVXPY problem.
    x = cp.Variable(N * 2)
    obj = cp.Minimize(-u.T @ x[:N] + rho * cp.sum(x[N: 2 * N]))
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
    is_success = (status == "optimal" or status == "optimal_inaccurate")
    w = x.value[:N]
    return w, is_success, status


def solve(u, lin_con, bnd_con, turn_con, solver_params):
    if turn_con is not None:
        rtn = _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params)
    else:
        rtn = _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params)
    if rtn[2] == "optimal_inaccurate":
        print("  warning::>>cvxpy>>linprog>>the result is optimal but inaccurate, increase max_iters/max_iter may help.")
    return rtn