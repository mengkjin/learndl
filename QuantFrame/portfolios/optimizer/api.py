from .inputs_utils.input_validator import validate_inputs
import copy
from .utils import *


def exec_linprog(engine_type, u, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True):
    validate_inputs(u, lin_con, bnd_con, turn_con)
    u, cov_info, turn_con, te = rescale_params(u, turn_con=turn_con)
    #
    if solver_params is None:
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAM[engine_type])
    else:
        solver_params = copy.deepcopy(solver_params)
    #
    if engine_type == "mosek":
        solve = linprog_mosek_solve
    elif engine_type == "cvxopt":
        solve = linprog_cvxopt_solve
    elif engine_type == "cvxpy.ecos":
        solve = linprog_cvxpy_solve
        solver_params["solver"] = "ECOS"
    elif engine_type == "cvxpy.scs":
        solve = linprog_cvxpy_solve
        solver_params["solver"] = "SCS"
    elif engine_type == "cvxpy.osqp":
        solve = linprog_cvxpy_solve
        solver_params["solver"] = "OSQP"
    else:
        assert False, "  error:>>optimizer>>exec_linprog>>Engine type is unknown!"
    w, is_success, status = solve(u, lin_con, bnd_con, turn_con, solver_params)
    if return_detail_infos:
        rtn = w, is_success, status, \
              calc_utility_func('linprog', w, u, turn_con), get_rslt_accuracy(w, lin_con, bnd_con, turn_con=turn_con)
    else:
        rtn = w, is_success, status
    return rtn


def exec_quadprog(engine_type, u, cov_info, wb, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True):
    validate_inputs(u, lin_con, bnd_con, turn_con=turn_con, cov_info=cov_info, wb=wb)
    u, cov_info, turn_con, te = rescale_params(u, cov_info=cov_info, turn_con=turn_con)
    #
    if solver_params is None:
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAM[engine_type])
    else:
        solver_params = copy.deepcopy(solver_params)
    #
    if engine_type == "mosek":
        solve = quadprog_mosek_solve
    elif engine_type == "cvxopt":
        solve = quadprog_cvxopt_solve
    elif engine_type == "cvxpy.ecos":
        solve = quadprog_cvxpy_solve
        solver_params["solver"] = "ECOS"
    elif engine_type == "cvxpy.scs":
        solve = quadprog_cvxpy_solve
        solver_params["solver"] = "SCS"
    elif engine_type == "cvxpy.osqp":
        solve = quadprog_cvxpy_solve
        solver_params["solver"] = "OSQP"
    else:
        assert False, "  error::>>optimizer>>exec_quadprog>>Engine type is unknown!"
    w, is_success, status = solve(u, cov_info, wb, lin_con, bnd_con, turn_con, solver_params)
    if return_detail_infos:
        rtn = w, is_success, status, calc_utility_func('quadprog', w, u, turn_con, cov_info, wb), \
              get_rslt_accuracy(w, lin_con, bnd_con, turn_con=turn_con)
    else:
        rtn = w, is_success, status
    return rtn


def exec_socp(engine_type, u, cov_info, wb, te: float, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True):
    validate_inputs(u, lin_con, bnd_con, turn_con=turn_con, cov_info=cov_info, wb=wb, te=te)
    u_adj, cov_info_adj, turn_con_adj, te_adj = rescale_params(u, cov_info=cov_info, turn_con=turn_con, te=te)
    #
    if solver_params is None:
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAM[engine_type])
    else:
        solver_params = copy.deepcopy(solver_params)
    #
    if engine_type == "mosek":
        solve = socp_mosek_solve
    elif engine_type == "cvxopt":
        solve = socp_cvxopt_solve
    elif engine_type == "cvxpy.ecos":
        solve = socp_cvxpy_solve
        solver_params["solver"] = "ECOS"
    elif engine_type == "cvxpy.scs":
        solve = socp_cvxpy_solve
        solver_params["solver"] = "SCS"
    elif engine_type == "cvxpy.osqp":
        assert False, "  error::>>optimizer>>exec_socp>>cvxpy.osqp could not solve socp!"
    else:
        assert False, "  error::>>optimizer>>exec_socp>>Engine type is unknown!"
    w, is_success, status = solve(u_adj, cov_info_adj, wb, te_adj, lin_con, bnd_con, turn_con_adj, solver_params)
    if return_detail_infos:
        rtn = w, is_success, status, calc_utility_func('socp', w, u, turn_con, cov_info, wb), \
              get_rslt_accuracy(w, lin_con, bnd_con, turn_con=turn_con, te=te, cov_info=cov_info, wb=wb)
    else:
        rtn = w, is_success, status
    return rtn