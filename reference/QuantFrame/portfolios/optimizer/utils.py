import numpy as np

try:
    from .mosek_solver.linprog import solve as linprog_mosek_solve
    from .mosek_solver.quadprog import solve as quadprog_mosek_solve
    from .mosek_solver.socp import solve as socp_mosek_solve
except:
    pass

try:
    from .cvxopt_solver.linprog import solve as linprog_cvxopt_solve
    from .cvxopt_solver.quadprog import solve as quadprog_cvxopt_solve
    from .cvxopt_solver.socp import solve as socp_cvxopt_solve
except:
    pass

try:
    from .cvxpy_solver.linprog import solve as linprog_cvxpy_solve
    from .cvxpy_solver.quadprog import solve as quadprog_cvxpy_solve
    from .cvxpy_solver.socp import solve as socp_cvxpy_solve
except:
    pass


def calc_utility_func(prob_type, w, u, turn_con, cov_info=None, wb=None):
    if prob_type == "linprog":
        if turn_con is not None:
            w0, to, rho = turn_con
            rtn = w.dot(u) - rho * np.abs(w - w0).sum()
        else:
            rtn = w.dot(u)
    elif prob_type == "quadprog" or prob_type == "socp":
        assert cov_info is not None and wb is not None
        if len(cov_info) == 4:
            lmbd, F, C, S = cov_info
            quad_term = (w - wb).T.dot(F.T).dot(C).dot(F).dot(w - wb) + ((w - wb) * S).dot(w - wb)
        elif len(cov_info) == 2:
            lmbd, cov = cov_info
            quad_term = (w - wb).T.dot(cov).dot(w - wb)
        else:
            assert False
        if turn_con is not None:
            w0, to, rho = turn_con
            rtn = w.dot(u) - 0.5 * lmbd * quad_term - rho * np.abs(w - w0).sum()
        else:
            rtn = w.dot(u) - 0.5 * lmbd * quad_term
    else:
        assert False
    return rtn


def get_rslt_accuracy(w, lin_con, bnd_con, cov_info=None, wb=None, te=None, turn_con=None):
    A, b = lin_con[0], lin_con[1]
    bl, bu = b[1], b[2]
    lb, ub = bnd_con[1], bnd_con[2]
    #
    min_lin_con_ub_bias = np.min(bu - A.dot(w))
    max_lin_con_lb_bias = np.max(A.dot(w) - bl)
    min_w_ub_bias = np.min(ub - w)
    max_w_lb_bias = np.max(w - lb)
    if turn_con is not None:
        w0, to, rho = turn_con
        excess_to = np.abs(w - w0).sum() - to
    else:
        excess_to = 0.0
    if te is not None:
        assert cov_info is not None and wb is not None
        if len(cov_info) == 2:
            lmbd, cov = cov_info
            optimize_te = np.sqrt((w - wb).T.dot(cov).dot(w - wb))
        elif len(cov_info) == 4:
            lmbd, F, C, S = cov_info
            optimize_te = np.sqrt((w - wb).T.dot(F.T).dot(C).dot(F).dot(w - wb) + ((w - wb) * S).dot(w - wb))
        else:
            assert False
        excess_te = optimize_te - te
    else:
        excess_te = 0.0
    return min_lin_con_ub_bias, max_lin_con_lb_bias, min_w_ub_bias, max_w_lb_bias, excess_to, excess_te


def rescale_params(u=None, cov_info=None, te=None, turn_con=None):
    a = 1.
    if u is not None:
        a = u.std()
        if a < 1.0e-6: a = 1.0  # TODO: temporary
        assert a > 0.0
        u = u / a
    #
    if cov_info is not None:
        if len(cov_info) == 4:
            lmbd, F, C, S = cov_info
            c = S.mean()
            #
            S_scl = S / c
            C_scl = C / c
            lmbd_scl = lmbd * c / a
            cov_info = (lmbd_scl, F, C_scl, S_scl)
        elif len(cov_info) == 2:
            lmbd, cov = cov_info
            c = np.abs(np.diagonal(cov)).mean()
            assert c > 0.
            cov_scl = cov / c
            lmbd_scl = lmbd * c / a
            cov_info = (lmbd_scl, cov_scl)
        else:
            assert False, "  error::optimizer>>utils>>unknown cov_info type."
        if te is not None:
            te = te / np.sqrt(c)
    #
    if turn_con is not None:
        w0, to, rho = turn_con
        rho_scl = rho / a
        turn_con = (w0, to, rho_scl)
    return u, cov_info, turn_con, te


DEFAULT_SOLVER_PARAM = {
    "cvxpy.ecos": {'max_iters': 200, 'bnd_inf': 100.0},
    "cvxpy.osqp": {'bnd_inf': 100.0},
    "cvxpy.scs": {'eps': 1e-6, 'bnd_inf': 100.0},
    "mosek": {},
    "cvxopt": {"show_progress": False}
}