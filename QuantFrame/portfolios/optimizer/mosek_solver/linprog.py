import numpy as np
import mosek
from .util import from_str_to_mosek_bnd_key


def _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params):
    #
    bnd_con = [[from_str_to_mosek_bnd_key(k) for k in bnd_con[0]], bnd_con[1], bnd_con[2]]
    lin_con = (lin_con[0], [[from_str_to_mosek_bnd_key(k) for k in lin_con[1][0]], lin_con[1][1], lin_con[1][2]])
    #
    N = u.shape[0]
    numvar = N
    line_con_num = len(lin_con[0])
    num_lcon = line_con_num
    #
    with mosek.Env() as env:
        with env.Task() as task:
            # setting
            for key, val in solver_params.items():
                task.putparam(key, val)
            #
            task.appendvars(numvar)
            for j in range(N):
                task.putcj(j, -u[j])
                task.putvarbound(j, bnd_con[0][j], bnd_con[1][j], bnd_con[2][j])
            #
            task.appendcons(num_lcon)
            for i in range(line_con_num):
                task.putconbound(i, lin_con[1][0][i], lin_con[1][1][i], lin_con[1][2][i])
                task.putarow(i, np.arange(N), lin_con[0][i, :])
            # optimize
            task.putobjsense(mosek.objsense.minimize)
            trmcode = task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            sol_sta = task.getsolsta(mosek.soltype.itr)
            xx = [0.0] * numvar
            task.getxx(mosek.soltype.itr, xx)
            if sol_sta == mosek.solsta.optimal:
                is_success = True
                status = 'optimal'
            elif sol_sta == mosek.solsta.unknown and trmcode == mosek.rescode.trm_max_iterations:
                is_success = True
                status = 'max_iteration'
            elif sol_sta == mosek.solsta.unknown and trmcode == mosek.rescode.trm_stall:
                is_success = True
                status = 'stall'
            else:
                is_success = False
                status = ''
    w = np.array(xx[:N])
    return w, is_success, status


def _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params):
    w0, to, rho = turn_con
    bnd_con = [[from_str_to_mosek_bnd_key(k) for k in bnd_con[0]], bnd_con[1], bnd_con[2]]
    lin_con = (lin_con[0], [[from_str_to_mosek_bnd_key(k) for k in lin_con[1][0]], lin_con[1][1], lin_con[1][2]])
    #
    N = u.shape[0]
    inf = 0.0  # for symbol
    numvar = 2 * N
    line_con_num = len(lin_con[0])
    num_lcon = 1 + 2 * N + line_con_num
    #
    with mosek.Env() as env:
        with env.Task() as task:
            # setting
            for key, val in solver_params.items():
                task.putparam(key, val)
            #
            task.appendvars(numvar)
            for j in range(N):
                task.putcj(j, -u[j])
                task.putcj(j + N, rho)
                task.putvarbound(j, bnd_con[0][j], bnd_con[1][j], bnd_con[2][j])
                task.putvarbound(j + N, mosek.boundkey.lo, 0.0, +inf)
            #
            task.appendcons(num_lcon)
            for i in range(line_con_num):
                task.putconbound(i, lin_con[1][0][i], lin_con[1][1][i], lin_con[1][2][i])
                task.putarow(i, list(range(N)), lin_con[0][i, :])
            task.putconbound(line_con_num, mosek.boundkey.up, -inf, to)
            task.putarow(line_con_num, list(range(N, N * 2)), [1.0] * N)
            for i in range(N):
                task.putconbound(line_con_num + 1 + i, mosek.boundkey.up, -inf, w0[i])
                task.putarow(line_con_num + 1 + i, [i, N + i], [1.0, -1.0])
            for i in range(N):
                task.putconbound(line_con_num + 1 + N + i, mosek.boundkey.up, -inf, -w0[i])
                task.putarow(line_con_num + 1 + N + i, [i, N + i], [-1.0, -1.0])
            # optimize
            task.putobjsense(mosek.objsense.minimize)
            trmcode = task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            sol_sta = task.getsolsta(mosek.soltype.itr)
            xx = [0.0] * numvar
            task.getxx(mosek.soltype.itr, xx)
            if sol_sta == mosek.solsta.optimal:
                is_success = True
                status = 'optimal'
            elif sol_sta == mosek.solsta.unknown and trmcode == mosek.rescode.trm_max_iterations:
                is_success = True
                status = 'max_iteration'
            elif sol_sta == mosek.solsta.unknown and trmcode == mosek.rescode.trm_stall:
                is_success = True
                status = 'stall'
            else:
                is_success = False
                status = ''
    w = np.array(xx[:N])
    return w, is_success, status


def solve(u, lin_con, bnd_con, turn_con, solver_params):
    if turn_con is not None:
        rtn = _solve_with_turn(u, lin_con, bnd_con, turn_con, solver_params)
    else:
        rtn = _solve_without_turn(u, lin_con, bnd_con, turn_con, solver_params)
    return rtn