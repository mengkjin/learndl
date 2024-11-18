import numpy as np
import cvxpy as cp

from typing import Literal
from ..interpreter import SolverInput , SolveCond , SolveVars

_SOLVER_PARAM = {
    'ECOS': {'max_iters': 200},
    'SCS': {'eps': 1e-6},
}

class Solver:
    def __init__(self , input : SolverInput , 
                 prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'socp' ,
                 cvxpy_solver : Literal['mosek','ecos','osqp','scs','clarabel'] = 'mosek' , **kwargs):
        self.input = input
        self.prob_type : Literal['linprog' , 'quadprog' , 'socp'] = prob_type
        self.solver_name = cvxpy_solver.upper()

    def parse_input(self):
        self.alpha    = self.input.alpha
        self.w0       = self.input.w0 if self.input.w0 is not None else np.zeros_like(self.alpha)
        self.wb       = self.input.wb if self.input.wb is not None else np.zeros_like(self.alpha)

        if self.prob_type != 'linprog' and self.input.cov_con and self.input.wb is not None:
            self.cov_con   = self.input.cov_con
        else:
            self.cov_con   = None

        if self.input.turn_con and self.input.w0 is not None:
            self.turn_con  = self.input.turn_con
        else:
            self.turn_con  = None

        if self.input.short_con:
            self.short_con  = self.input.short_con
        else:
            self.short_con  = None

        # variable sequence:
        # num_N , num_T (0 or num_N) , num_S (0 or num_N) , num_L (0 or len(self.F)) , num_Q (0 or 2)
        num_N = len(self.alpha)
        num_T = 0 if not self.conds.turn or not self.turn_con else num_N
        num_S = 0 if not self.conds.short or not self.short_con else num_N
        if (not self.conds.qobj and not self.conds.qcon) or not self.cov_con or self.cov_con.cov_type != 'model':
            num_L = 0
        else: num_L = len(self.cov_con.F)

        self.num_vars = SolveVars(num_N , num_T , num_S , num_L)
        return self

    def solve(self , turn = True , qobj = True , qcon = True , short = True):
        self.conds = SolveCond(turn , qobj , qcon , short)
        self.parse_input()

        x = cp.Variable(self.num_vars.N)
        objective = -self.input.alpha.T @ x
        constraints :list = [
            x <= self.input.bnd_con.ub ,
            x >= self.input.bnd_con.lb ,
        ]

        if self.cov_con:
            if self.num_vars.L:
                l = cp.Variable(self.num_vars.L)
                constraints.append(self.cov_con.F @ (x - self.wb) == l)
                if self.cov_con.lmbd:
                    S_sq = np.sqrt(self.cov_con.S)
                    objective = objective + self.cov_con.lmbd / 2.0 * \
                        (cp.sum_squares(cp.multiply(x - self.wb , S_sq)) + cp.quad_form(l , self.cov_con.C) )
                if self.cov_con.te:
                    constraints.append(cp.sum_squares(cp.multiply(x - self.wb , S_sq)) + 
                                cp.quad_form(l , self.cov_con.C) <= self.cov_con.te ** 2)
            else:
                if self.cov_con.lmbd:
                    objective = objective + self.cov_con.lmbd / 2.0 * cp.quad_form(x , self.cov_con.cov)
                if input.cov_con.te:
                    constraints.append(cp.quad_form(x , self.cov_con.cov) <= self.cov_con.te ** 2)
        
        eq_pos = self.input.lin_con.type == 'fx'
        if np.any(eq_pos):
            mat = self.input.lin_con.A[eq_pos]
            bnd = self.input.lin_con.lb[eq_pos]
            constraints.append(mat @ x == bnd)

        up_pos = np.isin(self.input.lin_con.type,['ra', 'up'])
        lo_pos = np.isin(self.input.lin_con.type,['ra', 'lo'])
        if np.any(up_pos) or np.any(lo_pos):
            mat = np.vstack((self.input.lin_con.A[up_pos], -self.input.lin_con.A[lo_pos]))
            bnd = np.hstack((self.input.lin_con.ub[up_pos], self.input.lin_con.lb[lo_pos]))
            constraints.append(mat @ x <= bnd)

        if self.turn_con and self.num_vars.T:
            t = cp.Variable(self.num_vars.T)
            constraints.append(t >= 0)
            constraints.append(x - t <= self.w0)
            constraints.append(-x - t <= -self.w0)

            if self.turn_con.dbl: constraints.append(cp.sum(t) <= self.turn_con.dbl)
            if self.turn_con.rho: objective = objective + self.turn_con.rho * cp.sum(t)  

        if self.short_con and self.num_vars.S:
            s = cp.Variable(self.num_vars.S)
            constraints.append(s >= 0)
            constraints.append(x - s >= 0)

            if self.short_con.pos:  constraints.append(cp.sum(s) <= self.short_con.pos)
            if self.short_con.cost: objective = objective + self.short_con.cost * cp.sum(s)  

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver = self.solver_name , **_SOLVER_PARAM.get(self.solver_name , {}))
        status = prob.status
        is_success = (status == 'optimal' or status == 'optimal_inaccurate')
        w = x.value
        return w, is_success, status