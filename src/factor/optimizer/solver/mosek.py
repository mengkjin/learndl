import numpy as np
import mosek

from typing import Any , Literal

from ..util import SolverInput , SolveCond , SolveVars

from ...basic.var import SYMBOL_INF as INF

_SOLVER_PARAM = {}

def mosek_bnd_key(bnd_key : str | list | np.ndarray):
    if isinstance(bnd_key , (list , np.ndarray)):
        return [mosek_bnd_key(k) for k in bnd_key]
    elif isinstance(bnd_key , str):
        if bnd_key == 'fx': return mosek.boundkey.fx
        elif bnd_key == 'lo': return mosek.boundkey.lo
        elif bnd_key == 'up': return mosek.boundkey.up
        elif bnd_key == 'ra': return mosek.boundkey.ra
        else: raise KeyError(bnd_key)
    else:
        return bnd_key
    
def enum(num : int , args : list[Any] , start = 0):         
    args = [arg if hasattr(arg , '__len__') else [arg] * num for arg in args]
    lens = np.array([len(arg) for arg in args])
    assert np.all(lens == num) , f'All lens must equal to num, but get {lens}'
    return enumerate(zip(*args) , start = start)

class Solver:
    def __init__(self , input : SolverInput , 
                 prob_type : Literal['linprog' , 'quadprog' , 'socp'] = 'socp' ,
                 **kwargs):
        self.input = input
        self.prob_type : Literal['linprog' , 'quadprog' , 'socp'] = prob_type

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
        if not self.conds.qcon or not self.cov_con or not self.cov_con.te:
            num_Q = 0
        else: num_Q = 2

        self.num_vars = SolveVars(num_N , num_T , num_S , num_L , num_Q)
        self.start_of = self.num_vars.start_of()

        return self

    def solve(self , turn = True , qobj = True , qcon = True , short = True):
        self.conds = SolveCond(turn , qobj , qcon , short)
        self.parse_input()

        with mosek.Env() as env:
            with env.Task() as task:
                # setting
                self.task_init(task)
                
                # linear objective
                self.task_add_lin_obj(task)

                # quad objective
                self.task_add_quad_obj(task)
                
                # lin constraints
                self.task_add_lin_con(task)
                # turnover constraint
                self.task_add_turn_con(task)
                # short constraint
                self.task_add_short_con(task)
                # L factor equivalent to F.dot(w)
                self.task_add_quad_model_con(task)
                # Quad TE constraint
                self.task_add_quad_te_con(task)
                
                # perform
                task.putobjsense(mosek.objsense.minimize) # mosek.objsense.maximize
                trmcode = task.optimize()
                task.solutionsummary(mosek.streamtype.msg)
                sol_sta = task.getsolsta(mosek.soltype.itr)
                xx = [0.0] * task.getnumvar()
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
                    status = str(sol_sta)

                self.task = task
        ww = np.array(xx , dtype=float)[:self.num_vars.N]

        self.optimal_x = xx
        self.optimal_w = ww

        return ww , is_success , status
    
    def task_init(self , task : mosek.Task): 
        for key, val in _SOLVER_PARAM.items() : task.putparam(key, val)
    
    def task_addvars(self , task : mosek.Task , num : int ,
                     bound_key : np.ndarray | Any , bound_lb : np.ndarray | Any , bound_ub : np.ndarray | Any ,
                     coef_obj : np.ndarray | Any = None):
        var_iter = enum(num , [bound_key , bound_lb , bound_ub , coef_obj] , start = task.getnumvar())
        task.appendvars(num)
        for j , (bkx , blx , bux , cj) in var_iter:
            task.putvarbound(j, bkx , blx , bux)
            if cj: task.putcj(j , cj)
    
    def task_addlcons(self , task : mosek.Task , num : int ,
                      bound_key : np.ndarray | Any , bound_lb : np.ndarray | Any , bound_ub : np.ndarray | Any ,
                      lcon_sub : np.ndarray | list , lcon_val : np.ndarray | list):
        con_iter = enum(num , [bound_key , bound_lb , bound_ub , lcon_sub , lcon_val] , start = task.getnumcon())
        task.appendcons(num)
        for i , (bkx , blx , bux , subi , vali) in con_iter:
            task.putconbound(i, bkx , blx , bux)
            task.putarow(i, subi , vali)

    def task_add_lin_obj(self , task : mosek.Task):
        # num_N stock weight
        bnd_type = mosek_bnd_key(self.input.bnd_con.type)
        self.task_addvars(task , self.num_vars.N , bnd_type , self.input.bnd_con.lb , self.input.bnd_con.ub , -self.alpha)

        # num_T absolute turnover
        self.task_addvars(task , self.num_vars.T , mosek.boundkey.lo , 0.0 , +INF , self.turn_con.rho if self.turn_con else 0)

        # num_S abs short weight
        self.task_addvars(task , self.num_vars.S , mosek.boundkey.lo , 0.0 , +INF , self.short_con.cost if self.short_con else 0)

        # num_L factor exposure
        self.task_addvars(task , self.num_vars.L , mosek.boundkey.fr , -INF , +INF , 0)

        # num_Q factor transformation
        self.task_addvars(task , self.num_vars.Q , mosek.boundkey.lo , 0.0 , +INF , 0)

    def task_add_quad_obj(self , task : mosek.Task):
        if not self.conds.qobj or not self.cov_con or not self.cov_con.lmbd: return
        
        if self.cov_con.cov_type == 'normal':
            u       = self.alpha.T + self.cov_con.lmbd * self.wb.dot(self.cov_con.cov)
            idx     = np.tril_indices(self.num_vars.N)
            qosubi  = idx[0]
            qosubj  = idx[1]
            qoval   = self.cov_con.lmbd * self.cov_con.cov[idx]
        else:
            u       = self.alpha.T + self.cov_con.lmbd * \
                (self.wb.dot(self.cov_con.F.T).dot(self.cov_con.C).dot(self.cov_con.F) + 
                 (0 if self.cov_con is None else self.wb * self.cov_con.S))
            idx     = np.tril_indices(self.num_vars.L)
            if self.cov_con.S is None:
                qosubi = idx[0] + self.start_of.L
                qosubj = idx[1] + self.start_of.L
                qoval  = self.cov_con.lmbd * self.cov_con.C[idx]
            else: 
                qosubi = np.concatenate([np.arange(self.num_vars.N) , idx[0] + self.start_of.L])
                qosubj = np.concatenate([np.arange(self.num_vars.N) , idx[1] + self.start_of.L])
                qoval  = self.cov_con.lmbd * np.concatenate([self.cov_con.S , self.cov_con.C[idx]])

        # override linear objective coefficient , mind the direction is negative for minimize
        [task.putcj(j, -u[j]) for j in range(self.num_vars.N)]
        # add quad objective
        task.putqobj(qosubi, qosubj, qoval)

    def task_add_lin_con(self , task : mosek.Task):
        if not self.input.lin_con: return
        # lin constraints
        K = len(self.input.lin_con)
        lin_type = mosek_bnd_key(self.input.lin_con.type)
        self.task_addlcons(task , K , lin_type , self.input.lin_con.lb , self.input.lin_con.ub , 
                           np.arange(self.num_vars.N)[None].repeat(K,0), self.input.lin_con.A)

    def task_add_turn_con(self , task : mosek.Task):
        if not self.num_vars.T or not self.turn_con or not self.turn_con.dbl: return
        # 1 : total turnover constraint
        self.task_addlcons(task , 1 , mosek.boundkey.up , -INF , self.turn_con.dbl , 
                           [self.start_of.T + np.arange(self.num_vars.T)] , [np.ones(self.num_vars.T)])
        
        # lcon_sub = [[i , self.start_of.T + i] for i in range(self.num_vars.T)]
        lcon_sub = np.arange(self.num_vars.T)[:,None] + np.array([0,self.start_of.T])
        lcon_vals = np.ones((self.num_vars.T,2)) * np.array([1,-1]) , -np.ones((self.num_vars.T,2))
        # N : turnover contrains w - delta <= w0
        self.task_addlcons(task , self.num_vars.T , mosek.boundkey.up , -INF ,  self.w0 , lcon_sub , lcon_vals[0]) # [[ 1.,-1.]] * self.num_vars.T
        # N : turnover contrains -w - delta <= -w0 
        self.task_addlcons(task , self.num_vars.T , mosek.boundkey.up , -INF , -self.w0 , lcon_sub , lcon_vals[1]) # [[-1.,-1.]] * self.num_vars.T
        
    def task_add_short_con(self , task : mosek.Task):
        if not self.num_vars.S or not self.short_con: return
        # 1 : total short constraint
        self.task_addlcons(task , 1 , mosek.boundkey.up , -INF , self.short_con.pos , 
                           [self.start_of.S + np.arange(self.num_vars.S)] , [np.ones(self.num_vars.S)])
        # N : turnover contrains w + short >= 0
        # lcon_sub = [[i , self.start_of.S + i] for i in range(self.num_vars.S)]
        # lcon_val = [[1.,1.]] * self.num_vars.S
        lcon_sub = np.arange(self.num_vars.S)[:,None] + np.array([0,self.start_of.S])
        lcon_val = np.ones((self.num_vars.S , 2))
        self.task_addlcons(task , self.num_vars.S , mosek.boundkey.lo , 0. , +INF , lcon_sub , lcon_val)
        
    def task_add_quad_model_con(self , task : mosek.Task):
        if not self.cov_con or not self.num_vars.L: return
        # num_L factors == self.F[i,:].dot(num_N variables)
        # lcon_sub : [[Range(n)] x L , Pos(L)]      lcon_val :  [F(L,N) , -1.0]
        #   [[0 1 2 ... n (start_of_L+0)]              [[f(a,0) f(a,1) f(a,2) ... f(a,n) -1.]
        #    [0 1 2 ... n (start_of_L+1)]               [f(b,0) f(b,1) f(b,2) ... f(b,n) -1.]
        #       ...                                         ...
        #    [0 1 2 ... n (start_of_L+L)]]              [f(L,0) f(L,0) f(L,0) ... f(L,n) -1.]] 
        lcon_sub = np.concatenate([np.arange(self.num_vars.N)[None].repeat(self.num_vars.L,0), 
                                   self.start_of.L + np.arange(self.num_vars.L)[:,None]] , axis=1)
        lcon_val = np.concatenate([self.cov_con.F,-np.ones((self.num_vars.L,1))] , axis=1)
        self.task_addlcons(task , self.num_vars.L , mosek.boundkey.fx , 0.0, 0.0 , lcon_sub , lcon_val)
        
    def task_add_quad_te_con(self , task : mosek.Task):
        if not self.conds.qcon or not self.cov_con or not self.cov_con.te: return
        te_sq = self.cov_con.te ** 2
        start = task.getnumcon()

        if self.cov_con.cov_type == 'normal': 
            idx  = np.tril_indices(self.num_vars.N)
            qcub = 0.5 * te_sq - 0.5 * self.wb.dot(self.cov_con.cov).dot(self.wb)

            task.appendcons(1)
            task.putconbound(start, mosek.boundkey.up, -INF, qcub)
            task.putarow(start, np.arange(self.num_vars.N), -self.wb.dot(self.cov_con.cov))
            task.putqconk(start, idx[0], idx[1], self.cov_con.cov[idx])
            
        elif self.cov_con.cov_type == 'model': 
            # total risk  
            task.appendcons(3) 
            task.putconbound(start, mosek.boundkey.up, -INF, te_sq)
            task.putarow(start, self.start_of.Q + np.arange(2), [1.0, 1.0])

            # common risk
            idx   = np.tril_indices(len(self.cov_con.F))
            qcub  = -0.5 * self.cov_con.F.dot(self.wb).T.dot(self.cov_con.C).dot(self.cov_con.F.dot(self.wb))
            qcsub = np.concatenate([self.start_of.L + np.arange(len(self.cov_con.F)) , [self.start_of.Q]])
            qcval = np.concatenate([-self.cov_con.F.dot(self.wb).T.dot(self.cov_con.C) , [-0.5]])

            task.putconbound(start + 1, mosek.boundkey.up, -INF, qcub)
            task.putarow(start + 1, qcsub, qcval)
            task.putqconk(start + 1, self.start_of.L + idx[0], self.start_of.L + idx[1] , self.cov_con.C[idx])
            
            # spec risk
            qcub  = -0.5 * (self.wb * self.cov_con.S).dot(self.wb)
            qcsub = np.concatenate([np.arange(self.num_vars.N) , [self.start_of.Q + 1]])
            qcval = np.concatenate([-self.wb * self.cov_con.S , [-0.5]])

            task.putconbound(start + 2, mosek.boundkey.up, -INF, qcub)
            task.putarow(start + 2, qcsub, qcval)
            task.putqconk(start + 2, np.arange(self.num_vars.N) , np.arange(self.num_vars.N) , self.cov_con.S)
