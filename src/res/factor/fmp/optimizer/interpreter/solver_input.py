import numpy as np

import src.res.factor.util.agency as AGENCY

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.proj import Logger
from .constr import LinearConstraint , BoundConstraint , TurnConstraint , CovConstraint , ShortConstraint

__all__ = ['SolverInput' , 'Relaxer' , 'SolveCond' , 'SolveVars']

def rescale_array(v : np.ndarray , scale_value : float = 1):
    return scale_value * v / (v.sum() + 1e-6)

@dataclass
class SolverInput:
    alpha : np.ndarray
    lin_con : LinearConstraint
    bnd_con : BoundConstraint
    turn_con : TurnConstraint | Any = None
    cov_con  : CovConstraint  | Any = None
    short_con : ShortConstraint | Any = None
    w0 : np.ndarray | Any = None
    wb : np.ndarray | Any = None
    utility_scaler : float = 1.
    relax_possible : bool = True

    def __post_init__(self):
        if self.turn_con is None: 
            self.turn_con   = TurnConstraint()
        if self.cov_con  is None: 
            self.cov_con    = CovConstraint()
        if self.short_con is None: 
            self.short_con = ShortConstraint()

        self.check()
        self.relaxer = Relaxer() if self.relax_possible else None

    @property
    def relaxable(self): return bool(self.relaxer)

    def check(self):
        assert self.alpha.ndim == 1 , self.alpha.ndim
        N = len(self.alpha)
        self.lin_con.check(N)
        self.bnd_con.check(N)
        if self.turn_con:  
            self.turn_con.check(N)
        if self.cov_con:   
            self.cov_con.check(N)
        if self.short_con: 
            self.short_con.check(N)
        assert self.w0 is None or self.w0.shape == (N , )
        assert self.wb is None or self.wb.shape == (N , )

    def copy(self): return deepcopy(self)

    def rescale(self , inplace = False):
        new = self if inplace else self.copy()
        scaler = new.alpha.std() + 1e-6
        new.alpha /= scaler
        new.cov_con.rescale(scaler)
        new.turn_con.rescale(scaler)
        new.short_con.rescale(scaler)
        new.utility_scaler *= scaler
        return new
    
    def relax(self):
        if self.relaxer: 
            self.relaxer(self)
        return self
    
    def utility(self , w : np.ndarray | None = None , prob_type = 'linprog' , turn = True ,  qobj = True , qcon = True , short = True):
        utility = AGENCY.PortCreateUtility()
        if w is not None:
            utility(alpha = w.dot(self.alpha))
            if (qobj and prob_type != 'linprog' and self.cov_con is not None and
                self.cov_con.lmbd is not None and self.wb is not None):
                utility(square = -0.5 * self.cov_con.lmbd * self.cov_con.variance(w - self.wb))
            if turn and self.turn_con is not None and self.turn_con.rho is not None and self.w0 is not None:
                utility(turnover = -self.turn_con.rho * np.abs(w - self.w0).sum())
            if short and self.short_con is not None and self.short_con.cost is not None:
                utility(short = self.short_con.cost * np.minimum(0 , w).sum())
            utility = utility * self.utility_scaler
        return utility
    
    def accuracy(self , w : np.ndarray | None = None):
        accuracy = AGENCY.PortCreateAccuracy()
        if w is not None:
            accuracy(lin_ub_bias = np.min(self.lin_con.ub - self.lin_con.A.dot(w)))
            accuracy(lin_lb_bias = np.min(self.lin_con.A.dot(w) - self.lin_con.lb))

            accuracy(bnd_ub_bias = np.min(self.bnd_con.ub - w))
            accuracy(bnd_lb_bias = np.min(w - self.bnd_con.lb))
            
            if self.turn_con and self.w0 is not None:
                accuracy(excess_turn = self.turn_con.dbl - np.abs(w - self.w0).sum())
                
            if self.cov_con is not None and self.cov_con.te is not None and self.wb is not None:
                optimize_te = np.sqrt(self.cov_con.variance(w - self.wb))
                accuracy(excess_te = self.cov_con.te - optimize_te)

            if self.short_con:
                accuracy(excess_short = self.short_con.pos + np.minimum(0 , w).sum())

        return accuracy

    @classmethod
    def rand(cls , N : int = 3 , with_turn = True , with_cov = True):
        return cls(
            alpha = np.random.randn(N) ,
            lin_con = LinearConstraint.rand(N) ,
            bnd_con = BoundConstraint.rand(N) ,
            turn_con = TurnConstraint.rand(N) if with_turn else None ,
            cov_con = CovConstraint.rand(N) if with_cov else None ,
            w0 = rescale_array(np.random.rand(N)) ,
            wb = rescale_array(np.random.rand(N)) ,
            short_con = ShortConstraint(0.1 , 0.002) ,
        )
    
class Relaxer:
    def __init__(self) -> None:
        self.queue = ['conflicted_lin_con',
                      'free_tracking_error' ,
                      'double_turnover' , 
                      'expand_portional_lin_con' ,
                      'expand_numerical_lin_con' ,
                      'free_turnover']
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(queue={str(self.queue)})'
    def __bool__(self): return bool(self.queue)
    def __call__(self, solver_input): return self.relax(solver_input)
        
    def relax(self , solver_input : SolverInput):
        relaxed = False
        while self and not relaxed:
            method = getattr(self , self.queue.pop(0))
            relaxed = method(bnd_con = solver_input.bnd_con ,
                             lin_con = solver_input.lin_con ,
                             turn_con = solver_input.turn_con ,
                             cov_con = solver_input.cov_con)

        return relaxed

    def conflicted_lin_con(self , bnd_con : BoundConstraint , lin_con : LinearConstraint , **kwargs):
        relaxed = False
        dup_groups = self.get_duplicated_rows(lin_con.A)
        for dup_rows in dup_groups:
            if lin_con.lb[dup_rows].max() <= lin_con.ub[dup_rows].min(): 
                continue
            Logger.stdout('Deal with conflicted duplicated linear constraints!')
            fx_b = lin_con.lb[dup_rows].max()
            lin_con.lb[dup_rows] = fx_b - 1e-6
            lin_con.ub[dup_rows] = fx_b + 1e-6
            lin_con.type[dup_rows] = 'ra'
            relaxed = True

        rows = lin_con.A.min(axis = 1) >= 0 # all positive rows
        if np.any(lin_con.ub[rows] < lin_con.A[rows].dot(bnd_con.lb)):
            # bnd lower bound > lin upper bound
            Logger.stdout('Lift up conflicted linear upper bound to weight bound!')
            lin_con.ub[rows] = np.maximum(lin_con.ub[rows] , lin_con.A[rows].dot(bnd_con.lb)) 
            relaxed = True

        if np.any(lin_con.lb[rows] > lin_con.A[rows].dot(bnd_con.ub)):
            # bnd upper bound < lin lower bound
            Logger.stdout('Draw down conflicted linear lower bound to weight bound!')
            lin_con.lb[rows] = np.minimum(lin_con.lb[rows] , lin_con.A[rows].dot(bnd_con.ub)) 
            relaxed = True

        rows = lin_con.A.max(axis = 1) <= 0 # all negatvie rows
        if np.any(lin_con.ub[rows] < lin_con.A[rows].dot(bnd_con.ub)):
            # bnd lower bound > lin upper bound
            Logger.stdout('Lift up conflicted linear upper bound to weight bound!')
            lin_con.ub[rows] = np.maximum(lin_con.ub[rows] , lin_con.A[rows].dot(bnd_con.ub))
            relaxed = True

        if np.any(lin_con.lb[rows] > lin_con.A[rows].dot(bnd_con.lb)):
            # bnd upper bound < lin lower bound
            Logger.stdout('Draw down conflicted linear lower bound to weight bound!')
            lin_con.lb[rows] = np.minimum(lin_con.lb[rows] , lin_con.A[rows].dot(bnd_con.lb))
            relaxed = True

        return relaxed

    def free_tracking_error(self , cov_con : CovConstraint , **kwargs):
        relaxed = False
        if cov_con.te is not None:
            cov_con.te = None
            Logger.stdout(f'Free turnover constraint!')
            relaxed = True
        return relaxed
    
    def double_turnover(self , turn_con : TurnConstraint , **kwargs):
        relaxed = False
        if turn_con.dbl is not None:
            turn_con.dbl = turn_con.dbl * 2
            Logger.stdout(f'Double turnover constraint to {turn_con.dbl :.2%}!')
            relaxed = True
        return relaxed
    
    def expand_portional_lin_con(self , lin_con : LinearConstraint , **kwargs):
        relaxed = False
        
        rng = (lin_con.ub - lin_con.lb) * 0.5
        rows = ((lin_con.A.min(axis = 1) >= 0) + (lin_con.A.max(axis = 1) <= 0)) * (lin_con.type == 'ra')
        if rows.any() > 0:
            Logger.stdout(f'Expand {rows.sum():d} portional constraint 2 times!')
            lin_con.ub[rows] = lin_con.ub[rows] + rng[rows]
            lin_con.lb[rows] = lin_con.lb[rows] - rng[rows]
            relaxed = True

        return relaxed
    
    def expand_numerical_lin_con(self , lin_con : LinearConstraint , **kwargs):
        relaxed = False

        rng = (lin_con.ub - lin_con.lb) * (lin_con.type == 'ra') * 0.5
        rows = (lin_con.A.min(axis = 1) < 0) * (lin_con.A.max(axis = 1) > 0) * (lin_con.type == 'ra')
        if rows.any() > 0:
            Logger.stdout(f'Expand {rows.sum():d} numerial constraint 2 times!')
            lin_con.ub[rows] = lin_con.ub[rows] + rng[rows]
            lin_con.lb[rows] = lin_con.lb[rows] - rng[rows]
            relaxed = True

        return relaxed
    
    def free_turnover(self , turn_con : TurnConstraint , **kwargs):
        relaxed = False
        if turn_con.dbl is not None:
            turn_con.dbl = None
            Logger.stdout(f'Free turnover constraint!')
            relaxed = True
        return relaxed

    @staticmethod
    def get_duplicated_rows(A : np.ndarray):
        _ , index , inverse = np.unique(A , axis=0 , return_index = True , return_inverse = True)
        dropped = np.setdiff1d(np.arange(len(A)) , index)
        remained = index[inverse[dropped]]
        duplicated = np.concatenate([dropped , remained])
        new_index = inverse[duplicated] 
        rslt = []
        for idx in np.unique(new_index):
            rslt.append(duplicated[new_index == idx])
        return rslt

@dataclass
class SolveCond:
    '''
    additional condition for solver
    turn : if consider turnover constrains
    qobj : if consider quad objective for risk aversion
    qcon : if consider quad constraint for tracking error
    short : if consider possible short position
    '''
    turn : bool = True
    qobj : bool = True
    qcon : bool = True
    short : bool = True

@dataclass
class SolveVars:
    '''
    record number of vars or start position of vars in solver
    N : weight variable to solve
    T : turnover variable , 0 or N , must be positive
    S : shortsell variable , 0 or N , must be positive
    L : risk model factor variable , 0 or len(cov_con.F) , equals portfolio risk factor exposure
    Q : risk model quad constraint variable , 0 or 2 , equals portfolio common risk and spec risk
    '''
    N : int         # normal variable
    T : int = 0     # turnover variable
    S : int = 0     # shortsell variable
    L : int = 0     # model factor
    Q : int = 0     # quadprog factor

    def start_of(self):
        start_of   = SolveVars(0)
        start_of.T = start_of.N + self.N
        start_of.S = start_of.T + self.T
        start_of.L = start_of.S + self.S
        start_of.Q = start_of.L + self.L
        return start_of