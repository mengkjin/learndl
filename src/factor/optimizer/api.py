
import numpy as np
from dataclasses import dataclass
from typing import Any , Literal , Optional

from src.environ import PATH

from .basic import DEFAULT_SOLVER_PARAM
from .solver.mosek import Solver as MosekSolver
from .util import Accuarcy , PortfolioOptimizerInput , SolverInput , Utility
from ..basic import Analytic , Port

SOLVER_CLASS = {
    'mosek' : MosekSolver
}

@dataclass
class PortfolioOptimizer:
    prob_type   : Literal['linprog' , 'quadprog' , 'socp'] = 'socp'
    engine_type : Literal['mosek' , 'cvxopt' , 'cvxpy'] = 'mosek'
    cvxpy_solver : Literal['mosek' , 'ecos' , 'osqp' , 'scs'] = 'mosek'
    auto_relax   : bool = True

    ignore_turn  : bool = False
    ignore_qobj  : bool = False
    ignore_qcon  : bool = False
    ignore_short : bool = False

    def __post_init__(self):
        key = f'{self.engine_type}.{self.cvxpy_solver}' if self.engine_type == 'cvxpy' else self.engine_type
        self.param = DEFAULT_SOLVER_PARAM[key]
        
    def setup_optimizer(self , config_path : Optional[str] = None):
        given_config = PATH.read_yaml(config_path) if config_path else {}
        self.opt_input = PortfolioOptimizerInput(given_config)

    def solve(self , solver_input : SolverInput):
        self.solver = SOLVER_CLASS[self.engine_type](solver_input, self.prob_type , self.param)

        while True:
            w, is_success, status = self.solver.solve(
                turn = not self.ignore_turn ,
                qobj = not self.ignore_qobj ,
                qcon = not self.ignore_qcon ,
                short = not self.ignore_short)
            if is_success or not self.auto_relax or solver_input.relaxable: break

        if not is_success and self.auto_relax:
            print('Failed optimization, even with relax, use w0 instead.')
            assert solver_input.w0 is not None , 'In this failed-with-relax case, w0 must not be None'
            w = solver_input.w0

        return w, is_success, status
        
    def optimize(self , model_date : int , initial_port : Any = None , secid : Any = None , detail_infos = True):
        self.solver_input = self.opt_input.to_solver_input(model_date , initial_port , secid).rescale()
        w , is_success , status = self.solve(self.solver_input)
        rslt = PortOptimResult(w , self.opt_input.secid , is_success , status)
        
        rslt.create_port(model_date , self.opt_input.portfolio_name , self.opt_input.initial_value)
        if detail_infos:
            rslt.utility  = self.solver_input.utility(w , self.prob_type , not self.ignore_turn , not self.ignore_qobj , not self.ignore_short) 
            rslt.accuracy = self.solver_input.accuracy(w)
            rslt.analytic = self.opt_input.analytic(rslt.port , self.opt_input.benchmark_port , self.opt_input.initial_port)
        return rslt

@dataclass
class PortOptimResult:
    w           : np.ndarray
    secid       : np.ndarray | Any
    is_success  : bool
    status      : Literal['optimal', 'max_iteration', 'stall'] | Any = ''
    utility     : Utility | Any = None
    accuracy    : Accuarcy | Any = None
    analytic    : Analytic | Any = None
    port        : Port | Any = None

    def __post_init__(self):
        if self.utility is None: self.utility = Utility()
        if self.accuracy is None: self.accuracy = Accuarcy()

    def __repr__(self):
        return '\n'.join([
            f'{self.__class__.__name__} ---------- ' ,
            f'Information : ' ,
            f'    is_success = {self.is_success} ,' ,
            f'    status     = {self.status} ,' ,
            f'Result : ' ,
            f'    utility    = {self.utility}',
            f'    accuracy   = {self.accuracy}',
            str(self.port) ,
            f'Analytic : (Only show style , access industry/risk mannually)' ,
            self.analytic.styler('style') ,
            f'Other components include [\'w\' , \'secid\'])'
        ])
    
    def create_port(self , date : int = -1 , name : Optional[str] = 'port' , value : float | Any = None , eps = 1e-6):
        w = np.where((self.w <= eps) * (self.w >= -eps) ,  0. , self.w)
        self.port = Port.create(self.secid , w , date = date , name = name , value = value)
        return self

    @staticmethod
    def trim_w(w : np.ndarray , eps = 1e-6) -> np.ndarray:
        w1 = w * 1.
        w1[(w1 <= eps) & (w1 >= -eps)] = 0
        return w1

'''
def exec_linprog(engine_type, u, lin_con, bnd_con, turn_con=None, solver_params=None, return_detail_infos=True):
    validate_inputs(u, lin_con, bnd_con, turn_con)
    u, cov_info, turn_con, te = rescale_params(u, turn_con=turn_con)
    #
    if solver_params is None:
        solver_params = copy.deepcopy(DEFAULT_SOLVER_PARAM[engine_type])
    else:
        solver_params = copy.deepcopy(solver_params)
    #
    if engine_type == 'mosek':
        solve = linprog_mosek_solve
    elif engine_type == 'cvxopt':
        solve = linprog_cvxopt_solve
    elif engine_type == 'cvxpy.ecos':
        solve = linprog_cvxpy_solve
        solver_params['solver'] = 'ECOS'
    elif engine_type == 'cvxpy.scs':
        solve = linprog_cvxpy_solve
        solver_params['solver'] = 'SCS'
    elif engine_type == 'cvxpy.osqp':
        solve = linprog_cvxpy_solve
        solver_params['solver'] = 'OSQP'
    else:
        assert False, '  error:>>optimizer>>exec_linprog>>Engine type is unknown!'
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
    if engine_type == 'mosek':
        solve = quadprog_mosek_solve
    elif engine_type == 'cvxopt':
        solve = quadprog_cvxopt_solve
    elif engine_type == 'cvxpy.ecos':
        solve = quadprog_cvxpy_solve
        solver_params['solver'] = 'ECOS'
    elif engine_type == 'cvxpy.scs':
        solve = quadprog_cvxpy_solve
        solver_params['solver'] = 'SCS'
    elif engine_type == 'cvxpy.osqp':
        solve = quadprog_cvxpy_solve
        solver_params['solver'] = 'OSQP'
    else:
        assert False, '  error::>>optimizer>>exec_quadprog>>Engine type is unknown!'
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
    if engine_type == 'mosek':
        solve = socp_mosek_solve
    elif engine_type == 'cvxopt':
        solve = socp_cvxopt_solve
    elif engine_type == 'cvxpy.ecos':
        solve = socp_cvxpy_solve
        solver_params['solver'] = 'ECOS'
    elif engine_type == 'cvxpy.scs':
        solve = socp_cvxpy_solve
        solver_params['solver'] = 'SCS'
    elif engine_type == 'cvxpy.osqp':
        assert False, '  error::>>optimizer>>exec_socp>>cvxpy.osqp could not solve socp!'
    else:
        assert False, '  error::>>optimizer>>exec_socp>>Engine type is unknown!'
    w, is_success, status = solve(u_adj, cov_info_adj, wb, te_adj, lin_con, bnd_con, turn_con_adj, solver_params)
    if return_detail_infos:
        rtn = w, is_success, status, calc_utility_func('socp', w, u, turn_con, cov_info, wb), \
              get_rslt_accuracy(w, lin_con, bnd_con, turn_con=turn_con, te=te, cov_info=cov_info, wb=wb)
    else:
        rtn = w, is_success, status
    return rtn
'''

if __name__ == '__main__':
    from src.factor.optimizer.api import PortfolioOptimizer
    config_path = 'custom_opt_config.yaml'

    optim = PortfolioOptimizer('socp')
    optim.setup_optimizer(config_path)

    s = optim.optimize(20240606)
    print(s)
