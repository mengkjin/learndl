import time
import numpy as np

from dataclasses import dataclass , field
from typing import Any , Literal , Optional

from .solver import SOLVER_CLASS
from .util import Accuarcy , PortfolioOptimizerInput , SolverInput , Utility
from ..util import Analytic , Port , AlphaModel , Amodel , Portfolio , Benchmark
from ..basic.var import DEFAULT_SOLVER_CONFIG
from ...env import PATH

@dataclass
class PortfolioOptimizer:
    prob_type   : Literal['linprog' , 'quadprog' , 'socp'] = 'socp'
    engine_type : Literal['mosek' , 'cvxopt' , 'cvxpy'] = 'mosek'
    cvxpy_solver : Literal['mosek' , 'ecos' , 'osqp' , 'scs'] = 'mosek'
    
    opt_relax : bool = True
    opt_turn  : bool = True
    opt_qobj  : bool = True
    opt_qcon  : bool = True
    opt_short : bool = True

    def __post_init__(self):
        key = f'{self.engine_type}.{self.cvxpy_solver}' if self.engine_type == 'cvxpy' else self.engine_type
        self.param = DEFAULT_SOLVER_CONFIG[key]
        
    def setup_optimizer(self , portfolio_name : str = 'port' , 
                        config_path : Optional[str] = None):
        given_config = PATH.read_yaml(config_path) if config_path else {}
        self.opt_input = PortfolioOptimizerInput(portfolio_name , given_config)
        return self

    def solve(self , solver_input : SolverInput):
        self.solver = SOLVER_CLASS[self.engine_type](solver_input, self.prob_type , self.param)

        while True:
            w, is_success, status = self.solver.solve(
                turn = self.opt_turn ,
                qobj = self.opt_qobj ,
                qcon = self.opt_qcon ,
                short = self.opt_short)
            if is_success or not self.opt_relax or solver_input.relaxable: break

        if not is_success and self.opt_relax:
            print(f'Failed optimization at {self.model_date}, even with relax, use w0 instead.')
            assert solver_input.w0 is not None , 'In this failed-with-relax case, w0 must not be None'
            w = solver_input.w0

        return w, is_success, status
        
    def optimize(self , model_date : int , alpha_model :AlphaModel|Amodel|Any = None , 
                 benchmark : Optional[Benchmark | Portfolio | Port] = None , init_port : Port | Any = None , 
                 detail_infos = True):
        t0 = time.time()
        self.model_date = model_date
        self.solver_input = self.opt_input.to_solver_input(model_date , alpha_model , benchmark , init_port).rescale()
        t1 = time.time()
        w , is_success , status = self.solve(self.solver_input)
        t2 = time.time()

        port = Port.create(self.secid , w , date = model_date , name = self.name , value = self.value)
        rslt = PortOptimResult(w , self.secid , port , is_success , status)

        if detail_infos:
            rslt.utility  = self.solver_input.utility(w , self.prob_type , self.opt_turn , self.opt_qobj , self.opt_short) 
            rslt.accuracy = self.solver_input.accuracy(w)
            if not rslt.accuracy and is_success:
                print('Not accurate but assessed as success!')
                print(rslt.accuracy)
            rslt.analytic = self.opt_input.analytic(port , self.opt_input.benchmark_port , self.opt_input.initial_port)

        t3 = time.time()
        rslt.time.update({'parse_input' : t1 - t0 , 'solve' : t2 - t1 , 'output' : t3 - t2})
        return rslt
    
    @property
    def secid(self): return self.opt_input.secid
    @property
    def name(self): return self.opt_input.portfolio_name
    @property
    def value(self): return self.opt_input.initial_value
    @property
    def benchport(self): return self.opt_input.benchmark_port

@dataclass
class PortOptimResult:
    w           : np.ndarray
    secid       : np.ndarray | Any = None
    port        : Port | Any = None
    is_success  : bool = False
    status      : Literal['optimal', 'max_iteration', 'stall'] | Any = ''
    utility     : Utility | Any = None
    accuracy    : Accuarcy | Any = None
    analytic    : Analytic | Any = None
    time        : dict[str,float] = field(default_factory=dict)

    def __post_init__(self):
        if self.utility is None: self.utility = Utility()
        if self.accuracy is None: self.accuracy = Accuarcy()
        if self.analytic is None: self.analytic = Analytic()

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
            self.analytic.styler('style').to_string() ,
            f'Other components include [\'w\' , \'secid\'])'
        ])
    
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
