from dataclasses import dataclass
from typing import Any , Literal , Optional

from src.basic import CONF
from src.res.factor.util import PortCreator , PortCreateResult , Port

from .interpreter import PortfolioOptimizerInput
from .solver import MosekSolver , CvxpySolver

PROB_TYPE = Literal['linprog' , 'quadprog' , 'socp']
ENGINE_TYPE = Literal['mosek' , 'cvxopt' , 'cvxpy']
CVXPY_SOLVER = Literal['mosek' , 'ecos' , 'osqp' , 'scs' , 'clarabel']

@dataclass(slots = True)
class PortfolioOptimizerConfig:
    prob_type : PROB_TYPE = 'quadprog'
    engine_type : ENGINE_TYPE = 'mosek'
    cvxpy_solver : CVXPY_SOLVER = 'mosek'
    optim_config : Optional[str] = None
    opt_relax : bool = True
    opt_turn  : bool = True
    opt_qobj  : bool = True
    opt_qcon  : bool = True
    opt_short : bool = True

    @classmethod
    def init_from(cls , print_info : bool = False , **kwargs):
        use_kwargs = {k: v for k, v in kwargs.items() if k in cls.__slots__ and v != cls.__dataclass_fields__[k].default}
        drop_kwargs = {k: v for k, v in kwargs.items() if k not in cls.__slots__}
        if print_info:
            if use_kwargs : print(f'In initializing {cls.__name__}, used kwargs: {use_kwargs}')
            if drop_kwargs: print(f'In initializing {cls.__name__}, dropped kwargs: {drop_kwargs}')
        return cls(**use_kwargs)

    @property
    def opt_config(self):
        return CONF.factor(f'{self.optim_config}_opt_config') if self.optim_config else {}

    @property
    def opt_cond(self):
        return {'turn' : self.opt_turn , 'qobj' : self.opt_qobj , 'qcon' : self.opt_qcon , 'short' : self.opt_short}

class PortfolioOptimizer(PortCreator):
    def __init__(self , name : str):
        super().__init__(name)

    def setup(self , print_info : bool = False , **kwargs): 
        self.conf = PortfolioOptimizerConfig.init_from(print_info = print_info , **kwargs)
        self.opt_input = PortfolioOptimizerInput(self.name , self.conf.opt_config)
        return self
    
    def parse_input(self):
        self.solver_input = self.opt_input.to_solver_input(self.model_date , self.alpha_model , self.bench_port , self.init_port).rescale()
        return self

    def solve(self):
        if self.conf.engine_type == 'mosek':
            self.solver = MosekSolver(self.solver_input, self.conf.prob_type)
        elif self.conf.engine_type == 'cvxpy':   
            self.solver = CvxpySolver(self.solver_input, self.conf.prob_type , cvxpy_solver = self.conf.cvxpy_solver)
        else:
            raise KeyError(self.conf.engine_type)

        while True:
            w, is_success, status = self.solver.solve(**self.conf.opt_cond)
            if is_success or not self.conf.opt_relax or self.solver_input.relaxable: break

        if not is_success and self.conf.opt_relax:
            print(f'Failed optimization at {self.model_date} , status is {status}, even with relax, use w0 instead.')
            assert self.solver_input.w0 is not None , 'In this failed-with-relax case, w0 must not be None'
            w = self.solver_input.w0

        self._optimized_tuple = w, is_success, status
        port = Port.create(self.opt_input.secid , w , date = self.model_date , name = self.name , value = self.value)
        self.create_result = PortCreateResult(port , is_success , status)
        return self

    def output(self):
        if self.detail_infos:
            w, is_success, status = self._optimized_tuple
            self.create_result.utility  = self.solver_input.utility(w , self.conf.prob_type , **self.conf.opt_cond) 
            self.create_result.accuracy = self.solver_input.accuracy(w)
            if not self.create_result.accuracy and is_success:
                print(f'Not accurate but assessed as success at {self.model_date} for [{self.opt_input.alpha_model.name}]!')
                print(self.create_result.accuracy)
            self.create_result.analyze(self.bench_port , self.init_port)
        return self
    
if __name__ == '__main__':
    from src.res.factor.fmp.optimizer.optim import PortfolioOptimizer
    config_path = 'custom_opt_config.yaml'

    optim = PortfolioOptimizer('test').setup(config_path = config_path , prob_type='socp')

    s = optim.create(20240606)
    print(s)
