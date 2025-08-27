import numpy as np

from copy import deepcopy
from typing import Any , Optional

from src.basic.conf import DEFAULT_OPT_CONFIG
from src.res.factor.util import AlphaModel , Amodel , Benchmark , Port , Portfolio , RISK_MODEL

from .solver_input import SolverInput
from .parser import (
    parse_config_benchmark , parse_config_board , parse_config_bound ,
    parse_config_component , parse_config_equity , parse_config_induspool ,
    parse_config_industry , parse_config_limitation , parse_config_pool ,
    parse_config_range , parse_config_short , parse_config_style ,
    parse_config_turnover , parse_config_utility
)
from .input_creator import (
    create_input_eq , create_input_benchmark , create_input_initial ,
    create_input_bnd_con , create_input_lin_con , create_input_turn_con ,
    create_input_cov_con ,  create_input_short_con
)

class PortfolioOptimizerInput:
    def __init__(
        self , 
        port_name : str = 'port' ,
        given_config : dict = {} ,
    ) -> None:
        
        self.portfolio_name  = port_name
        self.update_given_config(given_config)

        self.cfg_equity     = parse_config_equity(self.config)
        self.cfg_benchmark  = parse_config_benchmark(self.config)
        self.cfg_utility    = parse_config_utility(self.config)
        self.cfg_pool       = parse_config_pool(self.config)
        self.cfg_induspool  = parse_config_induspool(self.config)
        self.cfg_range      = parse_config_range(self.config)
        self.cfg_limitation = parse_config_limitation(self.config)
        self.cfg_bound      = parse_config_bound(self.config)
        self.cfg_board      = parse_config_board(self.config)
        self.cfg_industry   = parse_config_industry(self.config)
        self.cfg_style      = parse_config_style(self.config)
        self.cfg_component  = parse_config_component(self.config)
        self.cfg_turnover   = parse_config_turnover(self.config)
        self.cfg_short      = parse_config_short(self.config)

    def update_given_config(self , given_config : dict):
        self.config = deepcopy(DEFAULT_OPT_CONFIG)
        for key in self.config:
            if not (given := given_config.get(key)): continue
            assert isinstance(self.config[key] , dict) , self.config[key]
            if not isinstance(given , dict):
                assert len(self.config[key]) == 1 , f'If given is not dict, config must be of length 1'
                given = {next(iter(self.config[key])):given}
            self.config[key].update(given)

    def to_solver_input(self , model_date : int , alpha_model : AlphaModel | Amodel | Any = None , 
                        benchmark : Optional[Benchmark | Portfolio | Port] = None , init_port : Port | Any = None):

        self.model_date = model_date
        self.risk_model = RISK_MODEL.get_model(model_date)
        self.alpha_model = alpha_model.get_model(model_date).preprocess()
        
        self.initial_port = init_port
        self.secid = self.risk_model.universe

        self.benchmark = benchmark
        self.benchmark_port : Port | Any = None
        self.w0 : np.ndarray | Any = None
        self.wb : np.ndarray | Any = None

        self.eq     = create_input_eq(self)
        self.wb     = create_input_benchmark(self)
        self.w0     = create_input_initial(self)

        alpha       = self.alpha_model.align(self.secid).alpha
        bnd_con     = create_input_bnd_con(self)
        lin_con     = create_input_lin_con(self)
        turn_con    = create_input_turn_con(self)
        cov_con     = create_input_cov_con(self)
        short_con   = create_input_short_con(self)

        solver_input = SolverInput(alpha , lin_con , bnd_con , turn_con , cov_con , short_con , self.w0 , self.wb)
        return solver_input
    
    def analytic(self , port , bench = None , init = None):
        return self.risk_model.analyze(port , bench , init)
    
    @property
    def initial_position(self): 
        return None if self.initial_port is None else self.initial_port.position

    @property
    def initial_value(self): 
        return None if self.initial_port is None else self.initial_port.value