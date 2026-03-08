import torch
import pandas as pd
import numpy as np
from typing import Literal

from src.proj import Logger , Proj

from src.res.gp.func import factor_func as FF
from src.res.gp.util import EliteGroup , GeneticProgramming

class gpGenerator:
    '''
    构成因子生成器,返回输入因子表达式则输出历史因子值的函数
    input:
        kwargs:  specific gp parameters, suggestion is to leave it alone
    output:
        GP:      gp_generator
    '''
    def __init__(self , job_id : int | None = None , 
                 process_key : str = 'inf_winsor_norm' ,  
                 weight_scheme : Literal['ic' , 'ir' , 'ew'] = 'ic', 
                 window_type  : Literal['insample' , 'rolling'] = 'rolling', 
                 weight_decay : Literal['constant' , 'linear' , 'exp'] = 'exp' , 
                 ir_window : int = 40 , 
                 roll_window : int = 40 , 
                 halflife : int = 20 , 
                 min_coverage :float = 0.1 , 
                 **kwargs) -> None:
        with Proj.Silence:
            self.gp  = GeneticProgramming(job_id = job_id , train = False , **kwargs).load_data().preparation()
            self.process_key = process_key
            self.elitelog = self.load_log('elitelog').set_index('i_elite')
            self.secid : np.ndarray = self.gp.logger.load_state('secid')
            self.date  : np.ndarray = self.gp.logger.load_state('date')

        self.elite_multi_factor = FF.MultiFactor(
            weight_scheme = weight_scheme ,
            window_type   = window_type ,
            weight_decay  = weight_decay ,
            ir_window     = ir_window ,
            roll_window   = roll_window ,
            halflife      = halflife ,
            universe      = self.gp.input.universe , 
            insample      = self.gp.input.insample ,
            min_coverage  = min_coverage ,
            **kwargs)

    def __call__(self, syntax : str | FF.FactorValue , process_key : str | None = None , print_info = True) -> FF.FactorValue:
        '''
        Calcuate FactorValue of a syntax
        '''
        if process_key is None: 
            process_key = self.process_key
        factor = self.gp.evaluator.to_value(syntax , process_key = process_key) if isinstance(syntax , str) else syntax
        return factor

    def to_dataframe(self , syntax : str | FF.FactorValue , process_key : str | None = None , print_info = True) -> pd.DataFrame | None:
        '''
        Convert FactorValue to DataFrame
        '''
        if process_key is None: 
            process_key = self.process_key
        factor = self.gp.evaluator.to_value(syntax , process_key = process_key) if isinstance(syntax , str) else syntax
        if not factor.isnull():
            value = factor.to_dataframe(secid = self.secid , date = self.date)
        else:
            value = None
        if print_info: 
            Logger.stdout(f'{self.__class__.__name__} -> process_key : {process_key} , syntax : {syntax}')
        return value

    @property
    def elites(self) -> list[str]:
        return self.load_log('elitelog').syntax.tolist()

    @property
    def hof_elites(self) -> list[str]:
        return self.load_log('hoflog').syntax.tolist()

    def load_log(self , key : Literal['elitelog' , 'hoflog']) -> pd.DataFrame:
        return self.gp.logger.load_state(key)

    def entire_elites(self , block_len = 50 , process_key = None):
        '''
        Load all elite factors
        '''
        elite_log  = self.load_log('elitelog') 
        hof_log    = self.load_log('hoflog')
        hof_elites = EliteGroup(start_i_elite=0 , device=self.gp.device , block_max_len=block_len).assign_logs(hof_log=hof_log, elite_log=elite_log)
        for elite in elite_log.syntax: 
            hof_elites.append(self(elite , process_key = process_key))
        hof_elites.cat_all()
        Logger.success(f'Load {hof_elites.total_len()} Elites')
        return hof_elites

    def load_elites(self):
        self.elite_group = self.entire_elites()
        return self

    def load_elite(self , i_elite : int , factor = False):
        '''
        Load a single elite factor
        '''
        elite_syntax = str(self.elitelog.loc[i_elite , 'syntax'])
        return self(elite_syntax) if factor else elite_syntax
    
    def multi_factor(self , *factors , labels = None , **kwargs):
        '''
        Calculate MultiFactorValue given factors and labels
        None kwargs will use default values
        '''
        factor_list = list(factors)
        if len(factor_list) == 1 and isinstance(factor_list[0] , torch.Tensor) and factor_list[0].dim() == 3:
            factor = factor_list[0]
        else:
            for i , fac in enumerate(factor_list):
                if isinstance(fac , str): 
                    factor_list[i] = self(fac)
            factor = torch.stack(factor_list , dim = -1)
        if labels is None: 
            labels = self.gp.input.labels_raw
        metrics = self.elite_multi_factor.calculate_icir(factor , labels , **kwargs) # ic,ir
        multi = self.elite_multi_factor.multi_factor(factor , **metrics , **kwargs)
        return multi # multi对象拥有multi,weight,inputs三个自变量
    
    def multi_elite_factor(
            self , elites : EliteGroup | None = None , 
            process_key : str | None = None ,  
            weight_scheme : Literal['ic' , 'ir' , 'ew'] | None = None, 
            window_type  : Literal['insample' , 'rolling'] | None = None, 
            weight_decay : Literal['constant' , 'linear' , 'exp'] | None = None , 
            ir_window : int | None = None , 
            roll_window : int | None = None , 
            halflife : int | None = None , 
            min_coverage :float | None = None , 
            **kwargs):
        '''
        Load all elite factors and calculate MultiFactorValue.
        None kwargs will use default values
        '''
        if elites is None:
            elites = self.entire_elites(process_key = process_key)
        if isinstance(elites , EliteGroup):
            factors = elites.compile_elite_tensor()
        assert isinstance(factors , torch.Tensor) , type(factors)
        multi = self.multi_factor(
            factors , labels = self.gp.input.labels_raw , 
            weight_scheme = weight_scheme , window_type = window_type ,weight_decay = weight_decay ,
            ir_window = ir_window , roll_window = roll_window , halflife = halflife , min_coverage = min_coverage , 
            names = elites.all_names() , secid = self.secid , date = self.date , **kwargs)
        return multi