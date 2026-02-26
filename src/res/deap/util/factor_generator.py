import torch

from typing import Literal

from src.proj import Logger

from src.res.deap.func import factor_func as FF
from src.res.deap.util import EliteGroup , GeneticProgramming

class gpGenerator:
    '''
    构成因子生成器,返回输入因子表达式则输出历史因子值的函数
    input:
        kwargs:  specific gp parameters, suggestion is to leave it alone
    output:
        GP:      gp_generator
    '''
    def __init__(self , job_id , 
                 process_key : str = 'inf_winsor_norm' ,  
                 weight_scheme : Literal['ic' , 'ir' , 'ew'] = 'ic', 
                 window_type  : Literal['insample' , 'rolling'] = 'rolling', 
                 weight_decay : Literal['constant' , 'linear' , 'exp'] = 'exp' , 
                 ir_window : int = 40 , 
                 roll_window : int = 40 , 
                 halflife : int = 20 , 
                 min_coverage :float = 0.1 , 
                 **kwargs) -> None:
        self.gp_main  = GeneticProgramming(job_id = job_id , train = False , **kwargs)
        self.gp_main.update_toolbox()
        self.process_key = process_key
        self.elitelog   = self.gp_main.gp_logger.load_state('elitelog' , i_iter = -1).set_index('i_elite')
        self.df_axis    = self.gp_main.gp_logger.load_state('df_axis' , -1)

        self.Ensembler = FF.MultiFactor(
            universe      = self.gp_main.tensors['universe'] , 
            insample      = self.gp_main.tensors['insample'] ,
            weight_scheme = weight_scheme ,
            window_type   = window_type ,
            weight_decay  = weight_decay ,
            ir_window     = ir_window ,
            roll_window   = roll_window ,
            halflife      = halflife ,
            min_coverage  = min_coverage ,
            **kwargs)

    def __call__(self, syntax : str | FF.FactorValue , process_key : str | None = None , as_df = False , print_info = True) -> FF.FactorValue:
        '''
        Calcuate FactorValue of a syntax
        '''
        if isinstance(syntax , FF.FactorValue): 
            return syntax
        if process_key is None: 
            process_key = self.process_key
        factor = self.gp_main.to_value(syntax)
        if as_df and not factor.isnull():
            factor.value = factor.to_dataframe(index = self.df_axis['df_index'] , columns = self.df_axis['df_columns'])
        if print_info: 
            Logger.stdout(f'gpGenerator -> process_key : {process_key} , syntax : {syntax}')
        return factor

    def entire_elites(self , show_progress = True , block_len = 50 , process_key = None):
        '''
        Load all elite factors
        '''
        elite_log  = self.gp_main.gp_logger.load_state('elitelog' , i_iter = -1) 
        hof_log    = self.gp_main.gp_logger.load_state('hoflog'   , i_iter = -1)
        hof_elites = EliteGroup(start_i_elite=0 , device=self.gp_main.device , block_len=block_len).assign_logs(hof_log=hof_log, elite_log=elite_log)
        for elite in elite_log.syntax: 
            hof_elites.append(self(elite , process_key = process_key , print_info = show_progress))
        hof_elites.cat_all()
        Logger.stdout(f'Load {hof_elites.total_len()} Elites')
        return hof_elites

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
            labels = self.gp_main.tensors['labels_raw']
        metrics = self.Ensembler.calculate_icir(factor , labels , **kwargs) # ic,ir
        multi = self.Ensembler.multi_factor(factor , **metrics , **kwargs)
        return multi # multi对象拥有multi,weight,inputs三个自变量
    
    def multi_elite_factor(
            self , elites = None , show_progress = True , 
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
            elites = self.entire_elites(show_progress = show_progress , process_key = process_key)
        if isinstance(elites , EliteGroup):
            elites = elites.compile_elite_tensor()
        assert isinstance(elites , torch.Tensor) , type(elites)
        multi = self.multi_factor(
            elites , labels = self.gp_main.tensors['labels_raw'] , 
            weight_scheme = weight_scheme , window_type = window_type ,weight_decay = weight_decay ,
            ir_window = ir_window , roll_window = roll_window , halflife = halflife , min_coverage = min_coverage , **kwargs)
        return multi