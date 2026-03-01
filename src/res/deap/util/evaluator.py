import torch
import numpy as np
from typing import Callable , Literal
from deap import gp
from torch.multiprocessing import Pool
from tqdm import tqdm

from src.res.deap.func import math_func as MF , factor_func as FF
from src.res.deap.param import gpParameters
from .syntax import BaseIndividual , SyntaxRecord
from .input import gpInput
from .status import gpStatus
from .timer import gpTimer
from .logger import gpLogger
from .memory import MemoryManager
from .fitness import gpFitness


class gpEvaluator:
    def __init__(self , param : gpParameters , input : gpInput , status : gpStatus ,  
                timer : gpTimer | None = None , memory : MemoryManager | None = None , logger : gpLogger | None = None , **kwargs):
        self.param = param
        self.input = input
        self.status = status
        self.timer = timer if timer is not None else gpTimer()
        self.memory = memory if memory is not None else MemoryManager(0)
        self.logger = logger if logger is not None else gpLogger()
        self.fitness = gpFitness(param.fitness_wgt)

    @property
    def compiler(self) -> Callable[[BaseIndividual | str | SyntaxRecord], torch.Tensor]:
        if not hasattr(self , '_compiler'):
            def compiler(syntax : BaseIndividual | str | SyntaxRecord) -> torch.Tensor:
                if isinstance(syntax , str):
                    ind = BaseIndividual.get_class().from_syntax(syntax)
                elif isinstance(syntax , SyntaxRecord):
                    ind = syntax.to_ind()
                else:
                    ind = syntax
                return gp.compile(ind , ind.pset)
            self._compiler = compiler
        return self._compiler

    def to_value(self , individual : BaseIndividual | str | SyntaxRecord , * , neutral_type : Literal[0,1,2] = 0 , process_stream='inf_winsor_norm',
                **kwargs) -> FF.FactorValue:
        """
        根据迭代出的因子表达式,计算因子值
        计算因子时容易出现OutOfMemoryError,如果出现了异常处理一下,所以代码比较冗杂
        input:
            individual:   individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
            neutral_type: 0: no neutralize, 1: neutralize_1d, 2: neutralize_2d

        output:
            factor_value: 2d tensor
        """

        #Logger.stdout(individual)
        with self.timer.silent_timer('Compile'):
            func = self.compiler(individual)
            
        with self.timer.silent_timer('Eval'):
            func = self.memory.except_MemoryError(func, print_str=f'evaluating {str(individual)}')
            factor_value = func(*self.input.inputs)

        with self.timer.silent_timer('Process'):
            func = self.memory.except_MemoryError(FF.process_factor, print_str=f'processing {str(individual)}')
            factor_value = func(factor_value , process_stream , dim = 1)

        with self.timer.silent_timer('Neutralize'):
            assert neutral_type in [0,1,2] , neutral_type
            if factor_value is None or neutral_type == 0 or self.input.neutra is None:
                pass
            elif neutral_type == 1:
                assert isinstance(factor_value , torch.Tensor) , f'{type(factor_value)} is not a Tensor'
                func = self.memory.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
                shape2d = factor_value.shape
                factor_value = func(y = factor_value.reshape(-1) , 
                                    x = self.input.neutra.to(factor_value).reshape(-1,self.input.neutra.shape[-1]) , 
                                    insample = self.input.insample_2d.reshape(-1))
                if isinstance(factor_value , torch.Tensor): 
                    factor_value = factor_value.reshape(shape2d)
            elif neutral_type == 2:
                func = self.memory.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
                factor_value = func(factor_value , self.input.neutra.to(factor_value))

        return FF.FactorValue(name=str(individual) , process=process_stream , value=factor_value)

    def evaluate_individual(self , individual : BaseIndividual , pool_skuname : str , * , const_annual = 24 , min_coverage = 0.5 , **kwargs):
        """
        从因子表达式起步,生成因子并计算适应度
        input:
            individual:     individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
            pool_skuname:   pool skuname in pool.imap, e.g. iter0_gen0_0
            compiler:       compiler function to realize syntax computation, i.e. return factor function of given syntax
            const_annual:   constant of annualization
            min_coverage:   minimum daily coverage to determine if factor is valid
        output:
            tuple of (
                abs_rankir: (abs(insample_res), ) # !! Fitness definition 
                rankir:     (insample_res, outsample_res, insample_raw, outsample_raw)
            )
        """
        
        self.logger.update_sku(individual , pool_skuname)
        factor = self.to_value(individual , **kwargs)
        # self.memory.check('factor')
        
        metrics = torch.zeros(8)
        if isinstance(factor.value , torch.Tensor): 
            metrics = metrics.to(factor.value)
        if not factor.isnull(): 
            for i , labels in enumerate([self.input.labels_res , self.input.labels_raw]):
                rankic_full = MF.rankic_2d(factor.value , labels , dim = 1 , universe = self.input.universe , min_coverage = min_coverage)
                for j , sample in enumerate([self.input.insample , self.input.outsample]):
                    if rankic_full is None: 
                        continue
                    rankic = rankic_full[sample]
                    if rankic.isnan().sum() < 0.5 * len(rankic): # if too many nan rank_ic (due to low coverage)
                        rankic_avg  = rankic.nanmean()
                        rankic_std  = (rankic - rankic_avg).square().nanmean().sqrt() 
                        metrics[4*i + 2*j + 0] = rankic_avg.item() * const_annual
                        metrics[4*i + 2*j + 1] = (rankic_avg / (rankic_std + 1e-6) * np.sqrt(const_annual)).item()
        factor.infos['metrics'] = metrics.cpu().numpy()

        individual.if_valid = not factor.isnull()
        individual.metrics  = factor.infos['metrics']
        individual.fitness.values = self.fitness.fitness_value(factor.infos['metrics'] , as_abs=True)      
        # self.memory.check('rankic')
        return individual

    def evaluate_population(self , population : list[BaseIndividual] , * , i_iter = -1 , i_gen = -1 , desc = 'Evolve Generation' , **kwargs) -> list[BaseIndividual]:
        """
        计算整个种群的适应度
        input:
            population:     un-updated population of syntax
        output:
            population:     updated population of syntax
        """
        i_iter = i_iter if i_iter >= 0 else self.status.i_iter
        i_gen = i_gen if i_gen >= 0 else self.status.i_gen
        changed_pop   = [syx for syx in population if not syx.fitness.valid]
        pool_skunames = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(changed_pop))] # 'pool_skuname' arg for evaluate
        # record max_fit0 and show
        iterator = tuple(zip(changed_pop, pool_skunames))
        if self.param.worker_num > 1:
            with Pool(processes=self.param.worker_num) as pool:
                _ = list(tqdm(
                    pool.starmap(self.evaluate_individual, iterator), total=len(iterator) , 
                    desc=f'Parallel evalute population ({self.param.worker_num})'))
        else:
            def desc0(x):
                return (f'  --> {desc} {str(i_gen)} ' + 'MaxIRres{:+.2f}|MaxIRraw{:+.2f}'.format(*x))
            def maxir(ir , syx : BaseIndividual):
                if abs(syx.metrics[1]) > abs(ir[0]): 
                    ir[0] = syx.metrics[1] 
                if abs(syx.metrics[5]) > abs(ir[1]): 
                    ir[1] = syx.metrics[5]
                return ir
            iterator = tqdm(iterator, total = len(changed_pop))
            ir = [0.,0.]
            for syx , sku in iterator:
                syx = self.evaluate_individual(syx, sku)
                iterator.set_description(desc0(ir := maxir(ir,syx)))

            #iterator = tqdm(toolbox.map(toolbox.evaluate, changed_pop, pool_skunames), total=len(changed_pop))
            #[iterator.set_description(desc0(maxir := maxir_(maxir,syx))) for syx in iterator]

        assert all([syx.fitness.valid for syx in population])
        return population
