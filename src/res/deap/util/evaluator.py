import torch
import numpy as np
from typing import Callable , Literal , Any
from deap import gp
from torch.multiprocessing import Pool
from tqdm import tqdm

from src.proj import Logger
from src.math import tensor as T
from src.res.deap.func import factor_func as FF
from src.res.deap.param import gpParameters
from .syntax import BaseIndividual , SyntaxRecord , Population
from .input import gpInput
from .status import gpStatus
from .timer import gpTimer
from .logger import gpLogger
from .fitness import gpFitness

def except_MemoryError(func : Callable , out = None) -> Callable[..., Any]:
    def wrapper(*args , print_str = '' , **kwargs):
        try:
            value = func(*args , **kwargs)
        except torch.cuda.OutOfMemoryError:
            Logger.warning(f'OutOfMemoryError on {print_str or func.__name__}')
            torch.cuda.empty_cache()
            value = out
        return value
    return wrapper

def compiler(syntax : BaseIndividual | str | SyntaxRecord) -> Callable[..., torch.Tensor]:
    if isinstance(syntax , str):
        ind = BaseIndividual.get_class().from_syntax(syntax)
    elif isinstance(syntax , SyntaxRecord):
        ind = syntax.to_ind()
    else:
        ind = syntax
    return gp.compile(ind , ind.pset)

def raw_neutralize(y : torch.Tensor | None , x : torch.Tensor | None , * , insample : torch.Tensor | None = None , neutral_type : Literal[0,1,2] = 0) -> torch.Tensor | None:
    assert neutral_type in [0,1,2] , neutral_type
    if y is None or neutral_type == 0 or x is None:
        pass
    elif neutral_type == 1:
        assert isinstance(y , torch.Tensor) , f'{type(y)} is not a Tensor'
        shape2d = y.shape
        y = T.neutralize_1d(y.reshape(-1) , x.to(y).reshape(-1,x.shape[-1]) , insample = insample.reshape(-1,1).expand(y.shape)) if insample is not None else None
        if isinstance(y , torch.Tensor): 
            y = y.reshape(shape2d)
    elif neutral_type == 2:
        y = T.neutralize_2d(y , x.to(y))
    return y

processor = except_MemoryError(FF.process_factor)
neutralizer = except_MemoryError(raw_neutralize)

class gpEvaluator:
    def __init__(self , param : gpParameters , input : gpInput , status : gpStatus ,  
                timer : gpTimer , logger : gpLogger , **kwargs):
        self.param = param
        self.input = input
        self.status = status
        self.timer = timer
        self.logger = logger
        self.fitness = gpFitness(param.fitness_wgt)

    def to_value(self , individual : BaseIndividual | str | SyntaxRecord , * , neutral_type : Literal[0,1,2] = 0 , process_key='inf_winsor_norm',
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
        ind_name = str(individual)
        Logger.footnote(f'Evaluating {ind_name}' , vb_level = 'max' , indent = 1)

        with self.timer.timer('process' , 'Compile'):
            calculator = except_MemoryError(compiler(individual))
            
        with self.timer.timer('process' , 'Evaluate'):
            factor_value = calculator(*self.input.inputs , print_str=f'calculating {ind_name}')

        with self.timer.timer('process' , 'Process'):
            factor_value = processor(factor_value , process_key , dim = 1 , print_str=f'processing {ind_name}')

        with self.timer.timer('process' , 'Neutralize'):
            factor_value = neutralizer(factor_value , self.input.neutra , insample = self.input.insample , neutral_type = neutral_type , print_str=f'neutralizing {ind_name}')

        return FF.FactorValue(factor_value , ind_name , process_key)

    def assess(self , factor : FF.FactorValue , * , const_annual = 24 , min_coverage = 0.5 , **kwargs) -> tuple[np.ndarray, tuple | None]:
        """
        计算因子值的指标和适应度
        input:
            factor_value:   factor value
        output:
            metrics:        metrics
            fitness:        fitness
        """
        with self.timer.timer('process' , 'Assess'):
            if not factor.isnull(): 
                metrics = torch.zeros(8).to(factor.value)
                for i , labels in enumerate([self.input.labels_res , self.input.labels_raw]):
                    rankic_full = T.rankic_2d(factor.tensor_value , labels , universe = self.input.universe , min_coverage = min_coverage)
                    for j , sample in enumerate([self.input.insample , self.input.outsample]):
                        if rankic_full is None: 
                            continue
                        rankic = rankic_full[sample]
                        if rankic.isnan().sum() < 0.5 * len(rankic): # if too many nan rank_ic (due to low coverage)
                            rankic_avg  = rankic.nanmean()
                            rankic_std  = (rankic - rankic_avg).square().nanmean().sqrt() 
                            metrics[4*i + 2*j + 0] = rankic_avg.item() * const_annual
                            metrics[4*i + 2*j + 1] = (rankic_avg / (rankic_std + 1e-6) * np.sqrt(const_annual)).item()
                metrics  = metrics.cpu().numpy()
                fit_value = self.fitness.fitness_value(metrics , as_abs=True) 
            else:
                metrics  = np.zeros(8)
                fit_value = None
        return metrics, fit_value
        

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
        self.logger.update_sku(str(individual) , pool_skuname)
        factor = self.to_value(individual , **kwargs)
        metrics, fit_value = self.assess(factor , const_annual = const_annual , min_coverage = min_coverage , **kwargs)
    
        individual.metrics  = metrics
        individual.fit_value = fit_value
        return individual

    def evaluate_population(
        self , population : Population , * , i_iter = -1 , i_gen = -1 , 
        historybook : dict[str, SyntaxRecord] | None = None ,
        desc = 'Evolve Generation' , **kwargs) -> Population:
        """
        计算整个种群的适应度
        input:
            population:     un-updated population of syntax
        output:
            population:     updated population of syntax
        """
        i_iter = i_iter if i_iter >= 0 else self.status.i_iter
        i_gen = i_gen if i_gen >= 0 else self.status.i_gen
        if historybook is not None:
            for syx in population:
                record = historybook.get(str(syx), None)
                if record is None or not record.if_valid:
                    continue
                syx.metrics = record.metrics
                syx.fit_value = record.fit_value

        invalid_pop   = population.invalid_pop()
        pool_skunames = [f'iter{i_iter}_gen{i_gen}_{i}' for i in range(len(invalid_pop))] # 'pool_skuname' arg for evaluate
        # record max_fit0 and show
        iterator = tuple(zip(invalid_pop, pool_skunames))
        if self.param.worker_num > 1:
            with Pool(processes=self.param.worker_num) as pool:
                _ = list(tqdm(pool.starmap(self.evaluate_individual, iterator), total=len(iterator) , desc=f'Parallel evalute population ({self.param.worker_num})'))
        else:
            def desc0(x):
                return (f'  --> {desc} {str(i_gen)} ' + 'MaxIRres{:+.2f}|MaxIRraw{:+.2f}'.format(*x))
            def maxir(ir , syx : BaseIndividual):
                if syx.metrics is None:
                    return ir
                if abs(syx.metrics[1]) > abs(ir[0]): 
                    ir[0] = syx.metrics[1] 
                if abs(syx.metrics[5]) > abs(ir[1]): 
                    ir[1] = syx.metrics[5]
                return ir
            iterator = tqdm(iterator, total = len(invalid_pop))
            ir = [0.,0.]
            for syx , sku in iterator:
                self.evaluate_individual(syx, sku)
                iterator.set_description(desc0(ir := maxir(ir,syx)))

        return population
