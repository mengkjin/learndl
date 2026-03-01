import numpy as np
import pandas as pd

from datetime import datetime
from deap import tools
from deap.algorithms import varAnd
from typing import Sequence , Any

from src.proj import Logger
from src.res.deap.param import gpParameters
from .syntax import SyntaxRecord , BaseIndividual
from .toolbox import BaseToolbox
from .logger import gpLogger
from .elite import EliteGroup
from .memory import MemoryManager
from .timer import gpTimer
from .input import gpInput
from .status import gpStatus
from .evaluator import gpEvaluator

class GeneticProgramming:
    """遗传规划空间,包括参数、输入、输出、文件管理、内存管理、计时器、评价器、数据列"""
    _instance = None
    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self , job_id : int | None = None , train : bool = True , start_iter = 0 , start_gen = 0 , test_code : bool = False , noWith = False , **kwargs):
        self.param     = gpParameters(job_id , train , start_iter > 0 or start_gen > 0 , test_code , **kwargs)
        self.status    = gpStatus(self.param.n_iter , self.param.n_gen , start_iter , start_gen)
        self.input     = gpInput(self.param , self.status)
        
        self.logger   = gpLogger(self.param.job_dir , self.status)
        self.memory   = MemoryManager(0)
        self.timer    = gpTimer(not noWith)

        self.evaluator = gpEvaluator(self.param , self.input , self.status , self.timer , self.memory)

    @property
    def device(self):
        return self.param.device
    @property
    def gp_argnames(self):
        return self.param.gp_argnames
    @property
    def n_args(self):
        return self.param.n_args
    @property
    def df_index(self) -> np.ndarray:
        return self.input.df_index
    @property
    def df_columns(self) -> np.ndarray:
        return self.input.df_columns
    @property
    def i_iter(self) -> int:
        return self.status.i_iter
    @property
    def i_gen(self) -> int:
        return self.status.i_gen
    @property
    def start_iter(self) -> int:
        return self.status.start_iter
    @property
    def start_gen(self) -> int:
        return self.status.iter_start_gen

    def load_data(self):
        with self.timer.timer('Data' , print_str= '**Load Data'):
            self.input.load_data()

        if self.param.show_progress: 
            Logger.stdout(f'{len(self.param.gp_fac_list)} factors, {len(self.param.gp_raw_list)} raw data loaded!' , indent = 1)

    def preparation(self , **kwargs):
        """
        创建遗传算法基础模块Toolbox,以下参数不建议更改,如需更改,可参考deap官方文档
        计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化
        """
        self.toolbox = BaseToolbox(self.param , self.status , self.evaluator)
        self.input.update_residual()
        self.memory.check(showoff = True)

    def population(
            self , pop_num : int , max_round = 100 , 
            last_gen : Sequence[BaseIndividual | str | SyntaxRecord] | None = None , 
            forbidden : list[Any] | None = None , 
            **kwargs
        ) -> list[BaseIndividual]:
        """
        初始化种群
        input:
            pop_num:        population number
            max_round:      max iterations to approching 99% of pop_num
            last_gen:       starting population
            forbidden:      all individuals with null return (and those in the hallof fame) 
        output:
            population:     initialized pruned and deduplicated population of syntax
        """
        with self.timer.silent_timer('Population'):
            last_gen = last_gen or []
            forbidden = forbidden or []
            if last_gen: 
                population = self.toolbox.prune_pop(last_gen)
                population = self.toolbox.deduplicate(population , forbidden = forbidden)
            else:
                population = []

            for _ in range(max_round):
                new_comer  = self.toolbox.create_population(n = int(pop_num * 1.2) - len(population))
                new_comer  = self.toolbox.prune_pop(new_comer)
                new_comer  = self.toolbox.deduplicate(new_comer , forbidden = population + forbidden) 
                population = population + new_comer[:pop_num - len(population)]
                if len(population) >= 0.99 * pop_num: 
                    break
        return population

    def evolution(self , * , forbidden_lambda = None , show_progress=__debug__,stats=None,**kwargs):
        """
        变异/进化[小循环],从初始种群起步计算适应度并变异,重复n_gen次
        input:
            forbidden_lambda:   any function to determine if the result of evaluation indicate forbidden syntax
                                for instance: forbidden_lambda = lambda x:all(i for i in x)
        output:
            population:         updated population of syntax
            halloffame:         container of individuals with best fitness (no more than hof_num)
            forbidden:          all individuals with null return or goes in the the hof once
        """
  
        population , halloffame , forbidden = self.load_evolution()
            
        for i , i_gen in enumerate(self.status.iter_generation()):
            if show_progress and i > 0: 
                Logger.stdout(f'Survive {len(population)} Offsprings, try Populating to {self.param.pop_num} ones' , indent = 1)
            population = self.population(self.param.pop_num , last_gen = population , forbidden = forbidden + [str(ind) for ind in halloffame])
            #Logger.stdout([str(ind) for ind in population[:20]])
            population = self.toolbox.purify_pop(population)
            #Logger.stdout([str(ind) for ind in population[:20]])
            if show_progress and i == 0: 
                Logger.stdout(f'**A Population({len(population)}) has been Initialized at Generation {i_gen}')

            # Evaluate the new population
            population = self.evaluator.evaluate_population(population , **kwargs)
            self.current_population = population
            # check survivors
            survivors  = [ind for ind in population if ind.if_valid]
            forbidden += [ind for ind in population if not ind.if_valid or (False if forbidden_lambda is None else forbidden_lambda(ind.fitness.values))]
            # Update HallofFame with survivors 
            halloffame.update(survivors)

            # Selection of population to pass to next generation, consider surv_rate
            select_offspring = getattr(self.toolbox , f'select_{self.param.select_offspring}')
            
            if self.param.select_offspring in ['Tour' , '2Tour']: 
                # '2Tour' will incline to choose shorter ones
                # around 49% will survive
                k = len(population)
            else: 
                k = min(int(self.param.surv_rate * self.param.pop_num) , len(population))
            offspring = select_offspring(population , k)
            offspring = self.toolbox.revert_pop(list(set(offspring)))

            # Variation offsprings
            with self.timer.silent_timer('VarAnd'):
                population = varAnd(offspring, self.toolbox, self.param.cxpb , self.param.mutpb) # varAnd means variation part (crossover and mutation)
            # Dump population , halloffame , forbidden in logbooks of this generation
            other_dumps = stats.compile(survivors) if stats else {}
            self.dump_evolution(population, halloffame, forbidden , **other_dumps)

        Logger.stdout(f'**A HallofFame({len(halloffame)}) has been ' + ('Loaded' if self.start_gen >= self.param.n_gen else 'Evolutionized'))
        self.status.forbidden = [str(fbd) for fbd in forbidden]
        self.evolution_result = population , halloffame , forbidden

    def dump_evolution(self , population : Sequence , halloffame : Sequence | Any , forbidden : Sequence ,**kwargs):
        population = [self.toolbox.to_record(ind) for ind in population]
        halloffame = [self.toolbox.to_record(ind) for ind in halloffame]
        forbidden  = [str(fbd) for fbd in forbidden]
        self.logger.dump_generation(population, halloffame, forbidden , **kwargs)

    def load_evolution(self , **kwargs):
        population , halloffame , forbidden = self.logger.load_generation(**kwargs)
        population = self.toolbox.to_indpop(population)
        halloffame = self.toolbox.to_indpop(halloffame)
        forbidden = self.toolbox.to_indpop([str(fbd) for fbd in forbidden] + self.status.forbidden)

        halloffame = tools.HallOfFame(self.param.hof_num)
        halloffame.update(halloffame)
        return population , halloffame , forbidden

    def selection(self , * , show_progress=__debug__ , **kwargs):
        """筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英"""
        
        population , halloffame , forbidden = self.evolution_result
        elite_log  = self.logger.load_state('elitelog' , i_iter = self.i_iter) # 记录精英列表
        hof_log    = self.logger.load_state('hoflog'   , i_iter = self.i_iter) # 记录名人堂状态列表
        hof_elites = EliteGroup(start_i_elite = len(elite_log) , device=self.device).assign_logs(hof_log = hof_log , elite_log = elite_log)
        hof_inds = [self.toolbox.to_ind(ind) for ind in halloffame]
        infos   = pd.DataFrame(
            [[self.i_iter,-1,ind.syntax,ind.if_valid,False,0.] for ind in hof_inds] , 
            columns = pd.Index(['i_iter','i_elite','syntax','valid','elite','max_corr']))
        metrics = pd.DataFrame([ind.metrics for ind in hof_inds] , columns = list(self.param.fitness_wgt.keys())) #.reset_index(drop=True)
        new_log = pd.concat([infos , metrics] , axis = 1)

        ir_floor = self.param.ir_floor * (self.param.ir_floor_decay ** self.i_iter)
        new_log['elite'] = (
            new_log.valid & # valid factor value
            (new_log.rankir_in_res.abs() > ir_floor) & # higher rankir_res than threshold
            (new_log.rankir_in_raw.abs() > ir_floor) & # higher rankir_res than threshold
            (new_log.rankir_out_res != 0.))
        Logger.stdout(f'**HallofFame({len(halloffame)}) Contains {new_log.elite.sum()} Promising Candidates with RankIR >= {ir_floor:.2f}')
        if new_log.elite.sum() <= 0.1 * len(halloffame):
            # Failure of finding promising offspring , check if code has bug
            Logger.stdout(f'Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ' , indent = 1)
            Logger.stdout(f'Valid Hof({new_log.valid.sum()}), insample max ir({new_log.rankir_in_res.abs().max():.4f})' , indent = 1)

        for i , hof in enumerate(halloffame):
            # 若超过了本次循环的精英上限数,则后面的都不算,等到下一个循环再来(避免内存溢出)
            if hof_elites.i_elite - hof_elites.start_i_elite >= self.param.elite_num: 
                new_log.loc[i,'elite'] = False 
            if not new_log.loc[i,'elite']: 
                continue

            # 根据迭代出的因子表达式,计算因子值, 错误则进入下一循环
            factor_value = self.evaluator.to_value(hof,neutral_type=0 if self.i_iter == 0 else self.param.factor_neut_type,**kwargs)
            self.memory.check('factor')
            
            new_log.loc[i,'elite'] = not factor_value.isnull()
            if not new_log.loc[i,'elite']: 
                continue

            # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
            corr_values , exit_state = hof_elites.max_corr_with_me(factor_value, self.param.corr_cap, dim_valids=(self.input.insample,None), syntax = new_log.syntax[i])
            new_log.loc[i,'max_corr'] = round(corr_values[corr_values.abs().argmax()].item() , 4)
            new_log.loc[i,'elite'] = not exit_state
            self.memory.check('corr')

            if not new_log.loc[i,'elite']: 
                continue

            # 通过检验,加入因子库
            new_log.loc[i,'i_elite'] = hof_elites.i_elite
            forbidden.append(halloffame[i])
            hof_elites.append(factor_value , IR = new_log.rankir_in_res[i] , Corr = new_log.max_corr[i] , starter=f'  --> Hof{i:_>3d}/')
            if False and self.param.test_code: 
                self.logger.save_state(factor_value.to_dataframe(index = self.df_index , columns = self.df_columns) , 'parquet' , self.i_iter , i_elite = hof_elites.i_elite)
            self.memory.check(showoff = show_progress and self.param.test_code, starter = '  --> ')

        population = [self.toolbox.to_record(ind) for ind in population]
        halloffame = [self.toolbox.to_record(ind) for ind in halloffame]
        forbidden  = [str(fbd) for fbd in forbidden]
        self.logger.dump_generation(population, halloffame, forbidden)
        self.status.forbidden = [str(fbd) for fbd in forbidden]

        # 记录本次运行的名人堂与精英状态
        hof_elites.update_logs(new_log)
        self.logger.save_state(hof_elites.elite_log.round(6) , 'elitelog' , self.i_iter)
        self.logger.save_state(hof_elites.hof_log.round(6)   , 'hoflog'   , self.i_iter)
        
        elites = hof_elites.compile_elite_tensor(device=self.device)
        if elites is not None:
            self.logger.save_state(elites , 'elt' , self.i_iter)
            Logger.stdout(f'**An EliteGroup({elites.shape[-1]}) has been Selected')
        if self.device.type == 'cuda': 
            Logger.stdout(f'Cuda Memories of "input"     take {MemoryManager.object_memory(self.input.inputs):.4f}G' , indent = 1)
            Logger.stdout(f'Cuda Memories of "elites"    take {MemoryManager.object_memory(elites):.4f}G' , indent = 1)
            Logger.stdout(f'Cuda Memories of "tensors"   take {MemoryManager.object_memory(self.input.tensors):.4f}G' , indent = 1)

    def generation(self , **kwargs):
        """一次[大循环]的主程序,初始化种群、变异、筛选、更新残差labels"""
        init_time = datetime.now()

        # 准备工作,初始化遗传规划Toolbox, 更新残差标签
        with self.timer.timer('Setting' , print_str = f'**Prepare GP Toolbox and Update Residual Labels'):
            self.preparation()

        # 进行进化与变异,生成种群、精英和先祖列表
        with self.timer.timer('Evolution' , print_str = f'**Generations of Evolution' , memory_check = True):
            self.evolution() 
        
        # 衡量精英,筛选出符合所有要求的精英中的精英
        with self.timer.timer('Selection' , print_str = f'**Selection of HallofFame' , memory_check = True):
            self.selection()
                
        self.timer.append_time('Generation' , (datetime.now() - init_time).total_seconds())

    def iteration(self , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        """
        time0 = datetime.now()

        self.load_data()
        for i_iter in self.status.iter_iteration():
            with Logger.Paragraph(f'Iteration {i_iter} start from Generation {self.start_gen}' , level=2):
                self.generation()

        hours, secs = divmod((datetime.now() - time0).total_seconds(), 3600)
        Logger.stdout('=' * 20 + f' Total Time Cost :{hours:.0f} hours {secs/60:.1f} minutes ' + '=' * 20)
        self.logger.save_state(self.timer.time_table() , 'runtime' , 0)
        self.memory.print_memeory_record()

    @classmethod
    def main(cls , job_id : int | None = None , start_iter = 0 , start_gen = 0 , test_code : bool = False , noWith = False , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        input:
            job_id:         when test_code is not True, determines job_dir = f'{gpDefaults.DIR_pop}/{job_id}'   
            start_iter:     start iteration, when to start, any of them has positive value means continue training
            start_gen:      start generation, when to start, any of them has positive value means continue training
            test_code:      if True, will not save any state and the input / evolution scale will be reduced to very small
            noWith:         to shutdown all timers (with xxx expression)
        output:
            gp:             GeneticProgramming object
        """
        gp = cls(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , noWith = noWith , **kwargs)
        gp.iteration()
        return gp
