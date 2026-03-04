import numpy as np
import pandas as pd

from deap import tools
from typing import Sequence , Any

from src.proj import Logger , Proj
from src.res.deap.param import gpParameters
from .syntax import Population
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

    def __init__(self , job_id : int | None = None , train : bool = True , start_iter = 0 , start_gen = 0 , test_code : bool = False , timer = True , **kwargs):

        self.param     = gpParameters(job_id , train , start_iter > 0 or start_gen > 0 , test_code , **kwargs)
        self.status    = gpStatus(self.param.n_iter , self.param.n_gen , start_iter , start_gen , self.param.train)
        self.logger    = gpLogger(self.param.job_dir , self.status)
        self.input     = gpInput(self.param , self.status , self.logger)
        self.memory    = MemoryManager(0)
        self.timer     = gpTimer(timer)

        self.evaluator = gpEvaluator(self.param , self.input , self.status , self.timer , self.logger)

    @property
    def device(self):
        return self.param.device
    @property
    def argnames(self):
        return self.param.argnames
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
        with self.timer.timer('stage' , 'Data' , title= 'Load Data' , timer_level = 5):
            self.input.load_data()
            Logger.stdout(f'{len(self.param.gp_fac_list)} factors, {len(self.param.gp_raw_list)} raw data loaded!' , indent = 1 , vb_level = 2)

    def preparation(self , **kwargs):
        """
        创建遗传算法基础模块Toolbox,以下参数不建议更改,如需更改,可参考deap官方文档
        计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化
        """
        self.toolbox = BaseToolbox(self.param , self.status , self.evaluator)
        self.input.update_residual()
        self.memory.check(showoff = True)

    def population(
            self , last_gen : Sequence | None = None , 
            forbidden : list[Any] | None = None , 
            * , max_round = 100 , **kwargs
        ) -> Population:
        """
        初始化种群
        input:
            pop_num:        population number
            last_gen:       starting population
            forbidden:      all individuals with null return (and those in the hallof fame) 
            max_round:      max iterations to approching 99% of pop_num
        output:
            population:     initialized pruned and deduplicated population of syntax
        """
        with self.timer.timer('process' , 'Population'):
            last_gen = Population.from_list(last_gen or [])
            forbidden = forbidden or []
            
            population = last_gen.prune().deduplicate(forbidden = forbidden)
            for _ in range(max_round):
                new_comer  = self.toolbox.create_population(n = int(self.param.pop_num * 1.2) - len(population))
                new_pop = Population.from_list(new_comer).prune().deduplicate(forbidden = population.pure_str_list() + forbidden)
                population.extend(new_pop.slice(0 , self.param.pop_num - len(population)))
                if len(population) >= 0.99 * self.param.pop_num: 
                    break
        return population

    def evolution(self , * , forbidden_lambda = None , stats=None,**kwargs):
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

        
        population , halloffame = self.load_evolution()

        for i , i_gen in enumerate(self.status.iter_generation()):
            if i_gen == 0:
                restore_info = 'Initial Generation'
            elif i > 0: 
                restore_info = f'Survive Offsprings of {len(population)} '    
            else:
                restore_info = f'Restore Population of {len(population)}'   
            Logger.note(f'Generation {i_gen} : {restore_info}, Populate to {self.param.pop_num}' , indent = 1 , vb_level = 2)
                
            population = self.population(population , self.status.forbidden + [str(ind) for ind in halloffame]).purify()
            
            # Evaluate the new population
            self.evaluator.evaluate_population(population , historybook = self.logger.historybook , **kwargs)
            self.update_historybook(population)

            # check survivors and forbidden
            survivors = population.valid_pop()
            self.status.update_forbidden(population.invalid_pop(forbidden_lambda).str_list())
            halloffame.update(survivors)

            offspring = survivors.selection(self.param.select_offspring , int(self.param.surv_rate * self.param.pop_num))
            population = offspring.revert().variation(self.toolbox, self.param.cxpb , self.param.mutpb)
            self.dump_evolution(population , halloffame, stats = stats , **kwargs)
            
        self.halloffame = halloffame

    def variation(self , offspring : Population) -> Population:
        """variation part (crossover and mutation)"""
        with self.timer.timer('process' , 'VarAnd'):
            offspring = offspring.variation(self.toolbox, self.param.cxpb , self.param.mutpb)
        return offspring

    def dump_evolution(self , population : Population , halloffame : tools.HallOfFame , * , stats = None , **kwargs):
        """dump population , halloffame , forbidden in logbooks of this generation"""
        other_dumps = stats.compile(population) if stats else {}
        pop = population.record_list()
        hof = Population.from_list(halloffame).record_list()
        self.logger.dump_generation(pop, hof, self.status.forbidden, **other_dumps)

    def load_evolution(self):
        """load population , halloffame , forbidden from logbooks of this generation"""
        pop , hof , fbd = self.logger.load_generation(i_gen = self.start_gen)
        population = Population.from_list(pop)
        halloffame = Population.from_list(hof).to_hof(self.param.hof_num)
        self.status.update_forbidden(fbd)

        return population , halloffame 

    def update_historybook(self , population : Population , **kwargs):
        self.logger.update_historybook(population.record_list())

    def selection(self , **kwargs):
        """筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英"""

        elite_log  = self.logger.load_state('elitelog' , i_iter = self.i_iter) # 记录精英列表
        hof_log    = self.logger.load_state('hoflog'   , i_iter = self.i_iter) # 记录名人堂状态列表
        hof_elites = EliteGroup(start_i_elite = len(elite_log) , device=self.device).assign_logs(hof_log = hof_log , elite_log = elite_log)
        hof_inds = Population.from_list(self.halloffame)
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
        Logger.stdout(f'HallofFame({len(self.halloffame)}) Contains {new_log.elite.sum()} Promising Candidates with RankIR >= {ir_floor:.2f}' , indent = 1)
        if new_log.elite.sum() <= 0.1 * len(self.halloffame):
            # Failure of finding promising offspring , check if code has bug
            Logger.alert1(f'Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ' , indent = 1)
            Logger.alert1(f'Valid Hof({new_log.valid.sum()}), insample max ir({new_log.rankir_in_res.abs().max():.4f})' , indent = 1)

        elites = []
        for i , hof in enumerate(self.halloffame):
            # 若超过了本次循环的精英上限数,则后面的都不算,等到下一个循环再来(避免内存溢出)
            hof_state = True
            hof_msg = ''
            if hof_elites.elite_count >= self.param.elite_num: 
                hof_state = False
                hof_msg = f'Hof{i:_>3d} Fail Test [{str(hof)}] EliteGroup is Full'
                
            if hof_state: 
                # 根据迭代出的因子表达式,计算因子值, 错误则进入下一循环
                factor_value = self.evaluator.to_value(hof,neutral_type=0 if self.i_iter == 0 else self.param.factor_neut_type,**kwargs)
                self.memory.check('factor')
                
                hof_state = not factor_value.isnull()
                if not hof_state:
                    hof_msg = f'Hof{i:_>3d} Fail Test [{str(hof)}] Factor Value is Null'

            if hof_state: 
                # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
                corr_values , exit_state = hof_elites.max_corr_with_me(
                    factor_value, self.param.corr_cap, dim_valids=(None,self.input.insample), syntax = new_log.syntax[i])
                new_log.loc[i,'max_corr'] = round(corr_values[corr_values.abs().argmax()].item() , 4)
                self.memory.check('corr')
                hof_state = not exit_state
                if not hof_state:
                    hof_msg = f'Hof{i:_>3d} Fail Test [{str(hof)}] Corr with Existing Factors Too High ({new_log.max_corr[i]:+.2f})'

            new_log.loc[i,'elite'] = hof_state
            if hof_state:
                # 通过检验,加入因子库
                
                new_log.loc[i,'i_elite'] = hof_elites.i_elite
                elites.append(hof)
                infos = {'IR': new_log.rankir_in_res[i],'Corr': new_log.max_corr[i]}
                hof_elites.append(factor_value , **infos)
                hof_msg = f'Hof{i:_>3d} Pass Test [{str(hof)}] (Elite{hof_elites.i_elite:_>3d},{"|".join([f"{k}{v:+.2f}" for k,v in infos.items()])})'
                if False and self.param.test_code: 
                    self.logger.save_state(factor_value.to_dataframe(index = self.df_index , columns = self.df_columns) , 'parquet' , self.i_iter , i_elite = hof_elites.i_elite)
                self.memory.check(showoff = self.param.test_code and Proj.vb.is_max_level)
            if hof_state:
                Logger.success(f'{str(hof)} : {hof_msg}' , indent = 2 , vb_level = 2)
            else:
                Logger.alert1(f'{str(hof)} : {hof_msg}' , indent = 2 , vb_level = 2)

        self.status.update_forbidden(elites)
        self.logger.dump_generation([], [], self.status.forbidden , overall = True)

        # 记录本次运行的名人堂与精英状态
        hof_elites.update_logs(new_log)
        self.logger.save_state(hof_elites.elite_log.round(6) , 'elitelog' , self.i_iter)
        self.logger.save_state(hof_elites.hof_log.round(6)   , 'hoflog'   , self.i_iter)
        
        elites = hof_elites.compile_elite_tensor(device=self.device)
        if elites is not None:
            Logger.stdout(f'An EliteGroup of {elites.shape[-1]} has been Selected from HallofFame' , indent = 1)
            self.logger.save_state(elites , 'elt' , self.i_iter)
        if self.device.type == 'cuda': 
            Logger.stdout(f'Cuda Memories of "input"     take {MemoryManager.object_memory(self.input.inputs):.4f}G' , indent = 2)
            Logger.stdout(f'Cuda Memories of "elites"    take {MemoryManager.object_memory(elites):.4f}G' , indent = 2)
            Logger.stdout(f'Cuda Memories of "tensors"   take {MemoryManager.object_memory(self.input.tensors):.4f}G' , indent = 2)

    def iteration(self , **kwargs):
        """一次[大循环]的主程序,初始化种群、变异、筛选、更新残差labels"""
        # 准备工作,初始化遗传规划Toolbox, 更新残差标签
        with self.timer.timer('stage' ,'Setting' , title = 'Setting' , timer_level = 5):
            self.preparation()
        # 进行进化与变异,生成种群、精英和先祖列表
        with self.timer.timer('stage' , 'Evolution' , title = f'Evolution' , memory_check = True , timer_level = 5):
            self.evolution() 
        # 衡量精英,筛选出符合所有要求的精英中的精英
        with self.timer.timer('stage' , 'Selection' , title = f'Selection' , memory_check = True , timer_level = 5):
            self.selection()
           
    def gp(self , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        """

        with Logger.Paragraph('Genetic Programming' , level=2):
            self.timer.decorate_primas()
            self.load_data()
            self.logger.load_historybook()

            self.status.update_forbidden([k for k , v in self.logger.historybook.items() if not v.if_valid])
            for i_iter in self.status.iter_iteration():
                with self.timer.timer('stage' , 'Iteration' , title = f'Iter {i_iter} Start From Gen {self.start_gen}' , timer_level = 3):
                    self.iteration()

            self.logger.dump_historybook()
            self.logger.save_state(self.timer.time_table() , 'runtime' , 0)
            self.memory.print_memeory_record()
            self.timer.revert_primas()

    @classmethod
    def main(cls , job_id : int | None = None , start_iter = 0 , start_gen = 0 , test_code : bool = False , 
             timer = True , vb : int | None = None , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        input:
            job_id:         when test_code is not True, determines job_dir = f'{gpDefaults.DIR_pop}/{job_id}'   
            start_iter:     start iteration, when to start, any of them has positive value means continue training
            start_gen:      start generation, when to start, any of them has positive value means continue training
            test_code:      if True, will not save any state and the input / evolution scale will be reduced to very small
            timer:          to enable gpTimer , default is True
            vb:             verbosity level, default is None (use 'max' if test_code is True)
        output:
            gp:             GeneticProgramming object
        """
        with Proj.vb.WithVB('max' if test_code else vb):
            gp = cls(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , timer = timer , **kwargs)
            gp.gp()
        return gp
