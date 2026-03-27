import numpy as np

from deap import tools
from typing import Any , Sequence , Callable

from src.proj import Logger , Proj
from src.res.gp.param import gpParameters
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

    def __init__(self , job_id : int | None = None , train : bool = True , start_iter = 0 , start_gen = 0 , 
                 test_code : bool = False , timer = True , vb_level : Any = 2 , **kwargs):
        self.vb_level  = Proj.vb(vb_level)
        self.param     = gpParameters(job_id , train , start_iter > 0 or start_gen > 0 , test_code , vb_level = self.vb_level , **kwargs)
        self.status    = gpStatus(self.param.n_iter , self.param.n_gen , start_iter , start_gen , self.param.train)
        self.memory    = MemoryManager(self.param.device , vb_level = self.vb_level)
        self.logger    = gpLogger(self.param.job_dir , self.status , vb_level = self.vb_level + 1)
        self.input     = gpInput(self.param , self.status , self.logger , vb_level = self.vb_level + 1)
        
        self.timer     = gpTimer(timer , vb_level = self.vb_level + 2)
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
    def secid(self) -> np.ndarray:
        return self.input.records['secid']
    @property
    def date(self) -> np.ndarray:
        return self.input.records['date']
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
    @property
    def historybook(self):
        return self.status.historybook
    @property
    def forbidden(self):
        return self.status.forbidden
    @property
    def ir_floor(self):
        return self.param.ir_floor * (self.param.ir_floor_decay ** self.i_iter)

    def load_data(self):
        with self.timer.timer('stage' , 'Data' , title= 'Load Data' , timer_level = 5):
            self.input.load_data()
            Logger.stdout(f'{len(self.param.gp_raw_list)} raw datas , {len(self.param.gp_fac_list)} factors loaded!' , indent = 1 , vb_level = self.vb_level)
        return self

    def preparation(self , **kwargs):
        """
        创建遗传算法基础模块Toolbox,以下参数不建议更改,如需更改,可参考deap官方文档
        计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化
        检查显存占用情况(cuda),如果显存占用过高,则释放显存
        """
        self.toolbox = BaseToolbox(self.param , self.status , self.evaluator)
        self.input.update_residual()
        self.memory.check(showoff = True)
        return self

    def population(
            self , last_gen : Sequence | None = None , 
            forbidden : Sequence[str] | None = None , 
            * , max_round = 100 , **kwargs
        ) -> Population:
        """
        基于上一轮的种群,生成新的种群
        input:
            pop_num:        population number
            last_gen:       starting population
            forbidden:      all individuals with null return (and those in the hallof fame) 
            max_round:      max iterations to approching 99% of pop_num
        output:
            population:     pruned and deduplicated population of syntax
        """
        with self.timer.timer('process' , 'Population'):
            last_gen = Population.from_list(last_gen or [])
            forbidden = [str(ind) for ind in forbidden] if forbidden is not None else []
            
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
        变异/进化[小循环],从初始种群起步计算适应度,更新halloffame和forbidden,变异获得新的种群,将新种群恢复为raw状态再重新扩张到新的初始种群,重复n_gen次
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
            else:
                restore_info = f'Survive Offsprings of {len(population)}' if i > 0 else f'Restore Population of {len(population)}'     
    
            # Populate new population to pop_num
            population = self.population(population , self.forbidden + [str(ind) for ind in halloffame]).purify()
            Logger.note(f'Generation {i_gen} : {restore_info}, Populate to {len(population)}' , indent = 1 , vb_level = self.vb_level + 2)
            
            # Evaluate the new population , add evaluation result to historybook , update forbidden list
            self.evaluator.evaluate_population(population , historybook = self.historybook , **kwargs)
            self.update_historybook(population)
            self.update_forbidden(population , forbidden_lambda = forbidden_lambda)

            # Halloffame update with valid population, the offspring
            offspring = population.valid_pop()
            halloffame.update(offspring)

            # Offspring selection and variation, results in new generation's starting population
            population = offspring.selection(self.param.select_offspring , int(self.param.surv_rate * self.param.pop_num)).\
                revert().variation(self.toolbox, self.param.cxpb , self.param.mutpb)
            self.dump_evolution(population , halloffame, stats = stats , **kwargs)
            
        self.halloffame = halloffame

    def variation(self , offspring : Population) -> Population:
        """variation part (crossover and mutation)"""
        with self.timer.timer('process' , 'Variation'):
            offspring = offspring.variation(self.toolbox, self.param.cxpb , self.param.mutpb)
        return offspring

    def dump_evolution(self , population : Population , halloffame : tools.HallOfFame , * , stats = None , **kwargs):
        """dump population , halloffame , forbidden in logbooks of this generation"""
        other_dumps = stats.compile(population) if stats else {}
        pop = population.record_list()
        hof = Population.from_list(halloffame).record_list()
        self.logger.dump_generation(pop, hof, self.forbidden, **other_dumps)

    def load_evolution(self):
        """load population , halloffame , forbidden from logbooks of this generation"""
        pop , hof , fbd = self.logger.load_generation(i_gen = self.start_gen)
        population = Population.from_list(pop)
        halloffame = Population.from_list(hof).to_hof(self.param.hof_num)
        self.update_forbidden(fbd)

        return population , halloffame 

    def load_historybook(self):
        self.status.update_historybook(self.logger.load_historybook())
        self.update_forbidden([k for k , v in self.historybook.items() if not v.if_valid])

    def update_historybook(self , population : Population , **kwargs):
        self.status.update_historybook({str(ind) : ind for ind in population.record_list()})

    def dump_historybook(self):
        self.logger.dump_historybook(self.status.historybook)

    def update_forbidden(self , population_or_forbidden : Population | Sequence[str] , forbidden_lambda : Callable | None = None , **kwargs):
        if isinstance(population_or_forbidden , Population):
            forbidden = population_or_forbidden.invalid_pop(forbidden_lambda).str_list()
        else:
            forbidden = [str(ind) for ind in population_or_forbidden]
        self.status.update_forbidden(forbidden)

    def selection(self , **kwargs):
        """筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英"""
        elite_group = EliteGroup.new_from_logs(
            elite_log = self.logger.load_state('elitelog' , i_iter = self.i_iter - 1) , 
            hof_log = self.logger.load_state('hoflog'   , i_iter = self.i_iter - 1) , device=self.device)
        halloffame = Population.from_list(self.halloffame)

        hoflog = halloffame.log_df(list(self.param.fitness_wgt.keys())).\
            assign(i_iter = self.i_iter , i_elite = -1 , elite = False , elite_state = 0 , max_corr = 0.).\
            loc[:,['i_iter','i_elite','syntax','valid','elite','elite_state','max_corr',*self.param.fitness_wgt.keys()]]
        
        if not hoflog.valid.all():
            Logger.alert1(f'HallofFame({len(halloffame)}) contains invalid factors, please check the code' , indent = 1)
            Logger.display(hoflog[~hoflog.valid])
        
        elite_candidate_num = (hoflog.valid & (hoflog.rankir_in_res.abs() >= self.ir_floor) & (hoflog.rankir_in_raw.abs() >= self.ir_floor)).sum()
        Logger.stdout(f'HallofFame({len(halloffame)}) Contains {elite_candidate_num} Promising Candidates with RankIR >= {self.ir_floor:.2f}' , indent = 1)
        if elite_candidate_num <= 0.1 * len(halloffame):
            # Failure of finding promising offspring , check if code has bug
            Logger.alert1(f'Failure of Finding Enough Promising Candidates, Check if Code has Bugs ... ' , indent = 1)
            Logger.alert1(f'Valid Hof({hoflog.valid.sum()}), insample max ir({hoflog.rankir_in_res.abs().max():.4f})' , indent = 1)

        # hof states :
        # 0 : pass all tests
        # 1 : fail test : Factor Value is Null (silent)
        # 2 : fail test : IR too low (silent)
        # 3 : fail test : Corr with Existing Factors Too High
        # 4 : fail test : EliteGroup is Full
        for i , hof in enumerate(halloffame):
            elite_state = 0
            if not hoflog.valid[i]:
                elite_state = 1
            elif abs(hoflog.rankir_in_res[i]) < self.ir_floor or abs(hoflog.rankir_in_raw[i]) < self.ir_floor:
                elite_state = 2
            
            if elite_state != 0:
                # 没有通过基础的Null值或IR检验,则不进行后续的检验
                continue

            if elite_state == 0: 
                # 若超过了本次循环的精英上限数,则后面的都不算,等到下一个循环再来(避免内存溢出)
                if elite_group.elite_count >= self.param.elite_num:
                    elite_state = 4
                msg = f'Hof{i:_>3d} Fail Test : EliteGroup is Full'
                
            if elite_state == 0:
                # 根据迭代出的因子表达式,计算因子值, 错误则进入下一循环
                # recalculate factor value here in case loaded hof cannot be evaluated correctly (should not happend)
                factor_value = self.evaluator.to_value(hof,neutral_type=0 if self.i_iter == 0 else self.param.factor_neut_type,**kwargs)
                self.memory.check('factor')
                if factor_value.isnull() or hoflog.rankir_out_res[i] == 0.:
                    elite_state = 1
                msg = f'Hof{i:_>3d} Fail Test : Factor Value is Null'

            if elite_state == 0: 
                # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
                corr_values , exit_state = elite_group.max_corr_with_me(
                    factor_value, self.param.corr_cap, dim_valids=(None,self.input.insample), syntax = hoflog.syntax[i])
                hoflog.loc[i,'max_corr'] = round(corr_values[corr_values.abs().argmax()].item() , 4)
                self.memory.check('corr')
                if exit_state:
                    elite_state = 3
                msg = f'Hof{i:_>3d} Fail Test : Corr with Existing Factors Too High ({hoflog.max_corr[i]:+.2f})'

            hoflog.loc[i,'elite_state'] = elite_state
            if elite_state == 0:
                # 通过检验,加入因子库
                hoflog.loc[i,'elite'] = True
                hoflog.loc[i,'i_elite'] = elite_group.i_elite
                infos = {'IRraw': hoflog.rankir_in_raw[i],'IRres': hoflog.rankir_in_res[i],'Corr': hoflog.max_corr[i]}
                elite_group.append(factor_value , **infos)
                msg = f'Hof{i:_>3d} Pass Test [{str(hof)}] (Elite{elite_group.i_elite:_>3d},{"|".join([f"{k}{v:+.2f}" for k,v in infos.items()])})'
                self.memory.check(showoff = self.param.test_code and Proj.vb.is_max_level)
                Logger.success(f'{str(hof)} : {msg}' , indent = 2 , vb_level = self.vb_level)
            else:
                Logger.alert1(f'{str(hof)} : {msg}' , indent = 2 , vb_level = self.vb_level)

        self.update_forbidden(elite_group.all_names())
        self.logger.dump_generation([], [], self.forbidden , overall = True)

        # 记录本次运行的名人堂与精英状态
        elite_group.update_logs(hoflog)
        elites = elite_group.compile_elite_tensor(device=self.device)
        self.logger.save_states({'elitelog' : elite_group.elitelog , 'hoflog' : elite_group.hoflog})
        self.logger.save_state('elt', elites , self.i_iter)
        if elites is not None:
            Logger.stdout(f'An EliteGroup of {elites.shape[-1]} has been Selected from HallofFame' , indent = 1)
        else:
            Logger.alert1(f'EliteGroup is Empty, this iteration is futile' , indent = 1)
        self.memory.show_memories({'input' : self.input.inputs , 'elites' : elites , 'tensors' : self.input.tensors})

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

        with Logger.Paragraph('Genetic Programming' , level=2) , self:
            self.load_data()
            self.load_historybook()

            for i_iter in self.status.iter_iteration():
                with self.timer.timer('stage' , 'Iteration' , title = f'Iter {i_iter} Gen {self.start_gen} ~ {self.status.n_gen}' , timer_level = 3):
                    self.iteration()

            self.dump_historybook()
            self.logger.save_state('runtime', self.timer.time_table())

    def __enter__(self):
        self.timer.decorate_primas()
        return self

    def __exit__(self , exc_type , exc_value , exc_traceback):
        self.memory.print_memeory_record()
        self.timer.revert_primas()

    @classmethod
    def main(cls , job_id : int | None = None , start_iter = 0 , start_gen = 0 , test_code : bool = False , 
             timer = True , vb : int | None = None , vb_level : Any = 2 , **kwargs):
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
            gp = cls(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , timer = timer , vb_level = vb_level , **kwargs)
            gp.gp()
        return gp
