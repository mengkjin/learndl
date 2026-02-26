import operator , copy
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from deap import creator , tools , gp
from deap.algorithms import varAnd
from torch.multiprocessing import Pool
from typing import Any , Sequence
from tqdm import tqdm

from src.proj import Logger
from src.proj.func import torch_load
from src.res.deap.env import gpDefaults
from src.res.deap.func import math_func as MF , factor_func as FF
from src.res.deap.func import gp_labels_raw , gp_filename_converter , read_gp_data , df2ts

from .syntax import SyntaxRecord , BaseIndividual
from .toolbox import BaseToolbox
from .params import gpParameters
from .logger import gpLogger
from .elite import EliteGroup
from .memory import MemoryManager
from .timer import gpTimer
from .primative import Primative
from .fitness import FitnessObjectMin , gpFitness

class GeneticProgramming:
    """遗传规划空间,包括参数、输入、输出、文件管理、内存管理、计时器、评价器、数据列"""
    def __init__(self , job_id : int | None = None , train : bool = True , start_iter = 0 , start_gen = 0 , test_code : bool = False , noWith = False , **kwargs):
        self.param     = gpParameters(job_id , train , start_iter > 0 or start_gen > 0 , test_code , **kwargs)
        self.gp_inputs = []
        self.tensors :   dict[str , torch.Tensor] = {}
        self.records :   dict[str , Any] = {}
        
        self.gp_logger   = gpLogger(self.param.job_dir)
        self.memory      = MemoryManager(0)
        self.timer       = gpTimer(not noWith)
        self.fitness     = gpFitness(self.param.fitness_wgt)

        self.i_iter      = start_iter
        self.start_iter  = start_iter
        self.start_gen   = start_gen

    @property
    def device(self):
        return self.param.device
    @property
    def gp_argnames(self):
        return self.param.gp_fac_list + self.param.gp_raw_list
    @property
    def n_args(self):
        return (len(self.param.gp_fac_list) , len(self.param.gp_raw_list))
    @property
    def df_index(self) -> np.ndarray:
        return self.records.get('df_index' , np.array([]))
    @property
    def df_columns(self) -> np.ndarray:
        return self.records.get('df_columns' , np.array([]))

    def load_data(self):
        with self.timer('Data' , df_cols = False , print_str= '**Load Data'):
            package_path = gpDefaults.dir_pack.joinpath(f'gp_data_package' + '_test' * self.param.test_code + '.pt')
            package_require = ['gp_argnames' , 'gp_inputs' , 'size' , 'indus' , 'labels_raw' , 'df_index' , 'df_columns' , 'universe']

            load_finished = False
            package_data = torch_load(package_path) if package_path.exists() else {}

            if not np.isin(package_require , list(package_data.keys())).all() or not np.isin(self.gp_argnames , package_data['gp_argnames']).all():
                if self.param.show_progress: 
                    Logger.stdout(f'Exists "{package_path}" but Lack Required Data!' , indent = 1)
            else:
                assert np.isin(package_require , list(package_data.keys())).all() , np.setdiff1d(package_require , list(package_data.keys()))
                assert np.isin(self.gp_argnames , package_data['gp_argnames']).all() , np.setdiff1d(self.gp_argnames , package_data['gp_argnames'])
                assert package_data['df_index'] is not None

                if self.param.show_progress: 
                    Logger.stdout(f'Directly load "{package_path}"' , indent = 1)
                for gp_key in self.gp_argnames:
                    gp_val = package_data['gp_inputs'][package_data['gp_argnames'].index(gp_key)]
                    gp_val = df2ts(gp_val , gp_key , self.device)
                    self.gp_inputs.append(gp_val)

                for gp_key in ['size' , 'indus' , 'labels_raw' , 'universe']: 
                    gp_val = package_data[gp_key]
                    gp_val = df2ts(gp_val , gp_key , self.device)
                    self.tensors[gp_key] = gp_val

                for gp_key in ['df_index' , 'df_columns']: 
                    gp_val = package_data[gp_key]
                    self.records[gp_key] = gp_val

                load_finished = True

            if not load_finished:
                if self.param.show_progress: 
                    Logger.stdout(f'Load from Parquet Files:' , indent = 1)
                gp_filename = gp_filename_converter()
                nrowchar = 0
                for i , gp_key in enumerate(self.gp_argnames):
                    if self.param.show_progress and nrowchar == 0: 
                        Logger.stdout('' , end='', indent = 1)
                    gp_val = read_gp_data(gp_filename(gp_key),self.param.slice_date,self.df_columns)
                    if i == 0: 
                        self.records.update({'df_columns' : gp_val.columns.values , 'df_index' : gp_val.index.values})
                    gp_val = df2ts(gp_val , gp_key , self.device)
                    self.gp_inputs.append(gp_val)
                    
                    if self.param.show_progress:
                        Logger.stdout(gp_key , end=',')
                        nrowchar += len(gp_key) + 1
                        if nrowchar >= 100 or i == len(self.gp_argnames):
                            Logger.stdout()
                            nrowchar = 0

                for gp_key in ['size' , 'indus']: 
                    gp_val = read_gp_data(gp_filename(gp_key),self.param.slice_date,self.df_columns)
                    gp_val = df2ts(gp_val , gp_key , self.device)
                    self.tensors.update({gp_key:gp_val})

                if 'CP' in self.gp_argnames:
                    CP = self.gp_inputs[self.gp_argnames.index('CP')]      
                else:
                    CP = df2ts(read_gp_data(gp_filename('CP'),self.param.slice_date,self.df_columns) , 'CP' , self.device)    
                self.tensors['universe']   = ~CP.isnan() 
                self.tensors['labels_raw'] = gp_labels_raw(CP , self.tensors['size'] , self.tensors['indus'])
                gpDefaults.dir_pack.mkdir(parents=True, exist_ok=True)
                saved_data = {
                    'gp_argnames' : self.gp_argnames ,
                    'gp_inputs' : self.gp_inputs ,
                    'size' : self.tensors['size'] ,
                    'indus' : self.tensors['indus'] ,
                    'labels_raw' : self.tensors['labels_raw'] ,
                    'universe' : self.tensors['universe'] ,
                    'df_index' : self.df_index ,
                    'df_columns' : self.df_columns ,
                }
                torch.save(saved_data , package_path)

        if self.param.show_progress: 
            Logger.stdout(f'{len(self.param.gp_fac_list)} factors, {len(self.param.gp_raw_list)} raw data loaded!' , indent = 1)

        self.gp_logger.save_state(self.param, 'params', i_iter = 0) # useful to assert same index as package data
        self.gp_logger.save_state({'df_index' : self.df_index , 'df_columns' : self.df_columns},'df_axis' , i_iter = 0) # useful to assert same index as package data
        
        self.tensors['insample']  = torch.Tensor((self.records['df_index'] >= self.param.slice_date[0]) * 
                                                (self.records['df_index'] <= self.param.slice_date[1])).bool()
        self.tensors['outsample'] = torch.Tensor((self.records['df_index'] >= self.param.slice_date[2]) * 
                                                (self.records['df_index'] <= self.param.slice_date[3])).bool()
        if self.param.factor_neut_type == 1:
            self.tensors['insample_2d'] = self.tensors['insample'].reshape(-1,1).expand(self.tensors['labels_raw'].shape)

    def update_residual(self , **kwargs):
        """计算本轮需要预测的labels_res,基于上一轮的labels_res和elites,以及是否是完全中性化还是svd因子中性化"""
        assert self.param.labels_neut_type in ['svd' , 'all'] , self.param.labels_neut_type #  'all'
        assert self.param.svd_mat_method in ['coef_ts' , 'total'] , self.param.svd_mat_method

        self.tensors.pop('neutra', None)
        if self.i_iter == 0:
            labels_res = copy.deepcopy(self.tensors['labels_raw'])
            elites     = None
        else:
            labels_res = self.gp_logger.load_state('res' , self.i_iter - 1 , device = self.device)
            elites     = self.gp_logger.load_state('elt' , self.i_iter - 1 , device = self.device)
        neutra = elites

        if isinstance(elites , torch.Tensor) and self.param.labels_neut_type == 'svd': 
            assert isinstance(labels_res , torch.Tensor) , type(labels_res)
            if self.param.svd_mat_method == 'total':
                elites_mat = FF.factor_coef_total(elites[self.tensors['insample']],dim=-1)
            else:
                elites_mat = FF.factor_coef_with_y(elites[self.tensors['insample']], labels_res[self.tensors['insample']].unsqueeze(-1), corr_dim=1, dim=-1)
            neutra = FF.top_svd_factors(elites_mat, elites, top_n = self.param.svd_top_n ,top_ratio=self.param.svd_top_ratio, dim=-1 , inplace = True) # use svd factors instead
            Logger.stdout(f'  -> Elites({elites.shape[-1]}) Shrink to SvdElites({neutra.shape[-1]})')

        if isinstance(neutra , torch.Tensor) and neutra.numel(): 
            self.tensors.update({'neutra' : neutra.cpu()})
            Logger.stdout(f'  -> Neutra has {neutra.shape[-1]} Elements')

        assert isinstance(labels_res , torch.Tensor) , type(labels_res)
        labels_res = MF.neutralize_2d(labels_res, neutra , inplace = True) 
        self.gp_logger.save_state(labels_res, 'res', self.i_iter) 

        if self.param.factor_neut_type > 0 and self.param.labels_neut_type == 'svd':
            lastneutra = None if self.i_iter == 0 else self.gp_logger.load_state('neu' , self.i_iter - 1 , device = self.device)
            if isinstance(lastneutra , torch.Tensor): 
                lastneutra = lastneutra.cpu()
            if isinstance(neutra , torch.Tensor): 
                lastneutra = torch.cat([lastneutra , neutra.cpu()] , dim=-1) if isinstance(lastneutra , torch.Tensor) else neutra.cpu()
            self.gp_logger.save_state(lastneutra , 'neu', self.i_iter) 
            del lastneutra
        
        self.tensors['labels_res'] = labels_res
        self.memory.check(showoff = True)

    def update_toolbox(self , **kwargs):
        """创建遗传算法基础模块Toolbox,以下参数不建议更改,如需更改,可参考deap官方文档"""
        # https://zhuanlan.zhihu.com/p/72130823
        
        pset_raw , pset_pur = Primative.GetPrimSets(self.n_args , self.gp_argnames , '_' if self.i_iter < 0 else f'_{self.i_iter}')
    
        [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual' , 'Syntax']]
        fit_weights = self.fitness.fitness_weight() if self.fitness is not None else (+1.0,)
        creator.create('FitnessMin', FitnessObjectMin, weights=fit_weights)   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
        creator.create('Individual', BaseIndividual, fitness=getattr(creator , 'FitnessMin'), 
                       pset_raw = pset_raw , pset_pur = pset_pur) 
        
        toolbox = BaseToolbox()
        toolbox.register('generate_expr', gp.genHalfAndHalf, pset=pset_raw, min_=1, max_= self.param.max_depth)
        toolbox.register('create_individual', tools.initIterate, getattr(creator , 'Individual'), getattr(toolbox , 'generate_expr'))
        # toolbox.register('fitness_value', fitness.fitness_value)
        toolbox.register('create_population', tools.initRepeat, list, getattr(toolbox , 'create_individual')) 
        toolbox.register('evaluate_individual', self.evaluate_individual , compiler = toolbox.compiler , fitness = self.fitness , i_iter = self.i_iter , param = self.param , **kwargs) 
        toolbox.register('evaluate_population', self.evaluate_population , toolbox = toolbox) 
        toolbox.register('select_nsga2', tools.selNSGA2) 
        toolbox.register('select_best', tools.selBest) 
        toolbox.register('select_Tour', tools.selTournament, tournsize=3) # 锦标赛：随机选择3个，取最大, resulting around 49% of pop
        toolbox.register('select_2Tour', tools.selDoubleTournament, fitness_size=3 , parsimony_size=1.4 , fitness_first=True) # 锦标赛：第一轮随机选择3个，取最大
        toolbox.register('mate', gp.cxOnePoint)
        toolbox.register('expr_mut', gp.genHalfAndHalf, pset=pset_raw , min_=0, max_= self.param.max_depth)  # genFull
        toolbox.register('mutate', gp.mutUniform, expr = getattr(toolbox , 'expr_mut') , pset=pset_raw) 
        
        toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,self.param.max_depth)))  # max=3
        toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,self.param.max_depth)))  # max=3

        self.toolbox = toolbox
        self.gp_logger.update_toolbox(toolbox)

    def to_value(self , individual : BaseIndividual | str | SyntaxRecord , * , process_stream='inf_winsor_norm',
                 **kwargs) -> FF.FactorValue:
        """
        根据迭代出的因子表达式,计算因子值
        计算因子时容易出现OutOfMemoryError,如果出现了异常处理一下,所以代码比较冗杂
        input:
            individual:   individual syntax, e.g. sigmoid(rank_sub(ts_y_xbtm(turn, DP , 15, 4), hp)) 
        output:
            factor_value: 2d tensor
        """

        #Logger.stdout(individual)
        with self.timer.acc_timer('compile'):
            func = self.toolbox.compiler(individual)
            
        with self.timer.acc_timer('eval'):
            func = self.memory.except_MemoryError(func, print_str=f'evaluating {str(individual)}')
            factor_value = func(*self.gp_inputs)

        with self.timer.acc_timer('process'):
            func = self.memory.except_MemoryError(FF.process_factor, print_str=f'processing {str(individual)}')
            factor_value = func(factor_value , process_stream , dim = 1)

        with self.timer.acc_timer('neutralize'):
            factor_neut_type = self.param.factor_neut_type * (self.i_iter > 0) * (self.tensors.get('neutra') is not None)
            assert factor_neut_type in [0,1,2] , factor_neut_type
            if factor_value is None or factor_neut_type == 0:
                pass
            elif factor_neut_type == 1:
                assert isinstance(factor_value , torch.Tensor) , f'{type(factor_value)} is not a Tensor'
                func = self.memory.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
                shape2d = factor_value.shape
                factor_value = func(y = factor_value.reshape(-1) , 
                                    x = self.tensors['neutra'].to(factor_value).reshape(-1,self.tensors['neutra'].shape[-1]) , 
                                    insample = self.tensors['insample_2d'].reshape(-1))
                if isinstance(factor_value , torch.Tensor): 
                    factor_value = factor_value.reshape(shape2d)
            elif factor_neut_type == 2:
                func = self.memory.except_MemoryError(MF.neutralize_1d, print_str=f'neutralizing {str(individual)}')
                factor_value = func(factor_value , self.tensors['neutra'].to(factor_value))

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
        
        self.gp_logger.update_sku(individual , pool_skuname)
        factor = self.to_value(individual , **kwargs)
        # self.memory.check('factor')
        
        metrics = torch.zeros(8)
        if isinstance(factor.value , torch.Tensor): 
            metrics = metrics.to(factor.value)
        if not factor.isnull(): 
            for i , labels in enumerate([self.tensors['labels_res'] , self.tensors['labels_raw']]):
                rankic_full = MF.rankic_2d(factor.value , labels , dim = 1 , universe = self.tensors['universe'] , min_coverage = min_coverage)
                for j , sample in enumerate([self.tensors['insample'] , self.tensors['outsample']]):
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

    def evaluate_population(self , population : list[BaseIndividual] , * , i_iter = -1 , i_gen = 0, desc = 'Evolve Generation' , **kwargs) -> list[BaseIndividual]:
        """
        计算整个种群的适应度
        input:
            population:     un-updated population of syntax
        output:
            population:     updated population of syntax
        """
        changed_pop   = [syx for syx in population if not syx.fitness.valid]
        i_iter = i_iter if i_iter >= 0 else self.i_iter
        pool_skunames = [f'iter{self.i_iter}_gen{i_gen}_{i}' for i in range(len(changed_pop))] # 'pool_skuname' arg for evaluate
        # record max_fit0 and show
        iterator = tuple(zip(changed_pop, pool_skunames))
        if self.param.worker_num > 1:
            with Pool(processes=self.param.worker_num) as pool:
                _ = list(tqdm(
                    pool.starmap(self.toolbox.evaluate_individual, iterator), total=len(iterator) , 
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
                syx = self.toolbox.evaluate_individual(syx, sku)
                iterator.set_description(desc0(ir := maxir(ir,syx)))

            #iterator = tqdm(toolbox.map(toolbox.evaluate, changed_pop, pool_skunames), total=len(changed_pop))
            #[iterator.set_description(desc0(maxir := maxir_(maxir,syx))) for syx in iterator]

        assert all([syx.fitness.valid for syx in population])
        return population

    def population(self , pop_num : int , max_round = 100 , last_gen : Sequence[BaseIndividual | str | SyntaxRecord] | None = None , forbidden : list[BaseIndividual | str | SyntaxRecord] | None = None , **kwargs):
        """
        初始化种群
        input:
            pop_num:        population number
            max_round:      max iterations to approching 99% of pop_num
            last_gen:       starting population
            forbidden:      all individuals with null return (and those in the hallof fame) 
        output:
            population:     initialized population of syntax
        """
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
        start_gen = self.start_gen if self.i_iter == self.start_iter else 0
        population , halloffame , forbidden = self.gp_logger.load_generation(self.i_iter , start_gen-1 , hof_num = self.param.hof_num , stats = stats)
        forbidden : list = forbidden + self.records.get('forbidden' , [])
        population = self.toolbox.to_indpop(population)
        forbidden = self.toolbox.to_indpop(forbidden)
        for i_gen in range(start_gen, self.param.n_gen):
            if show_progress and i_gen > 0: 
                Logger.stdout(f'Survive {len(population)} Offsprings, try Populating to {self.param.pop_num} ones' , indent = 1)
            population = self.population(self.param.pop_num , last_gen = population , forbidden = forbidden + [ind for ind in halloffame])
            #Logger.stdout([str(ind) for ind in population[:20]])
            population = self.toolbox.purify_pop(population)
            #Logger.stdout([str(ind) for ind in population[:20]])
            if show_progress and i_gen == 0: 
                Logger.stdout(f'**A Population({len(population)}) has been Initialized')

            # Evaluate the new population
            population = self.toolbox.evaluate_population(population , i_gen = i_gen , **kwargs)

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
            offspring = self.toolbox.revive_pop(list(set(offspring)))

            # Variation offsprings
            with self.timer.acc_timer('varAnd'):
                population = varAnd(offspring, self.toolbox, self.param.cxpb , self.param.mutpb) # varAnd means variation part (crossover and mutation)
            # Dump population , halloffame , forbidden in logbooks of this generation
            self.gp_logger.dump_generation(population, halloffame, forbidden , self.i_iter , i_gen , **(stats.compile(survivors) if stats else {}))

        Logger.stdout(f'**A HallofFame({len(halloffame)}) has been ' + ('Loaded' if start_gen >= self.param.n_gen else 'Evolutionized'))
        self.evolve_result = population , halloffame , forbidden

    def selection(self , * , show_progress=__debug__ , **kwargs):
        """筛选精英群体中的因子表达式,以高ir、低相关为标准筛选精英中的精英"""
        
        population , halloffame , forbidden = self.evolve_result
        elite_log  = self.gp_logger.load_state('elitelog' , i_iter = self.i_iter) # 记录精英列表
        hof_log    = self.gp_logger.load_state('hoflog'   , i_iter = self.i_iter) # 记录名人堂状态列表
        hof_elites = EliteGroup(start_i_elite = len(elite_log) , device=self.device).assign_logs(hof_log = hof_log , elite_log = elite_log)
        hof_inds = [self.toolbox.to_ind(ind) for ind in halloffame]
        infos   = pd.DataFrame(
            [[self.i_iter,-1,ind.syntax,ind.if_valid,False,0.] for ind in hof_inds] , 
            columns = pd.Index(['i_iter','i_elite','syntax','valid','elite','max_corr']))
        metrics = pd.DataFrame([ind.metrics for ind in hof_inds] , columns = self.param.fitness_wgt.keys()) #.reset_index(drop=True)
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
            factor_value = self.to_value(hof,**kwargs)
            self.memory.check('factor')
            
            new_log.loc[i,'elite'] = not factor_value.isnull()
            if not new_log.loc[i,'elite']: 
                continue

            # 与已有的因子库"样本内"做相关性检验,如果相关性大于预设值corr_cap则进入下一循环
            corr_values , exit_state = hof_elites.max_corr_with_me(factor_value, self.param.corr_cap, dim_valids=(self.tensors['insample'],None), syntax = new_log.syntax[i])
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
                self.gp_logger.save_state(factor_value.to_dataframe(index = self.df_index , columns = self.df_columns) , 'parquet' , self.i_iter , i_elite = hof_elites.i_elite)
            self.memory.check(showoff = show_progress and self.param.test_code, starter = '  --> ')

        self.gp_logger.dump_generation(population, halloffame, forbidden , i_iter = self.i_iter , i_gen = -1)
        self.records['forbidden'] = forbidden

        # 记录本次运行的名人堂与精英状态
        hof_elites.update_logs(new_log)
        self.gp_logger.save_state(hof_elites.elite_log.round(6) , 'elitelog' , self.i_iter)
        self.gp_logger.save_state(hof_elites.hof_log.round(6)   , 'hoflog'   , self.i_iter)
        
        elites = hof_elites.compile_elite_tensor(device=self.device)
        if elites is not None:
            self.gp_logger.save_state(elites , 'elt' , self.i_iter)
            Logger.stdout(f'**An EliteGroup({elites.shape[-1]}) has been Selected')
        if True: 
            Logger.stdout(f'Cuda Memories of "gp_inputs" take {MemoryManager.object_memory(self.gp_inputs):.4f}G' , indent = 1)
            Logger.stdout(f'Cuda Memories of "elites"    take {MemoryManager.object_memory(elites):.4f}G' , indent = 1)
            Logger.stdout(f'Cuda Memories of "tensors"   take {MemoryManager.object_memory(self.tensors):.4f}G' , indent = 1)

    def generation(self , **kwargs):
        """一次[大循环]的主程序,初始化种群、变异、筛选、更新残差labels"""
        init_time = datetime.now()

        # 更新残差标签
        with self.timer('Residual' , print_str = f'**Update Residual Labels' , memory_check = True):
            self.update_residual()
            
        # 初始化遗传规划Toolbox
        with self.timer('Setting' , print_str = f'**Initialize GP Toolbox'):
            self.update_toolbox()

        # 进行进化与变异,生成种群、精英和先祖列表
        with self.timer('Evolution' , print_str = f'**Generations of Evolution' , memory_check = True):
            self.evolution() 
        
        # 衡量精英,筛选出符合所有要求的精英中的精英
        with self.timer('Selection' , print_str = f'**Selection of HallofFame' , memory_check = True):
            self.selection()
                
        self.timer.append_time('AvgVarAnd' , self.timer.acc_timer('varAnd').avgtime(pop_out = True))
        self.timer.append_time('AvgCompile', self.timer.acc_timer('compile').avgtime(pop_out = True))
        self.timer.append_time('AvgEval',    self.timer.acc_timer('eval').avgtime(pop_out = True))
        self.timer.append_time('All' , (datetime.now() - init_time).total_seconds())

        self.i_iter = self.i_iter + 1

    def iteration(self , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        """
        time0 = datetime.now()

        self.load_data()
        
        for i_iter in range(self.start_iter , self.param.n_iter):
            start_gen = self.start_gen if i_iter == self.start_iter else 0
            Logger.stdout('=' * 20 + f' Iteration {i_iter} start from Generation {start_gen} ' + '=' * 20)
            self.generation(start_gen = start_gen)

        hours, secs = divmod((datetime.now() - time0).total_seconds(), 3600)
        Logger.stdout('=' * 20 + f' Total Time Cost :{hours:.0f} hours {secs/60:.1f} ' + '=' * 20)
        self.gp_logger.save_state(self.timer.time_table(showoff=True) , 'runtime' , 0)
        self.memory.print_memeory_record()

    @classmethod
    def main(cls , job_id : int | None = None , start_iter = 0 , start_gen = 0 , test_code : bool = False , noWith = False , **kwargs):
        """
        训练的主程序,[大循环]的过程出发点,从start_iter的start_gen开始训练
        input:
            job_id:    when test_code is not True, determines job_dir = f'{gpDefaults.DIR_pop}/{job_id}'   
            start_iter , start_gen: when to start, any of them has positive value means continue training
            noWith:    to shutdown all timers (with xxx expression)
        output:
            pfr:       profiler to record time cost of each function (only available in test_code model)
        """
        gp = cls(job_id , start_iter = start_iter , start_gen = start_gen , test_code = test_code , noWith = noWith , **kwargs)
        gp.iteration()
        return gp
