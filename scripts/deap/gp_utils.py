import operator , time , joblib , os , gc , itertools , re , traceback , copy
import numpy as np
import pandas as pd
import torch
from argparse import Namespace
from deap import base , creator , tools , gp
from copy import deepcopy
import gp_math_func as MF
import gp_factor_func as FF

# ------------------------ gp terminals ------------------------
class Fac(torch.Tensor): pass
class Raw(torch.Tensor): pass
class int1(int): pass
class int2(int): pass
class int3(int): pass
class int4(int): pass
class float1(float): pass

class gpFitness(Namespace):
    def __init__(self, fitness_weights = {} , **kwargs) -> None:
        super().__init__(**kwargs)
        # assert len(weights) > 0, f'weights must have positive length'
        self.title = list(fitness_weights.keys())
        self.weights = tuple(fitness_weights.values())
        # deap.base.Fitness cannot deal 0 weights'
        self._idx  = [i for i,v in enumerate(self.weights) if v != 0]
        assert len(self._idx) > 0 , f'all fitness weights are 0!'
        self._keys = tuple(k for k,v in zip(self.title , self.weights) if v != 0)
        self._wgts = tuple(v for k,v in zip(self.title , self.weights) if v != 0)
    def fitness_value(self , metrics , as_abs = True , **kwargs):
        if as_abs: metrics = abs(metrics)
        return tuple(metrics[self._idx])
    def fitness_weight(self):
        return self._wgts
    def fitness_namespace(self , values):
        return Namespace(**{k:v for k,v in zip(self._keys,values)})

class gpHandler:
    '''
    ------------------------ gp primatives and terminals ------------------------
    '''
    @staticmethod
    def I(x): return x
    
    @classmethod
    def primatives_identity(cls):
        return [
            (cls.I,[Fac]    , Fac   , '_I_0_'),
            (cls.I,[Raw]    , Raw   , '_I_1_'),
            (cls.I,[int1]   , int1  , '_I_2_'),
            (cls.I,[int2]   , int2  , '_I_3_'),
            (cls.I,[int3]   , int3  , '_I_4_'),
            (cls.I,[float1] , float1, '_I_5_'),
        ]
    
    @staticmethod
    def primatives_1d():
        # (primative , in_types , out_type , name)
        return [
            (MF.log,        [(Fac,Raw)], Fac) ,
            (MF.sqrt,       [(Fac,Raw)], Fac) ,
            (MF.square ,    [Fac], Fac) ,
            (MF.rank_pct,   [Fac], Fac , 'rank') ,
            (MF.sigmoid,    [Fac], Fac) ,
            (MF.signedpower,[(Fac,Raw) , float1], Fac , 'power') ,
            (MF.sign,       [(Fac,Raw)], Fac) ,
        ]
    @staticmethod
    def primatives_2d():
        # (primative , in_types , out_type , name)
        return [
            (MF.add,        [Fac, Fac], Fac) ,
            (MF.sub,        [Fac, Fac], Fac) ,
            (MF.mul,        [Fac, Fac], Fac) ,
            (MF.div,        [Fac, Fac], Fac) ,
            (MF.rank_add,   [Fac, Fac], Fac) ,
            (MF.rank_sub,   [Fac, Fac], Fac) ,
            (MF.rank_div,   [Fac, Fac], Fac) ,
            (MF.rank_mul,   [Fac, Fac], Fac) ,
            (MF.ts_stddev,  [Fac, int2], Fac) ,
            (MF.ts_product, [Fac, int2], Fac) ,
            (MF.ts_delay,   [Fac, (int1,int2)], Fac) ,
            (MF.ts_delta,   [Fac, (int1,int2)], Fac) ,
            (MF.ts_lin_decay, [Fac, int2], Fac , 'ts_dw') ,
            (MF.ma,         [(Fac,Raw), (int2,int3)], Fac) ,
            (MF.pctchg,     [Raw, (int1,int2,int3)], Fac) ,
            (MF.ts_min,     [(Fac,Raw), int2], Fac) ,
            (MF.ts_max,     [(Fac,Raw), int2], Fac) ,
            (MF.ts_zscore,  [(Fac,Raw) , int3], Fac) ,
            (MF.ts_argmin,  [(Fac,Raw), int2], Fac) ,
            (MF.ts_argmax,  [(Fac,Raw), int2], Fac) ,
            (MF.ts_rank,    [(Fac,Raw), int2], Fac) ,
        ]
    @staticmethod
    def primatives_3d():
        # (primative , in_types , out_type , name)
        return [
            (MF.ts_rankcorr,        [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            (MF.ts_decay_pos_dif,   [(Fac,Raw), (Fac,Raw), int3], Fac , 'ts_dwdif') ,
            (MF.ts_cov,             [Fac, Fac, int2], Fac) ,
            (MF.ts_corr,            [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            (MF.ts_beta,            [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            (MF.ts_btm_avg,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
            (MF.ts_top_avg,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
            (MF.ts_rng_dif,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
        ]
    @staticmethod
    def primatives_4d():
        # (primative , in_types , out_type , name)
        return [
            (MF.ts_xbtm_yavg, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            (MF.ts_xtop_yavg, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            (MF.ts_xrng_ydif, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            #(MF.ts_xbtm_yavg, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xbtm_long') ,
            #(MF.ts_xtop_yavg, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xtop_long') ,
            #(MF.ts_xrng_ydif, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xdif_long') ,
        ]
    @classmethod
    def primatives_all(cls):
        # all primatives
        prims = [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['1d','2d','3d','4d']] for prima in plist]
        prims += cls.primatives_identity()
        return prims
    
    @classmethod
    def reach_base(cls , obj , base_types = [torch.Tensor,int,float,object]):
        return obj if obj in base_types else cls.reach_base(obj.__base__)

    @classmethod
    def sub_primatives(cls , prim):
        func_name = prim[3] if len(prim) > 3 else prim[0].__name__
        func , in_type_raw , out_type = prim[:3]

        in_types = []
        in_base   = []
        for i , _type in enumerate(in_type_raw):
            if isinstance(_type , (list,tuple)):
                _base = [cls.reach_base(_subtype) for _subtype in _type]
                assert all([_b == _base[0] for _b in _base]) , _base
                _base = _base[0]
            else:
                _base = cls.reach_base(_type)
                _type = [_type]
            in_types.append(_type) 
            in_base.append(_base)

        baseprim = [func , in_base , out_type , func_name]
        subprims = []
        for i , in_type in enumerate(list(itertools.product(*in_types))):
            func_name_sub = func_name + ('' if i == 0 else f'__{i-1}__')
            subprims.append([func , in_type , out_type , func_name_sub])
        return baseprim , subprims

    @staticmethod
    def ind2str(x):
        x = str(x)
        x = x.replace(' ','')
        return x
    
    @classmethod
    def syx2str(cls , x):
        x = str(x)
        x = x.replace(' ','')
        x = re.sub(r'_I_[0-9]+_','',x)
        x = re.sub(r'__[0-9]+__','',x)
        return x
    
    @staticmethod
    def str2ind(x , pset_ind):
        return getattr(creator , 'Individual').from_string(x , pset=pset_ind) 
        
    @staticmethod
    def str2syx(x , ind_str , pset_syx):
        syx = getattr(creator , 'Syntax').from_string(x , pset=pset_syx) 
        syx.ind_str = ind_str
        return syx
    
    @staticmethod
    def syx2ind(x , toolbox):
        return toolbox.str2ind(x.ind_str) 
    
    @classmethod
    def ind2syx(cls , ind , toolbox):
        ind_str = str(ind)
        ind = toolbox.ind_prune(ind)
        ind = toolbox.syx2str(ind)
        return toolbox.str2syx(ind , ind_str = ind_str) 
    
    @classmethod
    def ind_prune(cls , ind):
        assert isinstance(ind , getattr(creator , 'Individual')) , type(ind) 
        Ipos = [re.match(prim.name , r'^_I_[0-9]+_$') for prim in ind]
        new_prims = []
        for i , prim in enumerate(ind):
            if i > 0 and Ipos[i] and (ind[i] == ind[i-1]):
                pass
            else:
                new_prims.append(prim)
        return getattr(creator , 'Individual')(new_prims) 
    
    @classmethod
    def indpop_prune(cls , pop):
        return [cls.ind_prune(ind) for ind in pop]
    
    @staticmethod
    def indpop2syxpop(population , toolbox):
        # remove Identity primatives of population
        return [toolbox.ind2syx(ind) for ind in population]
        
    @staticmethod
    def syxpop2indpop(population , toolbox):
        # remove Identity primatives of population
        return [toolbox.syx2ind(ind) for ind in population]
    
    @classmethod
    def deduplicate(cls , population , forbidden = []):
        # return the unique population excuding specific ones (forbidden)
        ori = [cls.syx2str(ind) for ind in population]
        fbd = [cls.syx2str(ind) for ind in forbidden]
        index_maps = {value: index for index, value in enumerate(ori)} # remove duplicates
        index_keys = np.setdiff1d(list(index_maps.keys()) , fbd)       # remove forbidden members
        # return [population[i] for i in index_mapping.values()]
        return [population[index_maps[k]] for k in index_keys]
    
    @classmethod
    def Compiler(cls , pset):
        # return the compipler of individual sytax:
        # compiler can perform this:
        # factor_value = compiler(sytax) , where syntax is an instance of getattr(creator , 'Individual'), or simply a string such as 'add(cp,turn)'
        def compiler(individual):
            return gp.compile(individual , pset)
        return compiler
    
    @classmethod
    def Toolbox(cls , eval_func , eval_pop , param , gp_argnames , i_iter = -1 , n_args = (1,1) ,  
                fitness = None , **kwargs):
        '''
        ------------------------ create gp toolbox ------------------------
        input:
            gp_args:   initial gp factor names
            eval_func: evaluate function of individual, to register into toolbox
            i_iter:    i of outer loop, -1 as default mean no
            max_depth: [inner loop] max tree depth of gp
            n_args:    number of gp factors, (n_of_zscore_factors, n_of_raw_indicators)
            kwargs:    must include all args in eval_func
        output:
            toolbox:   toolbox that contains all gp utils
        '''
        assert sum(n_args) == len(gp_argnames)

        pset_individual = gp.PrimitiveSetTyped('main', [Fac] * n_args[0] + [Raw] * n_args[1], Fac)
        pset_syntax = gp.PrimitiveSetTyped('main', [torch.Tensor] * sum(n_args), torch.Tensor)
        for i , v in enumerate(gp_argnames): 
            pset_individual.renameArguments(**{f'ARG{i}':v})
            pset_syntax.renameArguments(**{f'ARG{i}':v})

        # ------------ set int and float terminals ------------
        str_iter = '_' if i_iter < 0 else f'_{i_iter}'
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'int{i}{str_iter}' for i in range(6)]]
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'float{i}{str_iter}' for i in range(6)]]
        for v in range(1,2): pset_individual.addTerminal(v, int1)
        for v in range(2,10): pset_individual.addTerminal(v, int2)
        for v in [10 , 15 , 20 , 30 , 40]:  pset_individual.addTerminal(v, int3)
        #for v in [60 , 120 , 180 , 200 , 240]: pset.addTerminal(v, int4)
        for v in range(0,101):  pset_individual.addTerminal(round(v*0.02,2), float1)
        #pset_individual.addEphemeralConstant(f'int1{str_iter}', lambda: np.random.randint(1,10+1), int1) # random int must > 0
        #pset_individual.addEphemeralConstant(f'int2{str_iter}', lambda: np.random.randint(2,10+1), int2) # random int must > 1
        #pset_individual.addEphemeralConstant(f'float1{str_iter}', lambda: round(np.random.random()*2+1,2), float1) # random int must > 1
        
        # add primatives (including their subprims)
        for prim in cls.primatives_all(): 
            baseprim , subprims = cls.sub_primatives(prim)
            pset_syntax.addPrimitive(*baseprim)
            for subprim in subprims: pset_individual.addPrimitive(*subprim)
        
        '''创建遗传算法基础模块，以下参数不建议更改，如需更改，可参考deap官方文档'''
        # https://zhuanlan.zhihu.com/p/72130823
        [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual' , 'Syntax']]
        fit_weights = fitness.fitness_weight() if fitness is not None else (+1.0,)
        creator.create('FitnessMin', base.Fitness, weights=fit_weights)   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
        creator.create('Individual', gp.PrimitiveTree, fitness=getattr(creator , 'FitnessMin'), 
                       pset=pset_individual , __hash__ = lambda self:hash(id(self))) 
        creator.create('Syntax'    , gp.PrimitiveTree, fitness=getattr(creator , 'FitnessMin'), ind_str = str , 
                       pset=pset_syntax , __hash__ = lambda self:hash(id(self)))
        
        toolbox = base.Toolbox()
        toolbox.register('generate_expr', gp.genHalfAndHalf, pset=pset_individual, min_=1, max_= param.max_depth)
        toolbox.register('individual', tools.initIterate, getattr(creator , 'Individual'), getattr(toolbox , 'generate_expr'))
        toolbox.register('population', tools.initRepeat, list, getattr(toolbox , 'individual')) 
        toolbox.register('ind2str', cls.ind2str)
        toolbox.register('syx2str', cls.syx2str)
        toolbox.register('str2ind', cls.str2ind , pset_ind=pset_individual) 
        toolbox.register('str2syx', cls.str2syx , pset_syx=pset_syntax) 
        toolbox.register('ind2syx', cls.ind2syx , toolbox = toolbox) 
        toolbox.register('syx2ind', cls.syx2ind , toolbox = toolbox) 
        toolbox.register('indpop2syxpop', cls.indpop2syxpop, toolbox = toolbox) 
        toolbox.register('deduplicate', cls.deduplicate)
        toolbox.register('ind_prune' , cls.ind_prune)
        toolbox.register('indpop_prune' , cls.indpop_prune)
        toolbox.register('compile', cls.Compiler(pset_syntax)) # use pset_syntax to compile
        toolbox.register('evaluate', eval_func , compiler = getattr(toolbox , 'compile') , fitness = fitness , i_iter = i_iter , param = param , **kwargs) 
        toolbox.register('evaluate_pop', eval_pop , toolbox = toolbox , param = param) 
        toolbox.register('select_best', tools.selBest) 
        toolbox.register('select_Tour', tools.selTournament, tournsize=3) # 锦标赛：随机选择3个，取最大, resulting around 49% of pop
        toolbox.register('select_2Tour', tools.selDoubleTournament, fitness_size=3 , parsimony_size=1.4 , fitness_first=True) # 锦标赛：第一轮随机选择3个，取最大
        toolbox.register('syxpop2indpop', cls.syxpop2indpop , toolbox = toolbox) 
        toolbox.register('mate', gp.cxOnePoint)
        toolbox.register('expr_mut', gp.genHalfAndHalf, pset=pset_individual , min_=0, max_= param.max_depth)  # genFull
        toolbox.register('mutate', gp.mutUniform, expr = getattr(toolbox , 'expr_mut') , pset=pset_individual) 
        toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3
        toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=min(10,param.max_depth)))  # max=3

        return toolbox
    
class gpContainer(Namespace):
    __reserved_names__ = ['get' , 'set' , 'update' , 'delete' , 'subset' , 'keys' , 'values' , 'items' , 'copy' , 'apply' , 'map']
    def __init__(self , inherit_from = None , **kwargs) -> None:
        if inherit_from is not None: 
            for k , v in inherit_from.items(): self.set(k , v)
        for k , v in kwargs.items(): self.set(k , v)
    def get(self , key , default = None , require = False):
        return getattr(self , key) if hasattr(self , key) else default
    def update(self , **kwargs):
        for k , v in kwargs.items(): self.set(k , v)
        return self
    def set(self , key  , v):
        assert key not in type(self).__reserved_names__ , key
        setattr(self , key , v)
        return self
    def delete(self , key):
        assert key not in type(self).__reserved_names__ , key
        if hasattr(self , key): delattr(self, key)
        return self
    def subset(self , keys , require = False):
        return {key:self.get(key , require = require) for key in keys}
    def keys(self): 
        return self.__dict__.keys()
    def values(self): 
        return self.__dict__.values()
    def items(self): 
        return self.__dict__.items()
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    def __add__(self , another):
        new = type(self)()
        for k , v in self.items(): new.set(k , v)
        for k , v in another.items(): new.set(k , v)
        return new
    def __getitem__(self , key):
        return self.__dict__.__getitem__(key)
    def __setitem__(self , key , value):
        return self.__dict__.__setitem__(key , value)
    def __len__(self):
        return self.__dict__.__len__()
    def copy(self):
        return deepcopy(self)
    def apply(self , key , func):
        self.set(key , func(self.__getitem__(key)))
        return self
    def map(self , keys , func):
        for key in keys: self.set(key , func(self.__getitem__(key)))
        return self

class gpFileManager:
    def __init__(self , job_dir = None , toolbox = None) -> None:
        self.job_dir = './pop/bendi' if job_dir is None else job_dir
        self.dir = Namespace(
            log = f'{self.job_dir}/logbook' ,
            sku = f'{self.job_dir}/skuname' ,
            pqt = f'{self.job_dir}/parquet' ,
            res = f'{self.job_dir}/labels_res' ,
            elt = f'{self.job_dir}/elites' ,
            neu = f'{self.job_dir}/neutra' ,
        )

        self.path = Namespace(
            elitelog = f'{self.job_dir}/elite_log.csv' ,
            hoflog   = f'{self.job_dir}/hof_log.csv' ,
            runtime  = f'{self.job_dir}/saved_times.csv' ,
            params   = f'{self.job_dir}/gp_params.pt' ,
            df_axis  = f'{self.job_dir}/df_axis.pt'
        )
        [os.makedirs(subfoler , exist_ok=True) for subfoler in self.dir.__dict__.values()]
        self.df_axis = {}

    def update_toolbox(self , toolbox):
        self.toolbox = toolbox
        return self

    def single_dumpable(self , ind):
        syx_str = self.toolbox.syx2str(ind)
        if isinstance(ind , getattr(creator , 'Individual')):
            ind_str = self.toolbox.ind2str(ind)
        else:
            ind_str = ind.ind_str
        fit = ind.fitness.values if hasattr(ind,'fitness') else ()
        # ind = syx_str, ind_str, fit
        ind = Namespace(syx_str = syx_str, ind_str = ind_str, fit = fit)
        return ind

    def dump_generation(self , population , halloffame , forbidden , i_iter = 0 , i_gen = 0 , **kwargs):
        if i_iter < 0: return self
        basename = self.record_basename(i_iter , i_gen)

        # input type: population as getattr(creator , 'Individual') , halloffame as creator.Syntax , forbidden as creator.Syntax (most likely)
        # save type: syntax list (syntax_str , ind_str , fitness)
        pop = [self.single_dumpable(ind) for ind in population]
        hof = [self.single_dumpable(ind) for ind in halloffame]
        fbd = [self.single_dumpable(ind) for ind in forbidden]

        self.logbook.record(i_gen = i_gen , 
                            population = pop, 
                            halloffame = hof, 
                            forbidden = fbd, 
                            **kwargs)

        joblib.dump(self.logbook[-1] , f'{self.dir.log}/{basename}.pkl')
        return self

    def load_generation(self , i_iter = 0 , i_gen = 0 , hof_num = 500 , **kwargs):
        self.logbook = tools.Logbook()
        if i_gen < 0: 
            pop = []
            hof = tools.HallOfFame(hof_num)
            if i_iter == 0:
                fbd  = []
            else:
                basename = self.record_basename(i_iter-1 , -1)
                log = joblib.load(f'{self.dir.log}/{basename}.pkl')
                fbd = [self.toolbox.str2syx(ind.syx_str , ind.ind_str) for ind in log['forbidden']] 
        else:
            basename = self.record_basename(i_iter , i_gen)
            self.logbook.record(**joblib.load(f'{self.dir.log}/{basename}.pkl'))
            pop = [self.toolbox.str2ind(ind.ind_str) for ind in self.logbook[-1]['population']] 
            hof = [self.toolbox.str2syx(ind.syx_str , ind.ind_str) for ind in self.logbook[-1]['halloffame']] 
            fbd = [self.toolbox.str2syx(ind.syx_str , ind.ind_str) for ind in self.logbook[-1]['forbidden']] 

            hof_ = self.toolbox.evaluate_pop(hof , i_iter = i_iter, i_gen = i_gen, desc = 'Load HallofFame')
            hof = tools.HallOfFame(hof_num)
            hof.update(hof_)
        
        return pop , hof , fbd

    def update_sku(self , individual , pool_skuname):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = time.time()
            output_path = f'{self.dir.sku}/z_{pool_skuname}.txt'
            with open(output_path, 'w', encoding='utf-8') as file1:
                print(gpHandler.syx2str(individual),'\n start_time',time.ctime(start_time_sku),file=file1)

    def record_basename(self , i_iter = 0 , i_gen = 0):
        iter_str = 'iteration' if i_iter < 0 else f'iter{i_iter}'
        gen_str  = 'overall' if i_gen < 0 else f'gen{i_gen}'
        return f'{iter_str}_{gen_str}'
    
    def load_state(self , key , i_iter , i_gen = 0 , i_elite = 0 , device = None):
        if key in ['res' , 'neu' , 'elt']:
            return torch.load(getattr(self.dir , key) + f'/iter{i_iter}.pt').to(device)
        elif key == 'parquet':
            return pd.read_parquet(f'{self.dir.pqt}/elite_{i_elite}.parquet', engine='fastparquet')
        else:
            path = getattr(self.path , key)
            if key == 'df_axis':
                self.df_axis = torch.load(path)
                return self.df_axis
            elif key in ['elitelog' , 'hoflog']:
                if os.path.exists(path):
                    df = pd.read_csv(path,index_col=0)
                    return df[df.i_iter < i_iter]
                return pd.DataFrame()
            elif path.endswith('.csv'):
                return pd.read_csv(path,index_col=0)
            elif path.endswith('.pt'):
                return torch.load(path)
            else:
                raise Exception(key)

    def save_state(self , data , key , i_iter , i_gen = 0 , i_elite = 0 , **kwargs):
        if key in ['res' , 'neu' , 'elt']:
            torch.save(data , getattr(self.dir , key) +f'/iter{i_iter}.pt')
        elif key == 'parquet':
            if isinstance(data , torch.Tensor): data = data.cpu().numpy()
            df = pd.DataFrame(data,index=self.df_axis['df_index'],columns=self.df_axis['df_columns'])
            df.to_parquet(f'{self.dir.pqt}/elite_{i_elite}.parquet',engine='fastparquet')
        else:
            path = getattr(self.path , key)
            if key == 'df_axis':
                torch.save(data , self.path.df_axis)
                self.df_axis = data
            elif isinstance(data , pd.DataFrame) and path.endswith('.csv'):
                data.reset_index(drop=True).to_csv(path)
            elif path.endswith('.pt'):
                torch.save(data , path)
            else:
                raise Exception(key , data)
            
    def load_states(self, keys , **kwargs):
        return [self.load_state(key , **kwargs) for key in keys]
    
    def save_states(self, datas , **kwargs):
        return [self.save_state(data , key , **kwargs) for key , data in datas.items()]
    
class gpTimer:
    '''
    ------------------------ gp timers ------------------------
    includes:
        PTimer     : record a process and its time cost
        AccTimer   : record a series of time costs, can average later
        EmptyTimer : do nothing
    '''
    def __init__(self , record = False) -> None:
        self.recording = record
        self.recorder = {}
        self.df_cols = {}

    class PTimer:
        def __init__(self , key , record = False , target_dict = {} , print = True , print_str = None , memory_check = False):
            self.key = key
            self.record = record
            self.target_dict = target_dict
            self.print = print
            self.print_str = key if print_str is None else print_str
            self.memory_check = memory_check and torch.cuda.is_available()
        def __enter__(self):
            if self.print: print('-' * 20 + f' {self.key} ' + '-' * 20)
            self.start_time = time.time()
            if self.memory_check and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.gmem_start = torch.cuda.mem_get_info()[0] / MemoryManager.unit
            return self
        def __exit__(self, type, value, trace):
            if type is not None:
                print(f'Error in PTimer {self.key}' , type , value)
                traceback.print_exc()
            else:
                time_cost = time.time() - self.start_time
                if self.memory_check:
                    torch.cuda.empty_cache()
                    mem_end  = torch.cuda.mem_get_info()[0] / MemoryManager.unit
                    mem_info = f', Free CudaMemory {self.gmem_start:.2f}G - > {mem_end:.2f}G' 
                else:
                    mem_info = ''
                if self.record: self.append_time(self.target_dict , self.key , time_cost)
                if self.print: print(f'{self.print_str} Done, Cost {time_cost:.2f} Secs' + mem_info)
                return self
        def add_string(self , new_str):
            self.print_str = self.print_str + new_str
        @staticmethod
        def append_time(target_dict , key , time_cost):
            if key not in target_dict.keys():
                target_dict[key] = [time_cost]
            else:
                target_dict[key].append(time_cost)

    class AccTimer:
        def __init__(self , key = ''):
            self.key   = key
            self.clear()
        def __enter__(self):
            self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if type is not None:
                print(f'Error in AccTimer {self.key}' , type , value)
                traceback.print_exc()
            else:
                self.time  += time.time() - self.start_time
                self.count += 1
        def __repr__(self) -> str:
            return f'time : {self.time} , count {self.count}'
        def avgtime(self , pop_out = False):
            avg = self.time if self.count == 0 else self.time / self.count
            if pop_out: self.clear()
            return avg
        def clear(self):
            self.time  = 0.
            self.count = 0

    class EmptyTimer:
        def __enter__(self):
            pass
        def __exit__(self, type, value, trace):
            if type is not None:
                print(f'Error in EmptyTimer ' , type , value)
                traceback.print_exc()
        
    def __call__(self , key , print = True , df_cols = True , print_str = None , memory_check = False):
        if df_cols: self.df_cols.update({key:True})
        return self.PTimer(key , self.recording , self.recorder , print = print , print_str = print_str , memory_check = memory_check)
    def __repr__(self):
        return self.recorder.__repr__()
    def __bool__(self): 
        return True
    def acc_timer(self , key):
        if key not in self.recorder.keys(): self.recorder[key] = self.AccTimer(key)
        assert isinstance(self.recorder[key] , self.AccTimer) , self.recorder[key]
        return self.recorder[key]
    def append_time(self , key , time_cost , df_cols = True):
        if df_cols: self.df_cols.update({key:True})
        self.PTimer.append_time(self.recorder , key , time_cost)
    def time_table(self , columns = None , showoff = False , dtype = float):
        if columns is None:
            df = pd.DataFrame(data = {k:self.recorder[k] for k,v in self.df_cols.items() if v and k in self.recorder.keys()} , dtype=dtype) 
        else:
            df = pd.DataFrame(data = {k:self.recorder[k] for k in columns} , dtype=dtype) 
        if showoff: 
            with pd.option_context('display.width' , 160 ,  'display.max_colwidth', 10 , 'display.precision', 4,):
                print(df)
        return df.round(6)

class MemoryManager():
    unit = 1024**3

    def __init__(self , device = 0) -> None:
        self.cuda_avail = torch.cuda.is_available()
        if self.cuda_avail:
            self.device = torch.device(device)
            self.unit = type(self).unit
            if self.cuda_avail: self.gmem_total = torch.cuda.mem_get_info(self.device)[1] / self.unit
            self.record = {}

    def check(self , key = None, showoff = False , critical_ratio = 0.5 , starter = '**'):
        if not self.cuda_avail: return 0.

        gmem_free = torch.cuda.mem_get_info(self.device)[0] / self.unit
        torch.cuda.empty_cache()
        if gmem_free > critical_ratio * self.gmem_total and not showoff: 
            # if showoff: print(f'**Cuda Memory: Free {gmem_free:.1f}G') 
            return gmem_free
        
        gmem_freed = torch.cuda.mem_get_info(self.device)[0] / self.unit - gmem_free
        gmem_free += gmem_freed
        gmem_allo  = torch.cuda.memory_allocated(self.device) / self.unit
        gmem_rsrv  = torch.cuda.memory_reserved(self.device) / self.unit
        
        if key is not None:
            if key not in self.record.keys(): self.record[key] = []
            self.record[key].append(gmem_freed)
        if showoff: 
            print(f'{starter}{time.ctime()}Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 
        
        return gmem_free
    
    def collect(self):
        torch.cuda.empty_cache() # collect graphic memory 
        # gc.collect() # collect memory, very slow
    
    def __bool__(self):
        return True
    
    @classmethod
    def object_memory(cls , object , cuda_only = True):
        if isinstance(object , torch.Tensor):
            return cls.tensor_memory(object , cuda_only = cuda_only)
        elif isinstance(object , (list,tuple)):
            return sum([cls.object_memory(obj) for obj in object])
        elif isinstance(object , dict):
            return sum([cls.object_memory(obj) for obj in object.values()])
        else:
            return 0.
    
    @classmethod
    def tensor_memory(cls , tensor , cuda_only = True):
        if cuda_only and not tensor.is_cuda: return 0.
        total_memory = tensor.element_size() * tensor.numel()
        return total_memory / cls.unit
    
    def print_memeory_record(self):
        if self.cuda_avail:
            print(f'  --> Avg Freed Cuda Memory: ')
            for key , value in self.record.items():
                print(f'  --> {key} : {len(value)} counts, on average freed {np.mean(value):.2f}G')
    
    @classmethod
    def clear_and_check(cls , silent = True):
        gmem_free = torch.cuda.mem_get_info()[0] / cls.unit
        torch.cuda.empty_cache()
        if not silent: 
            gmem_freed = torch.cuda.mem_get_info()[0] / cls.unit - gmem_free
            gmem_free += gmem_freed
            gmem_allo  = torch.cuda.memory_allocated() / cls.unit
            gmem_rsrv  = torch.cuda.memory_reserved() / cls.unit
            print(f'Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 

    @staticmethod
    def except_MemoryError(func , out = MF.null , print_str = ''):
        def wrapper(*args , **kwargs):
            try:
                value = func(*args , **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                print(f'OutOfMemoryError on {print_str}')
                torch.cuda.empty_cache()
                value = out
            except Exception as e:
                raise Exception(e)
            return value
        return wrapper

class gpEliteGroup:
    def __init__(self , start_i_elite = 0 , device = None , block_len = 50) -> None:
        self.start_i_elite = start_i_elite
        self.i_elite = start_i_elite
        self.device  = device
        self.block_len = block_len
        self.init_container()

    def init_container(self):
        self.container = [gpEliteBlock(self.block_len)]

    def assign_logs(self , hof_log , elite_log):
        self.hof_log = hof_log
        self.elite_log = elite_log
        return self

    def update_logs(self , new_log):
        if len(self.elite_log):
            self.elite_log = pd.concat([self.elite_log , new_log[new_log.elite]] , axis=0) 
        else:
            self.elite_log = new_log[new_log.elite]
        self.hof_log = pd.concat([self.hof_log , new_log] , axis=0) if len(self.hof_log) else new_log
        return self

    def max_corr_with_me(self , value , abs_corr_cap = 1.01 , dim = 1 , dim_valids = (None , None) , syntax = None):
        corr_values = torch.zeros((self.i_elite - self.start_i_elite + 1 ,)).to(value)
        exit_state  = False
        l = 0
        for block in self.container:
            corrs , exit_state = block.max_corr(value , abs_corr_cap , dim , dim_valids , syntax = syntax)
            corr_values[l:l+block.len()] = corrs[:block.len()]
            l += block.len()
            if exit_state: break
        return corr_values , exit_state

    def append(self , syntax , value , starter = None , **kwargs):
        if not self.container[-1].full:
            self.container[-1].append(syntax , value , **kwargs)
        else:
            if len(self.container) > 0: self.container[-1].cat2cpu()
            self.container.append(gpEliteBlock(self.block_len).append(syntax , value , **kwargs))
        if isinstance(starter,str): print(f'{starter}Elite{self.i_elite:_>3d} (' + '|'.join([f'{k}{v:+.2f}' for k,v in kwargs.items()]) + f'): {syntax}')
        self.i_elite += 1
        return self
    
    def cat_all(self):
        for block in self.container:
            block.cat2cpu()
        return self
    
    def compile_elite_tensor(self , device = None):
        self.cat_all()
        if device is None: device = self.device
        self.elite_tensor = torch.cat([block.data_at_device(device) for block in self.container] , dim = -1)
        # del self.container
        self.init_container()
        return self
    
class gpEliteBlock:
    def __init__(self , max_len = 50):
        self.max_len = max_len
        self.names = []
        self.infos = {}
        self.data  = []
        self.full = False

    def len(self):
        return len(self.names)
        
    def cat2cpu(self):
        if isinstance(self.data , list): 
            if len(self.data) == 0: 
                self.data = MF.null
            else:
                try:
                    self.data = MF.concat_factors(*self.data)
                    if isinstance(self.data , torch.Tensor): self.data = self.data.cpu()
                except MemoryError:
                    print('OutofMemory when concat gpEliteBlock, try use cpu to concat')
                    gc.collect()
                    self.data = MF.concat_factors(*self.data , device=torch.device('cpu')) # to cpu first
        return self

    def append(self , name , value , **kwargs):
        if not self.full and isinstance(self.data , list):
            self.names.append(name)
            self.infos.update({name:kwargs})
            self.data.append(value)
            self.full = self.len() >= self.max_len
        else:
            raise Exception('The EliteBlock is Full')   
        return self
    
    def max_corr(self , value , abs_corr_cap = 1.01 , dim = None , dim_valids = (None , None) , syntax = None):
        assert isinstance(self.data , (torch.Tensor , list))
        corr_values = torch.zeros((self.len()+1,)).to(value)
        exit_state  = False
        block = self.data.to(value) if isinstance(self.data , torch.Tensor) else self.data
        i = torch.arange(value.shape[0]) if dim_valids[0] is None else dim_valids[0]
        j = torch.arange(value.shape[1]) if dim_valids[1] is None else dim_valids[1]
        value = value[i][:,j]
        for k in range(self.len()):
            blk = block[i][:,j][...,k] if isinstance(block , torch.Tensor) else block[k][i][:,j]
            corr = MF.corrwith(value, blk , dim=dim).nanmean() 
            corr_values[k] = corr
            if exit_state := corr.abs() > abs_corr_cap: break 

        return corr_values , exit_state
    
    def data_at_device(self , device):
        assert isinstance(self.data , torch.Tensor) , type(self.data)
        return self.data.to(device)
