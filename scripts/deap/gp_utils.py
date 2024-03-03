import time
import numpy as np
import pandas as pd
from deap import base , creator , tools , gp
from copy import deepcopy
import operator
import gp_math_func as F
import torch

# ------------------------ gp terminals ------------------------
class Fac(): pass
class Raw(): pass
class int1(int): pass
class int2(int): pass
class int3(int): pass
class int4(int): pass
class int5(int): pass
class float1(float): pass

class gpContainer:
    __reserved_names__ = ['get' , 'set' , 'update' , 'delete' , 'subset' , 'keys' , 'values' , 'items' , 'copy' , 'apply' , 'map']
    def __init__(self , inherit_from = None , **kwargs) -> None:
        if inherit_from is not None: 
            for k , v in inherit_from.items(): self.set(k , v)
        for k , v in kwargs.items(): self.set(k , v)
    def get(self , key , default = None):
        return getattr(self , key) if hasattr(self , key) else default
    def update(self , another = {} , **kwargs):
        for k , v in another.items(): self.set(k , v)
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
    def subset(self , keys):
        return {key:self.get(key) for key in keys}
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

class gpHandler:
    '''
    ------------------------ gp primatives and terminals ------------------------
    includes:
        primatives_self
        primatives_1d
        primatives_2d
        primatives_3d
        primatives_4d
    '''
    @staticmethod
    def I(x): return x
    
    @classmethod
    def primatives_self(cls , use_class = [Fac,Raw,int1,int2,int3,float1]):
        return [(cls.I, [c], c) for c in use_class]
    
    @staticmethod
    def primatives_1d():
        # (primative , in_types , out_type , name)
        return [
            (F.log, [Fac], Fac) ,
            (F.sqrt, [Fac], Fac) ,
            (F.square , [Fac], Fac) ,
            #(F.neg, [Fac], Fac) ,
            (F.rank_pct, [Fac], Fac , 'rank') ,
            #(F.scale, [Fac], Fac) ,
            (F.sigmoid, [Fac], Fac) ,
            (F.signedpower, [Fac , float1], Fac , 'power') ,
            (F.sign, [Raw], Fac) ,
        ]
    @staticmethod
    def primatives_2d():
        # (primative , in_types , out_type , name)
        return [
            (F.add, [Fac, Fac], Fac) ,
            (F.sub, [Fac, Fac], Fac) ,
            (F.mul, [Fac, Fac], Fac) ,
            (F.div, [Fac, Fac], Fac) ,
            (F.rank_add, [Fac, Fac], Fac) ,
            (F.rank_sub, [Fac, Fac], Fac) ,
            (F.rank_div, [Fac, Fac], Fac) ,
            (F.rank_mul, [Fac, Fac], Fac) ,
            (F.ts_stddev, [Fac, int2], Fac) ,
            #(F.ts_sum, [Fac, int2], Fac) ,
            (F.ts_product, [Fac, int2], Fac) ,
            (F.ts_delay, [Fac, int1], Fac) ,
            (F.ts_delta, [Fac, int1], Fac) ,
            (F.ts_lin_decay, [Fac, int2], Fac , 'ts_dw') ,
            (F.ma, [Fac, int3], Fac) ,
            (F.ma, [Fac, int2], Fac) ,
            (F.pctchg, [Fac, int3], Fac) ,
            (F.pctchg, [Fac, int1], Fac) ,
            (F.ts_min, [Fac, int2], Fac) ,
            (F.ts_max, [Fac, int2], Fac) ,
            (F.ts_zscore, [Fac , int3], Fac) ,
            (F.ts_argmin, [Fac, int2], Fac) ,
            (F.ts_argmax, [Fac, int2], Fac) ,
            (F.ts_rank, [Fac, int2], Fac) ,
            (F.ts_argmin, [Raw, int2], Fac) ,
            (F.ts_argmax, [Raw, int2], Fac) ,
            (F.ts_rank, [Raw, int2], Fac) ,
            # (F.add_int, [Fac, int1], Fac) ,
            # (F.sub_int1, [Fac, int1], Fac) ,
            # (F.sub_int2, [int1, Fac], Fac) ,
            # (F.mul_int, [Fac, int1], Fac) ,
            # (F.div_int1, [Fac, int1], Fac) ,
            # (F.div_int2, [int1, Fac], Fac) ,
        ]
    @staticmethod
    def primatives_3d():
        # (primative , in_types , out_type , name)
        return [
            (F.ts_rankcorr, [Fac, Fac, int2], Fac) ,
            (F.ts_decay_pos_dif, [Fac, Fac, int3], Fac , 'ts_dwdif') ,
            (F.ts_cov, [Fac, Fac, int2], Fac) ,
            (F.ts_corr, [Raw, Raw, int2], Fac) ,
            (F.ts_beta, [Raw, Raw, int2], Fac) ,
            (F.ts_btm_avg, [Raw, int3, int1], Fac) ,
            (F.ts_top_avg, [Raw, int3, int1], Fac) ,
            (F.ts_rng_dif, [Raw, int3, int1], Fac) ,
        ]
    @staticmethod
    def primatives_4d():
        # (primative , in_types , out_type , name)
        return [
            (F.ts_xbtm_yavg, [Raw, Fac, int3, int1], Fac) ,
            (F.ts_xtop_yavg, [Raw, Fac, int3, int1], Fac) ,
            (F.ts_xrng_ydif, [Raw, Fac, int3, int1], Fac) ,
            #(F.ts_grouping_ascsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xbtm_long') ,
            #(F.ts_grouping_decsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xtop_long') ,
            #(F.ts_grouping_difsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xdif_long') ,
        ]
    @classmethod
    def primatives_all(cls , use_class = [Fac,Raw,int1,int2,int3,float1]):
        prims = [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['1d','2d','3d','4d']] for prima in plist]
        prims += cls.primatives_self(use_class)
        return prims
    
    @classmethod
    def prune(cls , population):
        if isinstance(population , creator.Individual): #type:ignore
            return creator.Individual([prim for prim in population if prim.name != 'I']) #type:ignore
        else:
            return [cls.prune(ind) for ind in population]
        
    @classmethod
    def duplicate(cls , population):
        original_strs = [str(ind).replace(' ','') for ind in population]
        index_mapping = {value: index for index, value in enumerate(original_strs)}
        return [population[i] for i in index_mapping.values()]
    
    @classmethod
    def Compiler(cls , pset):
        def compiler(individual):
            return gp.compile(individual , pset)
        return compiler
    
    @classmethod
    def Toolbox(cls , eval_func , gp_args , i_iter = - 1 , max_depth = 5 , n_args = (1,1) , **kwargs):
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
        pset = gp.PrimitiveSetTyped("main", [Fac] * n_args[0] + [Raw] * n_args[1], Fac)
        str_iter = '_' if i_iter < 0 else f'_{i_iter}'
        # ------------ rename initial input factors --------------
        assert sum(n_args) == len(gp_args)
        for i , v in enumerate(gp_args): pset.renameArguments(**{f'ARG{i}':v})
        
        # ------------ set int and float terminals ------------
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'int{i}{str_iter}' for i in range(6)]]
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'float{i}{str_iter}' for i in range(6)]]
        pset.addEphemeralConstant(f'int1{str_iter}', lambda: np.random.randint(1,10+1), int1) # random int must > 0
        pset.addEphemeralConstant(f'int2{str_iter}', lambda: np.random.randint(2,10+1), int2) # random int must > 1
        for v in [10 , 15 , 20 , 40]:  pset.addTerminal(v, int3)
        #for v in [10 , 20]:  pset.addTerminal(v, int4)
        #for v in [60 , 120 , 180 , 200 , 240]: pset.addTerminal(v, int5)
        pset.addEphemeralConstant(f'float1{str_iter}', lambda: round(np.random.random()*2+1,2), float1) # random int must > 1
        
        # add primatives
        for prims in cls.primatives_all(): pset.addPrimitive(*prims)
        
        '''创建遗传算法基础模块，以下参数不建议更改，如需更改，可参考deap官方文档'''
        # https://zhuanlan.zhihu.com/p/72130823
        [(delattr(creator , n) if hasattr(creator , n) else None) for n in ['FitnessMin' , 'Individual']]
        creator.create("FitnessMin", base.Fitness, weights=(+1.0,))   # 优化问题：单目标优化，weights为单元素；+1表明适应度越大，越容易存活
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset) # type:ignore 个体编码：pset，预设的

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_= max_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)# type:ignore
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)# type:ignore
        toolbox.register("prune", cls.prune)
        toolbox.register("duplicate", cls.duplicate)
        toolbox.register("compile", cls.Compiler(pset))
        toolbox.register("evaluate", eval_func , compiler = toolbox.compile , **kwargs) # type: ignore
        toolbox.register("select", tools.selTournament, tournsize=3) # 锦标赛：第一轮随机选择3个，取最大
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_= max_depth)  # genFull
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) # type:ignore
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=min(10,max_depth)))  # max=3
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=min(10,max_depth)))  # max=3

        return toolbox
    
class gpTimer:
    '''
    ------------------------ gp timers ------------------------
    includes:
        PTimer     : record a process and its time cost
        AccTimer   : record a series of time costs, can average later
        EmptyTimer : do nothing
    '''
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {}
        self.df_cols = {}

    class PTimer:
        def __init__(self , key , record = False , target_dict = {} , print = True , print_str = None):
            self.key = key
            self.record = record
            self.target_dict = target_dict
            self.print = print
            self.print_str = print_str
        def __enter__(self):
            self.start_time = time.time()
        def __exit__(self, type, value, trace):
            time_cost = time.time() - self.start_time
            if self.record: self.append_time(self.target_dict , self.key , time_cost)
            if self.print: print(f'{self.print_str if self.print_str else self.key} Done, cost {time_cost:.2f} secs')
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
            pass
        
    def __call__(self , key , print = True , df_cols = True , print_str = None):
        if df_cols: self.df_cols.update({key:True})
        return self.PTimer(key , self.recording , self.recorder , print = print , print_str = print_str)
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
    def save_to_csv(self , path , columns = None , print_out = False , dtype = float):
        if columns is None:
            df = pd.DataFrame(data = {k:self.recorder[k] for k,v in self.df_cols.items() if v} , dtype=dtype) 
        else:
            df = pd.DataFrame(data = {k:self.recorder[k] for k in columns} , dtype=dtype) 
        df.to_csv(path)
        if print_out: 
            with pd.option_context('display.width' , 160 ,  'display.max_colwidth', 10 , 'display.precision', 4,):
                print(df)
