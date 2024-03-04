import operator , time , joblib , os , gc
import numpy as np
import pandas as pd
import torch
from argparse import Namespace
from deap import base , creator , tools , gp
from copy import deepcopy
import gp_math_func as MF


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

class gpManager:
    def __init__(self , job_dir = None) -> None:
        self.job_dir = './pop/bendi' if job_dir is None else job_dir
        self.dir = Namespace(
            pop = f'{self.job_dir}/population' ,
            hof = f'{self.job_dir}/halloffame' ,
            log = f'{self.job_dir}/logbook' ,
            sku = f'{self.job_dir}/skuname' ,
            pqt = f'{self.job_dir}/parquet' ,
            res = f'{self.job_dir}/labels_res' ,
            vhl = f'{self.job_dir}/valhalla' ,
        )

        self.path = Namespace(
            goodlog = f'{self.job_dir}/good_log.csv' ,
            fulllog = f'{self.job_dir}/full_log.csv' ,
            runtime = f'{self.job_dir}/saved_times.csv' ,
            params  = f'{self.job_dir}/gp_params.pt' ,
            df_axis = f'{self.job_dir}/df_axis.pt'
        )
        [os.makedirs(subfoler , exist_ok=True) for subfoler in self.dir.__dict__.values()]
        self.df_axis = {}

    def dump_generation(self , population , halloffame , valhalla , i_iter = 0 , i_gen = 0 , verbose = False , **kwargs):
        if i_iter < 0: return self
        basename = self.record_basename(i_iter , i_gen)
        self.logbook.record(i_gen = i_gen , **kwargs)
        if verbose: [print('     -->   ' + s) for s in str(self.logbook.stream).split('\n')]
        joblib.dump(population, f'{self.dir.pop}/{basename}.pkl')
        joblib.dump(halloffame, f'{self.dir.hof}/{basename}.pkl')
        joblib.dump(valhalla  , f'{self.dir.vhl}/{basename}.pkl')
        joblib.dump(self.logbook , f'{self.dir.log}/{basename}.pkl')
        return self

    def load_generation(self , i_iter = 0 , i_gen = 0 , hof_num = 500 , **kwargs):
        if i_gen < 0: return self.new_generation(hof_num = hof_num , **kwargs)
        basename = self.record_basename(i_iter , i_gen)
        population = joblib.load(f'{self.dir.pop}/{basename}.pkl')
        halloffame = joblib.load(f'{self.dir.hof}/{basename}.pkl')
        valhalla   = joblib.load(f'{self.dir.vhl}/{basename}.pkl')
        self.logbook = joblib.load(f'{self.dir.log}/{basename}.pkl')
        return population , halloffame , valhalla
    
    def new_generation(self , hof_num = 500 , log_header = ['i_gen', 'n_evals'] , stats = None , **kwargs):
        population = []
        halloffame = tools.HallOfFame(hof_num)
        valhalla   = []
        self.logbook = tools.Logbook()
        self.logbook.header = log_header + (stats.fields if stats else []) #type:ignore
        return population , halloffame , valhalla

    def update_sku(self , individual , pool_skuname):
        poolid = int(pool_skuname.split('_')[-1])
        if poolid % 100 == 0:
            start_time_sku = time.time()
            output_path = f'{self.dir.sku}/z_{pool_skuname}.txt'
            with open(output_path, 'w', encoding='utf-8') as file1:
                print(str(individual).replace(' ',''),'\n start_time',time.ctime(start_time_sku),file=file1)

    def record_basename(self , i_iter = 0 , i_gen = 0):
        iter_str = 'iteration' if i_iter < 0 else f'iter{i_iter}'
        gen_str  = 'overall' if i_gen < 0 else f'gen{i_gen}'
        return f'{iter_str}_{gen_str}'
    
    def load_state(self , key , i_iter , i_gen = 0 , i_good = 0):
        if key == 'labels_res':
            return torch.load(f'{self.dir.res}/iter{i_iter}.pt')
        elif key == 'parquet':
            return pd.read_parquet(f'{self.dir.pqt}/good_{i_good}.parquet', engine='fastparquet')
        else:
            path = getattr(self.path , key)
            if key == 'df_axis':
                self.df_axis = torch.load(path)
                return self.df_axis
            elif key in ['goodlog' , 'fulllog']:
                if os.path.exists(path):
                    df = pd.read_csv(path,index_col=0)
                    return df[df.i_iter < i_iter]
                return pd.DataFrame()
            elif path.endswith('.csv'):
                return pd.read_csv(path)
            elif path.endswith('.pt'):
                return torch.load(path)
            else:
                raise Exception(key)

    def save_state(self , data , key , i_iter , i_gen = 0 , i_good = 0 , **kwargs):
        if key == 'labels_res':
            torch.save(data , f'{self.dir.res}/iter{i_iter}.pt')
        elif key == 'parquet':
            if isinstance(data , torch.Tensor): data = data.cpu().numpy()
            df = pd.DataFrame(data,index=self.df_axis['df_index'],columns=self.df_axis['df_columns'])
            df.to_parquet(f'{self.dir.pqt}/good_{i_good}.parquet' , engine='fastparquet')
        else:
            path = getattr(self.path , key)
            if key == 'df_axis':
                torch.save(data , self.path.df_axis)
                self.df_axis = data
            elif isinstance(data , pd.DataFrame) and path.endswith('.csv'):
                data.to_csv(path)
            elif path.endswith('.pt'):
                torch.save(data , path)
            else:
                raise Exception(key , data)
            
    def load_states(self, keys , **kwargs):
        return [self.load_state(key , **kwargs) for key in keys]
    
    def save_states(self, datas , **kwargs):
        return [self.save_state(data , key , **kwargs) for key , data in datas.items()]

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
            (MF.log, [Fac], Fac) ,
            (MF.sqrt, [Fac], Fac) ,
            (MF.square , [Fac], Fac) ,
            #(MF.neg, [Fac], Fac) ,
            (MF.rank_pct, [Fac], Fac , 'rank') ,
            #(MF.scale, [Fac], Fac) ,
            (MF.sigmoid, [Fac], Fac) ,
            (MF.signedpower, [Fac , float1], Fac , 'power') ,
            (MF.sign, [Raw], Fac) ,
        ]
    @staticmethod
    def primatives_2d():
        # (primative , in_types , out_type , name)
        return [
            (MF.add, [Fac, Fac], Fac) ,
            (MF.sub, [Fac, Fac], Fac) ,
            (MF.mul, [Fac, Fac], Fac) ,
            (MF.div, [Fac, Fac], Fac) ,
            (MF.rank_add, [Fac, Fac], Fac) ,
            (MF.rank_sub, [Fac, Fac], Fac) ,
            (MF.rank_div, [Fac, Fac], Fac) ,
            (MF.rank_mul, [Fac, Fac], Fac) ,
            (MF.ts_stddev, [Fac, int2], Fac) ,
            #(MF.ts_sum, [Fac, int2], Fac) ,
            (MF.ts_product, [Fac, int2], Fac) ,
            (MF.ts_delay, [Fac, int1], Fac) ,
            (MF.ts_delta, [Fac, int1], Fac) ,
            (MF.ts_lin_decay, [Fac, int2], Fac , 'ts_dw') ,
            (MF.ma, [Fac, int3], Fac) ,
            (MF.ma, [Fac, int2], Fac) ,
            (MF.pctchg, [Fac, int3], Fac) ,
            (MF.pctchg, [Fac, int1], Fac) ,
            (MF.ts_min, [Fac, int2], Fac) ,
            (MF.ts_max, [Fac, int2], Fac) ,
            (MF.ts_zscore, [Fac , int3], Fac) ,
            (MF.ts_argmin, [Fac, int2], Fac) ,
            (MF.ts_argmax, [Fac, int2], Fac) ,
            (MF.ts_rank, [Fac, int2], Fac) ,
            (MF.ts_argmin, [Raw, int2], Fac) ,
            (MF.ts_argmax, [Raw, int2], Fac) ,
            (MF.ts_rank, [Raw, int2], Fac) ,
            # (MF.add_int, [Fac, int1], Fac) ,
            # (MF.sub_int1, [Fac, int1], Fac) ,
            # (MF.sub_int2, [int1, Fac], Fac) ,
            # (MF.mul_int, [Fac, int1], Fac) ,
            # (MF.div_int1, [Fac, int1], Fac) ,
            # (MF.div_int2, [int1, Fac], Fac) ,
        ]
    @staticmethod
    def primatives_3d():
        # (primative , in_types , out_type , name)
        return [
            (MF.ts_rankcorr, [Fac, Fac, int2], Fac) ,
            (MF.ts_decay_pos_dif, [Fac, Fac, int3], Fac , 'ts_dwdif') ,
            (MF.ts_cov, [Fac, Fac, int2], Fac) ,
            (MF.ts_corr, [Raw, Raw, int2], Fac) ,
            (MF.ts_beta, [Raw, Raw, int2], Fac) ,
            (MF.ts_btm_avg, [Raw, int3, int1], Fac) ,
            (MF.ts_top_avg, [Raw, int3, int1], Fac) ,
            (MF.ts_rng_dif, [Raw, int3, int1], Fac) ,
        ]
    @staticmethod
    def primatives_4d():
        # (primative , in_types , out_type , name)
        return [
            (MF.ts_xbtm_yavg, [Raw, Fac, int3, int1], Fac) ,
            (MF.ts_xtop_yavg, [Raw, Fac, int3, int1], Fac) ,
            (MF.ts_xrng_ydif, [Raw, Fac, int3, int1], Fac) ,
            #(MF.ts_grouping_ascsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xbtm_long') ,
            #(MF.ts_grouping_decsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xtop_long') ,
            #(MF.ts_grouping_difsortavg, [Fac, Raw, int5, int4], Fac , 'ts_y_xdif_long') ,
        ]
    @classmethod
    def primatives_all(cls , use_class = [Fac,Raw,int1,int2,int3,float1]):
        prims = [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['1d','2d','3d','4d']] for prima in plist]
        prims += cls.primatives_self(use_class)
        return prims
    
    @classmethod
    def prune(cls , population):
        if isinstance(population , creator.Individual): #type:ignore
            return creator.Individual([prim for prim in population if prim.name != 'I']) #type:ignore , remove method of 'I'
        else:
            return [ind for ind in [cls.prune(ind) for ind in population] if ind is not None]
        
    @classmethod
    def deduplicate(cls , population , exclude = []):
        original_strs = [str(ind).replace(' ','') for ind in population]
        exclude_strs  = [str(ind).replace(' ','') for ind in exclude]
        index_mapping = {value: index for index, value in enumerate(original_strs)} # remove duplicates
        index_keys    = np.setdiff1d(list(index_mapping.keys()) , exclude_strs) # remove exclude members
        # return [population[i] for i in index_mapping.values()]
        return [population[index_mapping[k]] for k in index_keys]
    
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
        toolbox.register("deduplicate", cls.deduplicate)
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
    def __init__(self , record = False) -> None:
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
    def time_table(self , columns = None , showoff = False , dtype = float):
        if columns is None:
            df = pd.DataFrame(data = {k:self.recorder[k] for k,v in self.df_cols.items() if v} , dtype=dtype) 
        else:
            df = pd.DataFrame(data = {k:self.recorder[k] for k in columns} , dtype=dtype) 
        if showoff: 
            with pd.option_context('display.width' , 160 ,  'display.max_colwidth', 10 , 'display.precision', 4,):
                print(df)
        return df.round(6)

class MemoryManager():
    unit = 1024**3

    def __init__(self , device_no = 0) -> None:
        self.device_no = device_no
        self.cuda_avail = torch.cuda.is_available()
        self.unit = type(self).unit
        if self.cuda_avail: self.gmem_total = torch.cuda.mem_get_info()[1] / self.unit
        self.record = {}
        self.check(showoff = True)

    def check(self , key = None, showoff = False , critical_ratio = 0.5):
        
        if not self.cuda_avail: return 0.

        gmem_free = torch.cuda.mem_get_info(self.device_no)[0] / self.unit
        if gmem_free > critical_ratio * self.gmem_total: 
            # if showoff: print(f'**Cuda Memory: Free {gmem_free:.1f}G') 
            return gmem_free

        torch.cuda.empty_cache() # collect graphic memory 
        gmem_freed = torch.cuda.mem_get_info(self.device_no)[0] / self.unit - gmem_free
        gmem_free += gmem_freed
        gmem_allo  = torch.cuda.memory_allocated(self.device_no) / self.unit
        gmem_rsrv  = torch.cuda.memory_reserved(self.device_no) / self.unit
        
        if key is not None:
            if key not in self.record.keys(): self.record[key] = []
            self.record[key].append(gmem_freed)
        if showoff: print(f'**Cuda Memory: Free {gmem_free:.1f}G, Allocated {gmem_allo:.1f}G, Reserved {gmem_rsrv:.1f}G, Re-collect {gmem_freed:.1f}G Cache!') 
        
        # gc.collect() # collect memory, very slow
        
        return gmem_free

    def __bool__(self):
        return True
    
    @classmethod
    def object_memory(cls , object):
        if isinstance(object , torch.Tensor):
            return cls.tensor_memory(object)
        elif isinstance(object , (list,tuple)):
            return sum([cls.object_memory(obj) for obj in object])
        elif isinstance(object , dict):
            return sum([cls.object_memory(obj) for obj in object.values()])
        else:
            return 0.
    
    @classmethod
    def tensor_memory(cls , tensor):
        total_memory = tensor.element_size() * tensor.numel()
        return total_memory / cls.unit
    
    def print_memeory_record(self):
        if len(self.record):
            print(f'  --> Avg Freed Cuda Memory: ')
            for key , value in self.record.items():
                print(f'     --> {key} : {len(value)} counts, on average freed {np.mean(value):.2f}G')