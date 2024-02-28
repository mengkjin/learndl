import time
import numpy as np
import pandas as pd
from deap import gp
import math_func_gpu as F

# ------------------------ gp terminals ------------------------
class Fac(): pass
class Raw(): pass
class POSINT0(): pass
class POSINT1(): pass
class POSINT2(): pass
class POSINT3(): pass
class POSINT4(): pass
class POSINT5(): pass
def _returnself(input): return input

class gpPrimatives:
    '''
    ------------------------ gp primatives ------------------------
    includes:
        primatives_self
        primatives_1d
        primatives_2d
        primatives_3d
        primatives_4d
        new_pset
    '''
    @staticmethod
    def primatives_self():
        return [
            (_returnself, [Fac], Fac , '_fac') ,
            (_returnself, [Raw], Raw , '_raw') ,
            #(_returnself, [POSINT0], POSINT0, '_int0') ,
            (_returnself, [POSINT1], POSINT1, '_int1') ,
            (_returnself, [POSINT2], POSINT2, '_int2') ,
            (_returnself, [POSINT3], POSINT3, '_int3') ,
            #(_returnself, [POSINT4], POSINT4, '_int4') ,
            #(_returnself, [POSINT5], POSINT5, '_int5') ,
        ]
    @staticmethod
    def primatives_1d():
        # (primative , in_types , out_type , name)
        return [
            (F.log, [Fac], Fac) ,
            (F.sqrt, [Fac], Fac) ,
            (F.neg, [Fac], Fac) ,
            (F.rank_pct, [Fac], Fac) ,
            (F.sign, [Raw], Fac) ,
            (F.scale, [Fac], Fac) ,
            (F.sigmoid, [Fac], Fac) ,
            (F.signedpower, [Fac , POSINT2], Fac) ,
            # (F.neg_int, [POSINT0], POSINT0) ,
        ]
    @staticmethod
    def primatives_2d():
        # (primative , in_types , out_type , name)
        return [
            (F.add, [Fac, Fac], Fac) ,
            (F.sub, [Fac, Fac], Fac) ,
            (F.mul, [Fac, Fac], Fac) ,
            (F.div, [Fac, Fac], Fac) ,
            (F.rank_sub, [Fac, Fac], Fac) ,
            (F.rank_div, [Fac, Fac], Fac) ,
            (F.rank_add, [Fac, Fac], Fac) ,
            (F.ts_delaypct, [Fac, POSINT1], Fac) ,
            (F.ts_stddev, [Fac, POSINT2], Fac) ,
            (F.ts_sum, [Fac, POSINT2], Fac) ,
            (F.ts_product, [Fac, POSINT2], Fac) ,
            (F.ts_delay, [Fac, POSINT1], Fac) ,
            (F.ts_delta, [Fac, POSINT1], Fac) ,
            (F.ts_lin_decay, [Fac, POSINT2], Fac) ,
            (F.ts_min, [Fac, POSINT2], Fac) ,
            (F.ts_max, [Fac, POSINT2], Fac) ,
            (F.ts_zscore, [Fac , POSINT3], Fac) ,
            (F.ts_argmin, [Raw, POSINT2], Fac) ,
            (F.ts_argmax, [Raw, POSINT2], Fac) ,
            (F.ts_rank, [Raw, POSINT2], Fac) ,
            # (F.add_int, [Fac, POSINT1], Fac) ,
            # (F.sub_int1, [Fac, POSINT1], Fac) ,
            # (F.sub_int2, [POSINT1, Fac], Fac) ,
            # (F.mul_int, [Fac, POSINT1], Fac) ,
            # (F.div_int1, [Fac, POSINT1], Fac) ,
            # (F.div_int2, [POSINT1, Fac], Fac) ,
        ]
    @staticmethod
    def primatives_3d():
        # (primative , in_types , out_type , name)
        return [
            (F.ts_rankcorr, [Fac, Fac, POSINT2], Fac) ,
            (F.ts_lin_decay_pos, [Fac, Fac, POSINT3], Fac) ,
            (F.ts_covariance, [Fac, Fac, POSINT2], Fac) ,
            (F.ts_correlation, [Raw, Raw, POSINT2], Fac) ,
        ]
    @staticmethod
    def primatives_4d():
        # (primative , in_types , out_type , name)
        return [
            (F.ts_grouping_ascsortavg, [Fac, Raw, POSINT3, POSINT1], Fac) ,
            (F.ts_grouping_decsortavg, [Fac, Raw, POSINT3, POSINT1], Fac) ,
            (F.ts_grouping_difsortavg, [Fac, Raw, POSINT3, POSINT1], Fac) ,
            #(F.ts_grouping_ascsortavg, [Fac, Raw, POSINT5, POSINT4], Fac , 'ts_grouping_ascsortavg_long') ,
            #(F.ts_grouping_decsortavg, [Fac, Raw, POSINT5, POSINT4], Fac , 'ts_grouping_decsortavg_long') ,
            #(F.ts_grouping_difsortavg, [Fac, Raw, POSINT5, POSINT4], Fac , 'ts_grouping_difsortavg_long') ,
        ]
    @classmethod
    def primatives_all(cls):
        return [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['self','1d','2d','3d','4d']] for prima in plist]

    @classmethod
    def new_pset(cls , n_fac = 1 , n_raw = 1 , arg_names = None , i_iter = 0):
        pset = gp.PrimitiveSetTyped("main", [Fac] * n_fac + [Raw] * n_raw, Fac)

        # ------------rename initial input factors--------------
        if arg_names is not None:
            assert n_fac + n_raw == len(arg_names)
            for i , v in enumerate(arg_names): pset.renameArguments(**{f'ARG{i}':v})
        
        #--------set int----------
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'POSINT{i}_{i_iter}' for i in range(6)]]
        pset.addEphemeralConstant(f'POSINT1_{i_iter}', lambda: np.random.randint(1,10+1), POSINT1) # random int must > 0
        pset.addEphemeralConstant(f'POSINT2_{i_iter}', lambda: np.random.randint(2,10+1), POSINT2) # random int must > 1
        for v in [10 , 15 , 20 , 40]:  pset.addTerminal(v, POSINT3)
        
        #pset.addEphemeralConstant(f'POSINT0_{i_iter}', lambda: np.random.randint(0,10+1), POSINT0) # random int can be 0
        #for v in [10 , 20]:  pset.addTerminal(v, POSINT4)
        #for v in [60 , 120 , 180 , 200 , 240]: pset.addTerminal(v, POSINT5)

        # add primatives
        for prima in cls.primatives_all(): pset.addPrimitive(*prima)
        return pset
    
class gpTimer:
    '''
    ------------------------ gp timers ------------------------
    includes:
        PTimer
        AccTimer
    '''
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {}
        self.df_cols = {}

    class PTimer:
        def __init__(self , key , record = False , target_dict = {} , print = False):
            self.key = key
            self.record = record
            self.target_dict = target_dict
            self.print = print
        def __enter__(self):
            if self.print: print(f'{self.key} ... start!')
            self.start_time = time.time()
        def __exit__(self, type, value, trace):
            time_cost = time.time() - self.start_time
            if self.record: self.append_time(self.target_dict , self.key , time_cost)
            if self.print: print(f'{self.key} ... done, cost {time_cost:.2f} secs')
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
        
    def __call__(self , key , print = False , df_cols = False):
        if df_cols: self.df_cols.update({key:True})
        return self.PTimer(key , self.recording , self.recorder , print = print)
    def __repr__(self):
        return self.recorder.__repr__()
    def __bool__(self): 
        return True
    def acc_timer(self , key):
        if key not in self.recorder.keys(): self.recorder[key] = self.AccTimer(key)
        assert isinstance(self.recorder[key] , self.AccTimer) , self.recorder[key]
        return self.recorder[key]
    def append_time(self , key , time_cost , df_cols = False):
        if df_cols: self.df_cols.update({key:True})
        self.PTimer.append_time(self.recorder , key , time_cost)
    def save_to_csv(self , path , columns = None , print_out = False , dtype = float):
        if columns is None:
            df = pd.DataFrame(data = {k:self.recorder[k] for k,v in self.df_cols.items() if v} , dtype=dtype) 
        else:
            df = pd.DataFrame(data = {k:self.recorder[k] for k in columns} , dtype=dtype) 
        df.to_csv(path)
        if print_out: 
            with pd.option_context('display.max_colwidth', 11 , 'display.precision', 4,):
                print(df)
