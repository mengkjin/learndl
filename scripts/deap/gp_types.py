import numpy as np
from deap import gp
import math_func_gpu as F

class Fac(): pass
class Raw(): pass
class POSINT0(): pass
class POSINT1(): pass
class POSINT2(): pass
class POSINT3(): pass
class POSINT4(): pass
class POSINT5(): pass
def _returnself(input): return input

class gpTypes:
    # methods (or Primatives):
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