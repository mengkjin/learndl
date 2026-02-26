from dataclasses import dataclass
from typing import Callable , Sequence
import itertools
import torch
import numpy as np
from src.res.deap.func import math_func as MF
from .types import Fac, Raw, int1, int2, int3, float1

from deap import gp

@dataclass
class Primative:
    func : Callable
    in_types : list[type|tuple[type,...]]
    out_type : type
    name : str | None = None

    @property
    def func_name(self):
        return self.name if self.name else self.func.__name__

    def sub_primatives(self):
        in_types : list[list[type]] = []
        in_base  : list[type] = []
        for in_type in self.in_types:
            if isinstance(in_type , Sequence):
                in_type_bases = [self.reach_base(_subtype) for _subtype in in_type]
                assert all([b == in_type_bases[0] for b in in_type_bases]) , in_type_bases
                in_type_base = in_type_bases[0]
                in_type = list(in_type)
            else:
                in_type_base = self.reach_base(in_type)
                in_type = [in_type]
            in_types.append(in_type)
            in_base.append(in_type_base)

        baseprim = Primative(self.func , list(in_base) , self.out_type , self.func_name)
        subprims : list[Primative] = []
        for i , in_type in enumerate(list(itertools.product(*in_types))):
            func_name_sub = self.func_name + ('' if i == 0 else f'__{i-1}__')
            subprims.append(Primative(self.func , list(in_type) , self.out_type , func_name_sub))
        return baseprim , subprims

    def primative_args(self):
        return self.func , self.in_types , self.out_type , self.func_name

    @classmethod
    def reach_base(cls , obj , base_types : list[type] = [torch.Tensor,int,float]):
        """return deepest base type of the object"""
        return obj if obj in base_types else cls.reach_base(obj.__base__)

    @staticmethod
    def I(x): # noqa: E743
        return x
    
    @classmethod
    def primatives_identity(cls) -> list['Primative']:
        return [
            cls(cls.I,[Fac]    , Fac   , '_I_0_'),
            cls(cls.I,[Raw]    , Raw   , '_I_1_'),
            cls(cls.I,[int1]   , int1  , '_I_2_'),
            cls(cls.I,[int2]   , int2  , '_I_3_'),
            cls(cls.I,[int3]   , int3  , '_I_4_'),
            cls(cls.I,[float1] , float1, '_I_5_'),
        ]
    
    @classmethod
    def primatives_1d(cls) -> list['Primative']:
        # (primative , in_types , out_type , name)
        return [
            cls(MF.log,        [(Fac,Raw)], Fac) ,
            cls(MF.sqrt,       [(Fac,Raw)], Fac) ,
            cls(MF.square ,    [Fac], Fac) ,
            cls(MF.rank_pct,   [Fac], Fac , 'rank') ,
            cls(MF.sigmoid,    [Fac], Fac) ,
            cls(MF.signedpower,[(Fac,Raw) , float1], Fac , 'power') ,
            cls(MF.sign,       [(Fac,Raw)], Fac) ,
        ]
    @classmethod
    def primatives_2d(cls) -> list['Primative']:
        # (primative , in_types , out_type , name)
        return [
            cls(MF.add,        [Fac, Fac], Fac) ,
            cls(MF.sub,        [Fac, Fac], Fac) ,
            cls(MF.mul,        [Fac, Fac], Fac) ,
            cls(MF.div,        [Fac, Fac], Fac) ,
            cls(MF.rank_add,   [Fac, Fac], Fac) ,
            cls(MF.rank_sub,   [Fac, Fac], Fac) ,
            cls(MF.rank_div,   [Fac, Fac], Fac) ,
            cls(MF.rank_mul,   [Fac, Fac], Fac) ,
            cls(MF.ts_stddev,  [Fac, int2], Fac) ,
            cls(MF.ts_product, [Fac, int2], Fac) ,
            cls(MF.ts_delay,   [Fac, (int1,int2)], Fac) ,
            cls(MF.ts_delta,   [Fac, (int1,int2)], Fac) ,
            cls(MF.ts_lin_decay, [Fac, int2], Fac , 'ts_dw') ,
            cls(MF.ma,         [(Fac,Raw), (int2,int3)], Fac) ,
            cls(MF.pctchg,     [Raw, (int1,int2,int3)], Fac) ,
            cls(MF.ts_min,     [(Fac,Raw), int2], Fac) ,
            cls(MF.ts_max,     [(Fac,Raw), int2], Fac) ,
            cls(MF.ts_zscore,  [(Fac,Raw) , int3], Fac) ,
            cls(MF.ts_argmin,  [(Fac,Raw), int2], Fac) ,
            cls(MF.ts_argmax,  [(Fac,Raw), int2], Fac) ,
            cls(MF.ts_rank,    [(Fac,Raw), int2], Fac) ,
        ]
    @classmethod
    def primatives_3d(cls) -> list['Primative']:
        # (primative , in_types , out_type , name)
        return [
            cls(MF.ts_rankcorr,        [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            cls(MF.ts_decay_pos_dif,   [(Fac,Raw), (Fac,Raw), int3], Fac , 'ts_dwdif') ,
            cls(MF.ts_cov,             [Fac, Fac, int2], Fac) ,
            cls(MF.ts_corr,            [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            cls(MF.ts_beta,            [(Fac,Raw), (Fac,Raw), int2], Fac) ,
            cls(MF.ts_btm_avg,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
            cls(MF.ts_top_avg,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
            cls(MF.ts_rng_dif,         [(Raw,Fac), int3, (int1,int2)], Fac) ,
        ]
    @classmethod
    def primatives_4d(cls) -> list['Primative']:
        # (primative , in_types , out_type , name)
        return [
            cls(MF.ts_xbtm_yavg, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            cls(MF.ts_xtop_yavg, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            cls(MF.ts_xrng_ydif, [(Raw,Fac), (Fac,Raw), int3, (int1,int2)], Fac) ,
            #Primative(MF.ts_xbtm_yavg, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xbtm_long') ,
            #Primative(MF.ts_xtop_yavg, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xtop_long') ,
            #Primative(MF.ts_xrng_ydif, [(Raw,Fac), Fac, int4, int3], Fac , 'ts_y_xdif_long') ,
        ]
    @classmethod
    def all_primatives(cls) -> list['Primative']:
        # all primatives
        prims = [prima for plist in [getattr(cls , f'primatives_{m}')() for m in ['1d','2d','3d','4d']] for prima in plist]
        prims += cls.primatives_identity()
        return prims
    @classmethod
    def GetPrimSets(cls , n_args : tuple[int,int] , argnames : list[str] , additional_name_suffix : str = '' , 
                    int_range : dict[int,Sequence[int]] = {1:(1,2),2:(2,10),3:(10,15,20,30,40)} ,
                    float_range : Sequence[float] = (np.arange(0,101)*0.02).round(2).tolist()) -> tuple[gp.PrimitiveSetTyped,gp.PrimitiveSetTyped]:
        """create individual and syntax primitive set for individual and syntax
        input:
            n_args:       tuple of number of arguments for factor and raw , e.g. (17 , 16)
            argnames:     list of argument names , e.g. ['fac1' , 'fac2' , 'raw1' , 'raw2']
            additional_name_suffix:     suffix of additional name , e.g. '_' if gp_main.i_iter < 0 else f'_{gp_main.i_iter}'
            int_range:    range of int terminals , e.g. {1:(1,2),2:(2,10),3:(10,15,20,30,40)}
            float_range:  range of float terminals , e.g. (np.arange(0,101)*0.02).round(2).tolist()
        output:
            pset:         primitive set for syntax
        """
        ind_pset = cls.IndividualPrimSet(n_args , argnames)
        syx_pset = cls.SyntaxPrimSet(n_args , argnames)
        # ------------ set int and float terminals ------------
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'int{i}{additional_name_suffix}' for i in range(6)]]
        [(delattr(gp , n) if hasattr(gp , n) else None) for n in [f'float{i}{additional_name_suffix}' for i in range(6)]]
        [ind_pset.addTerminal(v, int1) for v in int_range.get(1,[])]
        [ind_pset.addTerminal(v, int2) for v in int_range.get(2,[])]
        [ind_pset.addTerminal(v, int3) for v in int_range.get(3,[])]
        [ind_pset.addTerminal(v, float1) for v in float_range]
        #ind_pset.addEphemeralConstant(f'int1{str_iter}', lambda: np.random.randint(1,10+1), int1) # random int must > 0
        #ind_pset.addEphemeralConstant(f'int2{str_iter}', lambda: np.random.randint(2,10+1), int2) # random int must > 1
        #ind_pset.addEphemeralConstant(f'float1{str_iter}', lambda: round(np.random.random()*2+1,2), float1) # random int must > 1
        
        # add primatives (including their subprims)
        for prim in cls.all_primatives(): 
            baseprim , subprims = prim.sub_primatives()
            syx_pset.addPrimitive(*baseprim.primative_args())
            for subprim in subprims: 
                ind_pset.addPrimitive(*subprim.primative_args())
            
        return ind_pset , syx_pset
    @classmethod
    def IndividualPrimSet(cls , n_args : tuple[int,int] , argnames : list[str]) -> gp.PrimitiveSetTyped:
        """create individual primitive set for individual
        example:
            n_args = (2 , 2)
            argnames = ['fac1' , 'fac2' , 'raw1' , 'raw2']
            additional_name_suffix = '_' if gp_main.i_iter < 0 else f'_{gp_main.i_iter}'
            return:
                pset:         primitive set for individual
                
        input:
            n_args:       tuple of number of arguments for factor and raw , e.g. (17 , 16)
            argnames:     list of argument names , e.g. ['fac1' , 'fac2' , 'raw1' , 'raw2']
        output:
            pset:         primitive set for individual
        """
        ind_pset =gp.PrimitiveSetTyped('main', [Fac] * n_args[0] + [Raw] * n_args[1], Fac)
        for i , v in enumerate(argnames): 
            ind_pset.renameArguments(**{f'ARG{i}':v})
        return ind_pset
    @classmethod
    def SyntaxPrimSet(cls , n_args : tuple[int,int] , argnames : list[str]) -> gp.PrimitiveSetTyped:
        """create syntax primitive set for syntax
        input:
            n_args:       tuple of number of arguments for factor and raw , but are both torch.Tensor , e.g. (17 , 16)
        output:
            pset:         primitive set for syntax
        """
        syx_pset = gp.PrimitiveSetTyped('main', [torch.Tensor] * sum(n_args), torch.Tensor)
        for i , v in enumerate(argnames): 
            syx_pset.renameArguments(**{f'ARG{i}':v})
        return syx_pset