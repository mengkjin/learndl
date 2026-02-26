from deap import gp , creator
import numpy as np
import re
from dataclasses import dataclass
from typing import Any ,Callable , Sequence
import torch

from .fitness import FitnessObjectMin

class BaseIndividual0(gp.PrimitiveTree):
    fitness : FitnessObjectMin
    pset_raw : gp.PrimitiveSetTyped
    pset_pur : gp.PrimitiveSetTyped

    def __hash__(self): 
        return hash(id(self))

    @property
    def purified(self) -> bool:
        return getattr(self , '_purified' , False)
    
    def purify(self):
        if self.purified:
            return self
        raw_name = str(self)
        pur_name = re.sub(r'__[0-9]+__','',re.sub(r'_I_[0-9]+_','',raw_name.replace(' ','')))
        new_ind = self.from_string(pur_name , pset=self.pset_pur)
        self.__init__(new_ind)
        
        self._raw_name = raw_name
        self._purified = True
        return self

    @property
    def pset(self) -> gp.PrimitiveSetTyped:
        return self.pset_pur if self.purified else self.pset_raw

    @property
    def if_valid(self) -> bool:
        return getattr(self , '_if_valid' , False)

    @if_valid.setter
    def if_valid(self , value : bool):
        self._if_valid = value

    @property
    def metrics(self) -> np.ndarray:
        metrics = getattr(self , '_metrics' , None)
        return metrics if metrics is not None else np.array([])

    @metrics.setter
    def metrics(self , value : np.ndarray | None):
        self._metrics = value

class BasePrimitiveTree(gp.PrimitiveTree):
    fitness : FitnessObjectMin
    pset : gp.PrimitiveSetTyped

    def __hash__(self): 
        return hash(id(self))

    @property
    def if_valid(self) -> bool:
        return getattr(self , '_if_valid' , False)

    @if_valid.setter
    def if_valid(self , value : bool):
        self._if_valid = value

    @property
    def metrics(self) -> np.ndarray:
        metrics = getattr(self , '_metrics' , None)
        return metrics if metrics is not None else np.array([])

    @metrics.setter
    def metrics(self , value : np.ndarray | None):
        self._metrics = value

class BaseIndividual(BasePrimitiveTree):
    """primitive tree that is directly generated from populating process , name can be redundant"""

class BaseSyntax(BasePrimitiveTree):
    """primitive tree that is shrinked from an individual , syntax string is shorter than individual"""
    ind_str : str

@dataclass
class SyntaxRecord:
    syx_str : str
    ind_str : str
    fit_value : tuple | None = None

    def __str__(self):
        return self.syx_str

    def __repr__(self):
        return f'SyntaxRecord(syx_str={self.syx_str}, ind_str={self.ind_str}, fit_value={self.fit_value})'

    def to_syx(self , toolbox):
        return toolbox.str2syx(self.syx_str , ind_str = self.ind_str , fit_value = self.fit_value)

    def to_ind(self , toolbox):
        return toolbox.str2ind(self.ind_str , fit_value = self.fit_value)

    @classmethod
    def create(cls , input : 'BaseIndividual | BaseSyntax | str | SyntaxRecord') -> 'SyntaxRecord':
        if isinstance(input , SyntaxRecord):
            return input
        syx_str = SyntaxControl.trim_syntax(str(input) , remove_placeholders = True)
        if isinstance(input , BaseIndividual):
            ind_str = SyntaxControl.ind2str(input)
        elif isinstance(input , BaseSyntax):
            ind_str = input.ind_str
        else:
            ind_str = input
        fit_value = None if isinstance(input , str) else input.fitness.values
        return cls(syx_str , ind_str , fit_value)

class SyntaxControl:
    @staticmethod
    def trim_syntax(tree_str : Any , remove_placeholders : bool = True) -> str:
        x = str(tree_str).replace(' ','')
        if remove_placeholders:
            x = re.sub(r'_I_[0-9]+_','',x)
            x = re.sub(r'__[0-9]+__','',x)
        return x

    @classmethod
    def ind2str(cls , ind : 'BaseIndividual') -> str:
        x = cls.trim_syntax(ind , remove_placeholders = False)
        return x

    @classmethod
    def syx2str(cls , syx : 'BaseSyntax') -> str:
        x = cls.trim_syntax(syx , remove_placeholders = False)
        return x

    @classmethod
    def str2ind(cls , x : str , ind_pset : gp.PrimitiveSetTyped , fit_value = None) -> 'BaseIndividual':
        assert fit_value is None or isinstance(fit_value , tuple) , fit_value
        ind = getattr(creator , 'Individual' , BaseIndividual).from_string(x , pset=ind_pset) 
        if fit_value: 
            ind.fitness.values = fit_value
        return ind
        
    @classmethod
    def str2syx(cls , x : str , ind_str : str , syx_pset : gp.PrimitiveSetTyped , fit_value = None) -> 'BaseSyntax':
        assert fit_value is None or isinstance(fit_value , tuple) , fit_value
        syx = getattr(creator , 'Syntax' , BaseSyntax).from_string(x , pset=syx_pset) 
        syx.ind_str = ind_str
        if fit_value: 
            syx.fitness.values = fit_value
        return syx

    @classmethod
    def syx2ind(cls , syx : 'BaseSyntax' , ind_pset : gp.PrimitiveSetTyped) -> 'BaseIndividual':
        ind = cls.str2ind(syx.ind_str , ind_pset = ind_pset , fit_value = syx.fitness.values) 
        return ind

    @classmethod
    def ind2syx(cls , ind : 'BaseIndividual' , syx_pset : gp.PrimitiveSetTyped) -> 'BaseSyntax':
        ind_str = str(ind)
        syx = cls.str2syx(cls.trim_syntax(cls.prune_ind(ind) , remove_placeholders = False) , ind_str = ind_str , syx_pset = syx_pset , fit_value = ind.fitness.values) 
        return syx

    @classmethod
    def prune_ind(cls , ind : 'BaseSyntax | BaseIndividual') -> 'BaseIndividual':
        assert isinstance(ind , (getattr(creator , 'Syntax' , BaseSyntax) , getattr(creator , 'Individual' , BaseIndividual))) , type(ind) 
        Ipos = [re.match(prim.name , r'^_I_[0-9]+_$') for prim in ind]
        new_prims = []
        for i , prim in enumerate(ind):
            if i > 0 and Ipos[i] and (ind[i] == ind[i-1]):
                pass
            else:
                new_prims.append(prim)
        return getattr(creator , 'Individual' , BaseIndividual)(new_prims) 

    @classmethod
    def prune_pop(cls , pop : Sequence['BaseIndividual | BaseSyntax']) -> list['BaseIndividual | BaseSyntax']:
        return [cls.prune_ind(syx) for syx in pop]

    @classmethod
    def deduplicate(cls , population : Sequence['BaseIndividual'] , forbidden : list | None = []) -> list['BaseIndividual']:
        # return the unique population excuding specific ones (forbidden)
        ori = [cls.trim_syntax(ind , remove_placeholders = True) for ind in population]
        fbd = [cls.trim_syntax(ind , remove_placeholders = True) for ind in forbidden] if forbidden else []
        allowed = [ind not in fbd for ind in ori]
        return [ind for ind , allowed in zip(population , allowed) if allowed]

    @classmethod
    def Compiler(cls , pset : gp.PrimitiveSetTyped) -> Callable[['BaseIndividual | BaseSyntax | str | SyntaxRecord'], torch.Tensor]:
        # return the compipler of individual sytax:
        # compiler can perform this:
        # factor_value = compiler(sytax) , where syntax is an instance of getattr(creator , 'Individual'), or simply a string such as 'add(cp,turn)'
        def compiler(syntax : BaseIndividual | BaseSyntax | str | SyntaxRecord) -> torch.Tensor:
            tree_or_str = str(syntax) if isinstance(syntax , SyntaxRecord) else syntax
            return gp.compile(tree_or_str , pset)
        return compiler

