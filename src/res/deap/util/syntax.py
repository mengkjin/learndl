from deap import gp , creator
import numpy as np
import re
from dataclasses import dataclass
from typing import Any
from src.proj import Logger
from .fitness import FitnessObjectMin

class BaseIndividual(gp.PrimitiveTree):
    """
    primitive tree that is directly generated from populating process , name can be redundant
    can use purify method to convert to a purified individual (can be compiled)
    can use revert method to convert from a purified individual to raw (can be varAnd)
    """
    fitness : FitnessObjectMin
    pset_raw : gp.PrimitiveSetTyped
    pset_pur : gp.PrimitiveSetTyped

    def __hash__(self): 
        return hash(id(self))

    def __init__(self , *args , **kwargs):
        super().__init__(*args , **kwargs)

    def __repr__(self):
        return f'BaseIndividual(syntax={self.syntax}, raw_syntax={self.raw_syntax}, purified={self.purified})'

    @property
    def purified(self) -> bool:
        return getattr(self , '_purified' , False)

    def prune(self):
        Ipos = [re.match(prim.name , r'^_I_[0-9]+_$') for prim in self]
        new_prims = []
        for i , prim in enumerate(self):
            if i > 0 and Ipos[i] and (self[i] == self[i-1]):
                pass
            else:
                new_prims.append(prim)
        self.__init__(new_prims)
        return self

    def update_syntax(self , syntax : str):
        self.__init__(syntax)
        return self
    
    def purify(self):
        if self.purified:
            return self
        raw_syntax = self.syntax
        new_ind = self.from_string(self.pure_syntax , pset=self.pset_pur)
        self.__init__(new_ind)
        self._purified = True
        self.raw_syntax = raw_syntax
        return self

    def revert(self):
        if not self.purified:
            return self
        new_ind = self.from_string(self.raw_syntax , pset=self.pset_raw)
        self.__init__(new_ind)
        self._purified = False
        self.raw_syntax = None
        return self

    @property
    def pset(self) -> gp.PrimitiveSetTyped:
        return self.pset_pur if self.purified else self.pset_raw

    @property
    def syntax(self) -> str:
        return str(self)

    @property
    def raw_syntax(self) -> str:
        if not hasattr(self , '_raw_syntax') or self._raw_syntax is None:
            return self.syntax
        return self._raw_syntax

    @raw_syntax.setter
    def raw_syntax(self , value : str | None):
        self._raw_syntax = value

    @property
    def fit_value(self) -> tuple | None:
        return self.fitness.values

    @fit_value.setter
    def fit_value(self , value : tuple | None):
        self.fitness.values = value

    @classmethod
    def trim_syntax(cls , syntax : Any) -> str:
        """trim syntax by replacing patterns"""
        # replace custom int1, int2, int3, float1 to int, int, int, float
        syntax = str(syntax)
        for prim in ['int1', 'int2', 'int3', 'float1']:
            syntax = re.sub(fr'{prim}#([+-]?\d*\.?\d+)#', r'\1', syntax)

        # remove primative suffix
        syntax = re.sub(r'__[0-9]+__','' , syntax)

        # remove identity primatives
        syntax = re.sub(r'_I_[0-9]+_','',syntax)
        return syntax

    @property
    def pure_syntax(self) -> str:
        return self.trim_syntax(self.syntax)

    def to_record(self) -> 'SyntaxRecord':
        self.purify()
        return SyntaxRecord(self.syntax , self.raw_syntax , self.fitness.values)

    @classmethod
    def from_syntax(cls , syntax : str , raw_syntax : str | None = None , fit_value : tuple | None = None) -> 'BaseIndividual':
        if raw_syntax:
            syntax = raw_syntax
        try:
            ind = cls.get_class().from_string(syntax , pset=cls.pset_raw).prune()
        except Exception:
            ind = cls.get_class().from_string(cls.trim_syntax(syntax) , pset=cls.pset_pur)
            ind.raw_syntax = syntax
        if fit_value is not None:
            ind.fit_value = fit_value
        return ind

    @classmethod
    def get_class(cls) -> type['BaseIndividual']:
        return getattr(creator , 'Individual' , BaseIndividual)

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

@dataclass
class SyntaxRecord:
    syntax : str
    raw_syntax : str | None = None
    fit_value : tuple | None = None

    def __str__(self):
        return self.syntax

    def __repr__(self):
        return f'SyntaxRecord(syntax={self.syntax}, raw_syntax={self.raw_syntax}, fit_value={self.fit_value})'

    def to_ind(self):
        return BaseIndividual.get_class().from_syntax(self.syntax , raw_syntax = self.raw_syntax , fit_value = self.fit_value)

    @classmethod
    def create(cls , input : 'BaseIndividual | str | SyntaxRecord') -> 'SyntaxRecord':
        if isinstance(input , SyntaxRecord):
            return input
        elif isinstance(input , BaseIndividual):
            return input.to_record()
        elif isinstance(input , str):
            Logger.warning(f'str to SyntaxRecord is not supported, because it is not unique from a purifed individul to raw (can be varAnd)')
            raise TypeError(f'str to SyntaxRecord is not supported, because it is not unique from a purifed individul to raw (can be varAnd)')
            return cls(input , input)
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

