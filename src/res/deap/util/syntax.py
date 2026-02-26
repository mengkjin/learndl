from deap import gp , creator
import numpy as np
import re
from dataclasses import dataclass

from .fitness import FitnessObjectMin

class BaseIndividual(gp.PrimitiveTree):
    """
    primitive tree that is directly generated from populating process , name can be redundant
    can use purify method to convert to a purified individual (can be used to convert a syntax or compare with others)
    """
    fitness : FitnessObjectMin
    pset_raw : gp.PrimitiveSetTyped
    pset_pur : gp.PrimitiveSetTyped

    def __hash__(self): 
        return hash(id(self))

    def __init__(self , *args , **kwargs):
        super().__init__(*args , **kwargs)
        self.raw_syntax = str(self)

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
    
    def purify(self):
        if self.purified:
            return self
        pure_syntax = self.pure_syntax
        if pure_syntax != self.raw_syntax:
            new_ind = self.from_string(self.pure_syntax , pset=self.pset_pur)
            self.__init__(new_ind)
        self._purified = True
        return self

    def revive(self):
        self._purified = False
        return self

    def revert(self):
        self.__init__(self.raw_syntax)
        return self

    @property
    def pset(self) -> gp.PrimitiveSetTyped:
        return self.pset_pur if self.purified else self.pset_raw

    @property
    def syntax(self) -> str:
        return str(self)

    @property
    def raw_syntax(self) -> str:
        return getattr(self , '_raw_syntax' , self.syntax)

    @raw_syntax.setter
    def raw_syntax(self , value : str):
        self._raw_syntax = value

    @property
    def fit_value(self) -> tuple | None:
        return self.fitness.values

    @fit_value.setter
    def fit_value(self , value : tuple | None):
        self.fitness.values = value

    @classmethod
    def trim_syntax(cls , syntax : str) -> str:
        return re.sub(r'__[0-9]+__','',re.sub(r'_I_[0-9]+_','',syntax.replace(' ','')))

    @property
    def pure_syntax(self) -> str:
        return self.trim_syntax(self.syntax)

    def to_record(self) -> 'SyntaxRecord':
        self.purify()
        return SyntaxRecord(self.syntax , self.raw_syntax , self.fitness.values)

    @classmethod
    def from_syntax(cls , syntax : str , raw_syntax : str | None = None , fit_value : tuple | None = None) -> 'BaseIndividual':
        syntax = cls.trim_syntax(syntax)
        try:
            ind = getattr(creator , 'Individual' , BaseIndividual).from_string(syntax , pset=cls.pset_pur)
        except Exception:
            ind = getattr(creator , 'Individual' , BaseIndividual).from_string(syntax , pset=cls.pset_raw)
        if raw_syntax is not None:
            ind.raw_syntax = raw_syntax
        if fit_value is not None:
            ind.fit_value = fit_value
        return ind

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
    raw_syntax : str
    fit_value : tuple | None = None

    def __str__(self):
        return self.syntax

    def __repr__(self):
        return f'SyntaxRecord(syntax={self.syntax}, raw_syntax={self.raw_syntax}, fit_value={self.fit_value})'

    def to_ind(self):
        return getattr(creator , 'Individual' , BaseIndividual).from_syntax(self.syntax , raw_syntax = self.raw_syntax , fit_value = self.fit_value)

    @classmethod
    def create(cls , input : 'BaseIndividual | str | SyntaxRecord') -> 'SyntaxRecord':
        if isinstance(input , SyntaxRecord):
            return input
        elif isinstance(input , BaseIndividual):
            return input.to_record()
        elif isinstance(input , str):
            return cls(input , input)
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

