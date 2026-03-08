import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from deap import gp , creator , tools , base
from deap.algorithms import varAnd
from typing import Any , Literal , Sequence , Callable
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
        self.metrics = None

    def __repr__(self):
        return f'BaseIndividual(syntax={self.syntax}, purified={self.purified}, metrics={self.metrics}, fit_value={self.fit_value})'

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
        return self.change_state('purify')

    def revert(self):
        return self.change_state('revert')

    def change_state(self , direction : Literal['purify' , 'revert']):
        if self.purified == (direction == 'purify'):
            return self
        new_syntax = self.pure_syntax if direction == 'purify' else self.raw_syntax
        raw_syntax = self.syntax if direction == 'purify' else None
        new_ind    = self.from_string(new_syntax , pset=self.pset_pur if direction == 'purify' else self.pset_raw)
        fit_value  = self.fit_value
        metrics    = self.metrics
        self.__init__([*new_ind])
        self._purified = direction == 'purify'
        self.raw_syntax = raw_syntax
        if metrics is not None:
            self.metrics = metrics
        if fit_value:
            self.fit_value = fit_value
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
        if value is None:
            self.fitness.delValues()
        else:
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
        return SyntaxRecord(self.syntax , self.raw_syntax , self.metrics , self.if_valid , self.fit_value)

    @classmethod
    def from_syntax(cls , syntax : str , raw_syntax : str | None = None , 
                    metrics : np.ndarray | None = None , fit_value : tuple | None = None) -> 'BaseIndividual':
        if raw_syntax:
            syntax = raw_syntax
        try:
            ind = cls.get_class().from_string(syntax , pset=cls.pset_raw).prune()
        except Exception:
            ind = cls.get_class().from_string(cls.trim_syntax(syntax) , pset=cls.pset_pur)
            ind.raw_syntax = syntax
        if metrics is not None:
            ind.metrics = metrics
        if fit_value is not None:
            ind.fit_value = fit_value
        return ind

    @classmethod
    def get_class(cls) -> type['BaseIndividual']:
        return getattr(creator , 'Individual' , BaseIndividual)

    @property
    def if_valid(self) -> bool:
        return self.fitness.valid

    @property
    def metrics(self) -> np.ndarray | None:
        metrics = getattr(self , '_metrics' , None)
        return metrics

    @metrics.setter
    def metrics(self , value : np.ndarray | None):
        self._metrics = value

    @classmethod
    def from_object(cls , syntax : 'BaseIndividual | str | SyntaxRecord' , **kwargs) -> 'BaseIndividual':
        if isinstance(syntax , str):
            return cls.get_class().from_syntax(syntax , **kwargs)
        elif isinstance(syntax , BaseIndividual):
            return syntax
        elif isinstance(syntax , SyntaxRecord):
            return syntax.to_ind(**kwargs)
        else:
            raise ValueError(f'Invalid input type: {type(syntax)}')

@dataclass
class SyntaxRecord:
    syntax : str
    raw_syntax : str | None = None
    metrics : np.ndarray | None = None
    if_valid : bool = True
    fit_value : tuple | None = None

    def __str__(self):
        return self.syntax

    def to_ind(self):
        return BaseIndividual.get_class().from_syntax(self.syntax , raw_syntax = self.raw_syntax , metrics = self.metrics , fit_value = self.fit_value if self.if_valid else None)

    @classmethod
    def create(cls , input : 'BaseIndividual | str | SyntaxRecord') -> 'SyntaxRecord':
        if isinstance(input , SyntaxRecord):
            return input
        elif isinstance(input , BaseIndividual):
            return input.to_record()
        elif isinstance(input , str):
            raise TypeError(f'str to SyntaxRecord is not supported, because it is not unique from a purifed individul to raw (can be varAnd)')
        else:
            raise ValueError(f'Invalid input type: {type(input)}')

class Population(Sequence):
    def __init__(self , population : Sequence[BaseIndividual]):
        self.pop = list(set(population))

    def __iter__(self):
        return iter(self.pop)

    def __len__(self):
        return len(self.pop)

    def __bool__(self):
        return len(self.pop) > 0

    def __contains__(self , item : BaseIndividual) -> bool:
        return item in self.pop

    def __getitem__(self , index : int) -> BaseIndividual:
        return self.pop[index]

    def __setitem__(self , index : int , value : BaseIndividual):
        self.pop[index] = value

    def __repr__(self):
        return repr(self.pop)

    def to_list(self) -> list[BaseIndividual]:
        return self.pop

    def record_list(self) -> list[SyntaxRecord]:
        return [ind.to_record() for ind in self.pop]

    @classmethod
    def from_list(cls , population : 'Sequence[BaseIndividual | str | SyntaxRecord] | tools.HallOfFame | Population') -> 'Population':
        if isinstance(population , Population):
            return population
        return cls([BaseIndividual.from_object(ind) for ind in population])

    def valid_pop(self) -> 'Population':
        return self.from_list([ind for ind in self.pop if ind.if_valid])

    def invalid_pop(self , forbidden_lambda : Callable | None = None) -> 'Population':
        return self.from_list([ind for ind in self.pop if not ind.if_valid or (False if forbidden_lambda is None else forbidden_lambda(ind.fit_value))])

    def deduplicate(self , forbidden : Sequence[Any] | None = []) -> 'Population':
        ori = [ind.pure_syntax for ind in self.pop]
        fbd = [str(ind) for ind in forbidden] if forbidden is not None else []
        allowed = [ind not in fbd for ind in ori]
        self.pop = [ind for ind , allowed in zip(self.pop , allowed) if allowed]
        return self

    def purify(self) -> 'Population':
        [ind.purify() for ind in self.pop]
        return self

    def revert(self) -> 'Population':
        [ind.revert() for ind in self.pop]
        return self

    def prune(self) -> 'Population':
        [ind.prune() for ind in self.pop]
        return self

    def selection(self , method : Literal['nsga2' , 'best' , 'Tour' , '2Tour'] , selection_size : int) -> 'Population':
        # Selection of population to pass to next generation, consider surv_rate
        k = len(self) if method in ['Tour' , '2Tour'] else min(selection_size, len(self))
        if method == 'nsga2':
            offspring = tools.selNSGA2(self.pop, k)
        elif method == 'best':
            offspring = tools.selBest(self.pop, k)
        elif method == 'Tour':
            offspring = tools.selTournament(self.pop, k , tournsize=3)
        elif method == '2Tour':
            offspring = tools.selDoubleTournament(self.pop, k , fitness_size=3 , parsimony_size=1.4 , fitness_first=True)
        else:
            raise ValueError(f'Invalid method: {method}')
        return self.from_list(offspring)

    def variation(self , toolbox : base.Toolbox , cxpb : float , mutpb : float) -> 'Population':
        """variation part (crossover and mutation)"""
        assert all([not ind.purified for ind in self.pop]) , 'Variation must be raw individuals , please use revert first'
        population = varAnd(self.pop, toolbox, cxpb , mutpb)
        return self.from_list(population)

    def to_hof(self , hof_num : int) -> tools.HallOfFame:
        hof = tools.HallOfFame(hof_num)
        hof.update(self.pop)
        return hof

    def str_list(self) -> list[str]:
        return [str(ind) for ind in self.pop]

    def pure_str_list(self) -> list[str]:
        return [ind.pure_syntax for ind in self.pop]

    def extend(self , population : 'Population'):
        self.pop.extend(population.pop)
        return self

    def slice(self , start : int , end : int) -> 'Population':
        return self.from_list(self.pop[start:end])

    def log_df(self , metrics_keys : list[str] | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            [[ind.syntax,ind.if_valid] for ind in self] , 
            columns = pd.Index(['syntax','valid']))
        metric_lens = np.array([len(ind.metrics) for ind in self if ind.metrics is not None])
        assert metric_lens.size == 0 or all(metric_lens == metric_lens[0]) , metric_lens
        n_metric = metric_lens[0] if metric_lens.size > 0 else 0
        if n_metric > 0:
            metrics_keys = metrics_keys if metrics_keys is not None else [f'metric_{i}' for i in range(n_metric)]
            metrics = pd.DataFrame([ind.metrics if ind.metrics is not None else np.full(n_metric , np.nan) for ind in self] , columns = metrics_keys)
            df = pd.concat([df , metrics] , axis = 1)
        return df
        