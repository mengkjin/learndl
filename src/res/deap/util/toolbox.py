import torch
from typing import Callable , Sequence
from deap import base , creator , gp
from .syntax import BaseIndividual , SyntaxRecord

class BaseToolbox(base.Toolbox):
    evaluate_individual : Callable[..., BaseIndividual]
    evaluate_population : Callable[..., list[BaseIndividual]]
    create_individual : Callable[... , BaseIndividual]
    create_population : Callable[... , list[BaseIndividual]]

    def str2ind(self , x : str , fit_value = None) -> BaseIndividual:
        return self.to_ind(x , fit_value = fit_value)

    def ind2str(self , ind : BaseIndividual) -> str:
        return ind.syntax

    def to_ind(self , syntax : BaseIndividual | str | SyntaxRecord , **kwargs) -> BaseIndividual:
        if isinstance(syntax , str):
            return getattr(creator , 'Individual' , BaseIndividual).from_syntax(syntax , **kwargs)
        elif isinstance(syntax , BaseIndividual):
            return syntax
        elif isinstance(syntax , SyntaxRecord):
            return syntax.to_ind(**kwargs)
        else:
            raise ValueError(f'Invalid input type: {type(syntax)}')

    def to_record(self , ind : BaseIndividual | str | SyntaxRecord) -> SyntaxRecord:
        return SyntaxRecord.create(ind)

    def prune_ind(self , ind : BaseIndividual | str | SyntaxRecord) -> BaseIndividual:
        return self.to_ind(ind).prune()

    def to_indpop(self , population : Sequence[BaseIndividual | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.to_ind(ind) for ind in population]

    def prune_pop(self , population : Sequence[BaseIndividual | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.prune_ind(ind) for ind in population]

    def deduplicate(self , population : Sequence[BaseIndividual] , forbidden : list | None = []) -> list[BaseIndividual]:
        # return the unique population excuding specific ones (forbidden)
        ori = [ind.pure_syntax for ind in population]
        fbd = [str(ind) for ind in forbidden] if forbidden else []
        allowed = [ind not in fbd for ind in ori]
        return [ind for ind , allowed in zip(population , allowed) if allowed]


    def purify_pop(self , population : Sequence[BaseIndividual]) -> list[BaseIndividual]:
        # remove Identity primatives of population
        return [ind.purify() for ind in population]
        
    def revive_pop(self , population : Sequence[BaseIndividual]) -> list[BaseIndividual]:
        # remove Identity primatives of population
        return [ind.revive() for ind in population]

    @property
    def compiler(self) -> Callable[[BaseIndividual | str | SyntaxRecord], torch.Tensor]:
        if not hasattr(self , '_compiler'):
            def compiler(syntax : BaseIndividual | str | SyntaxRecord) -> torch.Tensor:
                if isinstance(syntax , str):
                    ind = getattr(creator , 'Individual' , BaseIndividual).from_syntax(syntax)
                elif isinstance(syntax , SyntaxRecord):
                    ind = syntax.to_ind()
                else:
                    ind = syntax
                return gp.compile(ind , ind.pset)
            self._compiler = compiler
        return self._compiler