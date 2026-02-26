import torch
from typing import Callable , Sequence
from deap import base , gp
from .syntax import BaseIndividual , BaseSyntax , SyntaxRecord , SyntaxControl

class BaseToolbox(base.Toolbox):
    evaluate_syx : Callable[..., BaseSyntax]
    evaluate_pop : Callable[..., list[BaseSyntax]]
    create_individual : Callable[... , BaseIndividual]
    create_population : Callable[... , list[BaseIndividual]]

    def __init__(self , ind_pset : gp.PrimitiveSetTyped , syx_pset : gp.PrimitiveSetTyped) -> None:
        super().__init__()
        self.ind_pset = ind_pset
        self.syx_pset = syx_pset

    def str2ind(self , x : str , fit_value = None) -> BaseIndividual:
        return SyntaxControl.str2ind(x , ind_pset = self.ind_pset , fit_value = fit_value)

    def str2syx(self , x : str , ind_str : str , fit_value = None) -> BaseSyntax:
        return SyntaxControl.str2syx(x , ind_str = ind_str , syx_pset = self.syx_pset , fit_value = fit_value)

    def ind2syx(self , ind : BaseIndividual) -> BaseSyntax:
        return SyntaxControl.ind2syx(ind , syx_pset = self.syx_pset)

    def ind2str(self , ind : BaseIndividual) -> str:
        return SyntaxControl.ind2str(ind)

    def syx2str(self , syx : BaseSyntax) -> str:
        return SyntaxControl.syx2str(syx)

    def syx2ind(self , syx : BaseSyntax) -> BaseIndividual:
        return SyntaxControl.syx2ind(syx , ind_pset = self.ind_pset)

    def to_ind(self , input : BaseIndividual | BaseSyntax | str | SyntaxRecord) -> BaseIndividual:
        if isinstance(input , SyntaxRecord):
            return input.to_ind(self)
        elif isinstance(input , str):
            return self.str2ind(input)
        elif isinstance(input , BaseSyntax):
            return self.syx2ind(input)
        return input

    def to_syx(self , input : BaseIndividual | BaseSyntax | str | SyntaxRecord) -> BaseSyntax:
        if isinstance(input , SyntaxRecord):
            return input.to_syx(self)
        elif isinstance(input , str):
            return self.str2syx(input , input)
        elif isinstance(input , BaseIndividual):
            return self.ind2syx(input)
        return input

    def to_record(self , ind : BaseIndividual | BaseSyntax | str | SyntaxRecord) -> SyntaxRecord:
        if isinstance(ind , SyntaxRecord):
            return ind
        else:
            return SyntaxRecord.create(ind)

    def prune_ind(self , ind : BaseIndividual | BaseSyntax | str | SyntaxRecord) -> BaseIndividual:
        return SyntaxControl.prune_ind(self.to_ind(ind))

    def to_indpop(self , population : Sequence[BaseIndividual | BaseSyntax | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.to_ind(ind) for ind in population]

    def prune_pop(self , population : Sequence[BaseIndividual | BaseSyntax | str | SyntaxRecord]) -> list[BaseIndividual]:
        return [self.prune_ind(ind) for ind in population]

    def deduplicate(self , population : Sequence[BaseIndividual] , forbidden : list | None = []) -> list[BaseIndividual]:
        return SyntaxControl.deduplicate(population , forbidden = forbidden)

    def indpop2syxpop(self , population : Sequence[BaseIndividual]) -> list[BaseSyntax]:
        # remove Identity primatives of population
        return [self.ind2syx(ind) for ind in population]
        
    def syxpop2indpop(self , population : Sequence[BaseSyntax]) -> list[BaseIndividual]:
        # remove Identity primatives of population
        return [self.syx2ind(ind) for ind in population]

    @property
    def compiler(self) -> Callable[[BaseIndividual | BaseSyntax | str | SyntaxRecord], torch.Tensor]:
        if not hasattr(self , '_compiler'):
            self._compiler = SyntaxControl.Compiler(self.syx_pset)
        return self._compiler