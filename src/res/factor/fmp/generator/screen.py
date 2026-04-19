from __future__ import annotations
from typing import Any , Literal

from src.proj import CONST
from .basic import BasicCreatorConfig , BasicPortfolioCreator

builder_type = 'screen'

class ScreeningPortfolioCreatorConfig(BasicCreatorConfig):
    '''
    Config for Screening Portfolio Generator
    Screen (by target alpha) + Sorting (by sorter) = Screening
    Screening is based on the target alpha to screen stocks (e.g. choose top 50% of stocks with the target alpha)
    Then, the stocks are sorted by a sorting alpha (e.g. choose top 50 of gru_day_V1)

    kwargs:
        sorter          : str | list[str] = CONST.Conf.Fmp.default[builder_type]['sorter'] , source and key of alpha to be used for sorting
    '''

    def __init__(
        self , 
        sorter : str | list[str] = CONST.Conf.Fmp.default[builder_type]['sorter'] , 
        screener : Literal['self'] = 'self' ,
        **kwargs
    ):
        super().__init__(sorter = sorter , screener = 'self' , **kwargs)
    
class ScreeningPortfolioCreator(BasicPortfolioCreator):
    def setup(self , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        self.conf = ScreeningPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self