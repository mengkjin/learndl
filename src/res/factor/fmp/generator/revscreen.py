from typing import Any , Literal

from src.proj import CONST

from .basic import BasicCreatorConfig , BasicPortfolioCreator

builder_type = 'revscreen'

class RevScreeningPortfolioCreatorConfig(BasicCreatorConfig):
    '''
    Config for Reverse Screening Portfolio Generator
    Screen (by screener) + Sorting (by target alpha) = (RevScreen) = (Reinforce)
    Screening is based on a screen alpha to screen stocks (e.g. choose top 50% of stocks with the gru_day_V1)
    Then, the stocks are sorted by the target alpha (e.g. choose top 50 of pred)

    kwargs:
        screener     : str | list[str] = CONST.Conf.Fmp.default[builder_type]['screener'] , source and key of alpha to be used for sorting
    '''
    slots = ['screener' , 'screen_ratio' , 'n_best' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control']

    def __init__(
        self , 
        screener : str | list[str] = CONST.Conf.Fmp.default[builder_type]['screener'] , 
        sorter : Literal['self'] = 'self' ,
        **kwargs
    ):
        super().__init__(screener = screener , sorter = 'self' , **kwargs)
    
class RevScreeningPortfolioCreator(BasicPortfolioCreator):
    def setup(self , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        self.conf = RevScreeningPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self