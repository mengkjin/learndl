from typing import Any , Literal
from src.proj import CONST

from .basic import BasicCreatorConfig , BasicPortfolioCreator

builder_type = 'top'

class TopStocksPortfolioCreatorConfig(BasicCreatorConfig):
    '''
    Config for PortfolioGenerator (Top Portfolio Generator)
    kwargs:
        no additioanl kwargs
    '''
    slots = ['n_best' , 'screener' , 'screen_ratio' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control']
    def __init__(
        self , 
        n_best : int = CONST.Conf.Fmp.default[builder_type]['n_best'] , 
        sorter : Literal['self'] = 'self' ,
        **kwargs
    ):
        super().__init__(n_best = n_best , sorter = 'self' , **kwargs)

class TopStocksPortfolioCreator(BasicPortfolioCreator):
    def setup(self , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        self.conf = TopStocksPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self