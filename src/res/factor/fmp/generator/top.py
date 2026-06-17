"""
Top portfolio creator for Factor Model Portfolio
"""
from __future__ import annotations

from src.proj import Const , Base
from src.proj.bases import TestType
from src.res.factor.fmp.generator.basic import BasicCreatorConfig , BasicPortfolioCreator

test_type = TestType.TOP

__all__ = ['TopStocksPortfolioCreatorConfig' , 'TopStocksPortfolioCreator']

class TopStocksPortfolioCreatorConfig(BasicCreatorConfig):
    """
    Config for PortfolioGenerator (Top Portfolio Generator)
    kwargs:
        no additioanl kwargs
    """
    slots = ['n_best' , 'screener' , 'screen_ratio' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control']
    def __init__(
        self , 
        n_best : int = Const.Factor.FMP.creator[test_type.value]['n_best'] , 
        sorter : Base.SELF = 'self' ,
        **kwargs
    ):
        super().__init__(n_best = n_best , sorter = 'self' , **kwargs)

class TopStocksPortfolioCreator(BasicPortfolioCreator):
    def setup(self , indent : int = 1 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs):
        self.conf = TopStocksPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self