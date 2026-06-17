"""
Screening portfolio creator for Factor Model Portfolio
"""
from __future__ import annotations

from src.proj import Const , Base
from src.res.factor.analytic.test_basics import TestType
from src.res.factor.fmp.generator.basic import BasicCreatorConfig , BasicPortfolioCreator

test_type = TestType.SCREEN

__all__ = ['ScreeningPortfolioCreatorConfig' , 'ScreeningPortfolioCreator']

class ScreeningPortfolioCreatorConfig(BasicCreatorConfig):
    """
    Config for Screening Portfolio Generator
    Screen (by target alpha) + Sorting (by sorter) = Screening
    Screening is based on the target alpha to screen stocks (e.g. choose top 50% of stocks with the target alpha)
    Then, the stocks are sorted by a sorting alpha (e.g. choose top 50 of gru_day_V1)

    kwargs:
        sorter : Base.alias.FeatureType = Const.Factor.FMP.creator[test_type.value]['sorter'] , source and key of alpha to be used for sorting
    """

    def __init__(
        self , 
        sorter : Base.alias.NamesType = Const.Factor.FMP.creator[test_type.value]['sorter'] , 
        screener : Base.SELF = 'self' ,
        **kwargs
    ):
        super().__init__(sorter = sorter , screener = 'self' , **kwargs)
    
class ScreeningPortfolioCreator(BasicPortfolioCreator):
    def setup(self , indent : int = 1 , vb_level : Base.lit.VerbosityLevel = 'max' , **kwargs):
        self.conf = ScreeningPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self