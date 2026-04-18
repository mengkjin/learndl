import pandas as pd

from typing import Any

from src.proj import CONST

from .basic import BasicCreatorConfig , generate_top_stock_port
from ...util import Port , PortCreateResult , PortCreator , AlphaScreener

builder_type = 'revscreen'

class RevScreeningPortfolioCreatorConfig(BasicCreatorConfig):
    '''
    Config for Reverse Screening Portfolio Generator
    Screen (by screener) + Sorting (by target alpha) = (RevScreen) = (Reinforce)
    Screening is based on a screen alpha to screen stocks (e.g. choose top 50% of stocks with the gru_day_V1)
    Then, the stocks are sorted by the target alpha (e.g. choose top 50 of pred)

    kwargs:
        screen_ratio : float = CONST.Conf.Fmp.default[builder_type]['screen_ratio'] , ratio of stocks to be screened (used as pool of top stocks)
        screener     : str | list[str] = CONST.Conf.Fmp.default[builder_type]['screener'] , source and key of alpha to be used for sorting
    '''
    slots = ['screener' , 'screen_ratio' , 'n_best' , 'turn_control' , 'buffer_zone' , 'no_zone' , 'indus_control']

    def __init__(
        self , 
        screener : str | list[str] = CONST.Conf.Fmp.default[builder_type]['screener'] , 
        screen_ratio : float = CONST.Conf.Fmp.default[builder_type]['screen_ratio'] , 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.screener = screener
        self.screen_ratio = screen_ratio

    def get_screen_alpha(self , model_date : int , indus = True) -> pd.DataFrame:
        return self.alpha_screener.get(model_date).get(model_date).to_dataframe(indus = indus , na_indus_as = 'unknown')

    @property
    def alpha_screener(self) -> AlphaScreener:
        if not hasattr(self , '_alpha_screener'):
            self._alpha_screener = AlphaScreener(self.screener , ratio = self.screen_ratio)
        return self._alpha_screener
    
class RevScreeningPortfolioCreator(PortCreator):

    def setup(self , indent : int = 1 , vb_level : Any = 3 , **kwargs):
        self.conf = RevScreeningPortfolioCreatorConfig.init_from(indent = indent , vb_level = vb_level , **kwargs)
        return self
    
    def parse_input(self):
        return self

    def solve(self):
        df = generate_top_stock_port(
            model_date = self.model_date , 
            sorting_alpha = self.alpha_model , 
            config = self.conf , 
            init_port = self.init_port , 
            bench_port = self.bench_port , 
            screen_alpha = self.conf.alpha_screener
        )
        port = Port(df , date = self.model_date , name = self.name , value = self.value).rescale()
        self.create_result = PortCreateResult(port , len(df) == self.conf.n_best)
        return self

    def output(self):
        if self.detail_infos: 
            self.create_result.analyze(self.bench_port , self.init_port)
        return self