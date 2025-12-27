

import pandas as pd

from src.proj.util import Plot

class PlotDfFigIterator:
    def __init__(self , default_prefix : str):
        self.default_prefix = default_prefix
        self.args_set = False

    def set_args(self , data : pd.DataFrame , show : bool , title_prefix : str | None , title : str , group_key : list[str] , drop_keys : bool = True , 
                 drop_cols : list[str] = ['suffix' , 'strategy' , 'prefix'] , num_groups_per_iter = 1 , num_pages : int | None = None , **kwargs):
        self.data = data
        self.show = show
        self.title_prefix = title_prefix
        self.title = title
        self.group_key = group_key
        self.drop = drop_cols + (group_key if drop_keys else [])
        self.num_groups_per_iter = num_groups_per_iter
        self.num_pages = num_pages
        assert 'drop' not in kwargs , '"drop" kwargs must not be set'
        assert 'full_title' not in kwargs , '"drop" kwargs must not be set'
        self.kwargs = kwargs
        self.args_set = True
    
    def iter(self):
        if not self.args_set:
            raise ValueError('Arguments not set')
        self.args_set = False
        self.group_plot = Plot.PlotMultipleData(self.data , self.group_key , self.num_groups_per_iter)
        title = f'{self.title_prefix or self.default_prefix} {self.title}'
        
        for i , sub_data in enumerate(self.group_plot):
            full_title = '' if self.num_pages is None else f'{title} (P{i+1}/{self.num_pages})'
            with Plot.PlotFactorData(sub_data , drop = self.drop , title = title , full_title = full_title , show=self.show and i==0 , **self.kwargs) as (df , fig):
                yield df , fig

    @property
    def figs(self):
        return self.group_plot.fig_dict