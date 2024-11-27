import itertools
import numpy as np
import pandas as pd
from contextlib import nullcontext

from typing import Any , Literal

from src.basic import CALENDAR , RegisteredModel , SILENT
from src.factor.util import StockFactor , Benchmark , Portfolio
from src.factor.builder import PortfolioBuilder

class ModelPortfolioBuilder:
    SUB_FMP_TYPE = ['indep' , 'conti'] # independent and continuous portfolios

    def __init__(self , reg_model : RegisteredModel):
        self.reg_model = reg_model
        self.fmp_tables : dict[int , pd.DataFrame] = {}
    
    def pred_factor(self , date : int):
        df = self.reg_model.load_pred(date)
        assert not df.empty , f'empty df returned for {date}'
        factor = StockFactor(df.assign(date = date)).normalize()
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor
    
    def alpha_model(self , date : int):
        return self.pred_factor(date).alpha_models()[0]

    def last_fmp_table(self , date : int):
        fmp_dates = CALENDAR.slice(self.reg_model.fmp_target_dates , None , CALENDAR.cd(date , -1))
        if len(fmp_dates) == 0: return pd.DataFrame()
        last_date = fmp_dates.max()
        if last_date not in self.fmp_tables:
            self.fmp_tables[last_date] = self.reg_model.load_fmp(last_date)
        return self.fmp_tables[last_date]
    
    def last_port(self , date : int , port_name : str):
        table = self.last_fmp_table(date)
        if table.empty: return None
        table = table[table['name'] == port_name]
        return Portfolio.from_dataframe(table , name = port_name)

    def get_builder_kwargs(self , date : int , fmp_type : Literal['top' , 'optim'] , 
                           subtype : str , benchmark : str , n_best : int = -1):
        assert subtype in self.SUB_FMP_TYPE , f'invalid subtype: {subtype}'
        alpha = self.alpha_model(date)
        kwargs : dict[str , Any] = {
            'category' : fmp_type , 'alpha' : alpha , 'benchmark' : benchmark , 'lag' : 0 , 
            'suffixes' : [subtype] , 'n_best' : n_best , 'build_on' : None
        }
        if subtype == 'conti': kwargs['build_on'] = self.last_port(date , PortfolioBuilder.get_full_name(**kwargs))
        return kwargs
    
    def iter_builders(self , date : int , verbosity : int = 0):
        for subtype , benchmark , n_best in itertools.product(self.SUB_FMP_TYPE , Benchmark.DEFAULTS , [20 , 30 , 50 , 100]):
            kwargs = self.get_builder_kwargs(date , 'top' , subtype , benchmark , n_best)
            yield PortfolioBuilder(verbosity = verbosity , **kwargs)

        for subtype , benchmark in itertools.product(self.SUB_FMP_TYPE , Benchmark.DEFAULTS):
            kwargs = self.get_builder_kwargs(date , 'optim' , subtype , benchmark)
            yield PortfolioBuilder(verbosity = verbosity , **kwargs)
        
    def update_fmps(self , update = True , overwrite = False , silent = True):
        '''get update dates and build portfolios'''
        assert update != overwrite , 'update and overwrite must be different here'
        with SILENT if silent else nullcontext():   
            dates = CALENDAR.diffs(self.reg_model.fmp_target_dates , self.reg_model.fmp_dates if update else [])
            self.build_fmps(dates , deploy = True)
        return dates
    
    def build_fmps(self , dates : list[int] | np.ndarray , deploy = True):
        for date in dates:
            self.fmp_tables[date] = self.build_day(date) 
            if not SILENT: print(f'Finished build fmps for {self.reg_model} at date {date}')
            if deploy: self.reg_model.save_fmp(self.fmp_tables[date] , date , False)
    
    def build_day(self , date : int):
        ports = [builder.setup().build(date).port.full_table for builder in self.iter_builders(date)]
        df = pd.concat(ports).reset_index(drop=True)
        assert df.columns.tolist() == ['name' , 'date' , 'secid' , 'weight'] , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        return df
    
    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , silent = True):
        '''Update pre-registered factors to '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha' '''
        if model_name is None: print(f'model_name is None, build fmps for all registered models')
        models = RegisteredModel.SelectModels(model_name)
        [print(f'  -->  build fmps for {model}') for model in models]
        for model in models:
            md = cls(model)
            md.update_fmps(update = update , overwrite = overwrite , silent = silent)
            print(f'Finish build fmps for [{model}] predicting!')
        return md
    
