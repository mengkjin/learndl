import itertools
import numpy as np
import pandas as pd

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any , Literal

from src.basic import CALENDAR , RegisteredModel , SILENT
from src.factor.util import StockFactor , Benchmark , Portfolio 
from src.factor.fmp import PortfolioBuilder , PortfolioAccountManager

class ModelPortfolioBuilder:
    FMP_TYPES  = ['top' , 'optim']
    SUB_TYPES  = ['indep' , 'conti'] # independent and continuous portfolios
    N_BESTS    = [-1 , 20 , 30 , 50 , 100]
    BENCHMARKS = Benchmark.DEFAULTS

    def __init__(self , reg_model : RegisteredModel):
        self.reg_model = reg_model
        self.fmp_tables : dict[int , pd.DataFrame] = {}
        self.accountant = PortfolioAccountManager(reg_model.account_dir)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.reg_model})'
    
    def pred_factor(self , dates : int | list[int] | np.ndarray):
        dates = [dates] if not isinstance(dates , Iterable) else dates
        df = pd.concat([self.reg_model.load_pred(d).assign(date = d) for d in dates])
        assert not df.empty , f'empty df returned for {dates}'
        factor = StockFactor(df).normalize()
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor
    
    def alpha_model(self , dates : int | list[int] | np.ndarray):
        return self.pred_factor(dates).alpha_model()

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
    
    def iter_builder_kwargs(self , date : int | None = None , verbosity : int = 0):
        alpha = self.reg_model.pred_name if date is None else self.alpha_model(date) # None only for name
        iter_args = itertools.product(self.FMP_TYPES , self.SUB_TYPES , self.BENCHMARKS , self.N_BESTS)
        for fmp_type , sub_type , benchmark , n_best in iter_args:
            # check legitimacy
            if (fmp_type == 'top' and n_best <= 0) or (fmp_type == 'optim' and n_best > 0): continue
            
            kwargs : dict[str , Any] = {
                'category' : fmp_type , 'alpha' : alpha , 'benchmark' : benchmark , 'lag' : 0 , 
                'suffixes' : [sub_type] , 'n_best' : n_best , 'build_on' : None , 'verbosity' : verbosity
            }
            if sub_type == 'conti' and date is not None: 
                kwargs['build_on'] = self.last_port(date , PortfolioBuilder.get_full_name(**kwargs))
            
            yield kwargs
    
    def iter_fmp_names(self):
        for kwargs in self.iter_builder_kwargs():
            yield PortfolioBuilder.get_full_name(**kwargs)

    def iter_builders(self , date : int , verbosity : int = 0):
        for kwargs in self.iter_builder_kwargs(date , verbosity):           
            yield PortfolioBuilder(**kwargs).setup()
        
    def update_fmps(self , update = True , overwrite = False , silent = True):
        '''get update dates and build portfolios'''
        assert update != overwrite , 'update and overwrite must be different here'
        self._update_fmps_record = []
        with SILENT if silent else nullcontext():   
            dates = CALENDAR.diffs(self.reg_model.fmp_target_dates , self.reg_model.fmp_dates if update else [])
            dates = [d for d in dates if d in self.reg_model.pred_dates]
            self.build_fmps(dates , deploy = True)
        return dates
    
    def build_fmps(self , dates : list[int] | np.ndarray , deploy = True):
        for date in dates:
            self.fmp_tables[date] = self.build_day(date) 
            if not SILENT: print(f'Finished build fmps for {self.reg_model} at date {date}')
            if deploy: self.reg_model.save_fmp(self.fmp_tables[date] , date , False)
            self._update_fmps_record.append(date)
    
    def build_day(self , date : int):
        ports = [builder.build(date).port.to_dataframe() for builder in self.iter_builders(date)]
        df = pd.concat(ports).reset_index(drop=True)
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        return df

    def load_accounts(self , resume = True , verbose = True):
        if resume: self.accountant.load_dir()
        if verbose: print(f'accounts include names: {self.accountant.account_names}')
        return self
    
    def last_account_dates(self):
        last_dates = self.accountant.last_account_dates
        for name in self.iter_fmp_names():
            if name not in last_dates:
                last_dates[name] = CALENDAR.td(self.reg_model.start_dt , -1)
        return last_dates
    
    def accounting(self , resume = True , deploy = True):
        if resume: self.load_accounts(verbose = False)

        fmp_names = list(self.iter_fmp_names())
        last_dates = self.last_account_dates()
        account_dates = [date for date in self.reg_model.fmp_dates if date >= min(list(last_dates.values()))]
        if len(account_dates) == 0: return

        all_fmp_dfs = pd.concat([self.reg_model.load_fmp(date) for date in account_dates])
        alpha_model = self.alpha_model(account_dates)

        self._update_account_record = account_dates

        for fmp_name in fmp_names:
            portfolio = Portfolio.from_dataframe(all_fmp_dfs[all_fmp_dfs['name'] == fmp_name])
            builder   = PortfolioBuilder.from_full_name(fmp_name , alpha_model , portfolio)
            builder.accounting(last_dates[fmp_name])
            self.accountant.append_accounts(**{fmp_name : builder.account})

        if deploy:
            self.accountant.deploy(True)

    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , silent = True):
        '''Update pre-registered models' factor model portfolios'''
        models = RegisteredModel.SelectModels(model_name)
        if model_name is None: print(f'model_name is None, build fmps for all registered models (len={len(models)})')
        for model in models:
            md = cls(model)
            md.update_fmps(update = update , overwrite = overwrite , silent = silent)
            if md._update_fmps_record:
                print(f'  -->  Finish updating model portfolios for {model} , len={len(md._update_fmps_record)}')
            else:
                print(f'  -->  No new updating model portfolios for {model}')

            md.accounting(resume = True)
            if md._update_account_record:
                print(f'  -->  Finish updating model accounting for {model} , len={len(md._update_account_record)}')
            else:
                print(f'  -->  No new updating model accounting for {model}')
        return md