import itertools
import numpy as np
import pandas as pd

from functools import cached_property
from typing import Any , Iterable

from src.proj import CALENDAR , Dates
from src.proj.util import BaseModule
from src.res.factor.util import StockFactor , Benchmark , Portfolio , PortfolioAccountManager
from src.res.factor.fmp import PortfolioBuilder , parse_full_name , get_port_index
from src.res.model.util import PredictorPath

class ModelPortfolioBuilder(BaseModule):
    FMP_TYPES  = ['top' , 'optim']
    SUB_TYPES  = ['indep' , 'conti'] # independent and continuous portfolios
    N_BESTS    = [-1 , 50]
    BENCHMARKS = Benchmark.DEFAULTS

    def __init__(self , pred_path : PredictorPath , * , indent : int = 0 , vb_level : Any = 1):
        self.set_vb(vb_level , indent)
        self.pred_path = pred_path
        self.fmp_tables : dict[int , pd.DataFrame] = {}
        self.account_manager = PortfolioAccountManager(self.pred_path.account_dir)
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.pred_path})'

    @cached_property
    def updated_fmp_dates(self) -> list[Any]:
        return []

    @cached_property
    def updated_account_dates(self) -> list[Any]:
        return []
    
    def pred_factor(self , dates : int | list[int] | np.ndarray):
        dates = [dates] if not isinstance(dates , Iterable) else dates
        df = pd.concat([self.pred_path.load_pred(d).assign(date = d) for d in dates])
        assert not df.empty , f'empty df returned for {dates}'
        factor = StockFactor(df).normalize()
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor
    
    def alpha_model(self , dates : int | list[int] | np.ndarray):
        return self.pred_factor(dates).alpha_model()

    def last_fmp_table(self , date : int):
        fmp_dates = CALENDAR.slice(self.pred_path.fmp_target_dates , None , CALENDAR.cd(date , -1))
        if len(fmp_dates) == 0: 
            return pd.DataFrame()
        last_date = fmp_dates.max()
        if last_date not in self.fmp_tables:
            self.fmp_tables[last_date] = self.pred_path.load_fmp(last_date)
        return self.fmp_tables[last_date]
    
    def last_port(self , date : int , port_name : str):
        table = self.last_fmp_table(date)
        if table.empty: 
            return None
        table = table[table['name'] == port_name]
        return Portfolio.from_dataframe(table , name = port_name)
    
    def iter_builder_kwargs(self , date : int | None = None):
        alpha = self.pred_path.pred_name if date is None else self.alpha_model(date) # None only for name
        iter_args = itertools.product(self.FMP_TYPES , self.SUB_TYPES , self.BENCHMARKS , self.N_BESTS)
        for fmp_type , sub_type , benchmark , n_best in iter_args:
            # check legitimacy
            if (fmp_type == 'top' and n_best <= 0) or (fmp_type == 'optim' and n_best > 0): 
                continue
            
            kwargs : dict[str , Any] = {
                'category' : fmp_type , 'alpha' : alpha , 'benchmark' : benchmark , 'lag' : 0 , 
                'suffixes' : [sub_type] , 'n_best' : n_best , 'build_on' : None , 
                'indent' : self.indent + 1 , 'vb_level' : self.vb_level + 2
            }
            if sub_type == 'conti' and date is not None: 
                kwargs['build_on'] = self.last_port(date , PortfolioBuilder.get_full_name(**kwargs))
            
            yield kwargs
    
    def iter_fmp_names(self , fmp_names : list[str] | None = None):
        if fmp_names is None:
            for kwargs in self.iter_builder_kwargs():
                yield PortfolioBuilder.get_full_name(**kwargs)
        else:
            for name in fmp_names:
                yield name

    def iter_builders(self , date : int):
        for kwargs in self.iter_builder_kwargs(date):           
            yield PortfolioBuilder(**kwargs).setup()
        
    def update_fmps(self , update = True , overwrite = False):
        '''get update dates and build portfolios'''
        assert update != overwrite , 'update and overwrite must be different here'
        
        dates = CALENDAR.diffs(self.pred_path.fmp_target_dates , self.pred_path.fmp_dates if update else [])
        dates = [d for d in dates if d in self.pred_path.pred_dates]
        self.build_fmps(dates , deploy = True)
        return dates
    
    def build_fmps(self , dates : list[int] | np.ndarray , deploy = True):
        for date in dates:
            self.fmp_tables[date] = self.build_day(date) 
            self.logger.stdout(f'Finished build fmps for {self.pred_path} at {date}' , idt = 1 , vb = 1)
            if deploy:
                self.pred_path.save_fmp(self.fmp_tables[date] , date , False , indent = self.indent + 1 , vb_level = self.vb_level + 2)
            self.updated_fmp_dates.append(date)
        
    def build_day(self , date : int):
        ports = [builder.build(date).port.to_dataframe() for builder in self.iter_builders(date)]
        df = pd.concat(ports).reset_index(drop=True)
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        return df

    def load_accounts(self , resume = True):
        if resume: 
            self.account_manager.load_dir()
            self.logger.stdout(f'accounts include names: {self.account_manager.account_names}' , idt = 1 , vb = 1)
        return self
    
    def account_last_model_dates(self , fmp_names : list[str] | None = None):
        last_dates = self.account_manager.account_last_model_dates()
        default_date = CALENDAR.td(self.pred_path.start , -1).as_int()
        return {name:last_dates.get(name , default_date) for name in self.iter_fmp_names(fmp_names)}
    
    def account_last_end_dates(self , fmp_names : list[str] | None = None):
        last_dates = self.account_manager.account_last_end_dates()
        default_date = self.pred_path.start
        return {name:last_dates.get(name , default_date) for name in self.iter_fmp_names(fmp_names)}
    
    def accounting(self , resume = True , deploy = True):
        if resume:
            last_end_dates   = self.account_last_end_dates()
            update_fmp_names = [name for name , end in last_end_dates.items() if end < CALENDAR.updated()]
            if not update_fmp_names: 
                return

            last_model_dates = self.account_last_model_dates(update_fmp_names)
            account_dates = [date for date in self.pred_path.fmp_dates if date >= min(list(last_model_dates.values()))]
            if not account_dates: 
                return

        self.load_accounts(resume = resume)
        all_fmp_dfs = pd.concat([self.pred_path.load_fmp(date) for date in account_dates])

        for fmp_name in update_fmp_names:
            elements = parse_full_name(fmp_name)
            portfolio = Portfolio.from_dataframe(all_fmp_dfs , name = fmp_name)
            portfolio.accounting(Benchmark(elements['benchmark']) , analytic = elements['lag'] == 0 , attribution = elements['lag'] == 0 , 
                                 with_index = get_port_index(fmp_name) , indent = self.indent + 1 , vb_level = self.vb_level + 2)
            self.account_manager.append_accounts(**{fmp_name : portfolio.account})
            self.logger.stdout(f'Finished accounting for {fmp_name} at {Dates(account_dates)}' , idt = 1 , vb = 1)

        if deploy:
            with self.logger.timer(f'Deploy accounts for {self.pred_path} at {Dates(account_dates)}' , vb = 1 , enter_vb = 2):
                self.account_manager.deploy(update_fmp_names , overwrite = True , indent = self.indent + 2 , vb_level = self.vb_level + 2)
            
        self.updated_account_dates.extend(account_dates)

    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , indent : int = 0 , vb_level : Any = 1):
        '''Update prediction models' factor model portfolios'''
        cls.SetClassVB(vb_level , indent)
        cls.logger.note(f'Update since last update!')
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, build fmps for all prediction models (len={len(models)})' , idt = 1)
        for model in models:
            md = cls(model , indent = indent + 1 , vb_level = vb_level + 1)
            md.update_fmps(update = update , overwrite = overwrite)
            if md.updated_fmp_dates:
                md.logger.success(f'Update model portfolios for {model} , len={len(md.updated_fmp_dates)}')
            else:
                md.logger.skipping(f'Model portfolios for {model} is up to date')

            md.accounting(resume = True , deploy = True)
            if md.updated_account_dates:
                md.logger.success(f'Update model portfolios accounting for {model} , len={len(md.updated_account_dates)}')
            else:
                md.logger.skipping(f'Model portfolios accounting for {model} is up to date')
        return md