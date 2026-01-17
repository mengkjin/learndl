import itertools
import numpy as np
import pandas as pd

from collections.abc import Iterable
from typing import Any

from src.proj import Logger , CALENDAR
from src.res.factor.util import StockFactor , Benchmark , Portfolio , PortfolioAccountManager
from src.res.factor.fmp import PortfolioBuilder , parse_full_name , get_port_index
from src.res.model.util import PredictionModel

class ModelPortfolioBuilder:
    FMP_TYPES  = ['top' , 'optim']
    SUB_TYPES  = ['indep' , 'conti'] # independent and continuous portfolios
    N_BESTS    = [-1 , 50]
    BENCHMARKS = Benchmark.DEFAULTS

    def __init__(self , reg_model : PredictionModel):
        self.reg_model = reg_model
        self.fmp_tables : dict[int , pd.DataFrame] = {}
        self.account_manager = PortfolioAccountManager(reg_model.account_dir)
        self._update_fmps_record = []
        self._update_account_record = []

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
        if len(fmp_dates) == 0: 
            return pd.DataFrame()
        last_date = fmp_dates.max()
        if last_date not in self.fmp_tables:
            self.fmp_tables[last_date] = self.reg_model.load_fmp(last_date)
        return self.fmp_tables[last_date]
    
    def last_port(self , date : int , port_name : str):
        table = self.last_fmp_table(date)
        if table.empty: 
            return None
        table = table.query('name == @port_name')
        return Portfolio.from_dataframe(table , name = port_name)
    
    def iter_builder_kwargs(self , date : int | None = None , indent : int = 0 , vb_level : int = 1):
        alpha = self.reg_model.pred_name if date is None else self.alpha_model(date) # None only for name
        iter_args = itertools.product(self.FMP_TYPES , self.SUB_TYPES , self.BENCHMARKS , self.N_BESTS)
        for fmp_type , sub_type , benchmark , n_best in iter_args:
            # check legitimacy
            if (fmp_type == 'top' and n_best <= 0) or (fmp_type == 'optim' and n_best > 0): 
                continue
            
            kwargs : dict[str , Any] = {
                'category' : fmp_type , 'alpha' : alpha , 'benchmark' : benchmark , 'lag' : 0 , 
                'suffixes' : [sub_type] , 'n_best' : n_best , 'build_on' : None , 'indent' : indent , 'vb_level' : vb_level
            }
            if sub_type == 'conti' and date is not None: 
                kwargs['build_on'] = self.last_port(date , PortfolioBuilder.get_full_name(**kwargs))
            
            yield kwargs
    
    def iter_fmp_names(self):
        for kwargs in self.iter_builder_kwargs():
            yield PortfolioBuilder.get_full_name(**kwargs)

    def iter_builders(self , date : int , indent : int = 0 , vb_level : int = 1):
        for kwargs in self.iter_builder_kwargs(date , indent = indent , vb_level = vb_level):           
            yield PortfolioBuilder(**kwargs).setup()
        
    def update_fmps(self , update = True , overwrite = False , indent = 1 , vb_level : int = 2):
        '''get update dates and build portfolios'''
        assert update != overwrite , 'update and overwrite must be different here'
        self._update_fmps_record = []
        
        dates = CALENDAR.diffs(self.reg_model.fmp_target_dates , self.reg_model.fmp_dates if update else [])
        dates = [d for d in dates if d in self.reg_model.pred_dates]
        self.build_fmps(dates , deploy = True , indent = indent , vb_level = vb_level)
        return dates
    
    def build_fmps(self , dates : list[int] | np.ndarray , deploy = True , indent = 1 , vb_level : int = 2):
        for date in dates:
            self.fmp_tables[date] = self.build_day(date , indent = indent + 1 , vb_level = vb_level + 2) 
            Logger.stdout(f'Finished build fmps for {self.reg_model} at {date}' , indent = indent + 1 , vb_level = vb_level + 1)
            if deploy:
                self.reg_model.save_fmp(self.fmp_tables[date] , date , False , indent = indent + 1 , vb_level = vb_level + 2)
            self._update_fmps_record.append(date)
        
    def build_day(self , date : int , indent : int = 1 , vb_level : int = 1):
        ports = [builder.build(date).port.to_dataframe() for builder in self.iter_builders(date , indent = indent , vb_level = vb_level)]
        df = pd.concat(ports).reset_index(drop=True)
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        return df

    def load_accounts(self , resume = True , indent : int = 2 , vb_level : int = 3):
        if resume: 
            self.account_manager.load_dir()
            Logger.stdout(f'accounts include names: {self.account_manager.account_names}' , indent = indent , vb_level = vb_level)
        return self
    
    def account_last_model_dates(self , fmp_names : list[str] | None = None):
        last_dates = self.account_manager.account_last_model_dates()
        if fmp_names is None: 
            fmp_names = list(self.iter_fmp_names())
        ret = {name:last_dates.get(name , CALENDAR.td(self.reg_model.start_dt , -1).as_int()) for name in fmp_names}
        return ret
    
    def account_last_end_dates(self , fmp_names : list[str] | None = None):
        last_dates = self.account_manager.account_last_end_dates()
        if fmp_names is None: 
            fmp_names = list(self.iter_fmp_names())
        ret = {name:last_dates.get(name , self.reg_model.start_dt) for name in fmp_names}
        return ret
    
    def accounting(self , resume = True , deploy = True , indent : int = 1 , vb_level : int = 3):
        self.load_accounts(resume = resume , indent = indent + 1 , vb_level = vb_level + 1)
        self._update_account_record = []

        last_end_dates   = self.account_last_end_dates()
        update_fmp_names = [name for name , end_date in last_end_dates.items() if end_date < CALENDAR.updated()]
        if len(update_fmp_names) == 0: 
            return

        last_model_dates = self.account_last_model_dates(update_fmp_names)
        account_dates = [date for date in self.reg_model.fmp_dates if date >= min(list(last_model_dates.values()))]
        if len(account_dates) == 0: 
            return
        all_fmp_dfs = pd.concat([self.reg_model.load_fmp(date) for date in account_dates])

        for fmp_name in update_fmp_names:
            elements = parse_full_name(fmp_name)
            portfolio = Portfolio.from_dataframe(all_fmp_dfs , name = fmp_name)
            portfolio.accounting(Benchmark(elements['benchmark']) , analytic = elements['lag'] == 0 , attribution = elements['lag'] == 0 , 
                                 indent = indent + 1 , vb_level = vb_level + 2)
            self.account_manager.append_accounts(**{fmp_name : portfolio.account.with_index(get_port_index(fmp_name))})
            Logger.stdout(f'Finished accounting for {fmp_name} at {CALENDAR.dates_str(account_dates)}' , indent = indent + 1 , vb_level=vb_level + 1)

        if deploy:
            with Logger.Timer(f'Deploy accounts for {self.reg_model} at {CALENDAR.dates_str(account_dates)}' , indent = indent , vb_level = vb_level + 1 , enter_vb_level = vb_level + 2):
                self.account_manager.deploy(update_fmp_names , overwrite = True , indent = indent + 2 , vb_level = vb_level + 2)
            
        self._update_account_record = account_dates

    @classmethod
    def update(cls , model_name : str | None = None , update = True , overwrite = False , indent : int = 0 , vb_level : int = 1):
        '''Update prediction models' factor model portfolios'''
        Logger.note(f'Update : {cls.__name__} since last update!' , indent = indent , vb_level = vb_level)
        models = PredictionModel.SelectModels(model_name)
        if model_name is None: 
            Logger.stdout(f'model_name is None, build fmps for all prediction models (len={len(models)})' , indent = indent + 1 , vb_level = vb_level)
        for model in models:
            md = cls(model)
            md.update_fmps(update = update , overwrite = overwrite , indent = indent + 1 , vb_level = vb_level + 2)
            if md._update_fmps_record:
                Logger.success(f'Update model portfolios for {model} , len={len(md._update_fmps_record)}' , indent = indent + 1 , vb_level = vb_level)
            else:
                Logger.skipping(f'Model portfolios for {model} is up to date' , indent = indent + 1 , vb_level = vb_level)

            md.accounting(resume = True , deploy = True , indent = indent + 1 , vb_level = vb_level + 2)
            if md._update_account_record:
                Logger.success(f'Update model portfolios accounting for {model} , len={len(md._update_account_record)}' , indent = indent + 1 , vb_level = vb_level)
            else:
                Logger.skipping(f'Model portfolios accounting for {model} is up to date' , indent = indent + 1 , vb_level = vb_level)
        return md