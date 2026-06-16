"""
Factor Model Portfolio builder for model , which can build the factor model portfolio for a given date
"""
from __future__ import annotations
import itertools
import pandas as pd

from functools import cached_property
from typing import Any  , Self
from collections.abc import Generator

from src.proj import CALENDAR , Base , Dates
from src.res.factor.util import StockFactor , Benchmark , Portfolio , PortfolioAccountManager , AlphaModel
from src.res.factor.fmp import PortfolioBuilder , parse_full_name , get_port_index
from src.res.model.util import PredictorPath

__all__ = ['ModelPortfolioBuilder']

class ModelPortfolioBuilder(Base.BoundLogger):
    FMP_TYPES  = ['top' , 'optim']
    SUB_TYPES  = ['indep' , 'conti'] # independent and continuous portfolios
    N_BESTS    = [-1 , 50]
    BENCHMARKS = Benchmark.DEFAULTS

    def __init__(self , pred_path : PredictorPath , * , indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.pred_path = pred_path
        self.pred_path.set_vb(vb_level , indent)
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
    
    def pred_factor(self , dates : Base.intDates) -> StockFactor:
        dates = Dates(dates)
        df = pd.concat([self.pred_path.load_pred(d).assign(date = d) for d in dates])
        assert not df.empty , f'empty df returned for {dates}'
        factor = StockFactor(df).normalize()
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor
    
    def alpha_model(self , dates : Base.intDates) -> AlphaModel:
        return self.pred_factor(dates).alpha_model()

    def last_fmp_table(self , date : int) -> pd.DataFrame:
        fmp_dates = self.pred_path.fmp_target_dates.slice(end = CALENDAR.cd(date , -1))
        if fmp_dates.empty: 
            return pd.DataFrame()
        last_date = fmp_dates.max
        if last_date not in self.fmp_tables:
            self.fmp_tables[last_date] = self.pred_path.load_fmp(last_date)
        return self.fmp_tables[last_date]
    
    def last_port(self , date : int , port_name : str) -> Portfolio | None:
        table = self.last_fmp_table(date)
        if table.empty: 
            return None
        table = table[table['name'] == port_name]
        return Portfolio.from_dataframe(table , name = port_name)
    
    def iter_builder_kwargs(self , date : int | None = None) -> Generator[dict[str , Any] , None, None]:
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
    
    def iter_fmp_names(self , fmp_names : Base.alias.NamesType = None) -> Generator[str , None, None]:
        fmp_names = Base.ensure_name_list(fmp_names)
        if fmp_names is None:
            for kwargs in self.iter_builder_kwargs():
                yield PortfolioBuilder.get_full_name(**kwargs)
        else:
            for name in fmp_names:
                yield name

    def iter_builders(self , date : int) -> Generator[PortfolioBuilder , None, None]:
        for kwargs in self.iter_builder_kwargs(date):           
            yield PortfolioBuilder(**kwargs).setup()
        
    def update_fmps(self , update = True , overwrite = False) -> Base.UpdateFlag:
        """get update dates and build portfolios"""
        assert update != overwrite , 'update and overwrite must be different here'
        dates = self.pred_path.fmp_target_dates.diff(self.pred_path.fmp_dates if update else [] , inplace = False)
        dates = dates.intersect(self.pred_path.pred_dates)
        self.build_fmps(dates , deploy = True)

        if self.updated_fmp_dates:
            self.logger.success(f'Update model portfolios for {self.pred_path.pred_name} , len={len(self.updated_fmp_dates)}')
            return Base.UpdateFlag.SUCCESS
        else:
            self.logger.skipping(f'Model portfolios for {self.pred_path.pred_name} is up to date')
            return Base.UpdateFlag.SKIPPED
    
    def build_fmps(self , dates : Base.intDates , deploy = True) -> None:
        for date in Dates(dates):
            self.fmp_tables[date] = self.build_day(date) 
            self.logger.stdout(f'Finished build fmps for {self.pred_path} at {date}' , idt = 1 , vb = 1)
            if deploy:
                self.pred_path.save_fmp(self.fmp_tables[date] , date , False)
            self.updated_fmp_dates.append(date)
        
    def build_day(self , date : int) -> pd.DataFrame:
        ports = [builder.build(date).port.to_dataframe() for builder in self.iter_builders(date)]
        df = pd.concat(ports).reset_index(drop=True)
        assert all(col in df.columns for col in ['name' , 'date' , 'secid' , 'weight']) , \
            f'expect columns: name , date , secid , weight , got {df.columns.tolist()}'
        return df

    def load_accounts(self , resume = True) -> Self:
        if resume: 
            self.account_manager.load_dir()
            self.logger.stdout(f'accounts include names: {self.account_manager.account_names}' , idt = 1 , vb = 2)
        return self
    
    def account_last_model_dates(self , fmp_names : Base.alias.NamesType = None) -> dict[str , int]:
        last_dates = self.account_manager.account_last_model_dates()
        default_date = CALENDAR.td(self.pred_path.start , -1).as_int()
        return {name:last_dates.get(name , default_date) for name in self.iter_fmp_names(fmp_names)}
    
    def account_last_end_dates(self , fmp_names : Base.alias.NamesType = None) -> dict[str , int]:
        last_dates = self.account_manager.account_last_end_dates()
        default_date = self.pred_path.start
        return {name:last_dates.get(name , default_date) for name in self.iter_fmp_names(fmp_names)}
    
    def accounting(self , resume = True , deploy = True) -> Base.UpdateFlag:
        if resume:
            last_end_dates   = self.account_last_end_dates()
            update_fmp_names = [name for name , end in last_end_dates.items() if end < CALENDAR.updated()]
            if not update_fmp_names: 
                return Base.UpdateFlag.SKIPPED

            last_model_dates = self.account_last_model_dates(update_fmp_names)
            account_dates = [date for date in self.pred_path.fmp_dates if date >= min(list(last_model_dates.values()))]
            if not account_dates: 
                return Base.UpdateFlag.SKIPPED

        self.load_accounts(resume = resume)
        all_fmp_dfs = pd.concat([self.pred_path.load_fmp(date) for date in account_dates])

        for fmp_name in update_fmp_names:
            with self.logger.timer(f'Accounting {fmp_name} at {Dates(account_dates)}' , vb = 2 , enter_vb = 4):
                elements = parse_full_name(fmp_name)
                portfolio = Portfolio.from_dataframe(all_fmp_dfs , name = fmp_name)
                portfolio.accounting(Benchmark(elements['benchmark']) , analytic = elements['lag'] == 0 , attribution = elements['lag'] == 0 , 
                                    with_index = get_port_index(fmp_name) , indent = self.indent + 1 , vb_level = self.vb_level + 2)
                self.account_manager.append_accounts(**{fmp_name : portfolio.account})

        if deploy:
            with self.logger.timer(f'Deploy accounts for {self.pred_path} at {Dates(account_dates)}' , vb = 1 , enter_vb = 4):
                self.account_manager.deploy(update_fmp_names , overwrite = True , indent = self.indent + 2 , vb_level = self.vb_level + 4)
            
        self.updated_account_dates.extend(account_dates)

        if self.updated_account_dates:
            self.logger.success(f'Update model portfolios accounting for {self.pred_path.pred_name} , len={len(self.updated_account_dates)}')
            return Base.UpdateFlag.SUCCESS
        else:
            self.logger.skipping(f'Model portfolios accounting for {self.pred_path.pred_name} is up to date')
            return Base.UpdateFlag.SKIPPED

    @classmethod
    def update(
        cls , model_name : str | None = None , update = True , 
        overwrite = False , indent : int = 0 , vb_level : Any = 1
    ) -> Base.UpdateFlag:
        """Update prediction models' factor model portfolios"""
        cls.SetClassVB(vb_level , indent)
        cls.logger.note(f'Update since last update!')
        models = PredictorPath.SelectModels(model_name)
        if model_name is None: 
            cls.logger.stdout(f'model_name is None, build fmps for all prediction models (len={len(models)})' , idt = 1)
    
        flags = Base.UpdateFlagList()
        for model in models:
            md = cls(model , indent = indent , vb_level = vb_level)
            with md.logger.subprocess(idt = 1 , vb = 1):
                flags += md.update_fmps(update = update , overwrite = overwrite)
                flags += md.accounting(resume = True , deploy = True)
        return flags.summarize()