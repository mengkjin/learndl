import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional

from src.basic.conf import ROUNDING_RETURN , ROUNDING_TURNOVER , TRADE_COST
from src.data import DATAVENDOR
from src.factor.util import Portfolio , Benchmark , RISK_MODEL , Port

class PortfolioAccountant:
    '''
    portfolio : Portfolio
    benchmark : Benchmark | str , must given
    daily : bool , or in portfolio dates
    analytic : bool
    attribution : bool
    '''
    def __init__(self , portfolio : Portfolio , benchmark : Optional[Portfolio | Benchmark | str] , 
                 account_path : Optional[str | Path] = None):
        self.portfolio = portfolio
        self.benchmark = self.get_benchmark(benchmark)
        self.account = pd.DataFrame()
        self.account_path = account_path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(portfolio={self.portfolio},benchmark={self.benchmark})'

    @staticmethod
    def get_benchmark(benchmark : Optional[Portfolio | Benchmark | str] = None): 
        if benchmark is None:
            benchmark = Portfolio()
        elif isinstance(benchmark , str):
            benchmark = Benchmark(benchmark)
        return benchmark

    def accounting(self , start : int = -1 , end : int = 99991231 , daily = False , 
                   analytic = True , attribution = True , index : dict = {}):
        '''Accounting portfolio through date, if resume is True, will resume from last account date'''
        port_min , port_max = self.portfolio.available_dates().min() , self.portfolio.available_dates().max()
        start = np.max([port_min , start])
        end   = np.min([DATAVENDOR.td(port_max,5).td , end , DATAVENDOR.td(DATAVENDOR.last_quote_dt,-1).td])

        model_dates = DATAVENDOR.td_within(start , end)
        if len(model_dates) == 0:
            return self
        
        if daily:
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = period_st
        else:
            model_dates = np.intersect1d(model_dates , self.portfolio.available_dates())
            period_st = DATAVENDOR.td_array(model_dates , 1)
            period_ed = np.concatenate([model_dates[1:] , [DATAVENDOR.td(end,1).td]])

        assert np.all(model_dates < period_st) , (model_dates , period_st)
        assert np.all(period_st <= period_ed) , (period_st , period_ed)

        account = pd.DataFrame({
            'model_date':np.concatenate([[-1],model_dates]) , 
            'start':np.concatenate([[model_dates[0]],period_st]) , 
            'end':np.concatenate([[model_dates[0]],period_ed]) ,
            'pf':0. , 'bm':0. , 'turn':0. , 'excess':0. ,
            'analytic':None , 'attribution':None}).set_index('model_date').sort_index()

        port_old = Port.none_port(model_dates[0])
        for date , ed in zip(model_dates , period_ed):
            port_new = self.portfolio.get(date) if self.portfolio.has(date) else port_old
            bench = self.benchmark.get(date , True)

            turn = np.round(port_new.turnover(port_old),ROUNDING_TURNOVER)
            account.loc[date , ['pf' , 'bm' , 'turn']] = \
                [np.round(port_new.fut_ret(ed) , ROUNDING_RETURN) , np.round(bench.fut_ret(ed) , ROUNDING_RETURN) , turn]
            
            if analytic: 
                account.loc[date , 'analytic']    = RISK_MODEL.get(date).analyze(port_new , bench , port_old) #type:ignore
            if attribution: 
                account.loc[date , 'attribution'] = RISK_MODEL.get(date).attribute(port_new , bench , ed , turn * TRADE_COST)  #type:ignore
            port_old = port_new.evolve_to_date(ed)

        account['pf']  = account['pf'] - account['turn'] * TRADE_COST
        account['excess'] = account['pf'] - account['bm']
        account = account.reset_index()

        self.account = account.assign(**index).set_index(list(index.keys())).sort_values('model_date')
        return self