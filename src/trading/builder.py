import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.basic import CALENDAR , RegisteredModel , PATH , Email
from src.data import DATAVENDOR
from src.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Port
from src.factor.fmp import PortfolioBuilder

EXCLUDE_BSE = True
TRADING_PORTS = [
    ('test0' , 'gru_day_V1' , 'csi800+csi1000+csi2000') , 
    ('test1' , 'sellside@dongfang.scores_v0@avg' , 'top-1000') , 
    ('master_daily', 'sellside@huatai.master_combined@master_combined' , 'top-1000' , 50 , 1) , 
    ('gru_daily', 'gru_day_V1' , 'top-1000' , 50 , 1) , 
]

@dataclass
class TradingPortfolioBuilder:
    name        : str 
    alpha       : str
    universe    : str # 'csi800+csi1000+csi2000' , 'top-1000'
    top_num     : int = 50
    step        : int = 5

    init_value  : float = 1e6
    
    def __post_init__(self):
        exchange = ['SZSE','SSE'] if EXCLUDE_BSE else ['SZSE','SSE','BSE']
        self.candidates = DATAVENDOR.INFO.get_desc(exchange=exchange).index

        if self.step == 1:
            self.turn_control = 0.1
        elif self.step <= 4:
            self.turn_control = 0.1
        elif self.step <= 10:
            self.turn_control = 0.2
        elif self.step <= 30:
            self.turn_control = 0.5
        else:
            self.turn_control = 1.0
        self.buffer_zone = 0.8
        self.indus_control = 0.1

    def first_date(self , date : int) -> int:
        return self.last_date(date) <= 0
    
    def last_date(self , date : int) -> int:
        dates = PATH.dir_dates(PATH.trade_port.joinpath(self.name))
        dates = dates[dates < date]
        if len(dates) > 0:
            return dates.max()
        else:
            return -1
        
    def port_path(self , date : int):
        return PATH.trade_port.joinpath(self.name , f'{self.name}.{date}.csv')
        
    def updatable(self , date : int , force = False) -> bool:
        if force: return True
        if (last_date := self.last_date(date)) < 0: return True
        return CALENDAR.td(last_date , self.step) <= date

    def get_alpha(self , date : int) -> AlphaModel:
        if self.alpha in RegisteredModel.MODEL_DICT:
            reg_model = RegisteredModel.SelectModels(self.alpha)[0]
            dates = reg_model.pred_dates
            df = reg_model.load_pred(dates[dates <= date].max())
        elif '@' in self.alpha:
            exprs = self.alpha.split('@')
            if exprs[0] == 'sellside':
                dates = PATH.db_dates('sellside' , exprs[1])
                df = PATH.db_load('sellside' , exprs[1] , dates[dates <= date].max() , verbose = False).loc[:,['secid' , exprs[2]]]
            else:
                raise Exception(f'{self.alpha} is not a valid alpha')
        else:
            raise Exception(f'{self.alpha} is not a valid alpha')
        factor = StockFactor(df.assign(date = date))
        assert len(factor.factor_names) == 1 , f'expect 1 factor name , got {factor.factor_names}'
        return factor.normalize().alpha_model()

    def get_universe(self , date : int , safety = True) -> Portfolio:
        if self.universe.startswith('top'):
            top_num = int(self.universe.split('.')[0].removeprefix('top'))
            val = DATAVENDOR.TRADE.get_val(DATAVENDOR.TRADE.latest_date('val' , date)).sort_values('circ_mv' , ascending=False)
            val = val[val['secid'].isin(self.candidates)].iloc[:top_num].loc[:,['secid']].\
                reset_index().assign(date = date , name = self.universe)
            val['weight'] = 1 / len(val)
            pf = Portfolio.from_dataframe(val , name = self.universe)
        elif self.universe in Benchmark.AVAILABLES:
            pf = Benchmark(self.universe)
        elif '+' in self.universe:
            univs = [Benchmark(univ).get(date) for univ in self.universe.split('+')]
            pf = Portfolio.from_ports(Port.sum(univs) , name = self.universe)
        else:
            raise Exception(f'{self.universe} is not a valid benchmark')
        
        assert isinstance(pf , Portfolio) , f'expect Portfolio , got {type(pf)}'

        if safety:
            st_list = DATAVENDOR.INFO.get_st(date)['secid'].to_numpy()
            small_cp = DATAVENDOR.TRADE.get_val(date).query('close < 2.0')['secid'].to_numpy()

            pf = pf.exclude(st_list , True).exclude(small_cp , True)

        return pf

    def get_last_port(self , date : int) -> Portfolio:
        if (last_date := self.last_date(date)) > 0:
            df = pd.read_csv(self.port_path(last_date)).assign(date = last_date , name = self.name)
            port = Portfolio.from_dataframe(df)
        else:   
            port = Portfolio(self.name)
        return port

    def get_builder(self , date : int , reset_port = False) -> PortfolioBuilder:
        alpha = self.get_alpha(date)
        universe = self.get_universe(date)
        last_port = self.get_last_port(date)
        if reset_port:
            print(f'Beware: reset port for new build! {self.name}')
            last_port = Portfolio(self.name)
        else:
            last_port = self.get_last_port(date)
        builder = PortfolioBuilder('top' , alpha , universe , build_on = last_port , 
                                   n_best = self.top_num , turn_control = self.turn_control , 
                                   buffer_zone = self.buffer_zone , indus_control = self.indus_control).setup(0)
        return builder

    def build(self , date : int , force = False , reset_port = False , save = False):
        if not self.updatable(date , force):
            return pd.DataFrame()
        builder = self.get_builder(date , reset_port)
        pf = builder.build(date).port.to_dataframe()
        if self.first_date(date):
            pf['value'] = self.init_value
        if save:
            path = self.port_path(date)
            path.parent.mkdir(parents=True, exist_ok=True)
            pf.loc[:,['secid' , 'weight' , 'value']].to_csv(path)
        return pf

    
    @classmethod
    def update(cls , force = False , reset_ports : list[str] = []):
        date = CALENDAR.updated()
        pfs : list[pd.DataFrame] = []
        if reset_ports:
            ports , reset = reset_ports , True
        else:
            ports , reset = range(len(TRADING_PORTS)) , False
            
        updated_ports = []
        for port in ports:
            task = cls.SelectBuilder(port)
            pf = task.build(date , force , reset , save = True)
            if pf.empty: continue
            pfs.append(pf)
            updated_ports.append(task.name)
            
        if len(updated_ports) == 0: 
            print(f'No trading portfolios updated on {date}')
            return
        else:
            print(f'Trading portfolios updated on {date}: {updated_ports}')
        total_port = pd.concat([df for df in pfs])
        total_path = PATH.log_update.joinpath('trading_ports',f'trading_ports.{date}.csv')
        total_path.parent.mkdir(parents=True, exist_ok=True)
        total_port.to_csv(total_path)
        Email.attach(total_path)


    @classmethod
    def SelectBuilder(cls , which : str | int):
        if isinstance(which , str):
            port_names = [p for p in TRADING_PORTS if p[0] == which]
            assert len(port_names) == 1 , f'{which} is not a valid builder for ports {port_names}'
            return cls(*port_names[0])
        elif isinstance(which , int):
            assert 0 <= which < len(TRADING_PORTS) , f'expect 0 <= which < {len(TRADING_PORTS)} , got {which}'
            return cls(*TRADING_PORTS[which])
        else:
            raise Exception(f'{which} is not a valid builder')