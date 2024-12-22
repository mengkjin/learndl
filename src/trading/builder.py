import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.basic import CALENDAR , RegisteredModel , PATH , send_email
from src.data import DATAVENDOR
from src.factor.util import StockFactor , Benchmark , Portfolio , AlphaModel , Port
from src.factor.fmp import PortfolioBuilder

EXCLUDE_BSE = True
TRADING_PORTS = [
    ('test0' , 'gru_day_V1' , 'csi800+csi1000+csi2000') , 
    ('test1' , 'sellside@dongfang.scores_v0@avg' , 'top-1000') , 
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
        if last_date := self.last_date(date) < 0: return True
        return CALENDAR.td(last_date , self.step) >= date

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

    def get_universe(self , date : int) -> Portfolio:
        if self.universe.startswith('top'):
            top_num = int(self.universe.split('.')[0].removeprefix('top'))
            val = DATAVENDOR.TRADE.get_val(DATAVENDOR.TRADE.latest_date('val' , date)).sort_values('circ_mv' , ascending=False)
            val = val[val['secid'].isin(self.candidates)].iloc[:top_num].loc[:,['secid']].\
                reset_index().assign(date = date , name = self.universe)
            val['weight'] = 1 / len(val)
            return Portfolio.from_dataframe(val , name = self.universe)
        elif self.universe in Benchmark.AVAILABLES:
            return Benchmark(self.universe)
        elif '+' in self.universe:
            univs = [Benchmark(univ).get(date) for univ in self.universe.split('+')]
            return Portfolio.from_ports(Port.sum(univs) , name = self.universe)
        else:
            raise Exception(f'{self.universe} is not a valid benchmark')

    def get_last_port(self , date : int) -> Portfolio:
        if (last_date := self.last_date(date)) > 0:
            df = pd.read_csv(self.port_path(last_date)).assign(date = last_date , name = self.name)
            port = Portfolio.from_dataframe(df)
        else:   
            port = Portfolio(self.name)
        return port

    def get_builder(self , date : int , reset_port = False) -> PortfolioBuilder:
        alpha = self.get_alpha(date)
        univers = self.get_universe(date)
        last_port = self.get_last_port(date)
        if reset_port:
            print(f'Beware: reset port for new build! {self.name}')
            last_port = Portfolio(self.name)
        else:
            last_port = self.get_last_port(date)
        builder = PortfolioBuilder('top' , alpha , univers , build_on = last_port , n_best = self.top_num).setup(0)
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
    def update(cls , force = False , reset_ports : list[str] = [] , email = True):
        date = CALENDAR.updated()
        pfs : list[pd.DataFrame] = []
        if reset_ports:
            trading_port_names = [p[0] for p in TRADING_PORTS]
            assert all(p in trading_port_names for p in reset_ports) , \
                f'not all reset_ports in TRADING_PORTS: {np.setdiff1d(reset_ports , trading_port_names)}'
            ports , reset = [p for p in TRADING_PORTS if p[0] in reset_ports] , True
        else:
            ports , reset = TRADING_PORTS , False
            
        for port in ports:
            task = cls(*port)
            pfs.append(task.build(date , force , reset , save = True))
            
        total_port = pd.concat([df for df in pfs if not df.empty])
        if not total_port.empty:
            total_path = PATH.log_update.joinpath('trading_ports',f'trading_ports.{date}.csv')
            total_path.parent.mkdir(parents=True, exist_ok=True)
            total_port.to_csv(total_path)
            if email:
                send_email(f'Trading Portfolios Updated: {date}' , 'update successfully' , total_path)