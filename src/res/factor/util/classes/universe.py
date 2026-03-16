import pandas as pd
import numpy as np

from typing import Literal

from src.data import DATAVENDOR , DataBlock
from src.proj import DB
from .portfolio import Portfolio # , Port
from .benchmark import Benchmark

# class Universe:
#     """
#     Universe for factor model portfolio
#     Parameters:
#         name : str
#         -all : all stocks
#         -top<num> : top <num> stocks
#         -top-<num> : exclude bottom <num> stocks
#         -benchmark : benchmark
#         -<benchmark1>+<benchmark2>+... : combination of benchmarks
#     """

#     _cache_name : str | None = None
#     _cache_portfolio : Portfolio | None = None

#     def __init__(self , name : str):
#         self.name = name

#     def __repr__(self):
#         return f'Universe({self.name})'

#     @classmethod
#     def get_cache_portfolio(cls , name : str) -> Portfolio:
#         if cls._cache_name == name and cls._cache_portfolio is not None:
#             port = cls._cache_portfolio
#         else:
#             port = Portfolio(name)
#         return port

#     @classmethod
#     def set_cache_portfolio(cls , portfolio : Portfolio):
#         cls._cache_portfolio = portfolio
#         cls._cache_name = portfolio.name
    
#     def get_one(self , date : int , safety = True , exclude_bse = True) -> Portfolio:
#         exchange = ['SZSE','SSE'] if exclude_bse else ['SZSE','SSE','BSE']
#         candidates = DATAVENDOR.INFO.get_desc(exchange=exchange).index

#         if self.name == 'all':
#             pf = Portfolio.from_dataframe(pd.DataFrame({'secid' : candidates , 'date' : date , 'name' : self.name}) , name = self.name)
#         elif self.name.startswith('top'):
#             top_num = int(self.name.split('.')[0].removeprefix('top'))
#             val = DATAVENDOR.TRADE.get_val(DATAVENDOR.TRADE.latest_date('val' , date)).sort_values('circ_mv' , ascending=False)
#             val = val.query('secid in @candidates').iloc[:top_num].loc[:,['secid']].\
#                 reset_index().assign(date = date , name = self.name)
#             pf = Portfolio.from_dataframe(val , name = self.name)
#         elif self.name in Benchmark.AVAILABLES:
#             pf = Benchmark(self.name)
#         elif '+' in self.name:
#             univs = [Benchmark(univ).get(date) for univ in self.name.split('+')]
#             pf = Portfolio.from_ports(Port.sum(univs) , name = self.name)
#         else:
#             raise Exception(f'{self.name} is not a valid benchmark')
        
#         assert isinstance(pf , Portfolio) , f'expect Portfolio , got {type(pf)}'

#         if safety:
#             st_list = DATAVENDOR.INFO.get_st(date)['secid'].to_numpy()
#             small_cp = DATAVENDOR.TRADE.get_val(date).query('close < 2.0')['secid'].to_numpy()

#             pf = pf.filter_secid(st_list , exclude = True , inplace = True).filter_secid(small_cp , exclude = True , inplace = True)

#         return pf
    
#     def get_port(self , date : int , safety = True , exclude_bse = True) -> Port:
#         return self.get_one(date , safety , exclude_bse).get(date)
    
#     def get(self , date : Literal['all'] | int | list[int] | np.ndarray | None = None , *args , **kwargs) -> Portfolio:
#         if date is None:
#             date = []
#         elif isinstance(date , str) and date == 'all':
#             date = []
#         elif not isinstance(date , (list , np.ndarray , pd.Series , pd.Index)):
#             date = [date]
#         port = self.get_cache_portfolio(self.name)
#         for d in np.setdiff1d(date, port.port_date):
#             port.append(self.get_port(d) , ignore_name = True)
#         self.set_cache_portfolio(port)
#         return port

class Universe:
    """
    Universe for factor model portfolio
    Parameters:
        name : str
        -all : all stocks
        -top<num> : top <num> stocks
        -top-<num> : exclude bottom <num> stocks
        -benchmark : benchmark
        -<benchmark1>+<benchmark2>+... : combination of benchmarks
    """
    _cache_max_length : int = 5
    _cache_universes : dict[str , dict[int , pd.DataFrame]] = {}

    def __init__(self , name : str , exclusion : str = 'st_lowprice_bse_loser'):
        self.name = name
        self.exclusion = exclusion

        if len(self._cache_universes) <= self._cache_max_length and self.name not in self._cache_universes:
            self._cache_universes[self.name] = {}

    def __repr__(self):
        return f'Universe({self.name})'

    @classmethod
    def _combine_caches_with_exclusion(cls , dfs : list[pd.DataFrame] , exclusions : str = 'st_lowprice_bse_loser') -> pd.DataFrame:
        dfs = [df for df in dfs if df is not None and not df.empty]
        if not dfs:
            return pd.DataFrame(columns = ['date' , 'secid' , 'weight'])
        df = pd.concat(dfs)
        for exclusion in UniverseExclusions.get_possible_exclusions():
            if exclusion not in exclusions or exclusion not in df.columns:
                continue
            df = df.query(f'~{exclusion}').reset_index(drop = True)
            df['weight'] = 1.0 / len(df)
        if df.empty:
            return pd.DataFrame(columns = ['date' , 'secid' , 'weight'])
        df = df.loc[:,['date' , 'secid' , 'weight']]
        return df

    @classmethod
    def _get_cache_portfolio(cls , name : str , exclusions : str = 'st_lowprice_bse_loser') -> Portfolio:
        if name in cls._cache_universes:
            dfs = [df.assign(date = date) for date , df in cls._cache_universes[name].items()]
            df = cls._combine_caches_with_exclusion(dfs , exclusions = exclusions)
            return Portfolio.from_dataframe(df , name = name)
        else:
            return Portfolio(name)

    def get_universe_df(self , date : int) -> pd.DataFrame:
        path = DB.path('universe' , self.name , date)
        if path.exists():
            return DB.load_df(path)
        
        desc = DATAVENDOR.INFO.get_desc()
        secid = desc.index

        if self.name == 'all':
            df = pd.DataFrame({'date' : date , 'secid' : secid})
        elif self.name.startswith('top'):
            top_num = int(self.name.split('.')[0].removeprefix('top'))
            df = DATAVENDOR.TRADE.get_val(DATAVENDOR.TRADE.latest_date('val' , date)).sort_values('circ_mv' , ascending=False)
            df = df.query('secid in @secid').iloc[:top_num].loc[:,['secid']].\
                reset_index().assign(date = date).loc[:,['date' , 'secid']]
        elif self.name in Benchmark.AVAILABLES:
            df = Benchmark(self.name).get(date).to_dataframe().loc[:,['date' , 'secid']]
        elif '+' in self.name:
            dfs = [Benchmark(univ).get(date).to_dataframe().loc[:,['date' , 'secid']] for univ in self.name.split('+')]
            df = pd.concat(dfs)
        else:
            raise Exception(f'{self.name} is not a valid benchmark')
 
        for exclusion in UniverseExclusions.get_possible_exclusions():
            df[exclusion] = df['secid'].isin(UniverseExclusions.get_exclusion_list(exclusion , date))

        DB.save_df(df , path)
        return df
    
    def get(self , date : Literal['all'] | int | list[int] | np.ndarray | None = None , exclusions : str = 'st_lowprice_bse_loser') -> Portfolio:
        if date is None:
            date = []
        elif isinstance(date , str) and date == 'all':
            date = list(self._cache_universes[self.name].keys())
        elif not isinstance(date , (list , np.ndarray , pd.Series , pd.Index)):
            date = [date]
        dfs = []
        for d in date:
            if self.name in self._cache_universes:
                if d not in self._cache_universes[self.name]:
                    self._cache_universes[self.name][d] = self.get_universe_df(d)
                universe_df = self._cache_universes[self.name][d]
            else:
                universe_df = self.get_universe_df(d)
            dfs.append(universe_df)
        df = self._combine_caches_with_exclusion(dfs , exclusions = exclusions)
        return Portfolio.from_dataframe(df , name = self.name)

class UniverseExclusions:
    """
    Exclusions for universe
    """
    _possible_exclusions: list[str] = ['bse' , 'st' , 'lowprice' , 'loser']
    _cache_exclusions: dict[str , dict[int , np.ndarray]] = {}
    _block_cache: dict[str , DataBlock] = {}
    @classmethod
    def get_possible_exclusions(cls):
        """
        Get the possible exclusions
        """
        return cls._possible_exclusions

    @classmethod
    def get_exclusion_list(cls , exclusion : str , date : int):
        """
        Get the exclusion list for the given exclusion and date
        """
        assert exclusion in cls._possible_exclusions , f'{exclusion} is not a valid exclusion'
        if exclusion not in cls._cache_exclusions:
            cls._cache_exclusions[exclusion] = {}
        if date not in cls._cache_exclusions[exclusion]:
            if exclusion == 'bse':
                exc =  cls.get_bse(date)
            elif exclusion == 'st':
                exc = cls.get_st(date)
            elif exclusion == 'lowprice':
                exc = cls.get_lowprice(date)
            else:
                exc = cls.get_loser(date)
            cls._cache_exclusions[exclusion][date] = exc
        return cls._cache_exclusions[exclusion][date]

    @classmethod
    def get_bse(cls , date : int):
        """
        Beijing Stock Exchange
        """
        desc = DATAVENDOR.INFO.get_desc(date)
        if desc.empty:
            return np.array([]) 
        return desc.reset_index().query('exchange_name == "BSE"')['secid'].to_numpy()
    @classmethod
    def get_st(cls , date : int):
        """
        Suspended or ST stocks
        """
        st = DATAVENDOR.INFO.get_st(date)
        if st.empty:
            return np.array([])
        return st['secid'].to_numpy()
    @classmethod
    def get_lowprice(cls , date : int):
        """
        Low price stocks (less than 2.0)
        """
        val = DATAVENDOR.TRADE.get_val(date)
        if val.empty:
            return np.array([])
        return val.query('close < 2.0')['secid'].to_numpy()
    @classmethod
    def get_loser(cls , end_dt : int):
        """
        Loser stocks (in the last 50 weeks, never top 5% but at least twice bottom 5%)
        """
        df = DB.load('exposure' , 'week_rank_loser' , end_dt)
        if df.empty:
            return np.array([])
        return df.query('loser')['secid'].to_numpy()