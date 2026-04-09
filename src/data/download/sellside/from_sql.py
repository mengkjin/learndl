from __future__ import annotations
import pandas as pd
import numpy as np
import threading
import sqlalchemy

from string import Template
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import create_engine , exc
from typing import Any , ClassVar , Literal , Iterable

from src.proj import MACHINE , Logger , Duration , CALENDAR , DB , Dates
from src.proj.util.func import parallel
from src.data.util import secid_adjust , chinese_to_pinyin

factor_settings : dict[str,tuple[tuple[Any,...],dict[str,Any]]] = {
    'dongfang.hfq_chars' : (('dongfang' , 'hfq_chars'  , 'tradingdate' , 20050101 , 99991231 , '%Y%m%d') , {}) ,
    'dongfang.l2_chars'  : (('dongfang' , 'l2_chars'   , 'trade_date'  , 20130930 , 99991231 , '%Y%m%d') , {}) ,
    'dongfang.ms_chars'  : (('dongfang' , 'ms_chars'   , 'trade_date'  , 20050101 , 99991231 , '%Y%m%d') , {}) ,
    'dongfang.order_flow': (('dongfang' , 'order_flow' , 'trade_date'  , 20130930 , 99991231 , '%Y%m%d') , {}) ,
    'dongfang.gp'        : (('dongfang' , 'gp'         , 'tradingdate' , 20170101 , 99991231 , '%Y-%m-%d') , {}) ,
    'dongfang.tra'       : (('dongfang' , 'tra'        , 'tradingdate' , 20200101 , 99991231 , '%Y-%m-%d') , {}) ,
    'dongfang.hist'      : (('dongfang' , 'hist'       , 'tradingdate' , 20200101 , 99991231 , '%Y-%m-%d') , {}) ,
    'dongfang.scores_v0' : (('dongfang' , 'scores_v0'  , 'tradingdate' , 20171229 , 20251231 , '%Y-%m-%d') , {}) ,
    'dongfang.scores_v2' : (('dongfang' , 'scores_v2'  , 'tradingdate' , 20171229 , 20251231 , '%Y-%m-%d') , {}) ,
    'dongfang.scores_v3' : (('dongfang' , 'scores_v3' ,  'tradingdate' , 20171229 , 20251231 , '%Y-%m-%d') , {'connection_key' : 'dongfang2'}) ,
    'dongfang.factorvae' : (('dongfang' , 'factorvae'  , 'tradingdate' , 20200101 , 99991231 , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v0'  : (('huayuan'  , 'scores_v0'  , 'trade_dt'    , 20260101 , 99991231 , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v2'  : (('huayuan'  , 'scores_v2'  , 'trade_dt'    , 20260101 , 99991231 , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v3'  : (('huayuan'  , 'scores_v3'  , 'trade_dt'    , 20260101 , 99991231 , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v3_fast'   : (('huayuan' , 'scores_v3_fast'   , 'trade_dt' , 20171229 , 99991231 , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v4_style'  : (('huayuan' , 'scores_v4_style'  , 'trade_dt' , 20171229 , 99991231 , '%Y-%m-%d') , {}) ,
    'huatai.dl_factors'        : (('huatai' , 'dl_factors'        , 'datetime' , 20170101 , 99991231 , '%Y-%m-%d') , {'sub_factors' : ['price_volume_nn','text_fadt_bert']}) ,
    'huatai.master_combined'   : (('huatai' , 'master_combined'   , 'datetime' , 20170101 , 99991231 , '%Y-%m-%d') , {}) ,
    'huatai.fundamental_value' : (('huatai' , 'fundamental_value' , 'datetime' , 20170101 , 99991231 , '%Y-%m-%d') , {}) ,
    # 'haitong.hf_factors' : (('haitong'  , 'hf_factors' , 'trade_dt'    , 20130101 , 99991231 , '%Y%m%d') , {}) ,
    # 'haitong.dl_factors' : (('haitong'  , 'dl_factors' , 'trade_dt'    , 20161230 , 99991231 , '%Y%m%d') , {}) ,
    # 'guosheng.gs_pv_set1': (('guosheng' , 'gs_pv_set1' , 'date'        , 20100129 , 99991231 , '%Y%m%d') , {}) ,
    # 'kaiyuan.positive'   : (('kaiyuan' , 'positive' , 'date' , 20140130 , None , '%Y%m%d') , 
    #                         {'sub_factors' : ['active_trading','apm','opt_synergy_effect','large_trader_ret_error','offense_defense','high_freq_shareholder','pe_change']}) ,
    # 'kaiyuan.negative'   : (('kaiyuan' , 'negative' , 'date' , 20140130 , None , '%Y%m%d') , 
    #                         {'sub_factors' : ['smart_money','ideal_vol','ideal_reverse','herd_effect','small_trader_ret_error']}) ,
}

MAX_MAX_WORKERS: int = 3
@dataclass
class Connection:
    """a connection to a sellside sql database"""
    dialect     : str
    username    : str
    password    : str
    host        : str
    port        : int | str
    database    : str 
    driver      : str | None = None
    stay_connect: bool = True

    mysql_timeout_connect : ClassVar[float] = 10.
    mysql_timeout_read  : ClassVar[float] = 300.

    def __post_init__(self):
        self.lock = threading.Lock()

    @property
    def conn(self) -> sqlalchemy.engine.base.Connection:
        if not hasattr(self , '_conn'):
            self._conn = self.engine().connect()
        return self._conn

    @property
    def url(self) -> str:
        connect_url = Template('${dialect}://${username}:${password}@${host}:${port}/${database}').substitute(
            dialect  = self.dialect,
            username = self.username,
            password = self.password,
            host     = self.host,
            port     = self.port,
            database = self.database,
        )
        if self.driver : 
            connect_url += f'?driver={self.driver}'
        return connect_url

    def engine(self) -> sqlalchemy.engine.base.Engine:
        if self.dialect.startswith('mysql'):
            connect_args = {'connect_timeout' : self.mysql_timeout_connect , 'read_timeout' : self.mysql_timeout_read}
        else:
            connect_args = {}
        return create_engine(self.url , connect_args = connect_args , pool_pre_ping = True , pool_recycle = 1800)

    def reconnect(self) -> Connection:
        with self.lock:
            self.close()
            self._conn = self.engine().connect()
        return self
    
    def close(self) -> Connection:
        if hasattr(self , '_conn'):
            self._conn.close()
        return self

    @classmethod
    def default_connections(cls , keys : list[str] | str | None = None):
        if keys is None:
            keys = cls.available_sources()
        elif isinstance(keys , str): 
            keys = [keys]
        connections = {key:cls.connection(key) for key in keys}
        return connections

    @classmethod
    def connection(cls , key : str):
        kwargs : dict[str,Any] = {**MACHINE.secret['accounts']['sellside'][key]}
        type : str = kwargs.pop('type')
        assert type.startswith('sql') , f'{key} is not a valid sql source'
        if type.endswith('.disabled'):
            Logger.alert1(f'{key} is disabled')
        
        for system in ['linux' , 'windows' , 'macos']:
            driver : str | None = kwargs.pop(f'driver.{system}' , None)
            if driver and system == MACHINE.system_name:
                kwargs['driver'] = driver
        
        return Connection(**kwargs)

    @classmethod
    def available_sources(cls) -> list[str]:
        return [key for key , value in MACHINE.secret['accounts']['sellside'].items() if value['type'] == 'sql']

    @classmethod
    def test_all_connections(cls) -> None:
        for key in cls.available_sources():
            try:
                connection = cls.connection(key)
                connection.conn
                connection.close()
                Logger.success(f'{key} connection test passed')
            except Exception as e:
                Logger.error(f'{key} connection test failed: {e}')
                Logger.print_exc(e)

@dataclass
class SellsideSQLDownloader:
    """a downloader for sellside sql data"""
    factor_src      : str
    factor_set      : str
    date_col        : str
    start_date      : int
    end_date        : int = 99991231
    date_fmt        : str | None = None
    sub_factors     : list | None = None
    connection_key  : str = ''

    DB_SRC : ClassVar[str] = 'sellside'
    MAX_WORKERS: ClassVar[int] = min(1 , MAX_MAX_WORKERS)

    def __post_init__(self):
        assert 19900101 <= self.start_date <= self.end_date <= 99991231 , f'start_date {self.start_date} must be greater than 19900101 and less than end_date {self.end_date} and less than 99991231'
        
    @property
    def db_key(self) -> str:
        return f'{self.factor_src}.{self.factor_set}'

    @property
    def use_connection_key(self) -> str:
        return self.connection_key if self.connection_key else self.factor_src

    @property
    def connection(self) -> Connection:
        if not hasattr(self , '_connection'):
            self._connection = Connection.connection(self.use_connection_key)
        return self._connection

    @property
    def conn(self) -> sqlalchemy.engine.base.Connection:
        return self.connection.conn

    def sqlline_start_dt(self) -> str:
        if self.factor_src == 'haitong':
            if self.factor_set == 'hf_factors':
                template = Template('select min(${date_col}) from daily_factor_db.dbo.JSJJHFFactors')
            else:
                template = Template('select min(${date_col}) from daily_factor_db.dbo.JSJJHFFactors2')
        elif self.factor_src == 'huatai':
            if self.factor_set == 'dl_factors':
                template = Template('select min(${date_col}) from price_volume_nn')
            else:
                template = Template('select min(${date_col}) from ${factor_set}')
        elif self.factor_src == 'kaiyuan':
            template = Template('select min(${date_col}) from public.smart_money')
        elif self.factor_src in ['dongfang' , 'huayuan' , 'guosheng' , 'guojin']:
            template = Template('select min(${date_col}) from ${factor_set}')
        else:
            raise ValueError(f'Undefined startdt query for factor source: {self.factor_src}')
        sqlline = template.substitute(date_col = self.date_col , factor_set = self.factor_set)
        return sqlline

    def sqlline_factor_values(self , start : int | str , end : int | str , sub_factor : str | None = None) -> str:
        if self.factor_src == 'haitong':
            if self.factor_set == 'hf_factors':
                template = Template('select * from daily_factor_db.dbo.JSJJHFFactors t where t.${date_col} between \'${start}\' and \'${end}\'')
            else:
                template = Template('select s_info_windcode , trade_dt , f_value , model from daily_factor_db.dbo.JSJJHFFactors2 t where t.${date_col} between \'${start}\' and \'${end}\'')
        elif self.factor_src == 'huatai':
            if self.factor_set == 'dl_factors':
                template = Template('select * from ${sub_factor} where ${date_col} >= \'${start}\' and ${date_col} <= \'${end}\'')
            else:
                template = Template('select * from ${factor_set} where ${date_col} >= \'${start}\' and ${date_col} <= \'${end}\'')
        elif self.factor_src == 'kaiyuan':
            template = Template('select * from public.${sub_factor} where ${date_col} >= \'${start}\' and ${date_col} <= \'${end}\'')
        elif self.factor_src in ['dongfang' , 'huayuan' , 'guosheng' , 'guojin']:
            template = Template('select * from ${factor_set} where ${date_col} between \'${start}\' and \'${end}\'')
        else:
            raise ValueError(f'Undefined factor values query for factor source: {self.factor_src}')
        sqlline = template.substitute(date_col = self.date_col , factor_set = self.factor_set , start = start , end = end , sub_factor = sub_factor)
        return sqlline

    def get_connection(self):
        return Connection.connection(self.use_connection_key)
    
    def download(self , option : Literal['since' , 'dates' , 'all'] ,
                 dates : Iterable[int] = () , trace = 1 , start = 20000101, end = 99991231):
        if option == 'dates':
            date_intervals = [(d,d) for d in dates]
        else:
            start = max(self.start_date , start)
            end = min(self.end_date , end)
            if option == 'since':
                old_dates = DB.dates(self.DB_SRC , self.db_key)
                if trace > 0 and len(old_dates) > trace: 
                    old_dates = old_dates[:-trace]
                if len(old_dates): 
                    last1_dt = CALENDAR.cd(old_dates[-1],1)
                    start = max(start , last1_dt)

            end = CALENDAR.td(min(end , CALENDAR.update_to())).as_int()
            date_intervals = CALENDAR.range_segments(start , end , 'td' , 60)
        if not date_intervals: 
            return 
        
        start , end = date_intervals[0][0] , date_intervals[-1][1]
        Logger.stdout(f'Download: {self.DB_SRC}/{self.db_key} at {Dates(start , end)}, total {len(date_intervals)} periods' , indent = 1 , vb_level = 3)

        if self.MAX_WORKERS == 1 or self.factor_src == 'dongfang': 
            for inter in date_intervals:
                self.download_period(*inter , True)
        else:
            parallel([(self.download_period, (*inter , True)) for inter in date_intervals], max_workers = self.MAX_WORKERS)
 
    def download_period(self , start : int , end : int , reconnect = False):
        t0 = datetime.now()
        try:
            df = self.query_factor_values(start , end , reconnect = reconnect)
        except Exception as e:
            Logger.error(f'In {self.__class__.__name__} : Error in download_period of {self.DB_SRC}/{self.db_key} at {Dates(start , end)}: {e}')
            Logger.print_exc(e)
            self.connection.reconnect()
            return False
        if (num_dates := self.save_data(df)) > 0:
            Logger.success(f'Download {self.DB_SRC}/{self.db_key} at {Dates(start , end)}, total {num_dates} dates, time cost {Duration(since = t0)}' , indent = 1 , vb_level = 1)
        else:
            Logger.skipping(f'No data for {self.DB_SRC}/{self.db_key} at {Dates(start , end)}' , indent = 1)
        return True

    def query_start_dt(self):
        return pd.read_sql_query(self.sqlline_start_dt() , self.conn)
    
    def query_factor_values(self , start : int = 20230101 , end : int = 20230131 ,  reconnect = False, attempts : int = 1):
        if reconnect:
            self.connection.reconnect()
        start_str = CALENDAR.reformat(start , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
        end_str   = CALENDAR.reformat(end   , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
        
        df_input = None
        i = 0
        while i <= attempts:
            try:
                if self.sub_factors is None:
                    df_input = pd.read_sql_query(self.sqlline_factor_values(start_str , end_str) , self.conn)
                else:
                    df_input = {sub_factor:pd.read_sql_query(self.sqlline_factor_values(start_str , end_str , sub_factor) , self.conn) for sub_factor in self.sub_factors}
            except exc.ResourceClosedError:
                Logger.alert1(f'{self.factor_src} Connection is closed, re-connect')
                self.connection.reconnect()
            i += 1
        df = self.df_process(df_input)
        return df

    def df_process(self , df_input : pd.DataFrame | dict[Any, pd.DataFrame] | None):
        if df_input is None: 
            return None
        elif isinstance(df_input , pd.DataFrame):
            df = df_input
            df.columns = [col.lower() for col in df.columns]
        elif isinstance(df_input , dict):
            dfs = df_input
            for k,v in dfs.items():
                v.columns = [col.lower() for col in v.columns]

        if self.factor_src == 'haitong':
            assert isinstance(df_input , pd.DataFrame) , f'haitong {type(df_input)} is not a pd.DataFrame'
            df = secid_adjust(df , 's_info_windcode' , decode_first=True)
            if self.factor_set == 'hf_factors':
                df = df.reset_index(drop = True)
            else:
                df.loc[:,'model'] = 'haitong_dl_' + df.loc[:,'model'].astype(str)
                df = df.pivot_table('f_value',['date','secid'],'model').reset_index()
        elif self.factor_src == 'dongfang':
            assert isinstance(df_input , pd.DataFrame) , f'dongfang {type(df_input)} is not a pd.DataFrame'
        elif self.factor_src == 'huayuan':
            assert isinstance(df_input , pd.DataFrame) , f'huayuan {type(df_input)} is not a pd.DataFrame'
            df = df.rename(columns = {'factor_value':self.factor_set})
            
        elif self.factor_src == 'kaiyuan':
            assert isinstance(df_input , dict) , f'kaiyuan {type(df_input)} is not a dict'
            df = pd.DataFrame(columns = pd.Index([self.date_col,'code'])).astype(int)
            for k,subdf in dfs.items():
                df = df.merge(subdf.rename(columns = {'factor':k}),how='outer',on=[self.date_col,'code'])
        elif self.factor_src == 'huatai' and self.factor_set == 'dl_factors':
            assert isinstance(df_input , dict) , f'huatai {type(df_input)} is not a dict'
            df = pd.DataFrame(columns = pd.Index([self.date_col,'instrument'])).astype(int)
            for k , subdf in dfs.items():
                assert self.date_col in subdf.columns , subdf.columns
                df = df.merge(subdf.rename(columns = {'value':k}),how='outer',on=[self.date_col,'instrument'])

        elif self.factor_src == 'huatai':
            assert isinstance(df_input , pd.DataFrame) , f'huatai {type(df_input)} is not a pd.DataFrame'
            df = df.rename(columns = {'value':self.factor_set})

        elif self.factor_src == 'guojin':
            ...
        elif self.factor_src == 'guosheng':
            assert isinstance(df_input , pd.DataFrame) , f'guosheng {type(df_input)} is not a pd.DataFrame'
            df.columns = [chinese_to_pinyin(col) for col in df.columns]
            
        df = secid_adjust(df , drop_old=True)
        df = df.rename(columns = {self.date_col.lower():'date'})
        df['date'] = df['date'].astype(str).str.replace('[-.a-zA-Z]','',regex=True).astype(int)

        int_columns = df.columns.to_numpy(str)[df.columns.isin(['date', 'secid'])]
        flt_columns = df.columns.difference(int_columns.tolist())  
        df = df.astype({col:int for col in int_columns} | {col:float for col in flt_columns}).fillna(np.nan)
        return df

    def save_data(self , data : pd.DataFrame | None) -> int:
        if data is None or data.empty or len(data) == 0: 
            return 0
        data = data.sort_values(['date' , 'secid']).set_index('date')
        status = 0
        for d in data.index.unique():
            data_at_d = data.loc[d]
            if len(data_at_d) == 0: 
                continue
            DB.save(data_at_d , self.DB_SRC , self.db_key , d , indent = 2 , vb_level = 3)
            status += 1
        return status

    @classmethod
    def default_factors(cls , keys : list[str] | str | None = None) -> dict[str,SellsideSQLDownloader]:
        if keys is None:
            keys = cls.available_factors()
        elif isinstance(keys , str):
            keys = [keys]
        downloaders : dict[str,SellsideSQLDownloader] = {}
        for key in keys:
            args , kwargs = factor_settings[key]
            downloaders[key] = cls(*args , **kwargs)
        return downloaders

    @classmethod
    def available_factors(cls) -> list[str]:
        return list(factor_settings.keys())

    @classmethod
    def test_all_factors(cls) -> None:
        for key , downloader in cls.factors_downloaders().items():
            try:
                downloader.query_start_dt()
                Logger.success(f'{downloader.factor_src}.{downloader.factor_set} start dt query passed')
            except Exception as e:
                Logger.error(f'{downloader.factor_src}.{downloader.factor_set} start dt query failed: {e}')

    @classmethod
    def factors_downloaders(cls , keys = None) -> dict[str,SellsideSQLDownloader]:
        return cls.default_factors(keys)
        
    @classmethod
    def update_since(cls , trace = 0 , keys = None):
        for key , downloader in cls.factors_downloaders(keys).items():  
            downloader.download('since' , trace = trace)

    @classmethod
    def update_dates(cls , start , end , keys = None):
        dates = CALENDAR.range(start , end , 'td')
        if len(dates) == 0: 
            return NotImplemented

        for key , downloader in cls.factors_downloaders(keys).items():  
            downloader.download('dates' , dates=dates)

    @classmethod
    def update_allaround(cls , keys = None):

        prompt = f'Download: {cls.__name__} allaround!'
        assert (x := input(prompt + ', input "yes" to confirm!')) == 'yes' , f'input {x} is not "yes"'
        assert (x := input(prompt + ', input "yes" again to confirm!')) == 'yes' , f'input {x} is not "yes"'
        Logger.note(prompt)

        for key , downloader in cls.factors_downloaders(keys).items():  
            downloader.download('all')

    @classmethod
    def update(cls):
        Logger.note(f'Download: {cls.__name__} since last update!')
        cls.update_since(trace = 0)
        
if __name__ == '__main__':
    from src.data.download.sellside.from_sql import SellsideSQLDownloader

    start = 20100901 
    end   = 20100915
    dates = CALENDAR.range(start , end , 'td')

    downloader = SellsideSQLDownloader.factors_downloaders('dongfang.hfq_chars')['dongfang.hfq_chars']
    df = downloader.query_factor_values(start , end)
