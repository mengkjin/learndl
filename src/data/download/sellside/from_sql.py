import pandas as pd
import numpy as np
import multiprocessing as mp  

from string import Template
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import create_engine , exc
from typing import Any , ClassVar , Literal , Iterable

from src.proj import MACHINE , Logger , Duration , CALENDAR , DB , Dates
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
    'huayuan.scores_v3_fast'  : (('huayuan' , 'scores_v3_fast'  , 'trade_dt' , 20171229 , None , '%Y-%m-%d') , {}) ,
    'huayuan.scores_v4_style' : (('huayuan' , 'scores_v4_style' , 'trade_dt' , 20171229 , None , '%Y-%m-%d') , {}) ,
    'huatai.dl_factors'        : (('huatai' , 'dl_factors'        , 'datetime' , 20170101 , None , '%Y-%m-%d') , {'sub_factors' : ['price_volume_nn','text_fadt_bert']}) ,
    'huatai.master_combined'   : (('huatai' , 'master_combined'   , 'datetime' , 20170101 , None , '%Y-%m-%d') , {}) ,
    'huatai.fundamental_value' : (('huatai' , 'fundamental_value' , 'datetime' , 20170101 , None , '%Y-%m-%d') , {}) ,
    # 'haitong.hf_factors' : (('haitong'  , 'hf_factors' , 'trade_dt'    , 20130101 , 99991231 , '%Y%m%d') , {}) ,
    # 'haitong.dl_factors' : (('haitong'  , 'dl_factors' , 'trade_dt'    , 20161230 , 99991231 , '%Y%m%d') , {}) ,
    # 'guosheng.gs_pv_set1': (('guosheng' , 'gs_pv_set1' , 'date'        , 20100129 , 99991231 , '%Y%m%d') , {}) ,
    # 'kaiyuan.positive'   : (('kaiyuan' , 'positive' , 'date' , 20140130 , None , '%Y%m%d') , 
    #                         {'sub_factors' : ['active_trading','apm','opt_synergy_effect','large_trader_ret_error','offense_defense','high_freq_shareholder','pe_change']}) ,
    # 'kaiyuan.negative'   : (('kaiyuan' , 'negative' , 'date' , 20140130 , None , '%Y%m%d') , 
    #                         {'sub_factors' : ['smart_money','ideal_vol','ideal_reverse','herd_effect','small_trader_ret_error']}) ,
}

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

    def __post_init__(self):
        self.conn = None

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

    def engine(self):
        return create_engine(self.url)
    
    def connect(self , reconnect = False):
        if self.stay_connect and reconnect:
            self.close()
        if self.conn is None:
            engine = self.engine()
            conn = engine.connect()
            if self.stay_connect: 
                self.conn = conn
        else:
            conn = self.conn
        return conn
    
    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
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
        kwargs : dict[str,Any] = MACHINE.secret['accounts']['sellside'][key]
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
                connection.connect()
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
    MAX_WORKERS: ClassVar[int] = 1

    def __post_init__(self):
        assert 19900101 <= self.start_date <= self.end_date <= 99991231 , f'start_date {self.start_date} must be greater than 19900101 and less than end_date {self.end_date} and less than 99991231'
        assert self.date_fmt is None or len(self.date_fmt) == 8 , f'date_fmt {self.date_fmt} must be 8 characters'
        assert self.sub_factors is None or all(isinstance(sub_factor , str) for sub_factor in self.sub_factors) , f'sub_factors {self.sub_factors} must be a list of strings'
        assert self.connection_key is None or isinstance(self.connection_key , str) , f'connection_key {self.connection_key} must be a string'
        assert self.connection_key is None or self.connection_key in Connection.available_sources() , f'connection_key {self.connection_key} must be a valid connection key'
        assert self.connection_key is None or self.connection_key in Connection.available_sources() , f'connection_key {self.connection_key} must be a valid connection key'

    @property
    def db_key(self) -> str:
        return f'{self.factor_src}.{self.factor_set}'

    @property
    def use_connection(self) -> str:
        return self.connection_key if self.connection_key else self.factor_src

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
        return Connection.connection(self.use_connection)
    
    def download(self , option : Literal['since' , 'dates' , 'all'] ,
                 dates : Iterable[int] = () , trace = 1 , start = 20000101, end = 99991231 , 
                 connection : Connection | None = None):
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

        if connection is None:
            connection = self.get_connection()
        
        start , end = date_intervals[0][0] , date_intervals[-1][1]
        Logger.stdout(f'Download: {self.DB_SRC}/{self.db_key} at {Dates(start , end)}, total {len(date_intervals)} periods' , indent = 1 , vb_level = 3)

        if self.MAX_WORKERS == 1 or self.factor_src == 'dongfang':
            connection.stay_connect = True
            for inter in date_intervals:
                self.download_period(*inter , connection)
        else:
            connection.stay_connect = False
            with mp.Pool(processes=self.MAX_WORKERS) as pool:  
                pool.starmap(self.download_period, [(*inter , connection) for inter in date_intervals])
 
    def download_period(self , start : int , end : int , connection : Connection | None = None):
        if connection is None:
            connection = self.get_connection()
        t0 = datetime.now()
        try:
            df = self.query_factor_values(start , end , connection)
        except Exception as e:
            Logger.error(f'In {self.__class__.__name__} : Error in download_period of {self.DB_SRC}/{self.db_key} at {Dates(start , end)}: {e}')
            Logger.print_exc(e)
            return False
        if (num_dates := self.save_data(df)) > 0:
            Logger.success(f'Download {self.DB_SRC}/{self.db_key} at {Dates(start , end)}, total {num_dates} dates, time cost {Duration(since = t0)}' , indent = 1 , vb_level = 1)
        else:
            Logger.skipping(f'No data for {self.DB_SRC}/{self.db_key} at {Dates(start , end)}' , indent = 1)
        return True

    def query_start_dt(self , connection : Connection | None = None):
        if connection is None:
            connection = self.get_connection()
        conn = connection.connect()
        return pd.read_sql_query(self.sqlline_start_dt() , conn)
    
    def query_factor_values(self , start : int = 20230101 , end : int = 20230131 , connection : Connection | None = None , retry = 1):
        if connection is None:
            connection = self.get_connection()
        conn = connection.connect()
        start_str = CALENDAR.reformat(start , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
        end_str   = CALENDAR.reformat(end   , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
        
        df_input = None
        i = 0
        while i <= retry:
            try:
                if self.sub_factors is None:
                    df_input = pd.read_sql_query(self.sqlline_factor_values(start_str , end_str) , conn)
                else:
                    df_input = {sub_factor:pd.read_sql_query(self.sqlline_factor_values(start_str , end_str , sub_factor) , conn) for sub_factor in self.sub_factors}
            except exc.ResourceClosedError:
                Logger.alert1(f'{self.factor_src} Connection is closed, re-connect')
                conn = connection.connect(reconnect = True)
            i += 1
        if not connection.stay_connect: 
            conn.close()
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
    def default_factors(cls , keys : list[str] | str | None = None):
        if keys is None:
            keys = cls.available_factors()
        elif isinstance(keys , str):
            keys = [keys]
        downloaders : dict[str,'SellsideSQLDownloader'] = {}
        for key in keys:
            args , kwargs = factor_settings[key]
            downloaders[key] = cls(*args , **kwargs)
        return downloaders

    @classmethod
    def available_factors(cls) -> list[str]:
        return list(factor_settings.keys())

    @classmethod
    def test_all_factors(cls) -> None:
        for factor , connection in cls.factors_and_conns():
            try:
                factor.query_start_dt(connection = connection)
                Logger.success(f'{factor.factor_src}.{factor.factor_set} start dt query passed')
            except Exception as e:
                Logger.error(f'{factor.factor_src}.{factor.factor_set} start dt query failed: {e}')

    @classmethod
    def factors_and_conns(cls , keys = None):
        factors = cls.default_factors(keys)
        conns   = Connection.default_connections()
        return [(factor , conns[factor.use_connection]) for factor in factors.values()]

    @classmethod
    def update_since(cls , trace = 0 , keys = None):
        for factor , connection in cls.factors_and_conns(keys):  
            factor.download('since' , trace = trace , connection = connection)

    @classmethod
    def update_dates(cls , start , end , keys = None):
        dates = CALENDAR.range(start , end , 'td')
        if len(dates) == 0: 
            return NotImplemented

        for factor , connection in cls.factors_and_conns(keys):  
            factor.download('dates' , dates=dates , connection = connection)

    @classmethod
    def update_allaround(cls , keys = None):

        prompt = f'Download: {cls.__name__} allaround!'
        assert (x := input(prompt + ', input "yes" to confirm!')) == 'yes' , f'input {x} is not "yes"'
        assert (x := input(prompt + ', input "yes" again to confirm!')) == 'yes' , f'input {x} is not "yes"'
        Logger.note(prompt)

        for factor , connection in cls.factors_and_conns(keys):  
            factor.download('all' , connection = connection)

    @classmethod
    def update(cls):
        Logger.note(f'Download: {cls.__name__} since last update!')
        cls.update_since(trace = 0)
        
if __name__ == '__main__':
    from src.data.download.sellside.from_sql import SellsideSQLDownloader

    start = 20100901 
    end   = 20100915
    dates = CALENDAR.range(start , end , 'td')

    factors_set = SellsideSQLDownloader.factors_and_conns('dongfang.hfq_chars')
    factor , connection = factors_set[0]
    df = factor.query_factor_values(start , end , connection)
