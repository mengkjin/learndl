import re
import pandas as pd
import numpy as np
import multiprocessing as mp  

from string import Template
from dataclasses import dataclass
from datetime import datetime
from pypinyin import lazy_pinyin
from sqlalchemy import create_engine , exc
from typing import Any , ClassVar , Literal , Iterable

from src.proj import MACHINE , Logger , Duration , CALENDAR , DB
from src.data.util import secid_adjust

_connections : dict[str , dict[str , Any]] = {
    'haitong': {
        'dialect':'mssql+pyodbc' , 
        'username' : 'JSJJDataReader' ,
        'password' : 'JSJJDataReader' ,
        'host' : 'rm-uf6777pp2098v0um6hm.sqlserver.rds.aliyuncs.com' ,
        'port' : 3433 ,
        'database' : 'daily_factor_db' ,
        'driver':'FreeTDS' if MACHINE.is_linux else 'SQL Server'} ,
    'dongfang': {
        'dialect':'mysql+pymysql' , 
        'username' : 'hfq_jiashi' ,
        'password' : 'LTBi94we' ,
        'host' : '139.196.77.199' ,
        'port' : 81 ,
        'database' : 'model'} ,
    'dongfang2': {
        'dialect':'mysql+pymysql' , 
        'username' : 'score' ,
        'password' : 'dfquant' ,
        'host' : '139.196.77.199' ,
        'port' : 81 ,
        'database' : 'score'} ,
    'kaiyuan': {
        'dialect':'postgresql' , 
        'username' : 'harvest_user' ,
        'password' : 'harvest' ,
        'host' : '1.15.124.26' ,
        'port' : 5432 ,
        'database' : 'kyfactor'} ,
    'guosheng': {
        'dialect':'postgresql' , 
        'username' : 'jsquant' ,
        'password' : '3hkpe89ksq' ,
        'host' : 'frp-hub.top' ,
        'port' : 43230 ,
        'database' : 'gs_alpha'} ,
    'huatai': {
        'dialect':'mysql+pymysql' ,
        'username' : 'client_jinm' ,
        'password' : 'password_JINMENG20240304' ,
        'host' : 'sh-cynosdbmysql-grp-jf1wgp6a.sql.tencentcdb.com' ,
        'port' : 22087 ,
        'database' : 'htfe_alpha_factors'} ,
    'guojin': {
        'dialect':'mysql+pymysql' ,
        'username' : 'jsfund' ,
        'password' : 'Gjquant_js!' ,
        'host' : 'quantstudio.tpddns.cn' ,
        'port' : 3306 ,
        'database' : 'gjquant'} ,
    'huayuan': {
        'dialect':'mysql+pymysql' ,
        'username' : 'zhengjintao' ,
        'password' : 'hyquant' ,
        'host' : '47.100.228.38' ,
        'port' : 3306 ,
        'database' : 'deeplearning'} ,



}

_default_factors : dict[str,dict[str,Any]] = {
    # 'haitong.hf_factors' :{ 
    #     'factor_src' : 'haitong' ,
    #     'factor_set' : 'hf_factors' ,
    #     'date_col' : 'trade_dt' ,
    #     'start_dt' : 20130101 , 
    # } ,
    # 'haitong.dl_factors' :{
    #     'factor_src' : 'haitong' ,
    #     'factor_set' : 'dl_factors' ,
    #     'date_col' : 'trade_dt' ,
    #     'start_dt' : 20161230 ,
    'dongfang.hfq_chars' :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'hfq_chars' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20050101 ,
        'date_fmt' : '%Y%m%d' ,
    } ,
    'dongfang.l2_chars'  :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'l2_chars' ,
        'date_col' : 'trade_date' ,
        'start_dt' : 20130930 ,
        'date_fmt' : '%Y%m%d'
    } ,
    'dongfang.ms_chars'  :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'ms_chars' ,
        'date_col' : 'trade_date' ,
        'start_dt' : 20050101 ,
        'date_fmt' : '%Y%m%d' ,
    } ,
    'dongfang.order_flow':{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'order_flow' ,
        'date_col' : 'trade_date' ,
        'start_dt' : 20130930 ,
        'date_fmt' : '%Y%m%d'
    } ,
    'dongfang.gp'        :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'gp' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20170101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'dongfang.tra'       :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'tra' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20200101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'dongfang.hist'      :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'hist' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20200101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'dongfang.scores_v0' :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'scores_v0' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20171229 ,
        'end_dt' : 20251231 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'dongfang.scores_v2' :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'scores_v2' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20171229 ,
        'end_dt' : 20251231 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'dongfang.scores_v3' :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'scores_v3' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20171229 ,
        'end_dt' : 20251231 ,
        'date_fmt' : '%Y-%m-%d' ,
        'connection_key' : 'dongfang2' ,
    } ,
    'dongfang.factorvae' :{
        'factor_src' : 'dongfang' ,
        'factor_set' : 'factorvae' ,
        'date_col' : 'tradingdate' ,
        'start_dt' : 20200101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'huayuan.scores_v0' :{
        'factor_src' : 'huayuan' ,
        'factor_set' : 'scores_v0' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20260101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'huayuan.scores_v2' :{
        'factor_src' : 'huayuan' ,
        'factor_set' : 'scores_v2' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20260101 ,
        'date_fmt' : '%Y-%m-%d'
    } ,
    'huayuan.scores_v3' :{
        'factor_src' : 'huayuan' ,
        'factor_set' : 'scores_v3' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20260101 ,
        'date_fmt' : '%Y-%m-%d' ,
    } ,
    'huayuan.scores_v3_fast' :{
        'factor_src' : 'huayuan' ,
        'factor_set' : 'scores_v3_fast' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20171229 ,
        'date_fmt' : '%Y-%m-%d' ,
    } ,
    'huayuan.scores_v4_style' :{
        'factor_src' : 'huayuan' ,
        'factor_set' : 'scores_v4_style' ,
        'date_col' : 'trade_dt' ,
        'start_dt' : 20171229 ,
        'date_fmt' : '%Y-%m-%d' ,
    } ,
    # 'guosheng.gs_pv_set1':{
    #     'factor_src' : 'guosheng' ,
    #     'factor_set' : 'gs_pv_set1' ,
    #     'date_col' : 'date' ,
    #     'start_dt' : 20100129 ,
    #     'date_fmt' : '%Y%m%d'
    # } ,
    # 'kaiyuan.positive' :{
    #     'factor_src' : 'kaiyuan' ,
    #     'factor_set' : 'positive' ,
    #     'date_col' : 'date' ,
    #     'start_dt' : 20140130 ,
    #     'sub_factors' : ['active_trading','apm','opt_synergy_effect','large_trader_ret_error',
    #               'offense_defense','high_freq_shareholder','pe_change'] ,
    # } ,
    # 'kaiyuan.negative' :{
    #     'factor_src' : 'kaiyuan' ,
    #     'factor_set' : 'negative' ,
    #     'date_col' : 'date' ,
    #     'start_dt' : 20140130 ,
    #     'sub_factors' : ['smart_money','ideal_vol','ideal_reverse','herd_effect','small_trader_ret_error'] ,
    # } ,
    'huatai.dl_factors' :{
        'factor_src' : 'huatai' ,
        'factor_set' : 'dl_factors' ,
        'date_col' : 'datetime' ,
        'start_dt' : 20170101 ,
        'date_fmt' : '%Y-%m-%d' ,
        'sub_factors' : ['price_volume_nn','text_fadt_bert'] ,
    } ,
    'huatai.master_combined' :{
        'factor_src' : 'huatai' ,
        'factor_set' : 'master_combined' ,
        'date_col' : 'datetime' ,
        'start_dt' : 20170101 ,
        'date_fmt' : '%Y-%m-%d' ,
    } ,
    'huatai.fundamental_value' :{
        'factor_src' : 'huatai' ,
        'factor_set' : 'fundamental_value' ,
        'date_col' : 'datetime' ,
        'start_dt' : 20170101 ,
        'date_fmt' : '%Y-%m-%d' ,
   } ,
}
        

def factor_name_pinyin_conversion(text : str):
    text = text.replace('\'' , '2').replace('因子' , '')
    hanzi_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    hanzi_parts = hanzi_pattern.findall(text)
    
    pinyin_parts = ['_'.join(lazy_pinyin(part)) for part in hanzi_parts]
    
    for hanzi, pinyin in zip(hanzi_parts, pinyin_parts):
        text = text.replace(hanzi, pinyin)
    
    return text

def _date_offset(date : Any , offset : int = 0 , astype = int):
    iterable_input = isinstance(date , Iterable)
    date = pd.DatetimeIndex(np.array(date).astype(str) if iterable_input else [str(date)])
    dseries : pd.DatetimeIndex = (date + pd.DateOffset(n=offset))
    new_date = dseries.strftime('%Y%m%d').to_numpy(astype)
    return new_date if iterable_input else new_date[0]

def _date_seg(start_dt , end_dt , freq='QE' , astype : Any = int) -> list[tuple[int,int]]:
    if start_dt >= end_dt: 
        return []
    dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
    dt_starts = [_date_offset(start_dt) , *_date_offset(dt_list[:-1],1)]
    dt_ends = [*dt_list[:-1] , _date_offset(end_dt)]
    return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]

@dataclass
class Connection:
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
        connect_url = '{dialect}://{username}:{password}@{host}:{port}/{database}'.format(
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
    def default_connections(cls , keys = None):
        if keys is None:
            keys = list(_connections.keys())
        elif isinstance(keys , str): 
            keys = [keys]
        connections = {key:cls.connection(key) for key in keys}
        return connections

    @classmethod
    def connection(cls , key : str):
        return Connection(**_connections[key])

@dataclass
class SellsideSQLDownloader:
    factor_src      : str
    factor_set      : str
    date_col        : str
    start_dt        : int
    end_dt          : int = 99991231
    date_fmt        : str | None = None
    sub_factors     : list | None = None
    connection_key  : str = ''
    DB_SRC : ClassVar[str] = 'sellside'
    FREQ   : ClassVar[str] = 'QE' if pd.__version__ >= '2.2.0' else 'Q'
    MAX_WORKERS: ClassVar[int] = 1

    def __post_init__(self):
        self.db_key = self.factor_src + '.' + self.factor_set
        if self.connection_key == '':
            self.connection_key = self.factor_src

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
        print(sqlline)
        return sqlline

    def sqlline_factor_values(self , start_dt : int | str , end_dt : int | str , sub_factor : str | None = None) -> str:
        if self.factor_src == 'haitong':
            if self.factor_set == 'hf_factors':
                template = Template('select * from daily_factor_db.dbo.JSJJHFFactors t where t.${date_col} between \'${start_dt}\' and \'${end_dt}\'')
            else:
                template = Template('select s_info_windcode , trade_dt , f_value , model from daily_factor_db.dbo.JSJJHFFactors2 t where t.${date_col} between \'${start_dt}\' and \'${end_dt}\'')
        elif self.factor_src == 'huatai':
            if self.factor_set == 'dl_factors':
                template = Template('select * from ${sub_factor} where ${date_col} >= \'${start_dt}\' and ${date_col} <= \'${end_dt}\'')
            else:
                template = Template('select * from ${factor_set} where ${date_col} >= \'${start_dt}\' and ${date_col} <= \'${end_dt}\'')
        elif self.factor_src == 'kaiyuan':
            template = Template('select * from public.${sub_factor} where ${date_col} >= \'${start_dt}\' and ${date_col} <= \'${end_dt}\'')
        elif self.factor_src in ['dongfang' , 'huayuan' , 'guosheng' , 'guojin']:
            template = Template('select * from ${factor_set} where ${date_col} between \'${start_dt}\' and \'${end_dt}\'')
        else:
            raise ValueError(f'Undefined factor values query for factor source: {self.factor_src}')
        sqlline = template.substitute(date_col = self.date_col , factor_set = self.factor_set , start_dt = start_dt , end_dt = end_dt , sub_factor = sub_factor)
        print(sqlline)
        return sqlline

    def get_connection(self):
        return Connection.connection(self.connection_key)
    
    def download(self , option : Literal['since' , 'dates' , 'all'] ,
                 dates = [] , trace = 1 , start_dt = 20000101, end_dt = 99991231 , 
                 connection : Connection | None = None):
        if option == 'dates':
            date_intervals = [(d,d) for d in dates]
        else:
            start_dt = max(self.start_dt , start_dt)
            end_dt = min(self.end_dt , end_dt)
            if option == 'since':
                old_dates = DB.dates(self.DB_SRC , self.db_key)
                if trace > 0 and len(old_dates) > trace: 
                    old_dates = old_dates[:-trace]
                if len(old_dates): 
                    last1_dt = CALENDAR.cd(old_dates[-1],1)
                    start_dt = max(start_dt , last1_dt)

            end_dt = CALENDAR.td(min(end_dt , CALENDAR.update_to())).as_int()
            date_intervals = _date_seg(start_dt , end_dt , self.FREQ , astype=int)
        if not date_intervals: 
            return 

        if connection is None:
            connection = self.get_connection()
        
        start , end = date_intervals[0][0] , date_intervals[-1][1]
        Logger.stdout(f'Download: {self.DB_SRC}/{self.db_key} at {CALENDAR.dates_str([start , end])}, total {len(date_intervals)} periods' , indent = 1)

        if self.MAX_WORKERS == 1 or self.factor_src == 'dongfang':
            connection.stay_connect = True
            for inter in date_intervals:
                self.download_period(*inter , connection)
        else:
            connection.stay_connect = False
            with mp.Pool(processes=self.MAX_WORKERS) as pool:  
                pool.starmap(self.download_period, [(*inter , connection) for inter in date_intervals])
 
    def download_period(self , start , end , connection : Connection | None = None):
        if connection is None:
            connection = self.get_connection()
        t0 = datetime.now()
        df = self.query_factor_values(start , end , connection)
        if (num_dates := self.save_data(df)) > 0:
            Logger.success(f'Download {self.DB_SRC}/{self.db_key} at {CALENDAR.dates_str([start , end])}, total {num_dates} dates, time cost {Duration(since = t0)}' , indent = 1)
        else:
            Logger.skipping(f'No data for {self.DB_SRC}/{self.db_key} at {CALENDAR.dates_str([start , end])}' , indent = 1)
        return True

    def query_start_dt(self , connection : Connection | None = None):
        if connection is None:
            connection = self.get_connection()
        conn = connection.connect()
        return pd.read_sql_query(self.sqlline_start_dt() , conn)
    
    def query_factor_values(self , start_dt = 20230101 , end_dt = 20230131 , connection : Connection | None = None , retry = 1):
        if connection is None:
            connection = self.get_connection()
        conn = connection.connect()
        if self.date_fmt is not None:
            if start_dt: 
                start_dt = CALENDAR.format(start_dt , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
            if end_dt:   
                end_dt   = CALENDAR.format(end_dt   , old_fmt = '%Y%m%d' , new_fmt = self.date_fmt)
        
        i = 0
        while i <= retry:
            try:
                if self.sub_factors is None:
                    df_input = pd.read_sql_query(self.sqlline_factor_values(start_dt , end_dt) , conn)
                else:
                    df_input = {sub_factor:pd.read_sql_query(self.sqlline_factor_values(start_dt , end_dt , sub_factor) , conn) for sub_factor in self.sub_factors}
            except exc.ResourceClosedError:
                Logger.alert1(f'{self.factor_src} Connection is closed, re-connect')
                conn = connection.connect(reconnect = True)
            except Exception as e:
                Logger.error(f'In {self.__class__.__name__} : Error in query_factor_values: {e}')
                Logger.print_exc(e)
                break
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
        
        elif self.factor_src == 'huatai':
            assert isinstance(df_input , dict) , f'huatai {type(df_input)} is not a dict'
            df = pd.DataFrame(columns = pd.Index([self.date_col,'instrument'])).astype(int)
            for k , subdf in dfs.items():
                assert self.date_col in subdf.columns , subdf.columns
                df = df.merge(subdf.rename(columns = {'factor':k}),how='outer',on=[self.date_col,'instrument'])

        elif self.factor_src == 'guojin':
            ...
        elif self.factor_src == 'guosheng':
            assert isinstance(df_input , pd.DataFrame) , f'guosheng {type(df_input)} is not a pd.DataFrame'
            df.columns = [factor_name_pinyin_conversion(col) for col in df.columns]
            
        df = secid_adjust(df , drop_old=True)
        df = df.rename(columns = {self.date_col.lower():'date'})
        df['date'] = df['date'].astype(str).str.replace('[-.a-zA-Z]','',regex=True).astype(int)

        int_columns = df.columns.to_numpy(str)[df.columns.isin(['date', 'secid'])]
        flt_columns = df.columns.difference(int_columns.tolist())  
        df[int_columns] = df[int_columns].astype(int)  
        df[flt_columns] = df[flt_columns].astype(float)
        df = df.fillna(np.nan)
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
    def default_factors(cls , keys = None):
        if keys is None:
            keys = list(_default_factors.keys())
        elif isinstance(keys , str):
            keys = [keys]
        downloaders : dict[str,'SellsideSQLDownloader'] = {}
        for key in keys:
            downloaders[key] = cls(**_default_factors[key])
        return downloaders

    @classmethod
    def factors_and_conns(cls , keys = None):
        factors = cls.default_factors(keys)
        conns   = Connection.default_connections()
        return [(factor , conns[factor.connection_key]) for factor in factors.values()]

    @classmethod
    def update_since(cls , trace = 0 , keys = None):
        for factor , connection in cls.factors_and_conns(keys):  
            factor.download('since' , trace = trace , connection = connection)

    @classmethod
    def update_dates(cls , start_dt , end_dt , keys = None):
        dates = CALENDAR.td_within(start_dt , end_dt)
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

    start_dt = 20100901 
    end_dt   = 20100915
    dates = CALENDAR.td_within(start_dt , end_dt)

    factors_set = SellsideSQLDownloader.factors_and_conns('dongfang.hfq_chars')
    factor , connection = factors_set[0]
    df = factor.query_factor_values(start_dt , end_dt , connection)
