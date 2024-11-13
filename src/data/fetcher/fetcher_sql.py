# %%
import platform , time
import pandas as pd
import numpy as np
import multiprocessing as mp  

from dataclasses import dataclass
from datetime import date , datetime
from sqlalchemy import create_engine , exc
from typing import ClassVar , Literal

from ...basic import PATH
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
        if self.driver : connect_url += f'?driver={self.driver}'
        return connect_url

    def engine(self):
        return create_engine(self.url)
    
    def connect(self , reconnect = False):
        if self.stay_connect and reconnect:
            self.close()
        if self.conn is None:
            engine = self.engine()
            conn = engine.connect()
            if self.stay_connect: self.conn = conn
        else:
            conn = self.conn
        return conn
    
    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @classmethod
    def default_connections(cls , keys = None):
        if isinstance(keys , str): keys = [keys]
        dft : dict[str,cls] = {
            'haitong': cls(
                dialect='mssql+pyodbc' , 
                username ='JSJJDataReader' ,
                password ='JSJJDataReader' ,
                host = 'rm-uf6777pp2098v0um6hm.sqlserver.rds.aliyuncs.com' ,
                port = 3433 ,
                database = 'daily_factor_db' ,
                driver='FreeTDS' if platform.system() == 'Linux' else 'SQL Server') ,
            'dongfang': cls(
                dialect='mysql+pymysql' , 
                username ='hfq_jiashi' ,
                password ='LTBi94we' ,
                host = '139.196.77.199' ,
                port = 81 ,
                database = 'model') ,
            'kaiyuan': cls(
                dialect='postgresql' , 
                username ='harvest_user' ,
                password ='harvest' ,
                host = '1.15.124.26' ,
                port = 5432 ,
                database = 'kyfactor') ,
            'huatai': cls(
                dialect='mysql+pymysql' ,
                username ='client_jinm' ,
                password ='password_JINMENG20240304' ,
                host = 'sh-cynosdbmysql-grp-jf1wgp6a.sql.tencentcdb.com' ,
                port = 22087 ,
                database = 'htfe_alpha_factors') ,
            'guojin': cls(
                dialect='mysql+pymysql' ,
                username ='jsfund' ,
                password ='Gjquant_js!' ,
                host = 'quantstudio.tpddns.cn' ,
                port = 3306 ,
                database = 'gjquant') ,
        }
        if keys is None: return dft
        return {k:dft[k] for k in keys}

@dataclass
class SQLFetcher:
    factor_src      : str
    factor_set      : str
    date_col        : str
    start_dt        : int
    date_fmt        : str | None = None
    factors         : list | None = None
    startdt_query   : str = 'select min({date_col}) from {factor_set}'
    default_query   : str = 'select * from {factor_set} where {date_col} between \'{start_dt}\' and \'{end_dt}\''

    DB_SRC : ClassVar[str] = 'sellside'
    FREQ   : ClassVar[str] = 'QE' if pd.__version__ >= '2.2.0' else 'Q'
    MAX_WORKERS: ClassVar[int] = 1

    def __post_init__(self):
        self.db_key = self.factor_src + '.' + self.factor_set
    
    def download(self , option : Literal['since' , 'dates' , 'all'] , connection : Connection ,
                 dates = [] , trace = 1 , start_dt = 20000101, end_dt = 99991231):
        if option == 'dates':
            date_intervals = [(d,d) for d in dates]
        else:
            start_dt = max(self.start_dt , start_dt)
            if option == 'since':
                old_dates = self.get_target_dates()
                if trace > 0 and len(old_dates) > trace: old_dates = old_dates[:-trace]
                if len(old_dates): 
                    last1_dt = min(self.date_offset(old_dates[-1],1,astype=int))
                    start_dt = max(start_dt , last1_dt)

            end_dt = min(end_dt , int(date.today().strftime('%Y%m%d')))
            date_intervals = self.date_seg(start_dt , end_dt)
        if not date_intervals: return 
        
        print(f'{time.ctime()} : {self.DB_SRC}/{self.db_key} from ' + 
            f'{date_intervals[0][0]} to {date_intervals[-1][1]}, total {len(date_intervals)} periods')

        if self.MAX_WORKERS == 1 or self.factor_src == 'dongfang':
            connection.stay_connect = True
            for inter in date_intervals:
                self.download_period(connection , *inter)
        else:
            connection.stay_connect = False
            with mp.Pool(processes=self.MAX_WORKERS) as pool:  
                pool.starmap(self.download_period, [(connection , *inter) for inter in date_intervals])
 
    def download_period(self , connection , start , end):
        print(f'Start {self.DB_SRC}/{self.db_key}:{start}-{end} ')
        t0 = time.time()
        df = self.query_default(connection , start , end)
        self.save_data(df)
        print(f'Done {self.DB_SRC}/{self.db_key}:{start}-{end}, cost {time.time()-t0:.1f} Secs')
        return True

    def query_start_dt(self , connection : Connection):
        conn = connection.connect()
        return self.make_query(self.startdt_query , conn)
    
    def query_default(self , connection : Connection , start_dt = 20230101 , end_dt = 20230131 , retry = 1):
        conn = connection.connect()
        i , df = 0 , None
        while i <= retry:
            try:
                df = self.make_query(self.default_query , conn , start_dt , end_dt)
            except exc.ResourceClosedError:
                print(f'{self.factor_src} Connection is closed, re-connect')
                conn = connection.connect(reconnect = True)
            i += 1
        if not connection.stay_connect: conn.close()
        df = self.df_process(df , self.factor_src , self.factor_set)
        return df

    def make_query(self , query , connection , start_dt = None , end_dt = None):
        if self.date_fmt is not None:
            if start_dt: start_dt = datetime.strptime(str(start_dt), '%Y%m%d').strftime(self.date_fmt)
            if end_dt:   end_dt   = datetime.strptime(str(end_dt)  , '%Y%m%d').strftime(self.date_fmt)
        kwargs = {'factor_src' : self.factor_src , 
                  'factor_set' : self.factor_set , 
                  'date_col'   : self.date_col ,
                  'start_dt'   : start_dt , 
                  'end_dt'     : end_dt}
        if self.factors is None:
            df = pd.read_sql_query(query.format(**kwargs) , connection)
        else:
            df = {factor:pd.read_sql_query(query.format(factor = factor , **kwargs) , 
                                           connection) for factor in self.factors}
        return df

    def get_target_dates(self):
        return PATH.db_dates(self.DB_SRC , self.db_key)

    def save_data(self , data):
        if len(data) == 0: return
        data = data.sort_values(['date' , 'secid']).set_index('date')
        for d in data.index.unique():
            data_at_d = data.loc[d]
            if len(data_at_d) == 0: continue
            PATH.db_save(data_at_d , self.DB_SRC , self.db_key , d , force_type='date')

    @classmethod
    def convert_id(cls , x):
        if isinstance(x , (bytes)):
            return int(x.decode('utf-8').split('.')[0].split('!')[0])
        if isinstance(x , (str)):
            return int(x.split('.')[0].split('!')[0])
        else:
            return type(x)([cls.convert_id(xx) for xx in x])

    @classmethod
    def df_process(cls , df , factor_src , factor_set):
        if df is None: return df
        if factor_src == 'haitong':
            assert isinstance(df , pd.DataFrame)
            df.columns = [col.lower() for col in df.columns.values]
            df = df.rename(columns={'s_info_windcode':'secid','trade_dt':'date'})
            df['date'] = df['date'].astype(int)
            df['secid'] = cls.convert_id(df['secid'])
            if factor_set == 'hf_factors':
                df = df.reset_index(drop = True)
            else:
                df.loc[:,'model'] = 'haitong_dl_' + df.loc[:,'model'].astype(str)
                df = df.pivot_table('f_value',['date','secid'],'model').reset_index()
        elif factor_src == 'dongfang':
            assert isinstance(df , pd.DataFrame)
            df.columns = [col.lower() for col in df.columns.values]
            df = df.rename(columns={'stockcode':'secid','ticker':'secid',
                                    'tradingdate':'date','trade_dt':'date' , 'trade_date':'date'})
            df['secid'] = df['secid'].str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df['date'] = df['date'].astype(str).str.replace('-','').astype(int)
        elif factor_src == 'kaiyuan':
            assert isinstance(df , dict)
            df0 = pd.DataFrame(columns = ['date','code']).astype(int)
            for k,subdf in df.items():
                subdf.rename(columns = {'factor':k})
                # print(subdf.iloc[:5])
                df0 = df0.merge(subdf.rename(columns = {'factor':k}),how='outer',on=['date','code'])
            df = df0.rename(columns={'code':'secid'})
            df['secid'] = df['secid'].str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df['date']  = df['date'].astype(int)
            df = df.set_index(['date','secid']).reset_index()
        elif factor_src == 'huatai':
            assert isinstance(df , dict)
            df0 = pd.DataFrame(columns = ['date','secid']).astype(int)
            for k , subdf in df.items():
                subdf = subdf.rename(columns = {'datetime':'date','instrument':'secid','value':k})
                df0 = df0.merge(subdf.rename(columns = {'factor':k}),how='outer',on=['date','secid'])
            df = df0
            df['secid'] = df['secid'].str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df['date']  = df['date'].astype(str).str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df = df.set_index(['date','secid']).reset_index()
        elif factor_src == 'guojin':
            ...

        df = cls.df_astype(df).fillna(np.nan)
        return df

    @classmethod
    def df_astype(cls , df , int_columns = ['date', 'secid']):
        int_columns = df.columns.values[df.columns.isin(int_columns)]
        float_columns = df.columns.difference(int_columns)  
        df[int_columns] = df[int_columns].astype(int)  
        df[float_columns] = df[float_columns].astype(float)
        return df

    @classmethod
    def date_seg(cls , start_dt , end_dt , astype = int):
        if start_dt >= end_dt: return []
        dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=cls.FREQ).strftime('%Y%m%d').astype(int)
        dt_starts = [cls.date_offset(start_dt) , *cls.date_offset(dt_list[:-1],1)]
        dt_ends = [*dt_list[:-1] , cls.date_offset(end_dt)]
        return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]
    
    @classmethod
    def date_between(cls , start_dt , end_dt , astype = int):
        if start_dt >= end_dt: return []
        dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=cls.FREQ).strftime('%Y%m%d').astype(int)
        return dt_list.values
    
    @staticmethod
    def date_offset(date , offset = 0 , astype = int):
        if isinstance(date , (np.ndarray,pd.Index,pd.Series,list,tuple,np.ndarray)):
            new_date = pd.DatetimeIndex(np.array(date).astype(str))
        else:
            new_date = pd.DatetimeIndex([str(date)])
        if offset == 0:
            new_date = new_date.strftime('%Y%m%d') #type: ignore
        else:
            new_date = (new_date + pd.DateOffset(offset)).strftime('%Y%m%d') #type: ignore
        new_date = new_date.astype(astype)
        return new_date.values

    @classmethod
    def default_factors(cls , keys = None):
        if isinstance(keys , str): keys = [keys]
        dft : dict[str,cls] = {
            #'haitong.hf_factors' :cls(    
            #    'haitong','hf_factors','trade_dt',20130101,
            #    startdt_query = 'select min(trade_dt) from daily_factor_db.dbo.JSJJHFFactors' ,
            #    default_query = ('select * from daily_factor_db.dbo.JSJJHFFactors t ' +             
            #                    'where t.trade_dt between \'{start_dt}\' and \'{end_dt}\'')) ,
            #'haitong.dl_factors' :cls(
            #    'haitong','dl_factors','trade_dt',20161230,
            #    startdt_query = 'select min(trade_dt) daily_factor_db.dbo.JSJJDeepLearnFactorsV2' ,
            #    default_query = ('select s_info_windcode , trade_dt , f_value , model ' +
            #                    'from daily_factor_db.dbo.JSJJDeepLearnFactorsV2 t where ' + 
            #                    't.trade_dt between \'{start_dt}\' and \'{end_dt}\'')) ,
            'dongfang.hfq_chars' :cls('dongfang','hfq_chars' ,'tradingdate',20050101,'%Y%m%d') ,
            'dongfang.l2_chars'  :cls('dongfang','l2_chars'  ,'trade_date' ,20130930,'%Y%m%d') ,
            'dongfang.ms_chars'  :cls('dongfang','ms_chars'  ,'trade_date' ,20050101,'%Y%m%d') ,
            'dongfang.order_flow':cls('dongfang','order_flow','trade_date' ,20130930,'%Y%m%d') ,
            'dongfang.gp'        :cls('dongfang','gp'        ,'tradingdate',20170101,'%Y-%m-%d') ,
            'dongfang.tra'       :cls('dongfang','tra'       ,'tradingdate',20200101,'%Y-%m-%d') ,
            'dongfang.hist'      :cls('dongfang','hist'      ,'tradingdate',20200101,'%Y-%m-%d') ,
            'dongfang.scores_v0' :cls('dongfang','scores_v0' ,'tradingdate',20171229,'%Y-%m-%d') ,
            'dongfang.factorvae' :cls('dongfang','factorvae' ,'tradingdate',20200101,'%Y-%m-%d') ,
            #'kaiyuan.positive' :cls(
            #    'kaiyuan','positive','date',20140130,
            #    factors = ['active_trading','apm','opt_synergy_effect','large_trader_ret_error',
            #            'offense_defense','high_freq_shareholder','pe_change',] ,
            #    startdt_query = 'select min({date_col}) from public.smart_money' ,
            #    default_query = ('select * from public.{factor} where {date_col} >= ' + 
            #                    '\'{start_dt}\' and {date_col} <= \'{end_dt}\'')) ,
            #'kaiyuan.negative' :cls(
            #    'kaiyuan','negative','date',20140130,
            #    factors = ['smart_money','ideal_vol','ideal_reverse','herd_effect','small_trader_ret_error',] ,
            #    startdt_query = 'select min({date_col}) from public.smart_money' ,
            #    default_query = ('select * from public.{factor} where {date_col} >= ' + 
            #                    '\'{start_dt}\' and {date_col} <= \'{end_dt}\'')) ,
            'huatai.dl_factors' :cls(
                'huatai','dl_factors','datetime',20170101,'%Y-%m-%d',
                factors = ['price_volume_nn','text_fadt_bert',] ,
                startdt_query = 'select min({date_col}) from price_volume_nn' ,
                default_query = ('select * from {factor} where {date_col} >= ' + 
                                '\'{start_dt}\' and {date_col} <= \'{end_dt}\'')) ,
        }
        if keys is None: return dft
        return {k:dft[k] for k in keys}

    @classmethod
    def factors_and_conns(cls , keys = None):
        factors = cls.default_factors(keys)
        conns   = Connection.default_connections()
        print(f'Connection and Factor preparation finished!')
        return [(factor , conns[factor.factor_src]) for factor in factors.values()]

    @classmethod
    def update_since(cls , trace = 0):
        prompt = f'{time.ctime()} : download since!'
        print(prompt)

        for factor , connection in cls.factors_and_conns():  
            factor.download('since' , connection , trace = trace)

    @classmethod
    def update_dates(cls , start_dt , end_dt):
        dates = cls.date_between(start_dt , end_dt)
        if len(dates) == 0: return NotImplemented

        for factor , connection in cls.factors_and_conns():  
            factor.download('dates' , connection , dates=dates)

    @classmethod
    def update_allaround(cls):

        prompt = f'{time.ctime()} : download allaround!'
        assert input(prompt + ', print "yes" to confirm!') == 'yes'
        assert input(prompt + ', print "yes" again to confirm!') == 'yes'
        print(prompt)

        for factor , connection in cls.factors_and_conns():  
            factor.download('all' , connection)
        
if __name__ == '__main__':
    from src.data.fetcher.fetcher_sql import SQLFetcher

    start_dt = 20100901 
    end_dt   = 20100915
    dates = SQLFetcher.date_between(start_dt , end_dt)

    factors_set = SQLFetcher.factors_and_conns('dongfang.hfq_chars')
    factor , connection = factors_set[0]
    df = factor.query_default(connection , start_dt , end_dt)
