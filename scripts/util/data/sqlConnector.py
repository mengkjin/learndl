# %%
from sqlalchemy import create_engine , exc
from datetime import date , datetime

import traceback
import pandas as pd
import numpy as np
import os , time
import platform
from .DataTank import DataTank , Data1D
from .DataUpdater import get_date_groups,get_db_file,get_date_groups,get_db_path,outer_path_join,get_date_groups

_db_key = 'SellSideFactors'

# %%
connect_dict = {
    'haitong':{
        'dialect'  : 'mssql+pyodbc' ,
        'driver'   : 'FreeTDS' if platform.system() == 'Linux' else 'SQL Server',
        'username' : 'JSJJDataReader' ,
        'password' : 'JSJJDataReader' ,
        'host'     : 'rm-uf6777pp2098v0um6hm.sqlserver.rds.aliyuncs.com' ,
        'port'     : 3433 ,
        'database' : 'daily_factor_db' ,
    },
    'dongfang':{
        'dialect'  : 'mysql+pymysql' ,
        'username' : 'hfq_jiashi' ,
        'password' : 'LTBi94we' ,
        'host'     : '139.196.77.199' ,
        'port'     : 81 ,
        'database' : 'model' ,
    },
    'kaiyuan':{
        'dialect'  : 'postgresql' ,
        'username' : 'harvest_user' ,
        'password' : 'harvest' ,
        'host'     : '1.15.124.26' ,
        'port'     : 5432 ,
        'database' : 'kyfactor' ,
    },
    'guojin':{
        'dialect'  : 'mysql+pymysql' ,
        'username' : 'jsfund' ,
        'password' : 'Gjquant_js!' ,
        'host'     : 'quantstudio.tpddns.cn' ,
        'port'     : 3306 ,
        'database' : 'gjquant' ,
    },
}

data_start_dt = {
    'haitong':{'hf_factors': 20130101,
               'dl_factors': 20161230},
    'dongfang':{'hfq_chars'     : 20050101 ,
                'l2_chars'      : 20130930 ,
                'ms_chars'      : 20050101 ,
                'order_flow'    : 20130930 ,
                'gp'            : 20170101 ,
                'tra'           : 20200101 ,
                'hist'          : 20200101 ,
                'scores_v0'     : 20171229} ,
    'kaiyuan':{
        'positive': 20140130,
        'negative': 20140130},
    'guojin':{}
}
query_params = {
    'haitong':{'hf_factors':{}, 'dl_factors':{},},
    'dongfang':{'hfq_chars' :{'date_col':'tradingdate','date_fmt':'%Y%m%d'},
                'l2_chars'  :{'date_col':'trade_date' ,'date_fmt':'%Y%m%d'},
                'ms_chars'  :{'date_col':'trade_date' ,'date_fmt':'%Y%m%d'},
                'order_flow':{'date_col':'trade_date' ,'date_fmt':'%Y%m%d'},
                'gp'        :{'date_col':'tradingdate','date_fmt':'%Y-%m-%d'},
                'tra'       :{'date_col':'tradingdate','date_fmt':'%Y-%m-%d'},
                'hist'      :{'date_col':'tradingdate','date_fmt':'%Y-%m-%d'},
                'scores_v0' :{'date_col':'tradingdate','date_fmt':'%Y-%m-%d'},
                } ,
    'kaiyuan':{
        'positive':{'factors' : 
                    ['active_trading','apm','opt_synergy_effect','large_trader_ret_error',
                     'offense_defense','high_freq_shareholder','pe_change',],},
        'negative':{'factors' : 
                    ['smart_money','ideal_vol','ideal_reverse','herd_effect','small_trader_ret_error',],},
        # +1 : 'traction_f' , 'traction_ns' , 'traction_si', , 'long_momentum2' , 'merge_sue', 'consensus_adjustment',
    },
    'guojin':{},
}

# class online_sql_connector
class online_sql_connector():
    def __init__(self):
        assert _db_key == 'SellSideFactors' , _db_key
        self.db_path = get_db_path(_db_key)
        self.connect_dict = connect_dict
        self.engines = dict()
        self.connections = dict()
        self.data_start_dt = data_start_dt
        self.query_params = query_params
        [self.create_engines(src) for src in self.connect_dict.keys()]

    def create_engines(self , srcs):
        if not isinstance(srcs , (list,tuple)): srcs = [srcs]
        for src in srcs:
            self.engines[src] = self._single_engine(self.connect_dict[src])
            
    def create_connections(self , srcs):
        if not isinstance(srcs , (list,tuple)): srcs = [srcs]
        for src in srcs:
            if self.connections.get(src) is not None:
                self.connections[src].close()
            self.connections[src] = self.engines[src].connect()

    def close_all(self):
        [connection.close() for _,connection in self.connections.items()]
        
    def _single_engine(self , src_dict):
        connect_url = '{dialect}://{username}:{password}@{host}:{port}/{database}'.format(
            dialect  = src_dict['dialect'],
            username = src_dict['username'],
            password = src_dict['password'],
            host     = src_dict['host'],
            port     = src_dict['port'],
            database = src_dict['database'],
        )
        
        if src_dict.get('driver'): connect_url += f'?driver={src_dict.get("driver")}'
        return create_engine(connect_url)
    
    def _single_default_query(self , src , query_type , start_dt , end_dt):
        assert query_type in self.query_params[src].keys()
        if src == 'haitong':
            if query_type == 'hf_factors':
                query = '''
                select *  
                from daily_factor_db.dbo.JSJJHFFactors t 
                where t.trade_dt between \'{}\' and \'{}\'
                '''.format(start_dt , end_dt)
            else:
                query = '''
                select s_info_windcode , trade_dt , f_value , model 
                from daily_factor_db.dbo.JSJJDeepLearnFactorsV2 t 
                where t.trade_dt between \'{}\' and \'{}\'
                '''.format(start_dt , end_dt)
        elif src == 'dongfang':
            date_col = self.query_params[src][query_type]['date_col']
            date_fmt = self.query_params[src][query_type]['date_fmt']
            query = '''
            select *  
            from {} t 
            where t.{} between \'{}\' and \'{}\'
            '''.format(query_type , date_col , 
                        datetime.strptime(str(start_dt) , '%Y%m%d').strftime(date_fmt) ,
                        datetime.strptime(str(end_dt) , '%Y%m%d').strftime(date_fmt))
        elif src == 'kaiyuan':
            query = {v:f'select * from public.{v} where date >= \'{start_dt}\' and date <= \'{end_dt}\''
                    for v in self.query_params[src][query_type]['factors']}
        elif src == 'guojin':
            query = 'select * from gjquant.factordescription'
        return query
    
    def query_start_dt(self , src , query_type):
        assert query_type in self.query_params[src].keys()
        if src == 'haitong':
            if query_type == 'hf_factors':
                query = 'select min(trade_dt) from daily_factor_db.dbo.JSJJHFFactors'
            else:
                query = 'select min(trade_dt) daily_factor_db.dbo.JSJJDeepLearnFactorsV2'
        elif src == 'dongfang':
            date_col = self.query_params[src][query_type]['date_col']
            query = 'select min({}) from {}'.format(date_col , query_type)
        elif src == 'kaiyuan':
            query = 'select min(date) from public.smart_money'
        elif src == 'guojin':
            return 99991231
        return pd.read_sql_query(query , self.connections[src])

    def default_query(self , src , query_type , start_dt = 20231101, end_dt = 20231103):
        query = self._single_default_query(src , query_type , start_dt , end_dt)
        if src not in self.connections.keys(): self.create_connections(src)
        try:
            if src == 'kaiyuan':
                assert isinstance(query , dict) , query
                df = {k:pd.read_sql_query(q , self.connections[src]) for k,q in query.items()}
            else:
                assert isinstance(query , str) , query
                df = pd.read_sql_query(query , self.connections[src])
        except exc.ResourceClosedError:
            print(f'{src} Connection is closed, re-connect')
            self.create_connections(src)
            if src == 'kaiyuan':
                pass
            else:
                assert isinstance(query , str) , query
                df = pd.read_sql_query(query , self.connections[src])
        except:
            raise Exception
        df = self.df_process(df , src , query_type)
        return df
    
    def df_process(self , df , src , query_type):
        if src == 'haitong':
            df.columns = map(str.lower , df.columns)
            df = df.rename(columns={'s_info_windcode':'secid','trade_dt':'date'})
            df['date'] = df['date'].astype(int)
            df['secid'] = self._IDconvert(df['secid'])
            if query_type == 'hf_factors':
                df = df.reset_index(drop = True)
            else:
                df.loc[:,'model'] = 'haitong_dl_' + df.loc[:,'model'].astype(str)
                df = df.pivot_table('f_value',['date','secid'],'model').reset_index()
        elif src == 'dongfang':
            df.columns = map(str.lower , df.columns)
            df = df.rename(columns={'stockcode':'secid','ticker':'secid',
                                    'tradingdate':'date','trade_dt':'date' , 'trade_date':'date'})
            df['secid'] = df['secid'].str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df['date'] = df['date'].astype(str).str.replace('-','').astype(int)
        elif src == 'kaiyuan':
            df0 = pd.DataFrame(columns = ['date','code'])
            for k,subdf in df.items():
                subdf.rename(columns = {'factor':k})
                # print(subdf.iloc[:5])
                df0 = df0.merge(subdf.rename(columns = {'factor':k}),how='outer',on=['date','code'])
            df = df0.rename(columns={'code':'secid'})
            df['secid'] = df['secid'].str.replace('[-.a-zA-Z]','',regex=True).astype(int)
            df['date']  = df['date'].astype(int)
            df = df.set_index(['date','secid']).reset_index()
        elif src == 'guojin':
            pass
        return df.fillna(np.nan)

    def _IDconvert(self , x):
        if isinstance(x , (bytes)):
            return int(x.decode('utf-8').split('.')[0].split('!')[0])
        if isinstance(x , (str)):
            return int(x.split('.')[0].split('!')[0])
        else:
            return type(x)([self._IDconvert(xx) for xx in x])

    def download_since(self , src , query_type , trace = 1 , ask = True):
        start_dt = int(self.data_start_dt[src][query_type])
        old_dates = self.old_dtank_dates(src , query_type)
        if trace > 0 and len(old_dates) > trace: old_dates = old_dates[:-trace]
        if len(old_dates): start_dt = max(start_dt , self.date_offset(old_dates[-1],1,astype=int)) #type: ignore
        end_dt = min(99991231 , int(date.today().strftime('%Y%m%d')))
        if end_dt < start_dt: return []
        
        freq = 'QE'
        date_segs = self.date_seg(start_dt , end_dt , freq = freq)
        prompt = f'{time.ctime()} : {src}/{query_type} since {start_dt} to {end_dt}, total {len(date_segs)} periods({freq})'
        if ask: assert input(prompt + ', print "yes" to confirm!') == 'yes'
        print(prompt)

        dtank = DataTank()
        group_file_list = []
        for (s , e) in date_segs:
            data = self.default_query(src , query_type , s , e)
            if len(data) == 0: continue
            data = data.sort_values(['date' , 'secid']).set_index('date')
            for d in data.index.unique():
                data_at_d = data.loc[d]
                if len(data_at_d) == 0: continue
                group = get_date_groups(d)
                group_file = get_db_file(self.db_path , group)
                if dtank.filename != group_file: 
                    dtank.close()
                    dtank = DataTank(group_file , 'r+' , compress=True)
                    group_file_list.append(group_file)
                dtank.write_data1D(path = [src , query_type , str(d)] , 
                                   data = Data1D(src = data_at_d) , 
                                   overwrite = True)
            print(f'{time.ctime()} : {src}/{query_type}:{s // 100}{" "*10}' , end='\r')
        dtank.close()
        return group_file_list

    def download_dates(self , src , query_type , dates = []):
        dtank = DataTank()
        group_file_list = []
        for d in dates:
            data = self.default_query(src , query_type , d , d)
            if len(data) == 0: continue
            data = data.sort_values(['date' , 'secid']).set_index('date')

            group = get_date_groups(d)
            group_file = get_db_file(self.db_path , group)
            if dtank.filename != group_file: 
                dtank.close()
                dtank = DataTank(group_file , 'r+' , compress=True)
                group_file_list.append(group_file)
            dtank.write_data1D(path = [src , query_type , str(d)] , 
                               data = Data1D(src = data) , 
                               overwrite = True)
            print(f'{time.ctime()} : {src}/{query_type}:{d}{" "*10}' , end='\r')
        dtank.close()
        return group_file_list

    def old_dtank_dates(self , src , query_type):
        dates = []
        for sub_path in os.listdir(self.db_path):
            dtank = DataTank(outer_path_join(self.db_path , sub_path) , 'r')
            portal = dtank.get_object([src , query_type])
            if portal is not None and len(portal.keys()) > 0: #type: ignore
                portal_dt = np.array(list(portal.keys())) #type: ignore
                portal_dt_valid = np.array([dtank.is_Data1D(portal[pdt]) for pdt in portal_dt]) #type: ignore
                dates = [*dates , *portal_dt[portal_dt_valid]]
            dtank.close()
        return sorted(np.array(dates).astype(int))

    def download_allaround(self , src , query_type , start_dt = 20000101, end_dt = 99991231 , ask = True):
        if ask: assert input('print "yes" to confirm!') == 'yes'
        start_dt = max(start_dt , self.data_start_dt[src][query_type])
        end_dt   = min(end_dt , int(date.today().strftime('%Y%m%d')))
        freq = 'QE'
        date_segs = self.date_seg(start_dt , end_dt , freq = freq)
        prompt = f'{time.ctime()} : {src}/{query_type} since {start_dt} to {end_dt}, total {len(date_segs)} periods({freq})'
        if ask: assert input(prompt + ', print "yes" to confirm!') == 'yes'
        print(prompt)
        dtank = DataTank()
        group_file_list = []
        for (s , e) in date_segs:
            data = self.default_query(src , query_type , s , e)
            if len(data) == 0: continue
            data = data.sort_values(['date' , 'secid']).set_index('date')
            for d in data.index.unique():
                data_at_d = data.loc[d]
                if len(data_at_d) == 0: continue
                group = get_date_groups(d)
                group_file = get_db_file(self.db_path , group)
                if dtank.filename != group_file: 
                    dtank.close()
                    dtank = DataTank(group_file , 'r+' , compress=True)
                    group_file_list.append(group_file)
                dtank.write_data1D(path = [src , query_type , str(d)] , 
                                   data = Data1D(src = data_at_d) , 
                                   overwrite = True)
            print(f'{time.ctime()} : {src}/{query_type}:{s // 100}{" "*10}' , end='\r')
        print('\n')
        return group_file_list

    @classmethod
    def date_seg(cls , start_dt , end_dt , freq='QE' , astype = int):
        dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
        dt_starts = [cls.date_offset(start_dt) , *cls.date_offset(dt_list[:-1],1)]
        dt_ends = [*dt_list[:-1] , cls.date_offset(end_dt)]
        return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]
    
    @staticmethod
    def date_between(start_dt , end_dt , freq='QE' , astype = int):
        dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
        return dt_list.values
    
    @staticmethod
    def date_offset(date , offset = 0 , astype = int):
        if isinstance(date , (np.ndarray,pd.Index,pd.Series,list,tuple,np.ndarray)):
            is_scalar = False
            new_date = pd.DatetimeIndex(np.array(date).astype(str))
        else:
            is_scalar = True
            new_date = pd.DatetimeIndex([str(date)])
        if offset == 0:
            new_date = new_date.strftime('%Y%m%d') #type: ignore
        else:
            new_date = (new_date + pd.DateOffset(offset)).strftime('%Y%m%d') #type: ignore
        new_date = new_date.astype(astype)
        return new_date[0] if is_scalar else new_date.values

def update_sql_since(trace = 0):
    connector = online_sql_connector()
    group_file_list = []
    dtank = DataTank()
    try:
        for src in connector.query_params.keys():
            for qtype in connector.query_params[src].keys():
                group_files = connector.download_since(src , qtype , trace = trace , ask = False)
                group_file_list = [*group_file_list , *group_files]
        if group_file_list:
            dtank = DataTank(sorted(group_file_list)[-1] , 'r')
            dtank.print_tree()
    except Exception as e:
        traceback.print_exc()
    finally:
        dtank.close()
        connector.close_all()

def update_sql_dates(start_dt , end_dt):
    connector = online_sql_connector()

    dates = online_sql_connector.date_between(start_dt , end_dt)
    if len(dates) == 0: return NotImplemented
    
    group_file_list = []
    try:
        for src in connector.query_params.keys():
            for qtype in connector.query_params[src].keys():
                group_files = connector.download_dates(src , qtype , dates)
                group_file_list = [*group_file_list , *group_files]
        if group_file_list:
            dtank = DataTank(sorted(group_file_list)[-1] , 'r')
            dtank.print_tree()
    except Exception as e:
        traceback.print_exc()
    finally:
        dtank.close()
        connector.close_all()

    