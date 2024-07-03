import os ,  tarfile, time
import numpy as np
import pandas as pd

from functools import reduce

from .fetcher import DataFetcher , SQLFetcher
from ..basic import PATH , THIS_IS_SERVER
from ..basic.db import DB_BY_DATE , DB_BY_NAME , save_df

UPDATER_TITLE = 'DB_updater'
class DataUpdater():
    def __init__(self) -> None:
        self.Updater = self.get_new_updater()
        self.Success = []
        self.Failed  = []
        
    @staticmethod
    def get_updater_paths():
        # order matters!
        search_dirs = [PATH.database , PATH.updater] + ['/home/mengkjin/Workspace/SharedFolder'] * THIS_IS_SERVER

        paths = []
        for sdir in search_dirs:
            add_paths = [os.path.join(sdir , p) for p in os.listdir(sdir) if p.startswith(UPDATER_TITLE + '.')]
            paths = np.concatenate([paths , sorted(add_paths)])
        return list(paths)
    
    @staticmethod
    def unpack_exist_updaters(del_after_dumping = True):
        assert THIS_IS_SERVER , f'must on server'
        search_dirs = [PATH.database , PATH.updater , '/home/mengkjin/Workspace/SharedFolder']
        paths = []
        for sdir in search_dirs:
            paths += [os.path.join(sdir , p) for p in os.listdir(sdir) if p.startswith(UPDATER_TITLE + '.') and p.endswith('.tar')]
        paths.sort()
        if del_after_dumping and paths:
            print(paths)
            del_after_dumping = input(f'''Delete {len(paths)} updaters after completion? (press yes/y) : {paths}''')[0].lower() == 'y'

        for tar_filename in paths:
            with tarfile.open(tar_filename, 'r') as tar:  
                tar.extractall(path = PATH.database , filter='data')  
                
        if del_after_dumping: [os.remove(tar_filename) for tar_filename in paths]

    @staticmethod
    def get_new_updater():
        stime = time.strftime('%y%m%d%H%M%S',time.localtime())
        return os.path.join(PATH.updater , f'{UPDATER_TITLE}.{stime}.tar')

    def get_db_params(self , db_src):
        # db_update_parameters
        if db_src == 'information':
            param_args : list[list] = [
                ['calendar',] ,    # market trading calendar
                ['description',] , # stock list_date and delist_date
                ['st',] ,          # st treatment of stocks
                ['industry',] ,    # SW 2021 industry criterion
                ['concepts',] ,    # wind concepts
            ]
        elif db_src == 'models':
            param_args = [
                ['risk_exp',] ,   # market, industry, style exposure of risk model jm2018
                ['risk_cov',] ,   # factor covariance matrix (annualized ocv) of risk model jm2018
                ['risk_spec',] ,  # specific risk (annualized standard deviation) of risk model jm2018
                ['longcl_exp',] , # sub alpha factor exposure of longcl
            ]
        elif db_src == 'trade':
            param_args = [
                ['day',] , 
                ['5day',[5],] ,
                ['10day',[10],] , 
                ['20day',[20],] , 
                ['min',] , 
                ['5min',[5],], 
                ['10min',[10],] , 
                ['15min',[15],] , 
                ['30min',[30],] , 
                ['60min',[60],] ,
            ]
        elif db_src == 'labels':
            param_args = [
                ['ret5',[5 ,False],] , 
                ['ret5_lag',[5 ,True],] , 
                ['ret10',[10 ,False],] , 
                ['ret10_lag',[10 ,True],] , 
                ['ret20',[20 ,False],] , 
                ['ret20_lag',[20 ,True],] , 
            ]
        elif db_src == 'benchmark':
            param_args = [
                ['csi300',['csi300']] , 
                ['csi500',['csi500']] , 
                ['csi1000',['csi1000']] , 
            ]
        elif db_src in ['information_ts' , 'membership_ts']:
            param_args = []
        else:
            raise KeyError(db_src)
        params = [DataFetcher(db_src , *args) for args in param_args]
        return params
    
    def handle_df_result(self , df , target_path , result_dict = None):
        if isinstance(df , pd.DataFrame):
            save_df(df , target_path)
            with tarfile.open(self.Updater, 'a') as tar:  
                tar.add(target_path, arcname = os.path.relpath(target_path, PATH.database))  
            self.Success.append(target_path)
            target_str = f'Updated ~ {os.path.relpath(target_path)}'
        elif df is None:
            target_str = None
        else:
            self.Failed.append(df)
            target_str = f'Failed ~ {os.path.relpath(target_path)}'
            
        if result_dict is not None: result_dict[target_path] = df
        return target_str

    def update_by_name(self , db_src):
        params = self.get_db_params(db_src)
        result = None # {}
        for param in params:
            start_time = time.time()
            df , target_path = param()
            target_str = self.handle_df_result(df , target_path , result)
            if target_str: print(f'{time.ctime()} : {target_str} Done! Cost {time.time() - start_time:.2f} Secs')
        return result
    
    def update_by_date(self , db_src , start_dt = None , end_dt = None , force = False):
        params = self.get_db_params(db_src)
        result = None # {}
        update_dates = [par.get_update_dates(start_dt=start_dt,end_dt=end_dt,force=force) for par in params]
        full_dates = reduce(np.union1d, update_dates)
        update_cond  = np.stack([np.isin(full_dates , dts) for dts in update_dates] , -1)
        for i , date in enumerate(full_dates):
            temporal = {}
            for cond , param in zip(update_cond[i] , params):
                if not cond: continue
                start_time = time.time()
                df , target_path = param(date , df_min = temporal.get('df_min'))
                target_str = self.handle_df_result(df , target_path , result)
                if param.db_src == 'trade' and param.db_key == 'min': temporal['df_min'] = df
                if target_str: print(f'{time.ctime()} : {target_str} Done! Cost {time.time() - start_time:.2f} Secs')
        return result

    def update_all(self , db_srcs = DB_BY_NAME + DB_BY_DATE , start_dt = None , end_dt = None , force = False):
        assert not THIS_IS_SERVER , f'must on laptop'
        # selected DB is totally refreshed , so delete first
        if not isinstance(db_srcs , (list,tuple)): db_srcs = [db_srcs]
        for db_src in db_srcs:
            if db_src in DB_BY_NAME:
                self.update_by_name(db_src)
            elif db_src in DB_BY_DATE:
                self.update_by_date(db_src , start_dt , end_dt , force = force)
            else:
                raise Exception(db_src)

    def print_unfetch(self):
        [print(fail) for fail in self.Failed]

    @classmethod
    def update_server(cls):
        assert THIS_IS_SERVER , f'must on server'
        start_time = time.time()

        print(f'Unpack Update Files') 
        cls.unpack_exist_updaters(del_after_dumping=True)
        SQLFetcher.update_since()
        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update_laptop(cls):
        assert not THIS_IS_SERVER , f'must on laptop'
        start_time = time.time()
        print(f'Update Files')
        Updater = cls()
        Updater.update_all()
        Updater.print_unfetch()
        SQLFetcher.update_since()
        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')