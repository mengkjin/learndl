import numpy as np
import pandas as pd
import os , socket , tarfile, time

from functools import reduce 

from ..environ import DIR
from .DataFetcher import DataFetcher
from .DataFetcher_sql import DataFetcher_sql

class DataUpdater():
    db_updater_title = 'DB_updater'

    def __init__(self) -> None:
        self.Updater = self.get_new_updater()
        self.Success = []
        self.Failed  = []
        
    @classmethod
    def get_updater_paths(cls):
        # order matters!
        search_dirs = [DIR.db , DIR.db_updater] + ['/home/mengkjin/Workspace/SharedFolder'] * (socket.gethostname() == 'mengkjin-server')

        paths = []
        for sdir in search_dirs:
            add_paths = [os.path.join(sdir , p) for p in os.listdir(sdir) if p.startswith(cls.db_updater_title + '.')]
            paths = np.concatenate([paths , sorted(add_paths)])
        return list(paths)
    
    @classmethod
    def unpack_exist_updaters(cls , del_after_dumping = True):
        assert socket.gethostname() == 'mengkjin-server' , socket.gethostname()
        search_dirs = [DIR.db , DIR.db_updater , '/home/mengkjin/Workspace/SharedFolder']
        paths = []
        for sdir in search_dirs:
            path = [os.path.join(sdir , p) for p in os.listdir(sdir) if p.startswith(cls.db_updater_title + '.') and p.endswith('.tar')]
            paths += path
        paths.sort()
        if del_after_dumping and paths:
            print(paths)
            if input(f'''Delete {len(paths)} updaters after completion? (press yes/y) : {paths}''')[0].lower() != 'y': 
                del_after_dumping = False

        for tar_filename in paths:
            with tarfile.open(tar_filename, 'r') as tar:  
                tar.extractall(path = DIR.db , filter='data')  
                
        if del_after_dumping:
            for tar_filename in paths: os.remove(tar_filename)

    @classmethod
    def get_new_updater(cls):
        stime = time.strftime('%y%m%d%H%M%S',time.localtime())
        return os.path.join(DIR.db_updater , f'{cls.db_updater_title}.{stime}.tar')

    def get_db_params(self , db_src):
        # db_update_parameters
        if db_src == 'information':
            params = [
                DataFetcher(db_src,'calendar') ,    # market trading calendar
                DataFetcher(db_src,'description') , # stock list_date and delist_date
                DataFetcher(db_src,'st') ,          # st treatment of stocks
                DataFetcher(db_src,'industry') ,    # SW 2021 industry criterion
                DataFetcher(db_src,'concepts') ,    # wind concepts
            ]
        elif db_src == 'models':
            params = [
                DataFetcher(db_src,'risk_exp') , # market, industry, style exposure of risk model jm2018
                DataFetcher(db_src,'longcl_exp') , # sub alpha factor exposure of longcl
            ]
        elif db_src == 'trade':
            params = [
                DataFetcher(db_src,'day') , 
                DataFetcher(db_src,'5day',[5]) ,
                DataFetcher(db_src,'10day',[10]) , 
                DataFetcher(db_src,'20day',[20]) , 
                DataFetcher(db_src,'min') , 
                DataFetcher(db_src,'5min',[5]), 
                DataFetcher(db_src,'10min',[10]) , 
                DataFetcher(db_src,'15min',[15]) , 
                DataFetcher(db_src,'30min',[30]) , 
                DataFetcher(db_src,'60min',[60]) ,
            ]
        elif db_src == 'labels':
            params = [
                DataFetcher(db_src,'ret5',[5 ,False]) , 
                DataFetcher(db_src,'ret5_lag',[5 ,True]) , 
                DataFetcher(db_src,'ret10',[10 ,False]) , 
                DataFetcher(db_src,'ret10_lag',[10 ,True]) , 
                DataFetcher(db_src,'ret20',[20 ,False]) , 
                DataFetcher(db_src,'ret20_lag',[20 ,True]) , 
            ]
        else:
            raise Exception(db_src)
        return params
    
    def handle_df_result(self , df , target_path , result_dict = None):
        if isinstance(df , pd.DataFrame):
            DataFetcher.save_df(df , target_path)
            with tarfile.open(self.Updater, 'a') as tar:  
                tar.add(target_path, arcname=os.path.relpath(target_path,DIR.db))  
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

    def update_all(self , db_srcs = DataFetcher.DB_by_name + DataFetcher.DB_by_date , start_dt = None , end_dt = None , force = False):
        # selected DB is totally refreshed , so delete first
        if not isinstance(db_srcs , (list,tuple)): db_srcs = [db_srcs]
        for db_src in db_srcs:
            if db_src in DataFetcher.DB_by_name:
                self.update_by_name(db_src)
            elif db_src in DataFetcher.DB_by_date:
                self.update_by_date(db_src , start_dt , end_dt , force = force)
            else:
                raise Exception(db_src)

    def print_unfetch(self):
        for fail in self.Failed: print(fail) 

    @classmethod
    def update_server(cls):
        assert socket.gethostname() == 'mengkjin-server' , socket.gethostname()
        start_time = time.time()

        print(f'Unpack Update Files') 
        cls.unpack_exist_updaters(del_after_dumping=True)

        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update_laptop(cls):
        assert socket.gethostname() != 'mengkjin-server' , socket.gethostname()

        start_time = time.time()
        print(f'Update Files')
        Updater = cls()
        Updater.update_all()
        Updater.print_unfetch()

        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

def main():
    if socket.gethostname() == 'mengkjin-server':
        DataUpdater.update_server()
    else:
        DataUpdater.update_laptop()
    DataFetcher_sql.update_since()