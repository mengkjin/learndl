import tarfile, time , socket
import numpy as np
import pandas as pd

from functools import reduce
from pathlib import Path

from src.basic import PATH , MY_SERVER , MY_LAPTOP
from src.func.display import print_seperator

from .jsfetcher import JSFetcher

UPDATER_TITLE = 'DB_updater'
SHARE_FOLDERS = [Path('/home/mengkjin/Workspace/SharedFolder')] if MY_SERVER else []
class JSDataUpdater():
    '''
    in JS environment, update js source data from jinmeng's laptop
    must update after the original R database is updated
    '''
    def __init__(self) -> None:
        self.Updater = self.get_new_updater()
        self.Success = []
        self.Failed  = []
        
    @staticmethod
    def get_updater_paths():
        # order matters!
        search_dirs = [PATH.database , PATH.updater] + SHARE_FOLDERS

        paths : list[Path] = []
        for sdir in search_dirs:
            add_paths = [p for p in sdir.iterdir() if p.name.startswith(UPDATER_TITLE + '.')]
            paths += add_paths
        return paths
    
    @staticmethod
    def unpack_exist_updaters(del_after_dumping = True):
        assert MY_SERVER , f'must on server'
        search_dirs = [PATH.database , PATH.updater] + SHARE_FOLDERS
        paths : list[Path] = []
        for sdir in search_dirs:
            paths += [p for p in sdir.iterdir() if p.name.startswith(UPDATER_TITLE + '.') and p.name.endswith('.tar')]
        paths.sort()
        if del_after_dumping and paths:
            print_seperator()
            print(f'Delete {len(paths)} updaters after completion')
            print(paths)
            # del_after_dumping = input(f'''Delete {len(paths)} updaters after completion? (press yes/y) : {paths}''')[0].lower() == 'y'

        for tar_filename in paths:
            with tarfile.open(tar_filename, 'r') as tar:  
                tar.extractall(path = str(PATH.database) , filter='data')  
                
        if del_after_dumping: [tar_filename.unlink() for tar_filename in paths]

    @staticmethod
    def get_new_updater():
        stime = time.strftime('%y%m%d%H%M%S',time.localtime())
        return PATH.updater.joinpath(f'{UPDATER_TITLE}.{stime}.tar')

    def get_db_params(self , db_src):
        # db_update_parameters
        if db_src == 'information_js':
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
        elif db_src == 'trade_js':
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
        elif db_src == 'labels_js':
            param_args = [
                ['ret5',[5 ,False],] , 
                ['ret5_lag',[5 ,True],] , 
                ['ret10',[10 ,False],] , 
                ['ret10_lag',[10 ,True],] , 
                ['ret20',[20 ,False],] , 
                ['ret20_lag',[20 ,True],] , 
            ]
        elif db_src == 'benchmark_js':
            param_args = [
                ['csi300',['csi300']] , 
                ['csi500',['csi500']] , 
                ['csi800',['csi800']] , 
                ['csi1000',['csi1000']] , 
            ]
        else:
            param_args = []

        params = [JSFetcher(db_src , *args) for args in param_args]
        return params
    
    def handle_df_result(self , df , target_path : Path , result_dict = None):
        abs_path = str(target_path.absolute())
        rel_path = str(target_path.relative_to(PATH.database))
        if isinstance(df , pd.DataFrame):
            PATH.save_df(df , target_path)
            with tarfile.open(self.Updater, 'a') as tar:  
                tar.add(abs_path , arcname = rel_path) 
            self.Success.append(target_path)
            target_str = f'Updated ~ {rel_path}'
        elif df is None:
            target_str = None
        else:
            self.Failed.append(df)
            target_str = f'Failed ~ {rel_path}'
            
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
        if not params: return
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
                if param.db_src == 'trade_js' and param.db_key == 'min': temporal['df_min'] = df
                if target_str: print(f'{time.ctime()} : {target_str} Done! Cost {time.time() - start_time:.2f} Secs')
        return result

    def update_all(self , db_srcs = PATH.DB_BY_NAME + PATH.DB_BY_DATE , start_dt = None , end_dt = None , force = False):
        assert MY_LAPTOP , f'must on my laptop'
        # selected DB is totally refreshed , so delete first
        if not isinstance(db_srcs , (list,tuple)): db_srcs = [db_srcs]
        for db_src in db_srcs:
            if db_src in PATH.DB_BY_NAME:
                self.update_by_name(db_src)
            elif db_src in PATH.DB_BY_DATE:
                self.update_by_date(db_src , start_dt , end_dt , force = force)
            else:
                raise Exception(db_src)

    def print_unfetch(self):
        [print(fail) for fail in self.Failed]

    @classmethod
    def update_laptop(cls):
        assert MY_LAPTOP , f'must on my laptop'
        start_time = time.time()
        print(f'Update Files')
        Updater = cls()
        Updater.update_all()
        Updater.print_unfetch()
        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update_server(cls):
        assert MY_SERVER , f'must on my server'
        start_time = time.time()

        print(f'Unpack Update Files') 
        cls.unpack_exist_updaters(del_after_dumping=True)

        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update(cls):
        '''
        in JS environment, update js source data from jinmeng's laptop
        1. In laptop, update js source data from R project to updaters
        2. In server, unpack update files and move to Database
        '''
        if MY_SERVER:
            cls.update_server()
        elif MY_LAPTOP:
            cls.update_laptop()
        else:
            print(f'Unidentified machine: {socket.gethostname()} , do nothing')