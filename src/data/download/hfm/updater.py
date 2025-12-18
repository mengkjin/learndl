import tarfile
import numpy as np
import pandas as pd

from datetime import datetime
from functools import reduce
from pathlib import Path

from src.proj import PATH , MACHINE , Logger , Duration
from src.basic import DB

from .task import JSFetcher , JSDownloader
from .minute_transform import main as minute_transform    

class JSDataUpdater():
    '''
    in JS environment, update js source data from jinmeng's terminal
    must update after the original R database is updated
    '''
    UPDATER_BASE        = PATH.data
    UPDATER_TITLE       = 'DB_updater'
    UPDATER_SEARCH_DIRS = [PATH.updater , Path('/home/mengkjin/workspace/SharedFolder')] if MACHINE.server else [PATH.updater]

    def __init__(self) -> None:
        self.Updater = self.get_new_updater()
        self.Success = []
        self.Failed  = []
        
    @classmethod
    def get_updater_paths(cls):
        # order matters!

        paths : list[Path] = []
        for sdir in cls.UPDATER_SEARCH_DIRS:
            add_paths = [p for p in sdir.iterdir() if p.name.startswith(cls.UPDATER_TITLE + '.')]
            paths += add_paths
        return paths
    
    @classmethod
    def unpack_exist_updaters(cls , del_after_dumping = True):
        assert MACHINE.server , f'must on server'

        paths : list[Path] = []
        for sdir in cls.UPDATER_SEARCH_DIRS:
            paths += [p for p in sdir.iterdir() if p.name.startswith(cls.UPDATER_TITLE + '.') and p.name.endswith('.tar')]
        paths.sort()
        if del_after_dumping and paths:
            Logger.divider()
            Logger.stdout(f'Delete {len(paths)} updaters after completion')
            Logger.stdout(paths)
            # del_after_dumping = input(f'''Delete {len(paths)} updaters after completion? (press yes/y) : {paths}''')[0].lower() == 'y'

        for tar_filename in paths:
            with tarfile.open(tar_filename, 'r') as tar:  
                tar.extractall(path = str(cls.UPDATER_BASE) , filter='data')  
                
        if del_after_dumping: 
            [tar_filename.unlink() for tar_filename in paths]

    @classmethod
    def transform_datas(cls):
        minute_transform()

    @classmethod
    def get_new_updater(cls):
        stime = datetime.now().strftime('%y%m%d%H%M%S')
        return PATH.updater.joinpath(f'{cls.UPDATER_TITLE}.{stime}.tar')

    def get_db_params(self , db_src):
        # db_update_parameters
        if db_src == 'information_js':
            param_args : list[list] = [
                #['calendar',] ,    # market trading calendar
                #['description',] , # stock list_date and delist_date
                #['st',] ,          # st treatment of stocks
                #['industry',] ,    # SW 2021 industry criterion
                #['concepts',] ,    # wind concepts
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
    
    def handle_result(self , result , target_path : Path , result_dict = None):
        abs_path = str(target_path.absolute())
        rel_path = str(target_path.relative_to(self.UPDATER_BASE))
        if isinstance(result , pd.DataFrame):
            DB.save_df(result , target_path)
            with tarfile.open(self.Updater, 'a') as tar:  
                tar.add(abs_path , arcname = rel_path) 
            self.Success.append(target_path)
            target_str = f'Updated ~ {rel_path}'
        elif isinstance(result , Path):
            with tarfile.open(self.Updater, 'a') as tar:  
                tar.add(result , arcname = rel_path) 
            self.Success.append(target_path)
            target_str = f'Updated ~ {rel_path}'
        elif result is None:
            target_str = None
        else:
            self.Failed.append(result)
            target_str = f'Failed ~ {rel_path}'
            
        if result_dict is not None: 
            result_dict[target_path] = result
        return target_str

    def fetch_by_name(self , db_src):
        params = self.get_db_params(db_src)
        result = None # {}
        for param in params:
            start_time = datetime.now()
            df , target_path = param()
            target_str = self.handle_result(df , target_path , result)
            if target_str: 
                Logger.success(f'Fetching: {target_str} Done! Cost {Duration(since = start_time)}')
        return result
    
    def fetch_by_date(self , db_src , start_dt = None , end_dt = None , force = False):
        params = self.get_db_params(db_src)
        if not params: 
            return
        result = None # {}
        update_dates = [par.get_update_dates(start_dt=start_dt,end_dt=end_dt,force=force) for par in params]
        full_dates = reduce(np.union1d, update_dates)
        update_cond  = np.stack([np.isin(full_dates , dts) for dts in update_dates] , -1)
        for i , date in enumerate(full_dates):
            temporal = {}
            for cond , param in zip(update_cond[i] , params):
                if not cond: 
                    continue
                start_time = datetime.now()
                df , target_path = param(date , df_min = temporal.get('df_min'))
                target_str = self.handle_result(df , target_path , result)
                if param.db_src == 'trade_js' and param.db_key == 'min': 
                    temporal['df_min'] = df
                if target_str: 
                    Logger.success(f'Fetching: {target_str} Done! Cost {Duration(since = start_time)}')
        return result

    def fetch_all(self ,start_dt = None , end_dt = None , force = False):
        # selected DB is totally refreshed , so delete first
        if not MACHINE.belong_to_jinmeng: 
            return

        for db_src in DB.iter_db_srcs():
            if DB.by_name(db_src):
                self.fetch_by_name(db_src)
            elif DB.by_date(db_src):
                self.fetch_by_date(db_src , start_dt , end_dt , force = force)
            else:
                raise Exception(db_src)

    def download_all(self):
        paths : list[Path] = []
        for path in JSDownloader.proceed():
            start_time = datetime.now()
            target_str = self.handle_result(path , path)
            if target_str: 
                Logger.success(f'Download: {target_str} Done! Cost {Duration(since = start_time)}')
            paths.append(path)
        return paths

    def print_unfetch(self):
        [Logger.stdout(fail) for fail in self.Failed]

    @classmethod
    def update_hfm_terminal(cls):
        assert not MACHINE.server , f'must on terminal machine'
        if not MACHINE.belong_to_hfm: 
            return
        with Logger.ParagraphIII('Update JSData Update Files'):
            Updater = cls()
            Updater.fetch_all()
            Updater.download_all()
            Updater.transform_datas()
            Updater.print_unfetch()
            
    @classmethod
    def update_server(cls):
        assert MACHINE.server , f'must on server machine'

        with Logger.ParagraphIII('Unpack JSData Update Files'):
            cls.unpack_exist_updaters(del_after_dumping=True)
            cls.transform_datas()

    @classmethod
    def update(cls):
        '''
        in JS environment, update js source data from jinmeng's terminal
        1. In terminal, update js source data from R project to updaters
        2. In server, unpack update files and move to Database
        '''
        if MACHINE.server:
            cls.update_server()
        elif MACHINE.belong_to_jinmeng and MACHINE.belong_to_hfm:
            cls.update_hfm_terminal()
        else:
            ...

