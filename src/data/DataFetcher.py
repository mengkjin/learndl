import numpy as np
import pandas as pd
import os , re , socket , tarfile, time

from dataclasses import dataclass , field
from functools import reduce 
from typing import Callable , ClassVar , List , Literal , Any

from ..environ import DIR
from .DataFetcher_R import DataFetcher_R as RFetcher

# %%
# def stime(compact = False): return time.strftime('%y%m%d%H%M%S' if compact else '%Y-%m-%d %H:%M:%S',time.localtime())
   
@dataclass
class DataFetcher:
    db_src      : str
    db_key      : str
    args        : list = field(default_factory=list)
    fetcher     : Callable | str = 'default'
    
    DB_by_name  : ClassVar[List[str]] = ['information']
    DB_by_date  : ClassVar[List[str]] = ['models' , 'trade' , 'labels']  
    save_option : ClassVar[Literal['feather' , 'parquet']] = 'feather'

    def __post_init__(self):
        if self.fetcher == 'default':
            self.fetcher = self.default_fetcher(self.db_src , self.db_key)

    def __call__(self , date = None , **kwargs) -> Any:
        return self.eval(date , **kwargs) , self.target_path(date)
    
    @classmethod
    def default_fetcher(cls , db_src , db_key):
        if db_src == 'information': return RFetcher.basic_info
        elif db_src == 'models':
            if db_key == 'risk_exp': return RFetcher.risk_model
            elif db_key == 'longcl_exp': return RFetcher.alpha_longcl
        elif db_src == 'trade':
            if db_key == 'day': return RFetcher.trade_day
            elif db_key == 'min': return RFetcher.trade_min
            elif re.match(r'^\d+day$' , db_key): return RFetcher.trade_Xday
            elif re.match(r'^\d+min$' , db_key): return RFetcher.trade_Xmin
        elif db_src == 'labels': return RFetcher.labels
        raise Exception('Unknown default_fetcher')

    def eval(self , date = None , **kwargs) -> Any:
        assert callable(self.fetcher)
        if self.db_src in self.DB_by_name:
            v = self.fetcher(self.db_key , *self.args , **kwargs)
        elif self.db_src in self.DB_by_date:
            v = self.fetcher(date , *self.args , **kwargs)  
        return v
    
    def target_path(self , date = None):
        return self.get_target_path(self.db_src , self.db_key , date , makedir=True)
    
    def source_dates(self):
        return self.get_source_dates(self.db_src , self.db_key)
    
    def target_dates(self):
        return self.get_target_dates(self.db_src , self.db_key)
    
    def get_update_dates(self , start_dt = None , end_dt = None , trace = 0 , incremental = True , force = False):
        source_dates = self.source_dates()
        target_dates = self.target_dates()
        if force:
            if start_dt is None or end_dt is None:
                raise ValueError(f'start_dt and end_dt must not be None with force update!')
            target_dates = []
        if incremental: 
            if len(target_dates):
                source_dates = source_dates[source_dates >= min(target_dates)]
        if trace > 0 and len(target_dates) > 0: target_dates = target_dates[:-trace]

        new_dates = np.setdiff1d(source_dates , target_dates)
        if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
        if end_dt   is not None: new_dates = new_dates[new_dates <= end_dt  ]

        return new_dates
    
    @classmethod
    def get_target_path(cls , db_src , db_key , date = None , makedir = False , 
                        force_type : Literal['name' , 'date'] | None = None):
        if db_src in cls.DB_by_name or force_type == 'name':
            db_path = os.path.join(DIR.db , f'DB_{db_src}')
            db_base = f'{db_key}.{cls.save_option}'
        elif db_src in cls.DB_by_date or force_type == 'date':
            assert date is not None
            year_group = int(date) // 10000
            db_path = os.path.join(DIR.db , f'DB_{db_src}' , db_key , str(year_group))
            db_base = f'{db_key}.{str(date)}.{cls.save_option}'
        else:
            raise KeyError(db_src)
        if makedir: os.makedirs(db_path , exist_ok=True)
        return os.path.join(db_path , db_base)
    
    @classmethod
    def get_source_dates(cls , db_src , db_key):
        assert db_src in cls.DB_by_date
        return RFetcher.source_dates('/'.join([db_src , re.sub(r'\d+', '', db_key)]))
    
    @classmethod
    def get_target_dates(cls , db_src , db_key):
        db_path = os.path.join(DIR.db , f'DB_{db_src}' , db_key)
        target_files = RFetcher.list_files(db_path , recur=True)
        target_dates = RFetcher.path_date(target_files)
        return np.array(sorted(target_dates) , dtype=int)
    
    @classmethod
    def load_target_file(cls , db_src , db_key , date = None):
        target_path = cls.get_target_path(db_src , db_key , date)
        if os.path.exists(target_path):
            return cls.load_df(target_path)
        else:
            return None
        
    @classmethod
    def save_df(cls , df , target_path):
        if cls.save_option == 'feather':
            df.to_feather(target_path)
        else:
            df.to_parquet(target_path , engine='fastparquet')

    @classmethod
    def load_df(cls , target_path):
        if cls.save_option == 'feather':
            return pd.read_feather(target_path)
        else:
            return pd.read_parquet(target_path , engine='fastparquet')

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