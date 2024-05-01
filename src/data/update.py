import numpy as np
import pandas as pd
import argparse , gc , os , socket , sys , tarfile, time

from dataclasses import dataclass , field
from functools import reduce
from typing import Optional

from .fetcher import DB_by_date , DB_by_name , DataFetcher , SQLFetcher , save_df
from .core import DataBlock
from ..classes import DataProcessCfg
from ..environ import DIR
from ..func.time import Timer , today

@dataclass(slots=True)
class PreProcessConfig:
    predict         : bool
    blocks          : list = field(default_factory=list)
    load_start_dt   : Optional[int] = None
    load_end_dt     : Optional[int] = None
    save_start_dt   : Optional[int] = 20070101
    save_end_dt     : Optional[int] = None
    hist_start_dt   : Optional[int] = None
    hist_end_dt     : Optional[int] = 20161231
    mask            : Optional[dict] = None

    def __post_init__(self):
        self.blocks = [blk.lower() for blk in self.blocks]
        if self.predict:
            self.load_start_dt = today(-181)
            self.load_end_dt   = None
            self.save_start_dt = None
            self.save_end_dt   = None
            self.hist_start_dt = None
            self.hist_end_dt   = None
        if self.mask is None:
            self.mask = self.default_mask()

    def get_block_params(self):
        for blk in self.blocks:
            yield blk , self.default_block_param(blk)

    @staticmethod
    def default_block_param(blk : str):
        if blk in ['y' , 'labels']:
            params : dict[str,list]= {
                'labels': ['labels' , ['ret10_lag' , 'ret20_lag']] ,
                'models': ['models' , 'risk_exp'] ,
            }
        elif blk in ['day' , 'trade_day']:
            params = {
                'trade_day' : ['trade' , 'day' , ['adjfactor', 'close', 'high', 'low', 'open', 'vwap' , 'turn_fl']] ,
            }
        elif blk in ['30m' , 'trade_30m']:
            params = {
                'trade_30m' : ['trade' , '30min' , ['close', 'high', 'low', 'open', 'volume', 'vwap']] ,
                'trade_day' : ['trade' , 'day' , ['volume' , 'turn_fl' , 'preclose']] ,
            }
        elif blk in ['15m' , 'trade_15m']:
            params = {
                'trade_15m' : ['trade' , '15min' , ['close', 'high', 'low', 'open', 'volume', 'vwap']] ,
                'trade_day' : ['trade' , 'day' , ['volume' , 'turn_fl' , 'preclose']] ,
            }
        elif blk in ['week' , 'trade_week']:
            params = {
                'trade_day' : ['trade' , 'day' , ['adjfactor', 'preclose' ,'close', 'high', 'low', 'open', 'vwap' , 'turn_fl']] ,
            }
        else:
            raise KeyError(blk)
        return {k:DataProcessCfg(*v) for k,v in params.items()}
        
    @staticmethod
    def default_mask(): return {'list_dt':True}

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
            paths += [os.path.join(sdir , p) for p in os.listdir(sdir) if p.startswith(cls.db_updater_title + '.') and p.endswith('.tar')]
        paths.sort()
        if del_after_dumping and paths:
            print(paths)
            del_after_dumping = input(f'''Delete {len(paths)} updaters after completion? (press yes/y) : {paths}''')[0].lower() == 'y'

        for tar_filename in paths:
            with tarfile.open(tar_filename, 'r') as tar:  
                tar.extractall(path = DIR.db , filter='data')  
                
        if del_after_dumping: [os.remove(tar_filename) for tar_filename in paths]

    @classmethod
    def get_new_updater(cls):
        stime = time.strftime('%y%m%d%H%M%S',time.localtime())
        return os.path.join(DIR.db_updater , f'{cls.db_updater_title}.{stime}.tar')

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
                ['risk_exp',] , # market, industry, style exposure of risk model jm2018
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
        else:
            raise Exception(db_src)
        params = [DataFetcher(db_src , *args) for args in param_args]
        return params
    
    def handle_df_result(self , df , target_path , result_dict = None):
        if isinstance(df , pd.DataFrame):
            save_df(df , target_path)
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

    def update_all(self , db_srcs = DB_by_name + DB_by_date , start_dt = None , end_dt = None , force = False):
        # selected DB is totally refreshed , so delete first
        if not isinstance(db_srcs , (list,tuple)): db_srcs = [db_srcs]
        for db_src in db_srcs:
            if db_src in DB_by_name:
                self.update_by_name(db_src)
            elif db_src in DB_by_date:
                self.update_by_date(db_src , start_dt , end_dt , force = force)
            else:
                raise Exception(db_src)

    def print_unfetch(self):
        [print(fail) for fail in self.Failed]

    @classmethod
    def update_server(cls):
        assert socket.gethostname() == 'mengkjin-server' , socket.gethostname()
        start_time = time.time()

        print(f'Unpack Update Files') 
        cls.unpack_exist_updaters(del_after_dumping=True)
        SQLFetcher.update_since()
        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update_laptop(cls):
        assert socket.gethostname() != 'mengkjin-server' , socket.gethostname()

        start_time = time.time()
        print(f'Update Files')
        Updater = cls()
        Updater.update_all()
        Updater.print_unfetch()
        SQLFetcher.update_since()
        print(f'{time.ctime()} : All Updates Done! Cost {time.time() - start_time:.2f} Secs')

    @classmethod
    def update_data(cls):
        if socket.gethostname() == 'mengkjin-server':
            DataUpdater.update_server()
        else:
            DataUpdater.update_laptop()

    @staticmethod
    def preprocess_data(predict = False, confirm = 0 , parser = None):
        if parser is None:
            parser = argparse.ArgumentParser(description='manual to this script')
            parser.add_argument("--confirm", type=str, default = confirm)
            args , _ = parser.parse_known_args()

        if not predict and not args.confirm and not input('Confirm update data? print "yes" to confirm!').lower()[0] == 'y' : 
            sys.exit()

        t1 = time.time()
        print(f'predict is {predict} , Data Processing start!')

        Configs = PreProcessConfig(predict , blocks = ['y' , 'trade_day' , 'trade_30m'])
        print(f'{len(Configs.blocks)} datas :' + str(list(Configs.blocks)))

        for key , param in Configs.get_block_params():
            tt1 = time.time()
            print(f'{time.ctime()} : {key} start ...')
            
            BlockDict = DataBlock.load_DB(param , Configs.load_start_dt, Configs.load_end_dt)
            
            with Timer(f'{key} blocks process'):
                ThisBlock = DataBlock.blocks_process(BlockDict , key)

            with Timer(f'{key} blocks masking'):   
                ThisBlock = ThisBlock.mask_values(mask = Configs.mask)

            with Timer(f'{key} blocks saving '):
                ThisBlock.save(key , predict , Configs.save_start_dt , Configs.save_end_dt)

            with Timer(f'{key} blocks norming'):
                ThisBlock.hist_norm(key , predict , Configs.hist_start_dt , Configs.hist_end_dt)
            
            tt2 = time.time()
            print(f'{time.ctime()} : {key} finished! Cost {tt2-tt1:.2f} Seconds')
        
            del ThisBlock
            gc.collect()

        t2 = time.time()
        print('Data Processing Finished! Cost {:.2f} Seconds'.format(t2-t1))