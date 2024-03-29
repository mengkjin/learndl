import h5py
import numpy as np
import os , time
import traceback
import socket , platform
from .DataTank import DataTank,copy_tree,DataFailed
from .DataTransmitter import (
    read_risk_model,read_alpha_longcl,
    get_basic_information,get_labels,
    get_trade_day,get_trade_Xday,
    get_trade_min,get_trade_Xmin,
    get_directory_files,get_path_date
)

from ..environ import DIR

# %%
l_do_updater = True
l_print_tree = False
l_update_by_key  = ['information']
l_update_by_date = ['models' , 'trade_day' ,  'trade_Xday' , 'trade_min' , 'trade_Xmin' , 'labels']
l_require_params = ['__information__' , '__create_time__' , '__date_source__' , '__data_func__']
        
def stime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

def inner_path_join(*args):
    return '/'.join([str(arg) for arg in args])

def outer_path_join(*args):
    return os.path.join(*args)

def updater_key_to_path(key):
    return outer_path_join(*key.split('\\'))
    
def updater_path_to_key(path):
    return '\\'.join(os.path.relpath(path).replace('\\','/').split('/'))
    
def get_db_dir(make = True):
    dir_db = outer_path_join(DIR.data , 'DB_data')
    if make: os.makedirs(dir_db , exist_ok=True)
    return dir_db

def get_db_path(db_key , db_dir = None , make = True):
    db_name = f'DB_{db_key}'
    if db_dir is None: db_dir = get_db_dir()
    db_path = outer_path_join(db_dir , db_name)
    if make: os.makedirs(db_path , exist_ok=True)
    return db_path

def get_db_file(db_path , group = None):
    db_name = os.path.basename(db_path)
    db_key  = db_name.replace('DB_','')
    if db_key in l_update_by_key:
        db_name = db_name + '.h5'
    else:
        assert group is not None , db_key
        db_name = db_name + f'.{group}.h5'
    db_file = outer_path_join(db_path , db_name)
    return db_file

def get_date_groups(dates):
    # return year
    return np.array(dates).astype(int) // 10000

class DataUpdater():
    def __init__(self , do_updater = False) -> None:
        self.do_updater = do_updater
        self.DIR_db  = get_db_dir()
        self.UPDATER = self.get_new_updater()
        self.UNFETCH = dict()
        self.update_order = []

    def get_updater_paths(self , title = 'DB_updater'):
        # order matters!
        search_dirs = [self.DIR_db]
        if socket.gethostname() == 'mengkjin-server':
            search_dirs.append('/home/mengkjin/Workspace/SharedFolder')
        paths = []
        for sdir in search_dirs:
            add_paths = [outer_path_join(sdir , p) for p in os.listdir(sdir) if p.startswith(title + '.')]
            paths = np.concatenate([paths , sorted(add_paths)])
        return list(paths)

    def get_new_updater(self, title = 'DB_updater'):
        if self.do_updater:
            old_updaters = self.get_updater_paths(title)
            updater_i = [int(os.path.basename(p).split('.')[1]) for p in old_updaters]
            updater_n = 0 if len(updater_i) == 0 else max(updater_i) + 1
            updater_path = outer_path_join(self.DIR_db , f'{title}.{updater_n:d}.h5')
            return DataTank(updater_path , 'w')
        else:
            return dict()

    def get_db_params(self , key):
        # db_file path
        db_path = get_db_path(key , db_dir=self.DIR_db)

        # db_update_parameters
        if key == 'information':
            db_update_params = {
                'basic/calendar'   : {'__information__': 'market trading calendar'},
                'stock/description': {'__information__': 'stock list_date and delist_date'},
                'stock/st'         : {'__information__': 'st treatment of stocks'},
                'stock/industry'   : {'__information__': 'SW 2021 industry criterion'},
                'stock/concepts'   : {'__information__': 'wind concepts'},}
            for k in db_update_params.keys():
                db_update_params[k].update({'__data_func__':get_basic_information,'compress':True}) #type:ignore
        elif key == 'models':
            db_update_params = {
                'risk_model/exposure' : {
                    '__information__': 'market, industry, style exposure of risk model jm2018' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
                    '__data_func__'  : read_risk_model,'compress':False} ,
                'alpha_longcl/exposure' : {
                    '__information__': 'sub alpha factor exposure of risk model jm2018' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst' ,
                    '__data_func__'  : read_alpha_longcl,'compress':False}
            }
        elif key == 'trade_day':
            db_update_params = {
                'day/trade' : {
                    '__information__': 'daily trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
                    '__data_func__'  : get_trade_day,'compress':False} ,
            }
        elif key == 'trade_Xday':
            db_update_params = {
                f'{x_day}day/trade' : {
                    '__information__': f'daily {x_day} days trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
                    '__data_func__'  : get_trade_Xday, 'compress' : True , 
                    'src_update' : self.UPDATER , 
                    'x_day':x_day , } for x_day in [5,10,20]
            }
        elif key == 'trade_min':
            db_update_params = {
                'minute/trade' : {
                    '__information__': 'daily minute trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
                    '__data_func__'  : get_trade_min, 'compress' : True} ,
            }
        elif key == 'trade_Xmin':
            db_update_params = {
                f'{x_minute}min/trade' : {
                    '__information__': f'daily {x_minute} minute trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
                    '__data_func__'  : get_trade_Xmin, 'compress' : True , 
                    'src_updater' : self.UPDATER , 
                    'x_minute':x_minute , } for x_minute in [5,10,15,30,60]
            }
        elif key == 'labels':
            db_update_params = {}
            for days in [5,10,20]:
                for lag1 in [True , False]:
                    db_update_params[f'{days}days/lag{int(lag1)}'] = {
                        '__information__': f'forward labels of {days} days, with lag {lag1}' ,
                        '__date_source__': f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
                        '__data_func__'  : get_labels , 'compress':False ,
                        'days' : days , 'lag1' : lag1 ,
                    }
        else:
            raise Exception(key)
        [param.update({'__create_time__' : stime()}) for param in db_update_params.values()]
        return db_path , db_update_params

    def update_unfetch(self , db_path , path , unfetch):
        if db_path not in self.UNFETCH.keys(): 
            self.UNFETCH[db_path] = {}
        if path not in self.UNFETCH[db_path].keys(): 
            self.UNFETCH[db_path][path] = DataFailed()
        self.UNFETCH[db_path][path].update(unfetch)

    def update_data(self , data , dtank , path , compress = None , **kwargs):
        if isinstance(path , (list , tuple)): path = inner_path_join(path)
        if isinstance(data , DataFailed):
            self.update_unfetch(dtank.filename , path , data)
            return NotImplemented
        if isinstance(self.UPDATER , DataTank):  
            inner_path = inner_path_join(updater_path_to_key(dtank.filename), path)
            self.update_order.append(updater_path_to_key(dtank.filename))
            self.UPDATER.write_guess(inner_path , data = data , overwrite = True , compress = compress)
        else:
            dtank.write_guess(path , data = data , overwrite = True , compress = compress)
            if dtank.filename not in self.UPDATER.keys(): self.UPDATER[dtank.filename] = {}

    def update_attr(self , attrs , dtank , path , **kwargs):
        _create = ['__information__' , '__create_time__'] # on creation
        _update = ['__last_date__'   , '__update_time__'] # End of Loop
        attrs_create = {k:v for k,v in attrs.items() if k in _create}
        attrs_update = {k:v for k,v in attrs.items() if k in _update}
        if isinstance(path , (list , tuple)): path = '/'.join(path)
        if isinstance(self.UPDATER , DataTank):  
            inner_path = inner_path_join(updater_path_to_key(dtank.filename), path)
            self.UPDATER.set_group_attrs(inner_path , overwrite = False , **attrs_create)
            self.UPDATER.set_group_attrs(inner_path , overwrite = True  , **attrs_update)
        else:
            dtank.set_group_attrs(path , overwrite = False , **attrs_create)
            dtank.set_group_attrs(path , overwrite = True  , **attrs_update)

    def update_by_key(self , db_path , update_params):
        dtank = None
        try:
            db_file = get_db_file(db_path)
            if os.path.exists(db_file) and not isinstance(self.UPDATER , DataTank):
                os.remove(db_file)
            dtank = DataTank(db_file , 'r+')
            for path , param in update_params.items():
                __start_time__ = time.time()

                func_kwargs = {k:v for k,v in param.items() if k not in l_require_params}
                data = param['__data_func__'](path , **func_kwargs)  
                param.update({'__update_time__':stime()})

                self.update_data(data, dtank, path, **func_kwargs)
                self.update_attr(param , dtank , path)

                if isinstance(self.UPDATER , DataTank):
                    target_str = f'UPDATER ~ {os.path.relpath(db_path)}/{path}'
                else:
                    target_str = f'{os.path.basename(db_path)} ~ {path}'
                print(f'{time.ctime()} : {target_str} Done! Cost {time.time() - __start_time__:.2f} Secs')
        except Exception as e:
            traceback.print_exc()
        finally:
            if isinstance(dtank , DataTank): dtank.close()

    def update_by_date(self , db_path , update_params , start_dt = None , end_dt = None):
        self.update_order.append(db_path)
        dtank = None
        try:
            for path , param in update_params.items():
                __start_time__ = time.time()

                func_kwargs = {k:v for k,v in param.items() if k not in l_require_params}
                new_dates = get_new_dates(param['__date_source__'] , db_path , path , start_dt = start_dt , end_dt = end_dt)
                new_group = get_date_groups(new_dates)
                
                for group in np.unique(new_group):
                    db_file , dates_group = get_db_file(db_path , group) , new_dates[new_group == group]
                    dtank = DataTank(db_file , 'r+')

                    valid_day = 0
                    for date in dates_group:
                        data = param['__data_func__'](date , **func_kwargs)         
                        self.update_data(data , dtank , inner_path_join(path , date) , **func_kwargs)
                        if not isinstance(data , DataFailed): 
                            valid_day += 1
                            print(f'{time.ctime()} : {path} {date} Done.' , end = '\r')
                    param.update({'__update_time__':stime(),'__last_date__':max(dates_group)})
                    if valid_day > 0: self.update_attr(param , dtank , path)
                    dtank.close()

                    if isinstance(self.UPDATER , DataTank):
                        target_str = f'UPDATER ~ {os.path.relpath(db_file)}/{path}'
                    else:
                        target_str = f'{os.path.relpath(db_file)}/{path}'
                    print(f'{time.ctime()} : {target_str} of {valid_day} dates Done! Cost {time.time() - __start_time__:.2f} Secs')
        except Exception as e:
            traceback.print_exc()
        finally:
            if isinstance(dtank , DataTank): dtank.close()

    def update_something(self , KEY , start_dt = None , end_dt = None):
        # selected DB is totally refreshed , so delete first
        if not isinstance(KEY , (list,tuple)): KEY = [KEY]
        for key in KEY:
            db_path , db_update_params = self.get_db_params(key)
            if key in l_update_by_key:
                self.update_by_key(db_path , db_update_params)
            elif key in l_update_by_date:
                self.update_by_date(db_path , db_update_params , start_dt , end_dt)
            else:
                raise Exception(key)

    def _update_information(self , KEY = 'information'):
        self.update_by_key(*self.get_db_params(KEY))

    def _update_models(self , start_dt = None , end_dt = None , KEY = 'models'):
        self.update_by_date(*self.get_db_params(KEY) , start_dt , end_dt)

    def close(self):
        if isinstance(self.UPDATER , DataTank): self.UPDATER.close()

    def print_current_unfetch(self):
        unfetch_keys = list(self.UNFETCH.keys())
        _order = [*self.update_order , *unfetch_keys]
        unfetch_keys.sort(key = lambda x:_order.index(x))
        for key in unfetch_keys:
            for _path , _unfetch in self.UNFETCH[key].items():
                if len(_unfetch) > 0: print(_path,_unfetch) 

    def dump_exist_updaters(self , del_after_dumping = True):
        old_updaters = self.get_updater_paths()
        if del_after_dumping:
            print(old_updaters)
            if input(f'''Delete {len(old_updaters)} updaters after completion? (press yes/y) : 
                        {old_updaters}''')[0].lower() != 'y': 
                del_after_dumping = False

        for updater_path in old_updaters:
            with DataTank(updater_path , 'r') as updater:
                dump_updater(updater , key_order = self.update_order)
                
        if del_after_dumping:
            for updater_path in old_updaters: os.remove(updater_path)

    def dump_current_updater(self):
        dump_updater(self.UPDATER , key_order = self.update_order)

def get_target_dates(db_path , inner_path):
    old_keys = []
    for db_basename in os.listdir(db_path):
        with DataTank(outer_path_join(db_path , db_basename) , 'r') as dtank:
            portal = dtank.get_object(inner_path)
            if portal is not None: old_keys.append(list(portal.keys())) # type: ignore
    if len(old_keys) == 0:
        target_dates = np.array([])
    else:
        target_dates = np.unique(np.concatenate(old_keys)).astype(int)
    return target_dates

def get_source_dates(date_source):
    if isinstance(date_source , str):
        source_dates = np.array(get_path_date(get_directory_files(date_source)))
    elif isinstance(date_source , h5py.Group):
        source_dates = np.array(list(date_source.keys())).astype(int)
    else:
        source_dates = np.array(date_source)
    return source_dates

def get_new_dates(date_source , db_path , inner_path , start_dt = None , end_dt = None , trace = 0 , incremental = True):
    source_dates = get_source_dates(date_source)
    target_dates = get_target_dates(db_path , inner_path)

    if incremental: 
        if len(target_dates) == 0:
            print(f'No {db_path} , do not know when to start updating! update all source dates')
        else:
            source_dates = source_dates[source_dates >= min(target_dates)]
    if trace > 0 and len(target_dates) > 0: target_dates = target_dates[:-trace]

    new_dates = np.setdiff1d(source_dates , target_dates)
    if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
    if end_dt   is not None: new_dates = new_dates[new_dates <= end_dt  ]

    return new_dates

def dump_updater(updater , print_tree = l_print_tree , key_order = []):
    assert isinstance(updater , (DataTank,dict)) , updater
    if isinstance(updater , dict) and not print_tree: return NotImplemented
    db_path_list = list(updater.keys())
    key_order += db_path_list
    db_path_list.sort(key = lambda x:key_order.index(x))
    copy_mode = isinstance(updater , DataTank)

    for db_path in db_path_list:
        true_path = updater_key_to_path(db_path)
        os.makedirs(os.path.dirname(true_path),exist_ok=True)
        key  = os.path.basename(true_path).split('.')[0].replace('DB_','')

        if copy_mode and (key in l_update_by_key) and os.path.exists(true_path): 
            os.remove(true_path)
        try:
            dtank = DataTank(true_path ,  'r+' if copy_mode else 'r')
            if copy_mode: 
                copy_tree(updater , db_path , dtank , '.' , print_pre = f'{os.path.basename(str(updater.filename))} > ')
            if print_tree:
                print(f'Current {db_path} Tree:')
                dtank.print_tree()
        except:
            traceback.print_exc()
        finally:
            dtank.close()

def repack_DB_bygroup(source_tank_path , db_key = None , compress = None):
    assert db_key is not None
    l_update_by_date.append(db_key)
    db_path = get_db_path(db_key , make = False)
    assert not os.path.exists(db_path) , f'{db_path} exists!'
    os.makedirs(db_path , exist_ok=True)

    if input(f'Repack to {db_path} , press yes to confirm') != 'yes': 
        return NotImplemented
    
    try:
        source_dtank = DataTank(source_tank_path , 'r')
        target_dtank = DataTank()
        
        source_leaves = source_dtank.leaf_list(call_list = ['is_Data1D' , 'is_compress'])
        # assert all([leaf['is_Data1D'] for leaf in source_leaves])
        print(f'Prepare Repacking into {db_path} groups')

        target_dtank_path = {}
        target_leaves = {}
        
        for leaf in source_leaves:
            group = str(get_date_groups(int(leaf.get('key'))))
            if group not in target_dtank_path.keys(): 
                target_dtank_path[group]  = get_db_file(db_path,group=group)
                target_leaves[group] = []
            target_leaves[group].append(leaf)

        print(f'{db_path} has {len(target_dtank_path)} groups')
        for group in sorted(list(target_dtank_path.keys())):
            print(f'{db_path} groups {group} ...' , end = '')
            target_dtank = DataTank(target_dtank_path[group] , 'w')
            for leaf in target_leaves[group]:
                if not leaf['is_Data1D']: continue
                data = source_dtank.read_data1D(leaf['path'] , none_if_incomplete = True)
                target_dtank.write_data1D(leaf['path'] , data = data , compress = compress or leaf['is_compress'])
            target_dtank.close()
            print(f'... Done')
    except:
        traceback.print_exc()
    finally:
        source_dtank.close()
        target_dtank.close()

def update_server_main():
    __start_time__ = time.time()
    assert socket.gethostname() == 'mengkjin-server' , socket.gethostname()
    try:
        print(f'Merge Update Files') 
        Updater = DataUpdater(False)
        Updater.dump_exist_updaters(del_after_dumping=True)
    except:
        traceback.print_exc()
    finally:
        Updater.close()
    print(f'{time.ctime()} : All Updates Done! Cost {time.time() - __start_time__:.2f} Secs')

def update_laptop_main():
    __start_time__ = time.time()
    assert socket.gethostname() != 'mengkjin-server' , socket.gethostname()
    try:
        print(f'Create Update Files')
        Updater = DataUpdater(l_do_updater)
        Updater.update_something(KEY = l_update_by_key + l_update_by_date)
        Updater.print_current_unfetch()
        Updater.dump_current_updater()
    except:
        traceback.print_exc()
    finally:
        Updater.close()
    print(f'{time.ctime()} : All Updates Done! Cost {time.time() - __start_time__:.2f} Secs')

def main():
    if socket.gethostname() == 'mengkjin-server':
        update_server_main()
    else:
        update_laptop_main()