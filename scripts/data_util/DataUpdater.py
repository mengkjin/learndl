import h5py
import numpy as np
import os , time
import traceback
import socket , platform

try:
    from .DataTransmitter import *
    from .DataTank import *
except:
    from DataTransmitter import *
    from DataTank import *

# %%
do_updater = True
print_tree = False
remake_key_list = ['information']
required_params = ['__information__' , '__create_time__' , '__date_source__' , '__data_func__']

def get_data_dir():
    if socket.gethostname() == 'mengkjin-server':
        try:
            from scripts.util.environ import DIR_data
        except:
            DIR_data = '.'
        return DIR_data
    else:
        return '.'
DIR_data = get_data_dir()
        
def stime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

def inner_path_join(*args):
    return '/'.join([str(arg) for arg in args])

def outer_path_join(*args):
    return os.path.join(*args)

def updater_key_to_path(key):
    os_name = platform.system()
    if os_name == 'Linux':
        return outer_path_join(*key.split('\\'))
    else:
        return key

class DataUpdater():
    def __init__(self , do_updater = False) -> None:
        self.do_updater = do_updater
        self.DIR_db  = self._get_db_dir()
        self.UPDATER = self.get_new_updater()
        self.UNFETCH = dict()
        self.update_order = []
    
    def _get_db_dir(self):
        dir_db = 'DB_data'
        os.makedirs(outer_path_join(DIR_data , dir_db) , exist_ok=True)
        return dir_db

    def get_updater_paths(self , title = 'DB_updater'):
        # order matters!
        search_dirs = [outer_path_join(DIR_data , self.DIR_db)]
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
            return DataTank(outer_path_join(DIR_data , self.DIR_db , f'{title}.{updater_n:d}.h5') , open = True , mode = 'guess')
        else:
            return dict()

    def get_db_path(self , key):
        db_key  = f'DB_{key}'
        if key == 'information':
            db_name = db_key + '.h5'
        else:
            db_name = outer_path_join(db_key , db_key+'.{}.h5')
        db_path = outer_path_join(self.DIR_db , db_name)
        os.makedirs(os.path.dirname(outer_path_join(DIR_data , db_path)) , exist_ok=True)
        return db_path
    
    def get_db_params(self , key):
        # db_file path
        db_path = self.get_db_path(key)

        # db_update_parameters
        if key == 'information':
            db_update_params = {
                'basic/calendar'   : {'__information__': 'market trading calendar'},
                'stock/description': {'__information__': 'stock list_date and delist_date'},
                'stock/st'         : {'__information__': 'st treatment of stocks'},
                'stock/industry'   : {'__information__': 'SW 2021 industry criterion'},
                'stock/concepts'   : {'__information__': 'wind concepts'},}
            [param.update({'__data_func__':get_basic_information,'compress':True}) for param in db_update_params.values()]
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
        elif key == 'trade_min':
            db_update_params = {
                'minute/trade' : {
                    '__information__': 'daily minute trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
                    '__data_func__'  : get_trade_min, 'compress' : True} ,
            }
        elif key == 'trade_Xmin':
            db_path_min = self.get_db_path('trade_min')
            db_update_params = {
                f'{x_minute}min/trade' : {
                    '__information__': f'daily {x_minute} minute trading data of A shares' ,
                    '__date_source__': 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
                    '__data_func__'  : get_trade_Xmin, 'compress' : True , 
                    'src_min' : db_path_min , 'src_update' : self.UPDATER , 'x_minute':x_minute , 
                } for x_minute in [5,10,15,30,60]
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
            self.UNFETCH[db_path][path] = unfetched_data()
        self.UNFETCH[db_path][path].update(unfetch)

    def update_data(self , data , dtank , path , compress = None , **kwargs):
        if isinstance(path , (list , tuple)): path = inner_path_join(path)
        if isinstance(data , unfetched_data):
            self.update_unfetch(dtank.filename , path , data)
            return NotImplemented
        if isinstance(self.UPDATER , DataTank):  
            self.UPDATER.write_guess(inner_path_join(dtank.filename, path) , data = data , overwrite = True , compress = compress)
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
            self.UPDATER.set_group_attrs(inner_path_join(dtank.filename, path)  , overwrite = False , **attrs_create)
            self.UPDATER.set_group_attrs(inner_path_join(dtank.filename, path)  , overwrite = True  , **attrs_update)
        else:
            dtank.set_group_attrs(path , overwrite = False , **attrs_create)
            dtank.set_group_attrs(path , overwrite = True  , **attrs_update)

    def update_by_key(self , db_path , update_params , remake = False):
        self.update_order.append(db_path)
        dtank = None
        try:
            if remake: os.remove(outer_path_join(DIR_data , db_path))
            dtank = DataTank(outer_path_join(DIR_data , db_path) , open = True , mode = 'r+')
            for path , param in update_params.items():
                __start_time__ = time.time()

                func_kwargs = {k:v for k,v in param.items() if k not in required_params}
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

    def update_by_date(self , db_path , update_params , start_dt = None , end_dt = None , remake = False):
        self.update_order.append(db_path)
        dtank = None
        try:
            for path , param in update_params.items():
                __start_time__ = time.time()

                func_kwargs = {k:v for k,v in param.items() if k not in required_params}
                new_dates = get_new_dates(param['__date_source__'] , db_path , path , start_dt = start_dt , end_dt = end_dt)
                new_group = get_date_groups(new_dates)
                
                for group in np.unique(new_group):
                    db_path_group , dates_group = db_path.format(group) , new_dates[new_group == group]
                    dtank = DataTank(db_path_group , open = True , mode = 'r+')

                    valid_day = 0
                    for date in dates_group:
                        data = param['__data_func__'](date , group = group , **func_kwargs)                      
                        self.update_data(data , dtank , inner_path_join(path , date) , **func_kwargs)
                        if not isinstance(data , unfetched_data): 
                            valid_day += 1
                            print(f'{time.ctime()} : {path} {date} Done.' , end = '\r')
                    param.update({'__update_time__':stime(),'__last_date__':max(dates_group)})
                    if valid_day > 0: self.update_attr(param , dtank , path)
                    dtank.close()

                    if isinstance(self.UPDATER , DataTank):
                        target_str = f'UPDATER ~ {db_path}/{path}'.format(group)
                    else:
                        target_str = f'{os.path.basename(db_path)} ~ {path}'.format(group)
                    print(f'{time.ctime()} : {target_str} of {valid_day} dates Done! Cost {time.time() - __start_time__:.2f} Secs')
        except Exception as e:
            traceback.print_exc()
        finally:
            if isinstance(dtank , DataTank): dtank.close()

    def update_something(self , KEY , start_dt = None , end_dt = None):
        # selected DB is totally refreshed , so delete first
        db_path , db_update_params = self.get_db_params(KEY)

        if KEY in ['information']:
            self.update_by_key(db_path , db_update_params , remake = KEY in remake_key_list)
        elif KEY in ['models' , 'trade_day' , 'trade_min' , 'trade_Xmin' , 'labels']:
            self.update_by_date(db_path , db_update_params , start_dt , end_dt , remake = KEY in remake_key_list)
        else:
            raise Exception(KEY)

    def _update_information(self , KEY = 'information'):
        self.update_by_key(*self.get_db_params(KEY))

    def _update_models(self , start_dt = None , end_dt = None , KEY = 'models'):
        self.update_by_date(*self.get_db_params(KEY) , start_dt , end_dt)

    def close(self):
        if isinstance(self.UPDATER , DataTank): self.UPDATER.close()

    def dump_exist_updaters(self):
        old_updaters = self.get_updater_paths()
        updater = None
        try:
            for updater_path in old_updaters:
                updater = DataTank(updater_path , True , 'r')
                dump_updater(updater)
                updater.close()
        except:
            traceback.print_exc()
        finally:
            if isinstance(updater , DataTank): updater.close()

    def del_exist_updaters(self):
        if not input('''You must confirm to delete existing updaters 
                     (press yes/y)''').lower() in ['y','yes']: 
            return NotImplemented
        old_updaters = self.get_updater_paths()
        print(old_updaters)
        if not input(f'''Delete {len(old_updaters)} updaters (press yes/y) : 
                     {old_updaters}''').lower() in ['y','yes']: 
            return NotImplemented
        for updater_path in old_updaters:
            os.remove(updater_path)

def get_target_dates(db_name , path):
    db_dir , old_keys = os.path.dirname(db_name) ,  []
    for db_basename in os.listdir(db_dir):
        with DataTank(outer_path_join(DIR_data , db_dir , db_basename) , True , 'r') as dtank:
            portal = dtank.get_object(path)
            if portal is not None: old_keys.append(list(portal.keys()))
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

def get_new_dates(date_source , db_name , path , start_dt = None , end_dt = None , trace = 0 , incremental = True):
    source_dates = get_source_dates(date_source)
    target_dates = get_target_dates(db_name , path)

    if incremental: 
        if len(target_dates) == 0:
            print(f'No {db_name} , do not know when to start updating! update all source dates')
        else:
            source_dates = source_dates[source_dates >= min(target_dates)]
    if trace > 0 and len(target_dates) > 0: target_dates = target_dates[:-trace]

    new_dates = np.setdiff1d(source_dates , target_dates)
    if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
    if end_dt   is not None: new_dates = new_dates[new_dates <= end_dt  ]

    return new_dates

def get_date_groups(dates):
    # return year
    return np.array(dates).astype(int) // 10000

def dump_updater(Updater , print_tree = print_tree):
    if isinstance(Updater , DataUpdater):
        updater = Updater.UPDATER
        unfetch = Updater.UNFETCH
        db_path_list = list(updater.keys())
        _order = [*Updater.update_order , *db_path_list]
        db_path_list.sort(key = lambda x:_order.index(x))
    elif isinstance(Updater , (DataTank , str)):
        updater = Updater if isinstance(Updater , DataTank) else DataTank(Updater , True , 'r')
        unfetch = {}
        db_path_list = list(updater.keys())

    for db_path in db_path_list:
        true_path = outer_path_join(DIR_data , updater_key_to_path(db_path))
        key  = os.path.basename(true_path).split('.')[0].replace('DB_','')
        mode ='guess' 
        if not isinstance(updater , DataTank): 
            mode = 'r'
        elif key in remake_key_list:
            if os.path.exists(true_path): os.remove(true_path)
            mode = 'w'
        try:
            dtank = DataTank(true_path ,  open = True , mode = mode)
            if isinstance(dtank , DataTank):
                copy_tree(updater , db_path , dtank , '.' , 
                        print_pre = f'{os.path.basename(updater.filename)} > ')
            if print_tree:
                print(f'Current {db_path} Tree:')
                dtank.print_tree()
            if db_path in unfetch.keys(): 
                [print(_path,_unfetch) for _path,_unfetch in unfetch[db_path].items() if len(_unfetch) > 0]
        except:
            traceback.print_exc()
        finally:
            dtank.close()

if __name__ == '__main__':
    __start_time__ = time.time()
    if socket.gethostname() == 'mengkjin-server':
        try:
            Updater = DataUpdater(False)
            Updater.dump_exist_updaters()
            Updater.del_exist_updaters()
        except:
            traceback.print_exc()
        finally:
            Updater.close()
    else:
        try:
            Updater = DataUpdater(do_updater)
            update_list = ['information' ,
                           'models' , 'trade_day' , 'trade_min' , 'trade_Xmin' , 'labels']
            for key in update_list:
                Updater.update_something(KEY = key)
            dump_updater(Updater)
        except:
            traceback.print_exc()
        finally:
            Updater.close()
    print(f'{time.ctime()} : All Updates Done! Cost {time.time() - __start_time__:.2f} Secs')

# %%

