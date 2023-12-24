import h5py
import numpy as np
import os , time
import traceback
import socket

from .DataTransmitter import *
from .DataTank import *

# %%
do_updater = True

def _get_data_dir():
    if socket.gethostname() == 'mengkjin-server':
        from scripts.util.environ import DIR_data
        return DIR_data
    else:
        return '.'

def _get_updater_path(title = 'DB_updater'):
    data_dir = _get_data_dir()
    return ['/'.join([data_dir,p]) for p in os.listdir('./') if p.startswith(title + '.')]

def _get_new_updater(title = 'DB_updater'):
    if do_updater:
        old_updaters = _get_updater_path(title)
        if len(old_updaters) == 0:
            n_updater = 0
        else:
            n_updater = max([int(os.path.basename(p).split('.')[1]) for p in old_updaters]) + 1
        return DataTank(f'{title}.{n_updater:d}.h5' , open = True , mode = 'guess')
    else:
        return None

def _get_db_path(key):
    DB_name = {
        'information' : 'DB_information.h5' ,
        'model_data'  : 'DB_models.h5' ,
        'day_trade'   : 'DB_trade_day.h5' , 
        'min_trade'   : 'DB_trade_min.h5' , 
        'Xmin_trade'  : 'DB_trade_Xmin.h5' , 
        'labels'      : 'DB_labels.h5' , 
    }[key]
    return '/'.join([_get_data_dir() , DB_name])

def _get_db_dtank(key , open = True , mode = 'guess'):
    return DataTank(_get_db_path(key) , open = open , mode = mode)

def _str_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

def _get_dates(date_source , old_keys , trace = 1):
    if isinstance(date_source , str):
        tar_dates = get_path_date(get_directory_files(date_source))
    elif isinstance(date_source , h5py.Group):
        tar_dates = np.array(list(date_source.keys())).astype(int)
    else:
        tar_dates = np.array(date_source)
    old_dates = np.array(list(old_keys)).astype(int)
    if trace > 0 and len(old_dates) > 0: old_dates = old_dates[:-trace]
    new_dates = np.setdiff1d(tar_dates , old_dates)
    return new_dates , old_dates

def _update_data(data , dtank , path , updater = None):
    if isinstance(path , (list , tuple)): path = '/'.join(path)
    if isinstance(data , pd.DataFrame):
        dtank.write_dataframe(path , data = data , overwrite = True)
    elif isinstance(data , Data1D):
        dtank.write_data1D(path , data = data , overwrite = True)
    else:
        raise Exception(TypeError) , type(data)
    if updater is not None: 
        _update_data(data,updater,[os.path.basename(dtank.filename), path])

def _update_attr(attrs , dtank , path , updater = None):
    _create = ['__information__' , '__create_time__'] # on creation
    _update = ['__last_date__'   , '__update_time__'] # End of Loop
    if isinstance(path , (list , tuple)): path = '/'.join(path)
    attrs_create = {k:v for k,v in attrs.items() if k in _create}
    attrs_update = {k:v for k,v in attrs.items() if k in _update}
    dtank.set_group_attrs(path , overwrite = False , **attrs_create)
    dtank.set_group_attrs(path , overwrite = True  , **attrs_update)
    if updater is not None: 
        _update_attr(attrs, updater, [os.path.basename(dtank.filename), path])

def _update_by_date(dtank , update_params , start_dt = None , end_dt = None , 
                    required_param_keys=['__information__','__create_time__','__date_source__','__data_func__']):
    try:
        assert isinstance(dtank , DataTank) , dtank
        param_keys = {k:list(param.keys()) for k,param in update_params.items()}
        for k,pk in param_keys.items(): assert np.isin(required_param_keys , pk).all() , (k,pk)

        unfetched = unfetched_data()
        for path , param in update_params.items():
            print(path)
            __start_time__ = time.time()
            func_kwargs = {k:v for k,v in param.items() if k not in required_param_keys}
            
            """
            if dtank.get_object(path) is None:
                dtank.create_object(path)
                dtank.set_group_attrs(path , **{k:param[k] for k in attr_create})
            """
            new_dates , old_dates = _get_dates(param['__date_source__'] , dtank.get_object(path).keys())
            if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
            if end_dt is not None: new_dates = new_dates[new_dates <= end_dt]
            for date in new_dates:
                data = param['__data_func__'](date , **func_kwargs)
                if isinstance(data , unfetched_data):
                    unfetched.update(data)
                else:
                    _update_data(data, dtank,[path,str(date)], updater)
                    """
                    dtank.write_data1D([path,str(date)] , data , overwrite=True)
                    if updater is not None: 
                        updater.write_data1D([dtank.filename,path,str(date)] , data , overwrite=True)
                    
                    """
                    print(f'{time.ctime()} : {date} finished' , end = '\r')
            param.update({'__update_time__':_str_time(),'__last_date__':max(*old_dates,*new_dates)})
            _update_attr(param , dtank , path , updater)
            # dtank.set_group_attrs(path , **{k:param.get(k) for k in attr_update})
            print(f'{path} {len(new_dates)} of dates finished! Cost {time.time() - __start_time__:.2f} Secs')

        print(f'Current {dtank.filename} Tree:')
        dtank.print_tree()
        print(f'Unfetched Dates:')
        print(unfetched.dates)
    
    except Exception as e:
        traceback.print_exc()
        if updater is not None: updater.close()
    finally :
        dtank.close()

def update_information():
    dtank = _get_db_dtank('information')
    _ctime , _dfunc = _str_time() , get_basic_information
    update_params = {
        'basic/calendar'   : {'__information__': 'market trading calendar'},
        'stock/description': {'__information__': 'stock list_date and delist_date'},
        'stock/st'         : {'__information__': 'st treatment of stocks'},
        'stock/industry'   : {'__information__': 'SW 2021 industry criterion'},
        'stock/concepts'   : {'__information__': 'wind concepts'},}
    for param in update_params.values(): param.update({'__create_time__' : _ctime , '__data_func__' : _dfunc ,})
    try :
        for path , param in update_params.items():
            key = path.split('/')[-1]
            
            """
            if dtank.get_object(path) is None:
                dtank.create_object(path)
                dtank.set_group_attrs(path , **{k:param[k] for k in attr_create})
            """
            data = param['__data_func__'](key)
            _update_data(data, dtank, path, updater)
            #dtank.write_dataframe(path , data = data , overwrite = True)
            #if updater is not None: updater.write_dataframe([dtank.filename,path] , data = data , overwrite = True)
            
            param.update({'__update_time__':_str_time()})
            _update_attr(param , dtank , path , updater)
            # dtank.set_group_attrs(path , **{k:param.get(k) for k in attr_update})
            print(f'{path} Done!')
    except Exception as e:
        traceback.print_exc()
        if updater is not None: updater.close()
    finally :
        dtank.close()

def update_model_data(start_dt = None , end_dt = None):
    dtank = _get_db_dtank('model_data')
    update_params = {
        'risk_model/exposure' : {
            '__information__': 'market, industry, style exposure of risk model jm2018' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
            '__create_time__': _str_time() , '__data_func__' : read_risk_model,} ,
        'alpha_longcl/exposure' : {
            '__information__': 'sub alpha factor exposure of risk model jm2018' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst' ,
            '__create_time__': _str_time() , '__data_func__' : read_alpha_longcl,}
    }
    _update_by_date(dtank , update_params , start_dt , end_dt)

def update_day_trade_data(start_dt = None , end_dt = None):
    dtank = _get_db_dtank('day_trade')
    update_params = {
        'day/trade' : {
            '__information__': 'daily trading data of A shares' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
            '__create_time__': _str_time() , '__data_func__' : get_day_trade} ,
    }
    _update_by_date(dtank , update_params , start_dt = start_dt , end_dt = end_dt)

def update_min_trade_data(start_dt = None , end_dt = None):
    dtank = _get_db_dtank('min_trade')
    update_params = {
        'minute/trade' : {
            '__information__': 'daily minute trading data of A shares' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
            '__create_time__': _str_time() , '__data_func__' : get_min_trade,} ,
    }
    _update_by_date(dtank , update_params , start_dt = start_dt , end_dt = end_dt)

def update_Xmin_trade_data(start_dt = None , end_dt = None):
    dtank = _get_db_dtank('Xmin_trade')
    min_dtank = _get_db_dtank('min_trade' , mode = 'r')
    update_params = {
        f'{x}min/trade' : {
            '__information__': f'daily {x} minute trading data of A shares' ,
            '__date_source__': min_dtank.get_object('/minute/trade') ,
            '__create_time__': _str_time() , '__data_func__' : get_Xmin_trade,
            'min_DataTank': min_dtank , 'x':x} for x in [5,10,15,30,60]
    }
    _update_by_date(dtank , update_params , start_dt = start_dt , end_dt = end_dt)

def update_labels(start_dt = None , end_dt = None):
    dtank = _get_db_dtank('labels')
    days , lag1 = [5,10,20] , [True,False]
    kwargs_list , update_params = {} , {}
    [[kwargs_list.update({f'{d}days/lag{int(l)}':{'days':d,'lag1':l}}) for l in lag1] for d in days]
    for k , v in kwargs_list.items():
        update_params[k] = {
            '__information__': f'forward labels of {v["days"]} days, with lag {v["lag1"]}' ,
            '__date_source__': f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
            '__create_time__': _str_time() , '__data_func__' : get_labels , **v}
    _update_by_date(dtank , update_params , start_dt = start_dt , end_dt = end_dt)

if __name__ == '__main__':
    if socket.gethostname() == 'mengkjin-server':
        pass
    else:
        updater = _get_new_updater()
        update_information()
        update_model_data()
        update_day_trade_data()
        update_min_trade_data()
        update_Xmin_trade_data()
        update_labels()



    