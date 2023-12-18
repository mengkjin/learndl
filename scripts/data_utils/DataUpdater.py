# %%

import h5py
import pandas as pd
import numpy as np
import random, string , os , time
import traceback
from .DataTank import *
from .DataTransmitter import *

# %%
target_files = {
    'information' : 'DB_information.h5' ,
    'model_data'  : 'DB_models.h5' ,
    'day_trade'   : 'DB_trade_day.h5' , 
    'min_trade'   : 'DB_trade_min.h5' , 
    'Xmin_trade'  : 'DB_trade_Xmin.h5' , 
    'labels'      : 'DB_labels.h5' , 
}
updater_file = 'DataUpdater.h5'
updater_mode = 'w'
do_updater = False
updater = None
attr_create = ['__information__' , '__create_time__'] # on creation
attr_EOL    = ['__last_date__'   , '__update_time__'] # End of Loop

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

def _update_by_date(dtank , file_params , start_dt = None , end_dt = None , 
                    required_param_keys=['__information__','__create_time__','__date_source__','__data_func__']):
    assert isinstance(dtank , DataTank) , dtank
    param_keys = {k:list(param.keys()) for k,param in file_params.items()}
    for k,pk in param_keys.items(): assert np.isin(required_param_keys , pk).all() , (k,pk)

    unfetched = unfetched_data()
    for dtank_file , param in file_params.items():
        print(dtank_file)
        __start_time__ = time.time()
        func_kwargs = {k:v for k,v in param.items() if k not in required_param_keys}
        
        if dtank.get_object(dtank_file) is None:
            dtank.create_object(dtank_file)
            dtank.set_group_attrs(dtank_file , **{k:param[k] for k in attr_create})
        
        new_dates , old_dates = _get_dates(param['__date_source__'] , dtank.get_object(dtank_file).keys())
        if start_dt is not None: new_dates = new_dates[new_dates >= start_dt]
        if end_dt is not None: new_dates = new_dates[new_dates <= end_dt]
        for date in new_dates:
            data = param['__data_func__'](date , **func_kwargs)
            if isinstance(data , unfetched_data):
                unfetched.update(data)
            else:
                dtank.write_data1D([dtank_file,str(date)] , data , overwrite=True)
                if updater is not None: 
                    updater.write_data1D([dtank.filename,dtank_file,str(date)] , data , overwrite=True)
                print(f'{time.ctime()} : {date} finished' , end = '\r')
        param.update({'__update_time__':_str_time(),'__last_date__':max(*old_dates,*new_dates)})
        dtank.set_group_attrs(dtank_file , **{k:param.get(k) for k in attr_EOL})
        print(f'{dtank_file} {len(new_dates)} of dates finished! Cost {time.time() - __start_time__:.2f} Secs')

    print(f'Current {dtank.filename} Tree:')
    dtank.print_tree()
    print(f'Unfetched Dates:')
    print(unfetched.dates)

def choose_data_tank(type , open = True , mode = 'guess'):
    return DataTank(target_files[type] , open = open , mode = mode)

def update_information():
    dtank = choose_data_tank('information')
    _ctime , _dfunc = _str_time() , get_basic_information
    file_params = {
        'basic/calendar'   : {'__information__': 'market trading calendar'},
        'stock/description': {'__information__': 'stock list_date and delist_date'},
        'stock/st'         : {'__information__': 'st treatment of stocks'},
        'stock/industry'   : {'__information__': 'SW 2021 industry criterion'},
        'stock/concepts'   : {'__information__': 'wind concepts'},}
    for param in file_params.values(): param.update({'__create_time__' : _ctime , '__data_func__' : _dfunc ,})
    try :
        for dtank_file , param in file_params.items():
            key = dtank_file.split('/')[-1]
            if dtank.get_object(dtank_file) is None:
                dtank.create_object(dtank_file)
                dtank.set_group_attrs(dtank_file , **{k:param[k] for k in attr_create})
            data = param['__data_func__'](key)
            dtank.write_dataframe(dtank_file , data = data , overwrite = True)
            param.update({'__update_time__':_str_time()})
            dtank.set_group_attrs(dtank_file , **{k:param.get(k) for k in attr_EOL})
            if updater is not None: updater.write_dataframe([dtank.filename,dtank_file] , data = data , overwrite = True)
            print(f'{dtank_file} Done!')
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

def update_model_data(start_dt = None , end_dt = None):
    dtank = choose_data_tank('model_data')
    _ctime = _str_time()
    file_params = {
        'risk_model/exposure' : {
            '__information__': 'market, industry, style exposure of risk model jm2018' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
            '__create_time__': _ctime , '__data_func__' : read_risk_model,} ,
        'alpha_longcl/exposure' : {
            '__information__': 'sub alpha factor exposure of risk model jm2018' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst' ,
            '__create_time__': _ctime , '__data_func__' : read_alpha_longcl,}
    }
    try:
        _update_by_date(dtank , file_params , start_dt = start_dt , end_dt = end_dt)
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

def update_day_trade_data(start_dt = None , end_dt = None):
    dtank = choose_data_tank('day_trade')
    file_params = {
        'day/trade' : {
            '__information__': 'daily trading data of A shares' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
            '__create_time__': _str_time() , '__data_func__' : get_day_trade} ,
    }
    try:
        _update_by_date(dtank , file_params , start_dt = start_dt , end_dt = end_dt)
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

def update_min_trade_data(start_dt = None , end_dt = None):
    dtank = choose_data_tank('min_trade')
    file_params = {
        'minute/trade' : {
            '__information__': 'daily minute trading data of A shares' ,
            '__date_source__' : 'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin' ,
            '__create_time__': _str_time() , '__data_func__' : get_min_trade,} ,
    }
    try:
        _update_by_date(dtank , file_params , start_dt = start_dt , end_dt = end_dt)
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

def update_Xmin_trade_data(start_dt = None , end_dt = None):
    dtank = choose_data_tank('Xmin_trade')
    min_dtank = choose_data_tank('min_trade' , mode = 'r')
    file_params = {
        f'{x}min/trade' : {
            '__information__': f'daily {x} minute trading data of A shares' ,
            '__date_source__': min_dtank.get_object('/minute/trade') ,
            '__create_time__': _str_time() , '__data_func__' : get_Xmin_trade,
            'min_DataTank': min_dtank , 'x':x} for x in [5,10,15,30,60]
    }
    try:
        _update_by_date(dtank , file_params , start_dt = start_dt , end_dt = end_dt)
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

def update_labels(start_dt = None , end_dt = None):
    dtank = choose_data_tank('labels')
    days , lag1 = [5,10,20] , [True,False]
    kwargs_list , file_params = {} , {}
    [[kwargs_list.update({f'{d}days/lag{int(l)}':{'days':d,'lag1':l}}) for l in lag1] for d in days]
    for k , v in kwargs_list.items():
        file_params[k] = {
            '__information__': f'forward labels of {v["days"]} days, with lag {v["lag1"]}' ,
            '__date_source__' : f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
            '__create_time__': _str_time() , '__data_func__' : get_labels , **v}
    try:
        _update_by_date(dtank , file_params , start_dt = start_dt , end_dt = end_dt)
    except Exception as e:
        traceback.print_exc()
    finally :
        dtank.close()

if __name__ == '__main__':
    if do_updater:
        if os.path.exists(updater_file): os.remove(updater_file)
        updater = DataTank(updater_file , open = True , mode = updater_mode)
        updater_mode = 'r+'

    update_information()
    update_model_data()
    update_day_trade_data()
    update_min_trade_data()
    update_Xmin_trade_data()
    update_labels()
    