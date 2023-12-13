# %%
import pyreadr
import pandas as pd
import numpy as np
import random, string , os , time
from .DataTank import *

# %%
target_files = {
    'information' : 'ModelDatabase.h5' ,
    'model_data'  : 'ModelDatabase.h5' ,
    'trade_data'  : 'TradeDatabase.h5' , 
    'labels'      : 'ModelLabels.h5' , 
}
updater_file = 'DataUpdater.h5'
updater_mode = 'w'
do_updater = False
updater = None
attr_create = ['__information__' , '__create_time__'] # on creation
attr_EOL    = ['__last_date__'   , '__update_time__'] # End of Loop

# %%
def get_path_date(path):
    if isinstance(path , (list,tuple)):
        return [int(os.path.basename(p).split('.')[-2][-8:]) for p in path]
    else:
        return int(os.path.basename(path).split('.')[-2][-8:])

def get_directory_files(directory , fullname = False):
    paths = os.listdir(directory)
    return [os.path.join(directory , p) for p in paths] if fullname else paths

def get_directory_dates(directory):
    return get_path_date(get_directory_files(directory))

def windid_to_secid(x):
    replace_dict = {
        'T00018' : '600018'
    }
    x = x.str.slice(0, 6).replace(replace_dict)
    x[x.str.isdigit() == 0] = '-1'
    return x.astype(int)

def col_reform(df , col , newcol = None , fillna = None , astype = None , use_func = None):
    if newcol is None: newcol = col
    if use_func is not None:
        df[newcol] = use_func(df[col])
    else:
        x = df[col]
        if fillna is not None: x = x.fillna(fillna)
        if astype is not None: x = x.astype(astype)
        df[newcol] = x 
    return df

def col_filter(df , remain = None):
    if remain is None:
        pass
    else:
        if isinstance(remain , str): remain = [remain]
        df = df.loc[:,remain]
    return df

def row_filter(df , col , cond_func = lambda x:x):
    return df[cond_func(df[col])]

def read_risk_model(date = None , path = None , tol = 1e-8 , **kwargs):
    path = f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model/jm2018_model_{date}.csv'
    if not os.path.exists(path): return NotImplemented
    df = pd.read_csv(path)
    df['wind_id'] = windid_to_secid(df['wind_id'])
    df = df.rename(columns={'wind_id':'secid'})
    df[df.abs() < tol] = 0
    df['secid'] = df['secid'].astype(int)
    return df

def read_alpha_longcl(date , tol = 1e-8 , **kwargs):
    a_names = {
        'value'       : 'A7_Value' ,
        'analyst'     : 'A1_Analyst' ,
        'momentum'    : 'A5_Momentum' ,
        'correlation' : 'A2_Corr' ,
        'growth'      : 'A3_Growth' ,
        'volatility'  : 'A8_Volatility' ,
        'liquidity'   : 'A4_Liquidity' ,
        'quality'     : 'A6_Quality' ,
        'industrymom' : 'IndustryFactor' ,
        'riskindex'   : 'RiskIndex' ,
        'pred_fundamental' : 'Fundamental' ,
        'pred_behavioral'  : 'Behavior' ,
        'pred_final'  : 'Final'  ,
        'pred_multi'  : 'MultiFactorAll'
    }
    df = pd.DataFrame(columns=['secid'],dtype = int).set_index('secid')
    for k,v in a_names.items():
        colnames = ['secid',v]
        path = f'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/{v}/{v}_{date}.txt'
        if not os.path.exists(path):
            df_new = pd.DataFrame(columns=colnames,dtype=float)
        else:
            df_new = pd.read_csv(path, header=None , delimiter='\t',dtype=float)
            df_new.columns = colnames
        df_new['secid'] = df_new['secid'].astype(int)
        df = pd.merge(df , df_new.set_index('secid') , how='outer' , on='secid')
    df[df.abs() < tol] = 0
    df = df.reset_index()
    return df

def get_basic_information(key = None , **kwargs):
    if key is None: return 
    assert key in ['stock_info','stock_st','industry','concepts','calendar']
    d_secid = {'wind_id'   : {'newcol' : 'secid' , 'use_func' : windid_to_secid}}
    d_entrm = {'entry_dt'  : {'fillna' : -1 , 'astype' : int} ,
                      'remove_dt' : {'fillna' : 99991231 , 'astype' : int}}
    params = {
        'stock_info': {
            'path': f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_description.csv' ,
            'remain_cols' : ['wind_id' , 'secid' , 'sec_name' , 'exchange_name' , 'list_dt' , 'delist_dt'] ,
            col_reform :  {**d_secid , 
                           'list_dt' : {'fillna' : -1 , 'astype' : int} ,
                           'delist_dt' : {'fillna' : 99991231 , 'astype' : int}} ,
        } ,
        'stock_st' : {
            'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_st.csv' ,
            'remain_cols' : ['wind_id' , 'secid' , 'st_type' , 'entry_dt' , 'remove_dt' , 'ann_dt'] ,
            col_reform : {**d_secid , **d_entrm , 'ann_dt' : {'fillna' : -1 , 'astype' : int}} ,
            row_filter : {'st_type' : {'cond_func' : lambda x:x != 'R'}}
        },
        'industry' : {
            'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_industries_class_sw_2021.csv' ,
            'remain_cols' : ['wind_id', 'secid' , 'entry_dt', 'remove_dt', 'ind_code', 
                             'ind_code_1', 'chn_name_1', 'abbr_1', 'indexcode_1' ,
                             'ind_code_2', 'chn_name_2', 'abbr_2', 'indexcode_2' ,
                             'ind_code_3', 'chn_name_3', 'abbr_3', 'indexcode_3'] ,
            col_reform : {**d_secid , **d_entrm} ,
        },
        'concepts' : {
            'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_wind_concepts.csv' ,
            'remain_cols' : ['wind_id' , 'secid' , 'concept' , 'entry_dt' , 'remove_dt'] ,
            col_reform : {**d_secid , **d_entrm , 'wind_sec_name' : {'newcol' : 'concept'}} ,
        },
        'calendar' : {
            'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_calendar.csv' ,
            'dtype' : int , 'remain_cols' : None ,
        },
    }
    df = pd.read_csv(params[key]['path'] , encoding='gbk' , dtype = params[key].get('dtype'))
    if key == 'industry':
        path_dict = f'D:/Coding/ChinaShareModel/ModelParameters/setting/indus_dictionary_sw.csv'
        ind_dict = pd.read_csv(path_dict , encoding='gbk')
        ind_dict = ind_dict[ind_dict['version'] == 2021]        
        for i in range(3): 
            df[f'ind_code_{i+1}'] = df['ind_code'].str.slice(0 , 4 + 2*i)
            tmp = {
                f'ind_code_{i+1}'  : ind_dict['ind_code'].str.slice(0 , 4 + 2*i) ,
                f'chn_name_{i+1}'  : ind_dict['chinese_name'] ,
                f'abbr_{i+1}'      : ind_dict['abbreviation'] ,
                f'indexcode_{i+1}' : ind_dict['projected_index'] ,
            }
            df = df.merge(pd.DataFrame(tmp) , on = f'ind_code_{i+1}' , how='left')

    if params[key].get(col_reform) is not None:
        for col , kwargs in params[key].get(col_reform).items(): 
            df = col_reform(df , col , **kwargs)
    if params[key].get(row_filter) is not None:
        for col , kwargs in params[key].get(row_filter).items(): 
            df = row_filter(df , col , **kwargs)
    df = col_filter(df , remain = params[key].get('remain_cols'))
    df = df.reset_index(drop=True)
    return df

def read_day_trading_data(date , tol = 1e-8 , **kwargs):
    data_params = {
        'wind_id'   : ['1_basic_info'  , 'wind_id'] ,
        'adjfactor' : ['2_market_data' , 'day_adjfactor'] ,
        'open'      : ['2_market_data' , 'day_open'] ,
        'high'      : ['2_market_data' , 'day_high'] ,
        'low'       : ['2_market_data' , 'day_low'] ,
        'close'     : ['2_market_data' , 'day_close'] ,
        'amount'    : ['2_market_data' , 'day_amount'] ,
        'volume'    : ['2_market_data' , 'day_volume'] ,
        'vwap'      : ['2_market_data' , 'day_vwap'] ,
        'status'    : ['2_market_data' , 'day_trade_status'] ,
        'limit'     : ['2_market_data' , 'day_up_down_limit_status'] ,
        'pctchange' : ['2_market_data' , 'day_pct_change'] ,
        'preclose'  : ['2_market_data' , 'day_preclose'] ,
        'turn_tt'   : ['2_market_data' , 'day_total_turnover'] ,
        'turn_fl'   : ['2_market_data' , 'day_float_turnover'] ,
        'turn_fr'   : ['2_market_data' , 'day_free_turnover'] ,
    }
    paths = {k:f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/{v[0]}/{v[1]}/{v[1]}_{date}.Rdata' for k,v in data_params.items()}
    paths_not_exists = {k:os.path.exists(p)==0 for k,p in paths.items()}
    if any(paths_not_exists.values()): return unfetched_data(date , [k for k,v in paths_not_exists.items() if v != 0])
    df = pd.concat([pyreadr.read_r(paths[k])['data'].rename(columns={'data':k}) for k in paths.keys()] , axis = 1)
    df['wind_id'] = windid_to_secid(df['wind_id'])
    df = df.rename(columns={'wind_id':'secid'})
    return df

def read_labels_data(date : (int,str) , days : int , lag1 : bool , tol = 1e-8 , **kwargs):
    path_param = {
        'id'  : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/1_basic_info/wind_id' ,
        'res' : f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
        'adj' : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_adjfactor' ,
        'cp'  : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_close' ,
    }

    _files = {k:get_directory_files(v) for k , v in path_param.items()}
    _dates = {k:get_path_date(v) for k , v in _files.items()}
    for v in _files.values(): v.sort()

    pos = list(_dates['id']).index(date)
    if pos + lag1 + days >= len(_dates['id']): return unfetched_data()
    if os.path.exists(path_param['res']+'/'+os.path.basename(path_param['res'])+f'_{date}.Rdata') == 0: return unfetched_data()
    
    f_read = lambda k,d,p='':pyreadr.read_r(path_param[k]+'/'+os.path.basename(path_param[k])+f'_{d}.Rdata')['data'].rename(columns={'data':k+p})
    wind_id = f_read('id',date)

    d0 , d1 = _dates['id'][pos + lag1] , _dates['id'][pos + lag1 + days]
    cp0 = pd.concat([f_read('id',d0),f_read('cp',d0,'0'),f_read('adj',d0,'0')]  , axis = 1)
    cp1 = pd.concat([f_read('id',d1),f_read('cp',d1,'1'),f_read('adj',d1,'1')]  , axis = 1)

    rtn = wind_id.merge(cp0,how='left',on='id').merge(cp1,how='left',on='id')
    rtn['rtn'] = rtn['adj1'] * rtn['cp1'] / rtn['adj0'] / rtn['cp0'] - 1
    rtn = rtn.loc[:,['id','rtn']]

    res_pos = list(_dates['res']).index(d0)
    res_dates = [_dates['res'][res_pos + i] for i in range(days)]
    res = wind_id
    for i,di in enumerate(res_dates): res = res.merge(pd.concat([f_read('id',di),f_read('res',di,str(i))],axis=1),how='left',on='id')
    res = pd.DataFrame({'id':res['id'],'res':res.set_index('id').fillna(np.nan).values.sum(axis=1)})

    df = pd.merge(rtn,res,how='left',on='id')
    df.columns = ['secid' , f'rtn_lag{int(lag1)}_{days}' , f'res_lag{int(lag1)}_{days}']
    df['secid'] = windid_to_secid(df['secid'])
    return df

class unfetched_data():
    def __init__(self , date = None , keys = []) -> None:
        self.dates = np.array([date]) if date is not None else np.array([])
        self.unfetched_detail = {str(date):keys} if date is not None else {}

    def update(self , new):
        assert isinstance(new , unfetched_data) , (type(new))
        self.dates = np.union1d(self.dates , new.dates)
        self.unfetched_detail.update(new.unfetched_detail)


def _str_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())

def _get_dates(date_directory , old_keys):
    dates = get_path_date(get_directory_files(date_directory))
    old_dates = np.array(list(old_keys)).astype(int)
    # if len(old_dates) > 0: old_dates = old_dates[:-1]
    new_dates = np.setdiff1d(dates , old_dates)
    return new_dates , old_dates

def _update_by_date(dtank , file_params , required_param_keys=['__information__','__create_time__','date_directory','read_data_func']):
    assert isinstance(dtank , DataTank) , dtank
    param_keys = [list(param.keys()) for param in file_params.values()]
    assert np.all([np.isin(required_param_keys , ks).all() for ks in param_keys]) , param_keys

    unfetched = unfetched_data()
    for dtank_file , param in file_params.items():
        print(dtank_file)
        __start_time__ = time.time()
        date_directory , read_data_func = param['date_directory'] , param['read_data_func']
        
        if dtank.get_object(dtank_file) is None:
            dtank.create_object(dtank_file)
            dtank.set_group_attrs(dtank_file , **{k:param[k] for k in attr_create})
        
        new_dates , old_dates = _get_dates(date_directory , dtank.get_object(dtank_file).keys())
        for date in new_dates:
            data = read_data_func(date , **param)
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

def choose_data_tank(type , open = True):
    return DataTank(target_files[type] , open = open)

def update_information():
    dtank = choose_data_tank('information')
    file_params = {
        'basic_information/stock_info': {
            '__information__': 'stock list_date and delist_date' , '__create_time__' : _str_time()},
        'basic_information/stock_st': {
            '__information__': 'st treatment of stocks' , '__create_time__' : _str_time()},
        'basic_information/industry': {
            '__information__': 'SW 2021 industry criterion' , '__create_time__' : _str_time()},
        'basic_information/concepts': {
            '__information__': 'wind concepts' , '__create_time__' : _str_time()},
        'basic_information/calendar': {
            '__information__': 'market trading calendar' , '__create_time__' : _str_time()}}
    for dtank_file , param in file_params.items():
        key = dtank_file.split('/')[-1]
        if dtank.get_object(dtank_file) is None:
            dtank.create_object(dtank_file)
            dtank.set_group_attrs(dtank_file , **{k:param[k] for k in attr_create})
        data = get_basic_information(key)
        dtank.write_dataframe(dtank_file , data = data , overwrite = True)
        param.update({'__update_time__':_str_time()})
        dtank.set_group_attrs(dtank_file , **{k:param.get(k) for k in attr_EOL})
        if updater is not None: updater.write_dataframe([dtank.filename,dtank_file] , data = data , overwrite = True)
        print(f'{dtank_file} Done!')
    dtank.close()

def update_model_data():
    dtank = choose_data_tank('model_data')
    file_params = {
        'risk_model/exposure' : {
            '__information__': 'market, industry, style exposure of risk model jm2018' ,
            '__create_time__': _str_time() ,
            'date_directory' : 'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model' ,
            'read_data_func' : read_risk_model
        } ,
        'alpha_longcl/exposure' : {
            '__information__': 'sub alpha factor exposure of risk model jm2018' ,
            '__create_time__': _str_time() ,
            'date_directory' : 'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/A1_Analyst' ,
            'read_data_func' : read_alpha_longcl
        }
    }
    _update_by_date(dtank , file_params)
    dtank.close()

def update_trade_data():
    dtank = choose_data_tank('trade_data')
    file_params = {
        'trading/day' : {
            '__information__': 'daily trading data of A shares' ,
            '__create_time__': _str_time() ,
            'date_directory' : 'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_vwap' ,
            'read_data_func' : read_day_trading_data
        } ,
    }
    _update_by_date(dtank , file_params)
    dtank.close()

def update_labels():
    dtank = choose_data_tank('labels')
    days , lag1 = [5,10,20] , [True,False]
    kwargs_list , file_params = {} , {}
    [[kwargs_list.update({f'label_{d}days/lag{int(l)}':{'days':d,'lag1':l}}) for l in lag1] for d in days]
    for k , v in kwargs_list.items():
        file_params[k] = {
            '__information__': f'forward labels of {v["days"]} days, with lag {v["lag1"]}' ,
            '__create_time__': _str_time() ,
            'date_directory' : f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
            'read_data_func' : read_labels_data ,  **v
        }
    _update_by_date(dtank , file_params)
    dtank.close()

# %%
if __name__ == '__main__':
    if do_updater:
        if os.path.exists(updater_file): os.remove(updater_file)
        updater = DataTank(updater_file , open = True , mode = updater_mode)
        updater_mode = 'r+'

    update_information()
    update_model_data()
    update_trade_data()
    update_labels()
    