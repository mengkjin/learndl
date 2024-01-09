import pyreadr
import pandas as pd
import numpy as np
import os
from ..util.environ import DIR_data

try:
    from .DataTank import *
except:
    from DataTank import *

def get_path_date(path , startswith = '' , endswith = ''):
    if isinstance(path , (list,tuple)):
        return [d for d in [get_path_date(p , startswith , endswith) for p in path] if d is not None]
    else:
        if not path.startswith(startswith): return None
        if not path.endswith(endswith): return None
        s = os.path.basename(path).split('.')[-2][-8:]
        return int(s) if s.isdigit() else None

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
        if isinstance(remain , (str,bytes,int,float)): remain = [remain]
        df = df.loc[:,remain]
    return df

def row_filter(df , col , cond_func = lambda x:x):
    if isinstance(col , str):
        return df[cond_func(df[col])]
    else:
        return df[cond_func(*[df[_c] for _c in col])]

def read_risk_model(date = None , path = None , tol = 1e-8 , **kwargs):
    path = f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model/jm2018_model_{date}.csv'
    if not os.path.exists(path): return NotImplemented
    df = pd.read_csv(path)
    df['wind_id'] = windid_to_secid(df['wind_id'])
    df = df.rename(columns={'wind_id':'secid'})
    df[df.abs() < tol] = 0
    df['secid'] = df['secid'].astype(int)
    return Data1D(src=df)

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
    return Data1D(src=df)

def get_basic_information(key = None , **kwargs):
    if key is None: return 
    key = key.split('/')[-1]
    d_secid = {'wind_id'   : {'newcol' : 'secid' , 'use_func' : windid_to_secid}}
    d_entrm = {'entry_dt'  : {'fillna' : -1 , 'astype' : int} ,
                      'remove_dt' : {'fillna' : 99991231 , 'astype' : int}}
    params = {
        'calendar' : {
            'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_calendar.csv' ,
            'dtype' : int , 'remain_cols' : None ,
        },
        'description': {
            'path': f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_description.csv' ,
            'remain_cols' : ['wind_id' , 'secid' , 'sec_name' , 'exchange_name' , 'list_dt' , 'delist_dt'] ,
            col_reform :  {**d_secid , 
                           'list_dt' : {'fillna' : -1 , 'astype' : int} ,
                           'delist_dt' : {'fillna' : 99991231 , 'astype' : int}} ,
        } ,
        'st' : {
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

def get_trade_day(date , tol = 1e-8 , **kwargs):
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
    if any(paths_not_exists.values()): 
        # something wrong
        return unfetched_data(date , [k for k,v in paths_not_exists.items() if v != 0])
    df = pd.concat([pyreadr.read_r(paths[k])['data'].rename(columns={'data':k}) for k in paths.keys()] , axis = 1)
    df['wind_id'] = windid_to_secid(df['wind_id'])
    df = df.rename(columns={'wind_id':'secid'}).reset_index(drop=True)
    return Data1D(src=df)

def get_trade_Xday(date , x_day , tol = 1e-8 , **kwargs):
    np.seterr(invalid='ignore')
    db_path_info = f'{DIR_data}/DB_data/DB_information/DB_information.h5'
    with DataTank(db_path_info , 'r') as info:
        calendar = info.read_dataframe('basic/calendar')
    rolling_dates = calendar.calendar[calendar.trade > 0].to_numpy().astype(int)
    rolling_dates = sorted(rolling_dates[rolling_dates <= int(date)])[-x_day:]
    assert rolling_dates[-1] == date , (rolling_dates[-1] , date)
    groups = np.array(rolling_dates).astype(int) // 10000
    
    db_path_day = DIR_data + '/DB_data/DB_trade_day/DB_trade_day.{}.h5'
    src_files = [db_path_day.format(group) for group in groups]
    if not all([os.path.exists(file) for file in src_files]): return unfetched_data()

    price_feat  = ['open','close','high','low','vwap']
    volume_feat = ['amount','volume','turn_tt','turn_fl','turn_fr']
    pctchg_feat = ['pctchange']
    all_feat = price_feat + volume_feat + pctchg_feat

    dtank , datas , secid = DataTank() , [] , None
    for d , src in zip(rolling_dates , src_files):
        src_path , inner_path = os.path.join(db_path_day , src) , f'/day/trade/{d}'
        if dtank.filename != src_path: 
            dtank.close()
            dtank = DataTank(src_path , 'r')
        if dtank.get_object(inner_path) is None:
            data = get_trade_day(date , tol = tol , **kwargs)
        else:
            data = dtank.read_data1D(inner_path)
        if isinstance(data , unfetched_data) or data is None:
            return unfetched_data()
        secid = data.secid if secid is None else np.intersect1d(secid , data.secid)
        datas.append(data)
    dtank.close()
    
    for i , data in enumerate(datas):
        data = data.to_dataframe().loc[secid]
        data.loc[:,price_feat] = data.loc[:,price_feat] * data.loc[:,'adjfactor'].values[:,None]
        datas[i] = data.loc[:,all_feat].values

    data = np.stack([data for data in datas] , axis = 0)
    del datas

    df = pd.DataFrame({'secid':secid})
    # price_feat
    df['open']   = data[...,all_feat.index('open')][0]
    df['close']  = data[...,all_feat.index('close')][-1]
    df['high']   = data[...,all_feat.index('high')].max(axis = 0)
    df['low']    = data[...,all_feat.index('low')].min(axis = 0)

    for feat in volume_feat: df[feat] = data[...,all_feat.index(feat)].sum(axis = 0)

    df['vwap']      = (data[...,all_feat.index('vwap')]*data[...,all_feat.index('volume')]).sum(axis = 0) / df['volume']
    df['pctchange'] = (data[...,all_feat.index('pctchange')] / 100 + 1).prod(axis = 0) * 100 - 100

    np.seterr(invalid='warn')
    return Data1D(src=df)

def get_trade_min(date , tol = 1e-8 , **kwargs):
    data_params = {
        'ticker'    : 'secid'  ,
        'secoffset' : 'minute' ,
        'openprice' : 'open' ,
        'highprice' : 'high' , 
        'lowprice'  : 'low' , 
        'closeprice': 'close' , 
        'value'     : 'amount' , 
        'volume'    : 'volume' , 
        'vwap'      : 'vwap' , 
    }
    path = f'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin/equity_pricemin_{date}.txt'
    df = pd.read_csv(path , sep='\t' , low_memory=False)
    if df['ticker'].dtype in (object,str): 
        df = df[df['ticker'].str.isdigit()] 
    df['ticker'] = df['ticker'].astype(int)
    cond_stock = lambda x,y:((600000<=x)&(x<=699999)&(y=='XSHG'))|((0<=x)&(x<=398999)&(y=='XSHE'))
    df = row_filter(df,('ticker','exchangecd'),cond_stock)
    df = df.loc[:,list(data_params.keys())].rename(columns=data_params)
    df['minute'] = df['minute']/60
    df['minute'] = (df['minute'] - 90) * (df['minute'] <= 240) + (df['minute'] - 180) * (df['minute'] > 240)
    df = df.sort_values(['secid','minute'])
    return Data1D(src=df)

def get_labels(date : (int,str) , days : int , lag1 : bool , tol = 1e-8 , **kwargs):
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
    if pos + lag1 + days >= len(_dates['id']): 
        return unfetched_data()
    if os.path.exists(path_param['res']+'/'+os.path.basename(path_param['res'])+f'_{date}.Rdata') == 0: 
        return unfetched_data()
    
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
    for i , di in enumerate(res_dates): 
        res = res.merge(pd.concat([f_read('id',di),f_read('res',di,str(i))],axis=1),how='left',on='id')
    res = pd.DataFrame({'id':res['id'],'res':res.set_index('id').fillna(np.nan).values.sum(axis=1)})

    df = pd.merge(rtn,res,how='left',on='id')
    df.columns = ['secid' , f'rtn_lag{int(lag1)}_{days}' , f'res_lag{int(lag1)}_{days}']
    df['secid'] = windid_to_secid(df['secid'])
    return Data1D(src=df)

def fill_na_min_data(data):
    #'amount' , 'volume' to 0
    imin = np.where(data.feature == 'minute')[0][0]
    iccp = np.where(data.feature == 'close')[0][0]
    for f in ['amount' , 'volume']: 
        icol = np.where(data.feature == f)[0][0]
        rnan = np.where(np.isnan(data.values[:,icol]))[0]
        if len(rnan) > 0: data.values[rnan,icol] = 0.

    # close price
    for f in ['close']: 
        icol = np.where(data.feature == f)[0][0]
        rnan = np.where(np.isnan(data.values[:,icol]))[0]
        if len(rnan) == 0: continue
        rfix = np.where((np.isnan(data.values[:,icol]) * (data.values[:,imin] == 0)) != 0)[0]
        if len(rfix) > 0: data.values[rfix,icol] = -1.
        v = forward_fillna(data.values[:,icol])
        v[v == -1] = np.nan
        data.values[:,icol] = v

    # prices to last time price (min != 0)
    for f in ['open','high','low','vwap']: 
        icol = np.where(data.feature == f)[0][0]
        rnan = np.where((np.isnan(data.values[:,icol]) * (data.values[:,imin] > 0)) != 0)[0]
        if len(rnan) > 0: data.values[rnan,icol] = data.values[rnan-1,iccp]
    return data

def filter_min_Data1D(data):
    assert isinstance(data , Data1D) , type(data)
    x1 = (data.secid>=0)*(data.secid<100000)+(data.secid>=300000)*(data.secid<=398999)+(data.secid>=600000)*(data.secid<=699999)
    x2 = data.values[:,0] <= 240
    return data.slice(secid = (x1*x2)>0)

def Data1D_to_kline(data):
    assert isinstance(data , Data1D) , type(data)
    assert data.feature[0] == 'minute' , data.feature[0]
    if np.isnan(data.values).sum() > 0: data = fill_na_min_data(data)
    assert np.isnan(data.values).sum() == 0
    minute = data.values[:,0].astype(int)
    u_secid , u_minute = np.unique(data.secid) , np.unique(minute)
    u_feature = data.feature[1:]
    if len(data.values) == len(u_secid) * len(u_minute):
        new_values = data.values.reshape(len(u_secid) , len(u_minute) , -1)
        new_secid = data.secid.reshape(len(u_secid) , len(u_minute))
        assert (new_secid[:,:] - u_secid.reshape(-1,1) == 0).all()
        assert (new_values[:,:,0] - u_minute == 0).all()
        new_values = new_values[:,:,1:]
    else:
        new_values = np.zeros((len(u_secid) , len(u_minute) , len(u_feature)))
        for mm in u_minute:
            msec , mpos = data.secid[minute == mm] , minute == mm
            if np.array_equal(data.secid[mpos] , u_secid):
                new_values[:,mm,:] = data.values[mpos , 1:]
            else:
                i_sec = np.intersect1d(u_secid , msec , return_indices=True)[1]
                new_values[i_sec,mm,:] = data.values[mpos , 1:]
    return new_values , (u_secid , u_minute , u_feature)

def kline_to_Data1D(data , index , columnname = 'minute'):
    secid   = np.repeat(index[0] , len(index[1]))
    feature = np.concatenate([[columnname],index[2]])
    values  = np.concatenate([np.tile(index[1],len(index[0])).reshape(-1,1),data.reshape(-1,len(index[2]))],axis=-1)
    return Data1D(secid , feature , values)

def kline_reform(data , index , by = 5):
    assert len(index[-1]) == data.shape[-1] , (len(index[-1]) , data.shape[-1])
    assert 240 % by == 0 and data.shape[1] in [240,241] , (data.shape[1] , by)
    n_kline = 240 // by
    index = (index[0] , np.arange(n_kline) , index[2])
    if data.shape[1] == 241:
        new_data = np.concatenate([kline_aggregate(data[:,0:6] , keys = index[-1])] + 
                                [kline_aggregate(data[:,k*by+1:(k+1)*by+1] , keys = index[-1]) for k in index[1][1:]], axis=1)
    else:
        new_data = np.concatenate([kline_aggregate(data[:,k*by:(k+1)*by] , keys = index[-1]) for k in index[1]], axis=1)
    return new_data , index

def kline_aggregate(data , keys):
    assert isinstance(keys , np.ndarray)
    assert data.shape[-1] == len(keys) , (data.shape[-1] , len(keys))
    new_data = data.sum(axis=1,keepdims=True) # 'amount' , 'volume' Done:
    if 'close' in keys: new_data[:,0,keys=='close'] = data[:,-1,keys=='close'] # close
    if 'open'  in keys: new_data[:,0,keys=='open']  = data[:,0,keys=='open'] # open
    if 'high'  in keys: new_data[:,0,keys=='high']  = data[:,:,keys=='high'].max(axis=1) # high
    if 'low'   in keys: new_data[:,0,keys=='low']   = data[:,:,keys=='low'].min(axis=1) # low
    if 'vwap'  in keys: 
        v = data[:,-1:,keys=='vwap']
        pos = new_data[:,:,keys=='volume'] != 0
        v[pos] = new_data[:,:,keys=='amount'][pos] / new_data[:,:,keys=='volume'][pos]
        new_data[:,-1:,keys=='vwap']  = v # vwap
    return new_data

def get_trade_Xmin(date , x_minute , src_updater = None , tol = 1e-8 , **kwargs):
    data = None
    inner_path = f'minute/trade/{date}'
    if isinstance(src_updater , DataTank):
        for key in src_updater.keys():
            if (os.path.basename(key).startswith('DB_trade_min') and 
                src_updater.get_object(f'{key}/{inner_path}') is not None): 
                data = src_updater.read_data1D(f'{key}/{inner_path}')
                break

    if data is None:
        src_path  = f'{DIR_data}/DB_data/DB_trade_min/DB_trade_min.{int(date) // 10000}.h5'
        if os.path.exists(src_path):
            dtank = DataTank(src_path , 'r')
            if dtank.get_object(inner_path) is not None: 
                data = dtank.read_data1D(inner_path)
            dtank.close()
        
    if data is None: data = get_trade_min(date , tol = tol , **kwargs)

    if x_minute == 1:
        return data
    else:
        data = filter_min_Data1D(data)
        data , index = Data1D_to_kline(data)
        data , index = kline_reform(data , index , by = x_minute)
        return kline_to_Data1D(data , index)

def forward_fillna(arr , axis = 0):
    shape = arr.shape
    if axis < 0: axis = len(shape) + axis
    if axis > 0:
        new_axes  = [axis , *[i for i in range(len(shape)) if i != axis]]
        new_shape = [shape[i] for i in new_axes]
        old_axes  = list(range(len(shape)))[1:]
        old_axes.insert(axis,0)
        arr = arr.transpose(*new_axes)
    arr = arr.reshape(shape[axis],-1).transpose(1,0)
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    idx = np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx].transpose(1,0)
    if axis > 0:
        out = out.reshape(new_shape).transpose(*old_axes)
    return out

class unfetched_data():
    def __init__(self , date = None , keys = []) -> None:
        self.dates = np.array([date]) if date is not None else np.array([])
        self.unfetched_detail = {str(date):keys} if date is not None else {}

    def update(self , new):
        assert isinstance(new , unfetched_data) , (type(new))
        self.dates = np.union1d(self.dates , new.dates)
        self.unfetched_detail.update(new.unfetched_detail)

    def __len__(self):
        return len(self.dates)
    
    def __repr__(self) -> str:
        if len(self) > 0:
            return f'{len(self)} Unfetched Dates: {str(list(self.dates))}'
        else:
            return ''
