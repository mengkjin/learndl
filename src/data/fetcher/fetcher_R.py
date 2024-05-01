import os , pyreadr
import pandas as pd
import numpy as np

from typing import Any , Optional

from .common import FailedReturn , list_files , R_path_date

class DataFetcher_R:
    '''Fetch data from R environment'''
    @classmethod
    def adjust_secid(cls , df : pd.DataFrame):
        '''switch old wind_id into secid'''
        if 'wind_id' not in df.columns.values: return df

        df['wind_id'] = df['wind_id'].astype(str)
        replace_dict = {'T00018' : '600018'}
        df['wind_id'] = df['wind_id'].str.slice(0, 6).replace(replace_dict)
        df['wind_id'] = df['wind_id'].where(df['wind_id'].str.isdigit() , '-1')
        df['wind_id'] = df['wind_id'].astype(int)
        return df.rename(columns={'wind_id':'secid'})

    @staticmethod
    def col_reform(df : pd.DataFrame , col : str , rename = None , fillna = None , astype = None , use_func = None):
        '''do certain processing to DataFrame columns: newcol(rename) , fillna , astype or use_func'''
        if use_func is not None:
            df[col] = use_func(df[col])
        else:
            x = df[col]
            if fillna is not None: x = x.fillna(fillna)
            if astype is not None: x = x.astype(astype)
            df[col] = x 
        if rename: df = df.rename(columns={col:rename})
        return df

    @staticmethod
    def row_filter(df : pd.DataFrame , col : str | list | tuple , cond_func = lambda x:x):
        '''filter pd.DataFrame rows: cond_func(col)'''
        if isinstance(col , str):
            return df[cond_func(df[col])]
        else:
            return df[cond_func(*[df[_c] for _c in col])]
    
    @staticmethod
    def adjust_precision(df : pd.DataFrame , tol = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
        '''adjust precision for df columns'''
        for col in df.columns:
            if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
                df[col] = df[col].astype(dtype_float)
                df[col] *= (df[col].abs() > tol)
            if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
                df[col] = df[col].astype(dtype_int)
        return df

    @classmethod
    def basic_info(cls , key = None , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get basic info data from R environment , basic_info('concepts')'''
        if key is None: raise KeyError(key) 
        key = key.split('/')[-1]
        d_entrm = {'entry_dt'  : {'fillna' : -1 , 'astype' : int} , 
                   'remove_dt' : {'fillna' : 99991231 , 'astype' : int}}
        params = {
            'calendar' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_calendar.csv' ,
                'dtype' : int , 'remain_cols' : None ,
            },
            'description': {
                'path': f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_description.csv' ,
                'remain_cols' : ['secid' , 'sec_name' , 'exchange_name' , 'list_dt' , 'delist_dt'] ,
                cls.col_reform :  {
                    'list_dt' : {'fillna' : -1 , 'astype' : int} ,
                    'delist_dt' : {'fillna' : 99991231 , 'astype' : int}} ,
            } ,
            'st' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_st.csv' ,
                'remain_cols' : ['secid' , 'st_type' , 'entry_dt' , 'remove_dt' , 'ann_dt'] ,
                cls.col_reform : {**d_entrm , 'ann_dt' : {'fillna' : -1 , 'astype' : int}} ,
                cls.row_filter : {'st_type' : {'cond_func' : lambda x:x != 'R'}}
            },
            'industry' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_industries_class_sw_2021.csv' ,
                'remain_cols' : ['secid' , 'entry_dt', 'remove_dt', 'ind_code', 
                                'ind_code_1', 'chn_name_1', 'abbr_1', 'indexcode_1' ,
                                'ind_code_2', 'chn_name_2', 'abbr_2', 'indexcode_2' ,
                                'ind_code_3', 'chn_name_3', 'abbr_3', 'indexcode_3'] ,
                cls.col_reform : d_entrm ,
            },
            'concepts' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_wind_concepts.csv' ,
                'remain_cols' : ['secid' , 'concept' , 'entry_dt' , 'remove_dt'] ,
                cls.col_reform : {**d_entrm , 'wind_sec_name' : {'rename' : 'concept'}} ,
            },
        }
        if not os.path.exists(params[key]['path']): return FailedReturn(key)
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

        df = cls.adjust_secid(df)
        if params[key].get(cls.col_reform) is not None:
            for col , kwargs in params[key].get(cls.col_reform).items(): 
                df = cls.col_reform(df , col , **kwargs)
        if params[key].get(cls.row_filter) is not None:
            for col , kwargs in params[key].get(cls.row_filter).items(): 
                df = cls.row_filter(df , col , **kwargs)
        if params[key].get('remain_cols'):
            df = df.loc[:,params[key]['remain_cols']]
        df = df.reset_index(drop=True)
        return df

    @classmethod
    def risk_model(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get risk model from R environment , risk_model(20240325)'''
        path = f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model/jm2018_model_{date}.csv'
        if not os.path.exists(path): 
            return FailedReturn('risk_exp' , date)
        df = pd.read_csv(path)
        df = cls.adjust_secid(df)
        df = cls.adjust_precision(df)
        if with_date: df['date'] = date
        return df

    @classmethod
    def alpha_longcl(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get alpha longcl model from R environment , alpha_longcl(20240325)'''
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
        df = pd.DataFrame(columns=['secid'] , dtype = int).set_index('secid')
        for k,v in a_names.items():
            colnames = ['secid',v]
            path = f'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/{v}/{v}_{date}.txt'
            if not os.path.exists(path):
                return FailedReturn('longcl_exp' , date)
            else:
                df_new = pd.read_csv(path, header=None , delimiter='\t',dtype=float)
                df_new.columns = colnames
            df_new['secid'] = df_new['secid'].astype(int)
            df = pd.merge(df , df_new.set_index('secid') , how='outer' , on='secid')
        df = cls.adjust_precision(df).reset_index()
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_day(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get basic info data from R environment , trade_day(20240324)'''
        np.seterr(invalid='ignore' , divide = 'ignore')
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
            print(f'Something wrong at date {date} on {cls.__name__}.trade_day')
            return FailedReturn('day' , date)
        df = pd.concat([pyreadr.read_r(paths[k])['data'].rename(columns={'data':k}) for k in paths.keys()] , axis = 1)
        df = cls.adjust_secid(df)
        df = cls.adjust_precision(df).reset_index(drop=True)
        np.seterr(invalid='warn' , divide = 'warn')
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_Xday(cls , date : int , x : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get consecutive x_day trade data from R environment , trade_Xday(20240324 , 5) '''
        #np.seterr(invalid='ignore' , divide = 'ignore')
        # read calendar
        calendar = cls.basic_info('calendar')
        assert isinstance(calendar , pd.DataFrame)
        rolling_dates = calendar.calendar[calendar.trade > 0].to_numpy().astype(int)
        rolling_dates = sorted(rolling_dates[rolling_dates <= int(date)])[-x:]
        assert rolling_dates[-1] == date , (rolling_dates[-1] , date)

        price_feat  = ['open','close','high','low','vwap']
        volume_feat = ['amount','volume','turn_tt','turn_fl','turn_fr']

        data = []
        for d in rolling_dates:
            tmp = cls.trade_day(d , with_date=True)
            if isinstance(tmp , pd.DataFrame): 
                data.append(tmp)
            else:
                return FailedReturn(f'{x}day' , date)
        data = pd.concat(data , axis = 0)

        data.loc[:,price_feat] = data.loc[:,price_feat] * data.loc[:,'adjfactor'].values[:,None]
        data['pctchange'] = data['pctchange'] / 100 + 1
        data['vwap'] = data['vwap'] * data['volume']
        agg_dict = {'open':'first','high':'max','low':'min','close':'last','pctchange':'prod','vwap':'sum',**{k:'sum' for k in volume_feat},}
        df = data.groupby('secid').agg(agg_dict)
        df['pctchange'] = (df['pctchange'] - 1) * 100
        df['vwap'] /= np.where(df['volume'] == 0 , np.nan , df['volume'])
        df['vwap'] = df['vwap'].where(~df['vwap'].isna() , df['close'])
        #np.seterr(invalid='warn' , divide = 'warn')
        if with_date: df['date'] = date
        return df

    @classmethod
    def labels(cls , date : int , days : int , lag1 : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get raw and res labels'''
        path_param = {
            'id'  : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/1_basic_info/wind_id' ,
            'res' : f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model' ,
            'adj' : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_adjfactor' ,
            'cp'  : f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_close' ,
        }

        _files = {k:list_files(v) for k , v in path_param.items()}
        for v in _files.values(): v.sort()
        _dates = {k:R_path_date(v) for k , v in _files.items()}

        pos = list(_dates['id']).index(date)
        if pos + lag1 + days >= len(_dates['id']):  return None
        if not os.path.exists(path_param['res']+'/'+os.path.basename(path_param['res'])+f'_{date}.Rdata'):  return None
        
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
        df.columns = ['wind_id' , f'rtn_lag{int(lag1)}_{days}' , f'res_lag{int(lag1)}_{days}']
        df = cls.adjust_secid(df)
        df = cls.adjust_precision(df).reset_index(drop=True)
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_min(cls , date : int , with_date = False , dtank_first = True , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get minute trade data from R environment , trade_min(20240324)'''
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
        if not os.path.exists(path): return FailedReturn('min' , date)
        df = pd.read_csv(path , sep='\t' , low_memory=False)
        if df['ticker'].dtype in (object,str): 
            df = df[df['ticker'].str.isdigit()] 
        df['ticker'] = df['ticker'].astype(int)
        cond_stock = lambda x,y:((600000<=x)&(x<=699999)&(y=='XSHG'))|((0<=x)&(x<=398999)&(y=='XSHE'))
        df = cls.row_filter(df,('ticker','exchangecd'),cond_stock)
        df = df.loc[:,list(data_params.keys())].rename(columns=data_params)
        df['minute'] = (df['minute']/60).astype(int)
        df['minute'] = (df['minute'] - 90) * (df['minute'] <= 240) + (df['minute'] - 180) * (df['minute'] > 240)
        df = df.sort_values(['secid','minute'])
        df = cls.trade_min_fillna(df)
        if with_date: df['date'] = date
        return df

    @staticmethod
    def trade_min_fillna(df : pd.DataFrame):
        '''fillna for minute trade data'''
        #'amount' , 'volume' to 0
        df['amount'] = df['amount'].where(~df['amount'].isna() , 0)
        df['volume'] = df['volume'].where(~df['volume'].isna() , 0)

        # close price
        df1 = df.loc[:,['secid','minute','close']].copy().rename(columns={'close':'fix'})
        df1['minute'] = df1['minute'] + 1
        df = df.merge(df1,on=['secid','minute'],how='left')
        df['fix'] = df['fix'].where(~df['fix'].isna() , df['open'])

        # prices to last time price (min != 0)
        for feat in ['open','high','low','vwap','close']: 
            if df[feat].isna().any():
                df[feat] = df[feat].where(~df[feat].isna() , df['fix'])
        del df['fix']
        return df

    @classmethod
    def trade_min_reform(cls , df : pd.DataFrame , by : int):
        '''from minute trade data to xmin trade data'''
        df = cls.trade_min_filter(df)
        assert len(df['minute'].unique()) in [240,241] , df['minute'].unique()
        assert 240 % by == 0 , by
        df['minute'] = (df['minute'].clip(lower=1) - 1) // by
        agg_dict = {'open':'first','high':'max','low':'min','close':'last','amount':'sum','volume':'sum'}
        data_new = df.groupby(['secid' , 'minute']).agg(agg_dict)
        if 'vwap' in df.columns: data_new['vwap'] = data_new['amount'] / data_new['volume']
        return data_new.reset_index(drop=False)

    @staticmethod
    def trade_min_filter(df : pd.DataFrame):
        '''fillna for minute trade data, remain ashare'''
        assert isinstance(df , pd.DataFrame) , type(df)
        secid , minute = df['secid'] , df['minute']
        x1 = (secid>=0)*(secid<100000)+(secid>=300000)*(secid<=398999)+(secid>=600000)*(secid<=699999)
        x2 = minute <= 240
        return df[(x1*x2)>0].copy()

    @classmethod
    def trade_Xmin(cls , date : int , x : int , df_min : Any = None , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedReturn]:
        '''get X minute trade data from R environment , trade_Xmin(20240324 , 5)'''
        df = df_min if df_min is not None else cls.trade_min(date , **kwargs)
        if df is None or isinstance(df , FailedReturn):
            return FailedReturn(f'{x}min' , date)
        if x != 1:
            df = cls.trade_min_reform(df , by = x)
        if df is None: return df
        if with_date: df['date'] = date
        return df