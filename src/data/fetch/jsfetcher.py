import pyreadr , re
import pandas as pd
import numpy as np

from dataclasses import dataclass , field
from pathlib import Path
from typing import Any , Callable , Literal , Optional

from src.basic import PATH
from src.data.util import secid_adjust , col_reform , row_filter , adjust_precision , trade_min_reform , trade_min_fillna

@dataclass
class FailedData:
    type: str
    date: Optional[int] = None
    def add_attr(self , key , value): 
        self.__dict__[key] = value
        return self
    
@dataclass
class JSFetcher:
    db_src      : str
    db_key      : str
    args        : list = field(default_factory=list)
    fetcher     : Callable | str = 'default'

    def __post_init__(self):
        if self.fetcher == 'default':
            self.fetcher = self.default_fetcher(self.db_src , self.db_key)

    def __call__(self , date = None , **kwargs) -> Any:
        return self.eval(date , **kwargs) , self.target_path(date)
    
    @classmethod
    def default_fetcher(cls , db_src , db_key):
        if db_src == 'information_js': return cls.basic_info
        elif db_src == 'models':
            if db_key == 'risk_exp': return cls.risk_exp
            elif db_key == 'risk_cov': return cls.risk_cov
            elif db_key == 'risk_spec': return cls.risk_spec
            elif db_key == 'longcl_exp': return cls.alpha_longcl
        elif db_src == 'trade_js':
            if db_key == 'day': return cls.trade_day
            elif db_key == 'min': return cls.trade_min
            elif re.match(r'^\d+day$' , db_key): return cls.trade_Xday
            elif re.match(r'^\d+min$' , db_key): return cls.trade_Xmin
        elif db_src == 'labels_js': return cls.labels
        elif db_src == 'benchmark_js': return cls.benchmark
        raise Exception('Unknown default_fetcher')

    def eval(self , date = None , **kwargs) -> Any:
        assert callable(self.fetcher)
        if self.db_src in PATH.DB_BY_NAME:
            v = self.fetcher(self.db_key , *self.args , **kwargs)
        elif self.db_src in PATH.DB_BY_DATE:
            v = self.fetcher(date , *self.args , **kwargs)  
        return v
    
    def target_path(self , date = None):
        return PATH.db_path(self.db_src , self.db_key , date)
    
    def source_dates(self):
        return PATH.get_source_dates(self.db_src , self.db_key)
    
    def target_dates(self):
        return PATH.db_dates(self.db_src , self.db_key)
    
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
    def basic_info(cls , key = None , **kwargs) -> Optional[pd.DataFrame | FailedData]:
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
                col_reform :  {
                    'list_dt' : {'fillna' : -1 , 'astype' : int} ,
                    'delist_dt' : {'fillna' : 99991231 , 'astype' : int}} ,
            } ,
            'st' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_st.csv' ,
                'remain_cols' : ['secid' , 'st_type' , 'entry_dt' , 'remove_dt' , 'ann_dt'] ,
                col_reform : d_entrm | {'ann_dt' : {'fillna' : -1 , 'astype' : int}} ,
                row_filter : {'st_type' : {'cond_func' : lambda x:x != 'R'}}
            },
            'industry' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_industries_class_sw_2021.csv' ,
                'remain_cols' : ['secid' , 'entry_dt', 'remove_dt', 'ind_code', 
                                 'ind_code_1', 'chn_name_1', 'abbr_1', 'indexcode_1' ,
                                 'ind_code_2', 'chn_name_2', 'abbr_2', 'indexcode_2' ,
                                 'ind_code_3', 'chn_name_3', 'abbr_3', 'indexcode_3'] ,
                col_reform : d_entrm ,
            },
            'concepts' : {
                'path' : f'D:/Coding/ChinaShareModel/ModelData/1_attributes/a_share_wind_concepts.csv' ,
                'remain_cols' : ['secid' , 'concept' , 'entry_dt' , 'remove_dt'] ,
                col_reform : d_entrm | {'wind_sec_name' : {'rename' : 'concept'}} ,
            },
        }
        if not Path(params[key]['path']).exists(): return FailedData(key)
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

        df = secid_adjust(df , 'wind_id' , drop_old=True , raise_if_no_secid=False)
        if params[key].get(col_reform) is not None:
            for col , kwargs in params[key].get(col_reform).items(): 
                df = col_reform(df , col , **kwargs)
        if params[key].get(row_filter) is not None:
            for col , kwargs in params[key].get(row_filter).items(): 
                df = row_filter(df , col , **kwargs)
        if params[key].get('remain_cols'):
            df = df.loc[:,params[key]['remain_cols']]
        df = df.reset_index(drop=True)
        return df

    @classmethod
    def risk_exp(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get risk model from R environment , risk_exp(20240325)'''
        path = Path(f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model/jm2018_model_{date}.csv')
        if not path.exists(): return FailedData('risk_exp' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.read_csv(path)
            df = adjust_precision(secid_adjust(df , 'wind_id' , drop_old=True))
        if with_date: df['date'] = date
        return df
    
    @classmethod
    def risk_cov(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get risk model from R environment , risk_cov(20240325)'''
        path = Path(f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/6_factor_return_covariance/jm2018_model/jm2018_model_{date}.Rdata')
        if not path.exists(): return FailedData('risk_cov' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pyreadr.read_r(path)['data'] * 252
            df = adjust_precision(df.reset_index().rename(columns={'index' :'factor_name'}))
        if with_date: df['date'] = date
        return df
    
    @classmethod
    def risk_spec(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get risk model from R environment , risk_spec(20240325)'''
        path = Path(f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/2_factor_exposure/jm2018_model/jm2018_model_{date}.csv')
        if not path.exists(): return FailedData('risk_spec' , date)

        paths = {'wind_id':Path(f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/1_basic_info/wind_id/wind_id_{date}.Rdata') , 
                 'spec_risk':Path(f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/C_specific_risk/jm2018_model/jm2018_model_{date}.Rdata')}
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.concat([pyreadr.read_r(paths[k])['data'].rename(columns={'data':k}) for k in paths.keys()] , axis = 1)
            df = adjust_precision(secid_adjust(df , 'wind_id' , drop_old=True))
            df['spec_risk'] = df['spec_risk'] * np.sqrt(252)
        if with_date: df['date'] = date
        return df

    @classmethod
    def alpha_longcl(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
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
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.DataFrame(columns=['secid'] , dtype = int).set_index('secid')
            for k,v in a_names.items():
                colnames = ['secid',v]
                path = Path(f'D:/Coding/ChinaShareModel/ModelData/H_Other_Alphas/longcl/{v}/{v}_{date}.txt')
                if not path.exists():
                    return FailedData('longcl_exp' , date)
                else:
                    df_new = pd.read_csv(path, header=None , delimiter='\t',dtype=float)
                    df_new.columns = colnames
                df_new['secid'] = df_new['secid'].astype(int)
                df = pd.merge(df , df_new.set_index('secid') , how='outer' , on='secid')
            df = adjust_precision(df).reset_index()
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_day(cls , date : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get basic info data from R environment , trade_day(20240324)'''        
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
        paths = {k:Path(f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/{v[0]}/{v[1]}/{v[1]}_{date}.Rdata') for k,v in data_params.items()}
        paths_not_exists = {k:p.exists()==0 for k,p in paths.items()}
        if any(paths_not_exists.values()): 
            # something wrong
            print(f'Something wrong at date {date} on {cls.__name__}.trade_day')
            return FailedData('day' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.concat([pyreadr.read_r(paths[k])['data'].rename(columns={'data':k}) for k in paths.keys()] , axis = 1)
            df = adjust_precision(secid_adjust(df , 'wind_id' , drop_old=True)).reset_index(drop=True)
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_Xday(cls , date : int , x : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get consecutive x_day trade data from R environment , trade_Xday(20240324 , 5) '''
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
                return FailedData(f'{x}day' , date)
        
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            data = pd.concat(data , axis = 0)
            data.loc[:,price_feat] = data.loc[:,price_feat] * data.loc[:,'adjfactor'].values[:,None]
            data['pctchange'] = data['pctchange'] / 100 + 1
            data['vwap'] = data['vwap'] * data['volume']
            agg_dict = {'open':'first','high':'max','low':'min','close':'last','pctchange':'prod','vwap':'sum',**{k:'sum' for k in volume_feat},}
            df = data.groupby('secid').agg(agg_dict)
            df['pctchange'] = (df['pctchange'] - 1) * 100
            df['vwap'] /= np.where(df['volume'] == 0 , np.nan , df['volume'])
            df['vwap'] = df['vwap'].where(~df['vwap'].isna() , df['close'])

        if with_date: df['date'] = date
        return df

    @classmethod
    def labels(cls , date : int , days : int , lag1 : int , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get raw and res labels'''
        path_param = {
            'id'  : Path(f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/1_basic_info/wind_id') ,
            'res' : Path(f'D:/Coding/ChinaShareModel/ModelData/6_risk_model/7_stock_residual_return_forward/jm2018_model') ,
            'adj' : Path(f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_adjfactor') ,
            'cp'  : Path(f'D:/Coding/ChinaShareModel/ModelData/4_cross_sectional/2_market_data/day_close') ,
        }

        _files = {k:PATH.list_files(v) for k , v in path_param.items()}
        for v in _files.values(): v.sort()
        _dates = {k:PATH.file_dates(v) for k , v in _files.items()}

        pos = list(_dates['id']).index(date)
        if pos + lag1 + days >= len(_dates['id']):  return None
        if not path_param['res'].joinpath(path_param['res'].name + f'_{date}.Rdata').exists(): return None
        
        f_read = lambda k,d,p='':pyreadr.read_r(path_param[k].joinpath(f'{path_param[k].name}_{d}.Rdata'))['data'].rename(columns={'data':k+p})
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
        
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.merge(rtn,res,how='left',on='id')
            df.columns = ['wind_id' , f'rtn_lag{int(lag1)}_{days}' , f'res_lag{int(lag1)}_{days}']
            df = adjust_precision(secid_adjust(df , 'wind_id' , drop_old=True)).reset_index(drop=True)
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_min(cls , date : int , with_date = False , dtank_first = True , **kwargs) -> Optional[pd.DataFrame | FailedData]:
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
        path = Path(f'D:/Coding/ChinaShareModel/ModelData/Z_temporal/equity_pricemin/equity_pricemin_{date}.txt')
        if not path.exists(): return FailedData('min' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.read_csv(path , sep='\t' , low_memory=False)
            if df['ticker'].dtype in (object,str):  df = df[df['ticker'].str.isdigit()] 
            df['ticker'] = df['ticker'].astype(int)
            cond_stock = lambda x,y:((600000<=x)&(x<=699999)&(y=='XSHG'))|((0<=x)&(x<=398999)&(y=='XSHE'))
            df = row_filter(df,('ticker','exchangecd'),cond_stock)
            df = df.loc[:,list(data_params.keys())].rename(columns=data_params)
            df['minute'] = (df['minute']/60).astype(int)
            df['minute'] = (df['minute'] - 90) * (df['minute'] <= 240) + (df['minute'] - 180) * (df['minute'] > 240)
            df = df.sort_values(['secid','minute'])
            df = trade_min_fillna(df)
        if with_date: df['date'] = date
        return df

    @classmethod
    def trade_Xmin(cls , date : int , x : int , df_min : Any = None , with_date = False , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get X minute trade data from R environment , trade_Xmin(20240324 , 5)'''
        df = df_min if df_min is not None else cls.trade_min(date , **kwargs)
        if df is None or isinstance(df , FailedData): return FailedData(f'{x}min' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            if x != 1: df = trade_min_reform(df , x)
            if df is None: return df
        if with_date: df['date'] = date
        return df
    
    @classmethod
    def benchmark(cls , date : int , bm : Literal['csi300' , 'csi500' , 'csi800' , 'csi1000'] , **kwargs) -> Optional[pd.DataFrame | FailedData]:
        '''get risk model from R environment , bm_any('CSI300' , 20240325)'''
        path = Path(f'D:/Coding/ChinaShareModel/ModelData/B_index_weight/1_csi_index/{bm.upper()}/{bm.upper()}_{date}.csv')
        if not path.exists():  return FailedData(f'bm_{bm.lower()}' , date)
        with np.errstate(invalid='ignore' , divide = 'ignore'):
            df = pd.read_csv(path)
            df = secid_adjust(df , 'wind_id' , drop_old=True)
            df['weight'] = df['weight'] / df['weight'].sum()
            df = adjust_precision(df)
        return df