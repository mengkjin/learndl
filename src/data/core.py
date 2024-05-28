import os , torch
import numpy as np
import pandas as pd

from dataclasses import dataclass
from torch import Tensor
from typing import Any , ClassVar , Optional

from .fetcher import get_target_dates , load_target_file
from ..classes import StockData4D
from ..environ import PATH , CONF
from ..func import index_union , index_intersect , forward_fillna
from ..func.time import date_offset , Timer

def data_type_abbr(key : str):
    key = key.lower()
    if (key.startswith('trade_') and len(key)>6):
        return key[6:]
    elif key.startswith(('rtn_lag','res_lag','std_lag')):
        return '{:s}{:d}'.format(key[:3] , sum([int(s) for s in key[7:].split('_')]))
    elif key in ['y' , 'labels']:
        return 'y'
    else:
        return key
    
def data_type_alias(path : str , key : str):
    alias_list = [key , f'trade_{key}' , key.replace('trade_','')]
    for alias in alias_list:
        if os.path.exists(path.format(alias)): 
            return path.format(alias)
    return path.format(key)

class GetData:
    @staticmethod
    def trade_dates(start_dt : int = -1 , end_dt : int = 99991231):
        calendar = load_target_file('information' , 'calendar')
        assert calendar is not None
        calendar = np.array(calendar['calendar'].values[calendar['trade'] == 1])
        calendar = calendar[(calendar >= start_dt) & (calendar <= end_dt)]
        return calendar
        
class DataBlock(StockData4D):   
    def save(self , key : str , predict=False , start_dt = None , end_dt = None):
        path = self.block_path(key , predict) 
        os.makedirs(os.path.dirname(path),exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        self.save_dict(data , path)
    
    @staticmethod
    def save_dict(data : dict , file_path : str):
        if file_path is None: return NotImplemented
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        if file_path.endswith(('.npz' , '.npy' , '.np')):
            np.savez_compressed(file_path , **data)
        elif file_path.endswith(('.pt' , '.pth')):
            torch.save(data , file_path , pickle_protocol = 4)
        else:
            raise Exception(file_path)
        
    @staticmethod
    def load_dict(file_path : str , keys = None) -> dict[str,Any]:
        if file_path.endswith(('.npz' , '.npy' , '.np')):
            file = np.load(file_path)
        elif file_path.endswith(('.pt' , '.pth')):
            file = torch.load(file_path)
        else:
            raise Exception(file_path)
        keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
        data = {k:file[k] for k in keys}
        return data
    
    @classmethod
    def load_path(cls , path : str): return cls(**cls.load_dict(path))
    
    @classmethod
    def load_paths(cls , paths : str | list[str], fillna = 'guess' , intersect_secid = True ,
                   start_dt = None , end_dt = None , dtype = torch.float):
        if isinstance(paths , str): paths = list(paths)
        _guess = lambda ls,excl:[os.path.basename(x).lower().startswith(excl) == 0 for x in ls]
        if fillna == 'guess':
            exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m','week')
            fillna = np.array(_guess(paths , exclude_list))
        elif fillna is None or isinstance(fillna , bool):
            fillna = np.repeat(fillna , len(paths))
        else:
            assert len(paths) == len(fillna) , (len(paths) , len(fillna))
        
        with Timer(f'Load  {len(paths)} DataBlocks'):
            blocks = [cls.load_path(path) for path in paths]

        with Timer(f'Align {len(paths)} DataBlocks'):
            # sligtly faster than .align(secid = secid , date = date)
            newsecid = None
            if intersect_secid:  newsecid = index_intersect([blk.secid for blk in blocks])[0]
            newdate  = index_union([blk.date for blk in blocks] , start_dt , end_dt)[0]
            
            for i , blk in enumerate(blocks):
                blk.align_secid_date(newsecid , newdate)
                if fillna[i]: blk.values = forward_fillna(blk.values , axis = 1)
                blk.as_tensor().as_type(dtype)

        return blocks
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        return cls.load_path(cls.block_path(key , predict , alias_search))

    @classmethod
    def load_keys(cls , keys : list[str] , predict = False , alias_search = True , **kwargs):
        paths = [cls.block_path(key , predict , alias_search) for key in keys]
        return cls.load_paths(paths , **kwargs)

    @classmethod
    def block_path(cls , key : str , predict=False, alias_search = True):
        train_mark = '.00' if predict else ''
        if key.lower() in ['y' , 'labels']: 
            return f'{PATH.block}/Y{train_mark}.{CONF.SAVE_OPT_BLK}'
        else:
            path = (f'{PATH.block}/X_'+'{}'+f'{train_mark}.{CONF.SAVE_OPT_BLK}')
            return data_type_alias(path , key) if alias_search else path.format(key)
    
    @classmethod
    def load_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , feature = None , **kwargs):
        target_dates = get_target_dates(db_src , db_key)
        if start_dt is not None: target_dates = target_dates[target_dates >= start_dt]
        if end_dt   is not None: target_dates = target_dates[target_dates <= end_dt]
        if len(target_dates) == 0: 
            return cls()
        if feature is not None:
            assert isinstance(feature , list) , feature
            feature = [f for f in feature if f not in ['secid' , 'date' , 'minute']]
        dfs = [cls.df_preprocess(load_target_file(db_src,db_key,date),date,feature) for date in target_dates]
        dfs = pd.concat([df for df in dfs if isinstance(df,pd.DataFrame)] , axis = 0)
        dfs = dfs.set_index(['secid' , 'date'] + ['minute'] * ('minute' in dfs.columns.values))
        return cls.from_dataframe(dfs)
    
    @classmethod
    def df_preprocess(cls , df : Optional[pd.DataFrame] , date : int , feature = None):
        if isinstance(df , pd.DataFrame): 
            if 'date' not in df.columns.values: df['date'] = date
            if feature is not None: 
                remain_cols = ['secid','date']+['minute']*('minute' in df.columns.values)+feature
                df = df.loc[:,remain_cols]
        else:
            df = None
        return df
    
    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
    
        adjfactor = adjfactor and ('adjfactor' in self.feature)
        if multiply is None and divide is None and (not adjfactor): return self  

        if isinstance(price_feat , (str,)): price_feat = [price_feat]
        i_price = np.where(np.isin(self.feature , price_feat))[0].astype(int)
        if len(i_price) == 0: return self
        v_price = self.values[...,i_price]

        if adjfactor : v_price *= self.loc(feature=['adjfactor'])
        if multiply  is not None: v_price *= multiply
        if divide    is not None: v_price /= divide

        self.values[...,i_price] = v_price 
        return self
    
    def adjust_volume(self , multiply = None , divide = None , 
                      vol_feat = ['volume' , 'amount', 'turn_tt', 'turn_fl', 'turn_fr']):
        if multiply is None and divide is None: return self

        if isinstance(vol_feat , (str,)): vol_feat = [vol_feat]
        i_vol = np.where(np.isin(self.feature , vol_feat))[0]
        if len(i_vol) == 0: return self
        v_vol = self.values[...,i_vol]
        if multiply is not None: v_vol *= multiply
        if divide   is not None: v_vol /= divide
        self.values[...,i_vol] = v_vol
        return self
    
    def mask_values(self , mask : dict , **kwargs):
        if not mask : return self
        mask_pos = np.full(self.shape , fill_value=False , dtype=bool)
        if mask_list_dt := mask.get('list_dt'):
            desc = load_target_file('information' , 'description')
            assert desc is not None
            desc = desc[desc['secid'] > 0].loc[:,['secid','list_dt','delist_dt']]
            if len(np.setdiff1d(self.secid , desc['secid'])) > 0:
                add_df = pd.DataFrame({
                        'secid':np.setdiff1d(self.secid , desc['secid']) ,
                        'list_dt':21991231 , 'delist_dt':21991231})
                desc = pd.concat([desc,add_df],axis=0).reset_index(drop=True)

            desc = desc.sort_values('list_dt',ascending=False).drop_duplicates(subset=['secid'],keep='first').set_index('secid') 
            secid , date = self.secid , self.date
            
            list_dt = np.array(desc.loc[secid , 'list_dt'])
            list_dt[list_dt < 0] = 21991231
            list_dt = date_offset(list_dt , mask_list_dt , astype = int)

            delist_dt = np.array(desc.loc[secid , 'delist_dt'])
            delist_dt[delist_dt < 0] = 21991231

            tmp = np.stack([(date <= l) + (date >= d) for l,d in zip(list_dt , delist_dt)] , axis = 0)
            mask_pos[tmp] = True

        assert (~mask_pos).sum() > 0
        self.values[mask_pos] = np.nan
        return self
    
    def hist_norm(self , key : str , predict = False ,
                  start_dt : Optional[int] = None , end_dt : Optional[int]  = 20161231 , 
                  step_day = 5 , **kwargs):
        return DataBlockNorm.calculate(self , key , predict , start_dt , end_dt , step_day , **kwargs)

@dataclass(slots=True)
class DataBlockNorm:
    avg : Tensor
    std : Tensor
    dtype : Any = None

    DIVLAST  : ClassVar[list[str]] = ['day']
    HISTNORM : ClassVar[list[str]] = ['day','15m','min','30m','60m','week']

    def __post_init__(self):
        self.avg = self.avg.to(self.dtype)
        self.std = self.std.to(self.dtype)

    @classmethod
    def calculate(cls , block : DataBlock , key : str , predict = False ,
                  start_dt : Optional[int] = None , end_dt : Optional[int]  = 20161231 , 
                  step_day = 5 , **kwargs):
        
        key = data_type_abbr(key)
        if predict or not (key in cls.HISTNORM): return None

        default_maxday = {'day' : 60}
        maxday = default_maxday.get(key , 1)

        date_slice = np.repeat(True , len(block.date))
        if start_dt is not None: date_slice[block.date < start_dt] = False
        if end_dt   is not None: date_slice[block.date > end_dt]   = False

        secid , date , inday , feat = block.secid , block.date , block.shape[2] , block.feature

        len_step = len(date[date_slice]) // step_day
        len_bars = maxday * inday

        x = torch.tensor(block.values[:,date_slice])
        pad_array = (0,0,0,0,maxday,0,0,0)
        x = torch.nn.functional.pad(x , pad_array , value = torch.nan)
        
        avg_x , std_x = torch.zeros(len_bars , len(feat)) , torch.zeros(len_bars , len(feat))

        x_endpoint = x.shape[1]-1 + step_day * np.arange(-len_step + 1 , 1)
        x_div = torch.ones(len(secid) , len_step , 1 , len(feat)).to(x)
        re_shape = (*x_div.shape[:2] , -1)
        if key in cls.DIVLAST: # divide by endpoint , day dataset
            x_div.copy_(x[:,x_endpoint,-1:])
            
        nan_sample = (x_div == 0).reshape(*re_shape).any(dim = -1)
        nan_sample += x_div.isnan().reshape(*re_shape).any(dim = -1)
        for i in range(maxday):
            nan_sample += x[:,x_endpoint-i].reshape(*re_shape).isnan().any(dim=-1)

        for i in range(maxday):
            vijs = ((x[:,x_endpoint - maxday+1 + i]) / (x_div + 1e-6))[nan_sample == 0]
            avg_x[i*inday:(i+1)*inday] = vijs.mean(dim = 0)
            std_x[i*inday:(i+1)*inday] = vijs.std(dim = 0)

        assert avg_x.isnan().sum() + std_x.isnan().sum() == 0 , ((nan_sample == 0).sum())
        
        data = cls(avg_x , std_x)
        data.save(key)
        return data

    def save(self , key):
        DataBlock.save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_path(cls , path : str , dtype = None):
        if not os.path.exists(path): return None
        data = DataBlock.load_dict(path)
        return cls(data['avg'] , data['std'] , dtype)

    @classmethod
    def load_paths(cls , paths : str | list[str] , dtype = None):
        if isinstance(paths , str): paths = [paths]
        norms = [cls.load_path(path , dtype) for path in paths]
        return norms
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        path = cls.norm_path(key , predict , alias_search)
        return cls.load_path(path)

    @classmethod
    def load_keys(cls , keys : str | list[str] , predict = False , alias_search = True , dtype = None):
        if isinstance(keys , str): keys = [keys]
        return [cls.load_key(key , predict , alias_search , dtype) for key in keys]
    
    @classmethod
    def norm_path(cls , key : str , predict = False, alias_search = True):
        if key.lower() == 'y': return f'{PATH.norm}/Y.{CONF.SAVE_OPT_NORM}'
        path = (f'{PATH.norm}/X_'+'{}'+f'.{CONF.SAVE_OPT_NORM}')
        return data_type_alias(path , key) if alias_search else path.format(key)

@dataclass(slots=True)
class ModuleData:
    '''load datas / norms / index'''
    x : dict[str,DataBlock]
    y : DataBlock
    norms : dict[str,DataBlockNorm]
    secid : np.ndarray
    date  : np.ndarray

    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def load(cls , data_type_list : list[str] , y_labels : Optional[list[str]] = None , 
             predict : bool = False , dtype : Optional[str | Any] = torch.float , save_upon_loading : bool = True):
        if dtype is None: dtype = torch.float
        if isinstance(dtype , str): dtype = getattr(torch , dtype)
        if predict: 
            dataset_path = 'no_dataset'
        else:
            last_date = max(DataBlock.load_dict(DataBlock.block_path('y'))['date'])
            dataset_code = '+'.join(data_type_list)
            dataset_path = f'{PATH.dataset}/{dataset_code}.{last_date}.pt'

        if y_labels is not None and os.path.exists(dataset_path):
            data = cls(**torch.load(dataset_path))
            if (np.isin(data_type_list , list(data.x.keys())).all() and
                np.isin(y_labels , list(data.y.feature)).all()):
                print(f'try using {dataset_path} , success!')
            else:
                print(f'try using {dataset_path} , but incompatible, load raw blocks!')
                data = None
        else:
            data = None

        if data is None:
            data_type_list = ['y' , *data_type_list]
            
            blocks = DataBlock.load_keys(data_type_list, predict , alias_search=True,dtype = dtype)
            norms  = DataBlockNorm.load_keys(data_type_list, predict , alias_search=True,dtype = dtype)

            y : DataBlock = blocks[0]
            x : dict[str,DataBlock] = {cls.abbr(key):blocks[i] for i,key in enumerate(data_type_list) if i != 0}
            norms = {cls.abbr(key):val for key,val in zip(data_type_list , norms) if val is not None}
            secid , date = blocks[0].secid , blocks[0].date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x' : x , 'y' : y , 'norms' : norms , 'secid' : secid , 'date' : date}
            if not predict and save_upon_loading: 
                torch.save(data , dataset_path , pickle_protocol = 4)
            data = cls(**data)

        data.y.align_feature(y_labels)
        return data
    
    @staticmethod
    def abbr(data_type : str): return data_type_abbr(data_type)