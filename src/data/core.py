import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from torch import Tensor
from typing import Any , ClassVar , Literal , Optional

from .classes import Stock4DData
from ..basic import PATH , SILENT
from ..basic.util import Timer
from ..func import index_union , index_intersect , forward_fillna
from ..func.time import date_offset , today

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

def data_type_alias(key : str):
    return [key , f'trade_{key}' , key.replace('trade_','')]

class DataBlock(Stock4DData):
    DEFAULT_INDEX = ['secid','date','minute','factor_name']

    def save(self , key : str , predict=False , start_dt = None , end_dt = None):
        path = self.block_path(key , predict) 
        path.parent.mkdir(exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: date_slice[self.date < start_dt] = False
        if end_dt   is not None: date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        self.save_dict(data , path)

    def ffill(self):
        self.values = forward_fillna(self.values , axis = 1)
        return self
    
    @staticmethod
    def save_dict(data : dict , file_path : Path):
        if file_path is None: return NotImplemented
        Path(file_path).parent.mkdir(exist_ok=True)
        if file_path.suffix in ['.npz' , '.npy' , '.np']:
            np.savez_compressed(file_path , **data)
        elif file_path.suffix in ['.pt' , '.pth']:
            torch.save(data , file_path , pickle_protocol = 4)
        else:
            raise Exception(file_path)
        
    @staticmethod
    def load_dict(file_path : Path , keys = None) -> dict[str,Any]:
        if file_path.suffix in ['.npz' , '.npy' , '.np']:
            file = np.load(file_path)
        elif file_path.suffix in ['.pt' , '.pth']:
            file = torch.load(file_path)
        else:
            raise Exception(file_path)
        keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
        data = {k:file[k] for k in keys}
        return data
    
    @classmethod
    def load_path(cls , path : Path): return cls(**cls.load_dict(path))
    
    @classmethod
    def load_paths(cls , paths : Path | list[Path], fillna = 'guess' , intersect_secid = True ,
                   start_dt = None , end_dt = None , dtype = torch.float):
        if not isinstance(paths , list): paths = [paths]
        _guess = lambda ls,excl:[Path(x).name.lower().startswith(excl) == 0 for x in ls]
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
            for blk in blocks: newdate = newdate[newdate >= min(blk.date)]
            
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
            return PATH.block.joinpath(f'Y{train_mark}.{PATH.SAVE_OPT_BLK}')
        else:
            alias_list = data_type_alias(key) if alias_search else []
            for new_key in alias_list:
                path = PATH.block.joinpath(f'X_{new_key}{train_mark}.{PATH.SAVE_OPT_BLK}')
                if path.exists(): return path
            return PATH.block.joinpath(f'X_{key}{train_mark}.{PATH.SAVE_OPT_BLK}')
    
    @classmethod
    def load_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , feature = None , 
                parallel : Literal['thread' , 'process'] | None = 'thread' , max_workers = 20):
        dates = PATH.get_target_dates(db_src , db_key , start_dt=start_dt , end_dt=end_dt)
        dfs = PATH.load_target_file_dates(db_src , db_key , dates , parallel = parallel, max_workers=max_workers)
        dfs = [df.assign(date = date) for date,df in dfs.items() if df is not None and not df.empty]
        df  = pd.concat(dfs) if len(dfs) else pd.DataFrame()
        if len(df) == 0: return cls()

        use_index = [f for f in cls.DEFAULT_INDEX if f in df.columns]
        assert len(use_index) <= 3 , use_index
        if feature is not None:  df = df.loc[:,use_index + [f for f in feature if f not in use_index]]

        return cls.from_dataframe(df.set_index(use_index))
    
    @property
    def price_adjusted(self): return getattr(self , '_price_adjusted' , False)

    @property
    def volume_adjusted(self): return getattr(self , '_volume_adjusted' , False)

    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
        if self.price_adjusted: return self
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

        if 'vwap' in self.feature:
            i_vp = np.where(self.feature == 'vwap')[0].astype(int)
            nan_idx = self.values[...,i_vp].isnan() if isinstance(self.values , Tensor) else np.isnan(self.values[...,i_vp])
            nan_idx = nan_idx.squeeze(-1)
            pcols = [col for col in ['close', 'high', 'low', 'open' , 'preclose'] if col in self.feature]
            if pcols: 
                i_cp = np.where(self.feature == pcols[0])[0].astype(int)
                self.values[nan_idx , i_vp] = self.values[nan_idx , i_cp]
            else:
                ...
        
        self._price_adjusted = True
        return self
    
    def adjust_volume(self , multiply = None , divide = None , 
                      vol_feat = ['volume' , 'amount', 'turn_tt', 'turn_fl', 'turn_fr']):
        if self.volume_adjusted: return self
        if multiply is None and divide is None: return self

        if isinstance(vol_feat , (str,)): vol_feat = [vol_feat]
        i_vol = np.where(np.isin(self.feature , vol_feat))[0]
        if len(i_vol) == 0: return self
        v_vol = self.values[...,i_vol]
        if multiply is not None: v_vol *= multiply
        if divide   is not None: v_vol /= divide
        self.values[...,i_vol] = v_vol
        self._volume_adjusted = True
        return self
    
    def mask_values(self , mask : dict , **kwargs):
        if not mask : return self
        mask_pos = np.full(self.shape , fill_value=False , dtype=bool)
        if mask_list_dt := mask.get('list_dt'):
            desc = PATH.load_target_file('information' , 'description')
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
    def load_path(cls , path : Path , dtype = None):
        if not path.exists(): return None
        data = DataBlock.load_dict(path)
        return cls(data['avg'] , data['std'] , dtype)

    @classmethod
    def load_paths(cls , paths : Path | list[Path] , dtype = None):
        if not isinstance(paths , list): paths = [paths]
        norms = [cls.load_path(path , dtype) for path in paths]
        return norms
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        path = cls.norm_path(key , predict , alias_search)
        return cls.load_path(path)

    @classmethod
    def load_keys(cls , keys : str | list[str] , predict = False , alias_search = True , dtype = None):
        if not isinstance(keys , list): keys = [keys]
        return [cls.load_key(key , predict , alias_search , dtype) for key in keys]
    
    @classmethod
    def norm_path(cls , key : str , predict = False, alias_search = True):
        if key.lower() == 'y': return PATH.norm.joinpath(f'Y.{PATH.SAVE_OPT_NORM}')
        alias_list = data_type_alias(key) if alias_search else []
        for new_key in alias_list:
            path = PATH.norm.joinpath(f'X_{new_key}.{PATH.SAVE_OPT_BLK}')
            if path.exists(): return path
        return PATH.norm.joinpath(f'X_{key}.{PATH.SAVE_OPT_BLK}')

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
             fit : bool = True , predict : bool = False , 
             dtype : Optional[str | Any] = torch.float , 
             save_upon_loading : bool = True):
        
        assert fit or predict , (fit , predict)
        if not predict: 
            return cls.load_datas(data_type_list , y_labels , False , dtype , save_upon_loading)
        elif not fit:
            return cls.load_datas(data_type_list , y_labels , True  , dtype , save_upon_loading)
        else:
            hist_data = cls.load_datas(data_type_list , y_labels , False , dtype , save_upon_loading)
            pred_data = cls.load_datas(data_type_list , y_labels , True  , dtype , save_upon_loading)

            hist_data.y = hist_data.y.merge_others([pred_data.y])
            hist_data.secid , hist_data.date = hist_data.y.secid , hist_data.y.date
            for x_key in hist_data.x:
                hist_data.x[x_key] = hist_data.x[x_key].merge_others([pred_data.x[x_key]]).\
                    align_secid_date(hist_data.secid , hist_data.date)

            return hist_data

    @classmethod
    def load_datas(cls , data_type_list : list[str] , y_labels : Optional[list[str]] = None , 
                   predict : bool = False , dtype : Optional[str | Any] = torch.float , 
                   save_upon_loading : bool = True):
        if dtype is None: dtype = torch.float
        if isinstance(dtype , str): dtype = getattr(torch , dtype)
        if predict: 
            dataset_path = 'no_dataset'
        else:
            last_date = max(DataBlock.load_dict(DataBlock.block_path('y'))['date'])
            dataset_code = '+'.join(data_type_list)
            dataset_path = f'{PATH.dataset}/{dataset_code}.{last_date}.pt'

        if y_labels is not None and Path(dataset_path).exists():
            data = cls(**torch.load(dataset_path))
            if (np.isin(data_type_list , list(data.x.keys())).all() and
                np.isin(y_labels , list(data.y.feature)).all()):
                if not SILENT: print(f'try using {dataset_path} , success!')
            else:
                if not SILENT: print(f'try using {dataset_path} , but incompatible, load raw blocks!')
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
            secid , date = y.secid , y.date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x' : x , 'y' : y , 'norms' : norms , 'secid' : secid , 'date' : date}
            if not predict and save_upon_loading and data_type_list: 
                torch.save(data , dataset_path , pickle_protocol = 4)
            data = cls(**data)

        data.y.align_feature(y_labels)
        return data
    
    @staticmethod
    def abbr(data_type : str): return data_type_abbr(data_type)