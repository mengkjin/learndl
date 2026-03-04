import torch
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Any , ClassVar , Literal

from src.proj import PATH , Logger , CALENDAR , DB , Proj
from src.proj.func import torch_load
from src.math import index_union , index_intersect , forward_fillna

from . import Stock4DData
from ..stock_info import INFO

__all__ = ['DataBlock' , 'DataBlockNorm']

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

def save_dict(data : dict , file_path : Path):
    if file_path is None: 
        return NotImplemented
    Path(file_path).parent.mkdir(exist_ok=True)
    if file_path.suffix in ['.npz' , '.npy' , '.np']:
        np.savez_compressed(file_path , **data)
    elif file_path.suffix in ['.pt' , '.pth']:
        torch.save(data , file_path , pickle_protocol = 4)
    else:
        raise Exception(file_path)

def load_dict(file_path : Path , keys = None) -> dict[str,Any]:
    if file_path.suffix in ['.npz' , '.npy' , '.np']:
        file = np.load(file_path)
    elif file_path.suffix in ['.pt' , '.pth']:
        file = torch_load(file_path)
    else:
        raise Exception(file_path)
    keys = file.keys() if keys is None else np.intersect1d(keys , list(file.keys()))
    data = {k:file[k] for k in keys}
    return data

class DataBlock(Stock4DData):
    DEFAULT_INDEX = ['secid','date','minute','factor_name']
    FREQUENT_DBS = ['trade_ts.day' , 'trade_ts.val']
    FREQUENT_MIN_DATES = 500

    @staticmethod
    def data_type_abbr(key : str): 
        return data_type_abbr(key)

    @staticmethod
    def data_type_alias(key : str): 
        return data_type_alias(key)

    @classmethod
    def last_modified_date(cls , key , predict):
        return PATH.file_modified_date(cls.key_path(key , predict))
    
    @classmethod
    def last_modified_time(cls , key , predict):
        return PATH.file_modified_time(cls.key_path(key , predict))

    @staticmethod
    def key_path(key : str , predict=False, alias_search = True):
        type = 'predict' if predict else 'fit'
        if key.lower() in ['y' , 'labels']: 
            return DB.block_path(type , 'Y')
        else:
            alias_list = data_type_alias(key) if alias_search else []
            for new_key in alias_list:
                path = DB.block_path(type , f'X_{new_key}')
                if path.exists(): 
                    return path
            return DB.block_path(type , f'X_{key}')
        
    @staticmethod
    def norm_path(key : str , predict = False, alias_search = True):
        return DataBlockNorm.norm_path(key , predict , alias_search)

    @classmethod
    def raw_path(cls , src : str , key : str):
        return DB.block_path('raw' , f'{src}.{key}')

    @classmethod
    def last_data_date(cls , key : str = 'y' , predict = False):
        try:
            path = cls.key_path(key , predict)
            if not path.exists():
                return None
            return max(load_dict(path)['date'])
        except ModuleNotFoundError as e:
            Logger.error(f'last_data_date({key , predict}) error: ModuleNotFoundError: {e}')
            return None

    def save_keys(self , key : str , predict=False , start_dt = None , end_dt = None):
        path = self.key_path(key , predict) 
        path.parent.mkdir(exist_ok=True)
        date_slice = np.repeat(True,len(self.date))
        if start_dt is not None: 
            date_slice[self.date < start_dt] = False
        if end_dt   is not None: 
            date_slice[self.date > end_dt]   = False
        data = {'values'  : self.values[:,date_slice] , 
                'date'    : self.date[date_slice].astype(int) ,
                'secid'   : self.secid.astype(int) , 
                'feature' : self.feature}
        save_dict(data , path)

    def save_raw(self , src : str , key : str):
        path = self.raw_path(src , key)
        path.parent.mkdir(exist_ok=True)
        save_dict({'values' : self.values , 'date' : self.date.astype(int) , 'secid' : self.secid.astype(int) , 'feature' : self.feature} , path)

    def ffill(self , if_fill : bool = True):
        if if_fill:
            self.values = forward_fillna(self.values , axis = 1)
        return self

    def fillna(self , value : Any = 0):
        if isinstance(self.values , torch.Tensor):
            self.values = self.values.nan_to_num(value)
        elif isinstance(self.values , np.ndarray):
            self.values = np.nan_to_num(self.values , value)
        else:
            raise TypeError(f'Unsupported type: {type(self.values)} for {self.__class__.__name__} values')
        return self
        
    @staticmethod
    def guess_fillna(name : str , fillna : Literal['guess'] | bool | None = 'guess' , 
                     excl : tuple[str,...] = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m','week')) -> bool:
        if fillna == 'guess':
            return name.startswith(excl) == 0
        else:
            return bool(fillna)

    @classmethod
    def load_keys(cls , keys : list[str] | str , predict = False , alias_search = True , 
                  fillna : Literal['guess'] | bool | None = 'guess' , intersect_secid = True ,
                  start_dt = None , end_dt = None , dtype : str | Any = torch.float , vb_level = 2 , **kwargs):
        if isinstance(keys , str):
            keys = [keys]
        paths = [cls.key_path(key , predict , alias_search) for key in keys]
        fillnas = {key:cls.guess_fillna(path.stem.lower() , fillna) for key , path in zip(keys , paths)}
        
        block_title = f'{len(keys)} DataBlocks' if len(keys) > 3 else f'DataBlock [{",".join(keys)}]'
        with Logger.Timer(f'Load {block_title} (predict={predict})' , vb_level = vb_level):
            blocks : dict[str,'DataBlock'] = {}
            for key , path in zip(keys , paths):
                block = cls(**load_dict(path))
                if predict and key == 'y' and not block.empty:
                    block = block.align_date(CALENDAR.td_within(min(block.date) , CALENDAR.updated()))
                blocks[key] = block

        if len(blocks) <= 1:
            return blocks

        with Logger.Timer(f'Align {block_title} (predict={predict})' , vb_level = vb_level):
            # sligtly faster than .align(secid = secid , date = date)
            if intersect_secid:  
                newsecid = index_intersect([blk.secid for blk in blocks.values()])[0]
            else:
                newsecid = None
            
            max_min_date = max([min(blk.date) for blk in blocks.values() if not blk.empty])
            newdate : np.ndarray | Any = index_union([blk.date for blk in blocks.values()] , start_dt , end_dt)[0]
            newdate = newdate[newdate >= max_min_date]
            
            for key , blk in blocks.items():
                blk.align_secid_date(newsecid , newdate , inplace = True).ffill(fillnas[key]).as_tensor().as_type(dtype)

        return blocks

    @classmethod
    def load_keys_norms(cls , keys : list[str] | str , predict = False , alias_search = True , dtype = None):
        if isinstance(keys , str):
            keys = [keys]
        return DataBlockNorm.load_keys(keys, predict , alias_search = alias_search , dtype = dtype)

    @classmethod
    def load_from_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , * , 
                dates = None , feature = None , use_alt = True , vb_level = Proj.vb.max):

        if dates is None:
            dates = CALENDAR.td_within(start_dt , end_dt)

        if len(dates) == 0:
            return cls()

        df = DB.loads(db_src , db_key , dates = dates , use_alt=use_alt , fill_datavendor=True , vb_level=vb_level)

        if len(df) == 0: 
            return cls()
            
        if len(df.index.names) > 1 or df.index.name: 
            df = df.reset_index()
        use_index = [f for f in cls.DEFAULT_INDEX if f in df.columns]
        assert 2 <= len(use_index) <= 3 , use_index
        if feature is not None:  
            df = df.loc[:,use_index + [f for f in feature if f not in use_index]]
        return cls.from_dataframe(df.set_index(use_index))

    @classmethod
    def load_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , * , 
                dates = None , feature = None , use_alt = True , vb_level = Proj.vb.max):
        if dates is None:
            dates = CALENDAR.td_within(start_dt , end_dt)

        if f'{db_src}.{db_key}' in cls.FREQUENT_DBS:
            block = cls()
            raw_path = cls.raw_path(db_src , db_key)
            if len(dates) >= cls.FREQUENT_MIN_DATES and raw_path.exists():
                try:
                    block = cls(**load_dict(raw_path))
                except Exception as e:
                    Logger.error(f'load_db({db_src} , {db_key}) error: {e}')
                    raw_path.unlink()
            saved_dates = block.date
            block = block.extend_to(db_src , db_key , start_dt , end_dt , feature = feature , use_alt = use_alt , vb_level = vb_level)
            if not np.array_equal(saved_dates , block.date) or not raw_path.exists():
                block.save_raw(db_src , db_key)
            block = block.align(feature = feature , inplace = True)
        else:
            block = cls.load_from_db(db_src , db_key , dates = dates , feature = feature , use_alt = use_alt , vb_level = vb_level)
        return block
    
    @property
    def price_adjusted(self): 
        return getattr(self , '_price_adjusted' , False)

    @property
    def volume_adjusted(self): 
        return getattr(self , '_volume_adjusted' , False)

    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
        if self.price_adjusted or self.empty: 
            return self
        adjfactor = adjfactor and ('adjfactor' in self.feature)
        if multiply is None and divide is None and (not adjfactor): 
            return self  

        if isinstance(price_feat , (str,)): 
            price_feat = [price_feat]
        i_price = np.where(np.isin(self.feature , price_feat))[0].astype(int)
        if len(i_price) == 0: 
            return self
        v_price = self.values[...,i_price]

        if adjfactor :  
            v_price *= self.loc(feature=['adjfactor'] , fillna = 1)
        if multiply  is not None: 
            v_price *= multiply
        if divide    is not None: 
            v_price /= divide
        self.values[...,i_price] = v_price 

        if 'vwap' in self.feature:
            i_vp = np.where(self.feature == 'vwap')[0].astype(int)
            nan_idx = self.values[...,i_vp].isnan() if isinstance(self.values , torch.Tensor) else np.isnan(self.values[...,i_vp])
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
        if self.volume_adjusted: 
            return self
        if multiply is None and divide is None: 
            return self

        if isinstance(vol_feat , (str,)): 
            vol_feat = [vol_feat]
        i_vol = np.where(np.isin(self.feature , vol_feat))[0]
        if len(i_vol) == 0: 
            return self
        v_vol = self.values[...,i_vol]
        if multiply is not None: 
            v_vol *= multiply
        if divide   is not None: 
            v_vol /= divide
        self.values[...,i_vol] = v_vol
        self._volume_adjusted = True
        return self
    
    def mask_values(self , mask : dict , **kwargs):
        if not mask: 
            return self
        mask_pos = np.full(self.shape , fill_value=False , dtype=bool)
        if mask_list_dt := mask.get('list_dt'):
            desc = INFO.get_desc(set_index=False)
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
            list_dt = CALENDAR.cd_array(list_dt , mask_list_dt).astype(int)

            delist_dt = np.array(desc.loc[secid , 'delist_dt'])
            delist_dt[delist_dt < 0] = 21991231

            tmp = np.stack([(date <= lst) + (date >= dls) for lst,dls in zip(list_dt , delist_dt)] , axis = 0)
            mask_pos[tmp] = True

        assert (~mask_pos).sum() > 0 , 'all values are masked'
        self.values[mask_pos] = np.nan
        return self
    
    def hist_norm(self , key : str , predict = False ,
                  start_dt : int | None = None , end_dt : int | None  = 20161231 , 
                  step_day = 5 , **kwargs):
        return DataBlockNorm.calculate(self , key , predict , start_dt , end_dt , step_day , **kwargs)

    def extend_to(self , src : str , key : str , start_dt : int | None = None , end_dt : int | None = None , * , 
                  dates = None , feature : list[str] | None = None , use_alt = True , inplace = True , vb_level = Proj.vb.max):
        if dates is None:
            dates = CALENDAR.td_within(start_dt , end_dt)
        block = self.load_from_db(src , key , dates = CALENDAR.diffs(dates , self.date) , feature = feature , use_alt = use_alt , vb_level = vb_level)
        block = block.adjust_price()
        self = self.merge_others(block , inplace = inplace)
        return self

@dataclass(slots=True)
class DataBlockNorm:
    avg : torch.Tensor
    std : torch.Tensor
    dtype : Any = None

    DIVLAST  : ClassVar[list[str]] = ['day']
    HISTNORM : ClassVar[list[str]] = ['day','15m','min','30m','60m','week']

    def __post_init__(self):
        self.avg = self.avg.to(self.dtype)
        self.std = self.std.to(self.dtype)

    @classmethod
    def calculate(cls , block : DataBlock , key : str , predict = False ,
                  start_dt : int | None = None , end_dt : int | None  = 20161231 , 
                  step_day = 5 , **kwargs):
        
        key = data_type_abbr(key)
        if predict or (key not in cls.HISTNORM): 
            return None

        default_maxday = {'day' : 60}
        maxday = default_maxday.get(key , 1)

        date_slice = np.repeat(True , len(block.date))
        if start_dt is not None: 
            date_slice[block.date < start_dt] = False
        if end_dt   is not None: 
            date_slice[block.date > end_dt]   = False

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
        path = self.norm_path(key)
        path.parent.mkdir(exist_ok=True)
        save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_keys(cls , keys : str | list[str] , predict = False , alias_search = True , dtype = None) -> dict[str,'DataBlockNorm']:
        if not isinstance(keys , list): 
            keys = [keys]
        norms = {}
        for key in keys:
            path = cls.norm_path(key , predict , alias_search)
            if not path.exists(): 
                continue
            data = load_dict(path)
            norms[key] = cls(data['avg'] , data['std'] , dtype)
        return norms
    
    @classmethod
    def norm_path(cls , key : str , predict = False, alias_search = True):
        if key.lower() == 'y':
            return DB.norm_path('fit' , 'Y')
        alias_list = data_type_alias(key) if alias_search else []
        for new_key in alias_list:
            path = DB.norm_path('fit' , f'X_{new_key}')
            if path.exists():
                return path
        return DB.norm_path('fit' , f'X_{key}')