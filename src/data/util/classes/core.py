import torch , json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from typing import Any , ClassVar

from src.proj import PATH , Logger , SILENT , Timer
from src.basic import CALENDAR , DB , torch_load
from src.func import index_union , index_intersect , forward_fillna

from . import Stock4DData
from ..stock_info import INFO

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

def block_file_path(key : str , predict=False, alias_search = True):
    train_mark = '.00' if predict else ''
    if key.lower() in ['y' , 'labels']: 
        return DB.block_path(f'Y{train_mark}')
    else:
        alias_list = data_type_alias(key) if alias_search else []
        for new_key in alias_list:
            path = DB.block_path(f'X_{new_key}{train_mark}')
            if path.exists(): 
                return path
        return DB.block_path(f'X_{key}{train_mark}')

def norm_file_path(key : str , predict = False, alias_search = True):
    if key.lower() == 'y':
        return DB.norm_path(f'Y')
    alias_list = data_type_alias(key) if alias_search else []
    for new_key in alias_list:
        path = DB.norm_path(f'X_{new_key}')
        if path.exists():
            return path
    return DB.norm_path(f'X_{key}')

class DataBlock(Stock4DData):
    DEFAULT_INDEX = ['secid','date','minute','factor_name']

    @staticmethod
    def data_type_abbr(key : str): 
        return data_type_abbr(key)

    @staticmethod
    def data_type_alias(key : str): 
        return data_type_alias(key)

    @classmethod
    def last_modified_date(cls , key , predict):
        return PATH.file_modified_date(cls.block_path(key , predict))
    
    @classmethod
    def last_modified_time(cls , key , predict):
        return PATH.file_modified_time(cls.block_path(key , predict))

    @staticmethod
    def block_path(key : str , predict=False, alias_search = True):
        return block_file_path(key , predict , alias_search)
        
    @staticmethod
    def norm_path(key : str , predict = False, alias_search = True):
        return norm_file_path(key , predict , alias_search)

    @classmethod
    def last_data_date(cls , key : str = 'y' , predict = False):
        try:
            return max(DataBlock.load_dict(DataBlock.block_path(key , predict))['date'])
        except ModuleNotFoundError as e:
            Logger.error(f'last_data_date({key , predict}) error: ModuleNotFoundError: {e}')
            return None

    def save(self , key : str , predict=False , start_dt = None , end_dt = None):
        path = self.block_path(key , predict) 
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
        self.save_dict(data , path)

    def ffill(self , fill : bool | Any = True):
        if bool(fill):
            self.values = forward_fillna(self.values , axis = 1)
        return self
    
    @staticmethod
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
        
    @staticmethod
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
    
    @classmethod
    def load_path(cls , path : Path): 
        return cls(**cls.load_dict(path))
    
    @classmethod
    def load_paths(cls , paths : Path | list[Path], fillna = 'guess' , intersect_secid = True ,
                   start_dt = None , end_dt = None , dtype = torch.float):
        if not isinstance(paths , list): 
            paths = [paths]
        def _guess(ls,excl):
            return [Path(x).name.lower().startswith(excl) == 0 for x in ls]
        if fillna == 'guess':
            exclude_list = ('y','x_trade','x_day','x_15m','x_min','x_30m','x_60m','week')
            fillna = np.array(_guess(paths , exclude_list))
        elif fillna is None or isinstance(fillna , bool):
            fillna = np.repeat(fillna , len(paths))
        else:
            assert len(paths) == len(fillna) , (len(paths) , len(fillna))
        
        with Timer(f'Load {len(paths)} DataBlocks'):
            blocks = [cls.load_path(path) for path in paths]

            if len(blocks) == 1:
                blocks[0].ffill(fillna[0]).as_tensor().as_type(dtype)
                return blocks

        with Timer(f'Align DataBlocks'):
            # sligtly faster than .align(secid = secid , date = date)
            if intersect_secid:  
                newsecid = index_intersect([blk.secid for blk in blocks])[0]
            else:
                newsecid = None
            
            newdate : np.ndarray | Any = index_union([blk.date for blk in blocks] , start_dt , end_dt)[0]
            for blk in blocks: 
                newdate = newdate[newdate >= min(blk.date)]
            
            for i , blk in enumerate(blocks):
                blk.align_secid_date(newsecid , newdate).ffill(fillna[i]).as_tensor().as_type(dtype)

        return blocks
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        return cls.load_path(cls.block_path(key , predict , alias_search))

    @classmethod
    def load_keys(cls , keys : list[str] , predict = False , alias_search = True , **kwargs):
        paths = [cls.block_path(key , predict , alias_search) for key in keys]
        return {key:val for key,val in zip(keys , cls.load_paths(paths , **kwargs))}
    
    @classmethod
    def load_db(cls , db_src : str , db_key : str , start_dt = None , end_dt = None , feature = None , use_alt = True):
        dates = CALENDAR.td_within(start_dt , end_dt)
        main_dates = np.intersect1d(dates , DB.dates(db_src , db_key , use_alt=use_alt))
        df = DB.load_multi(db_src , db_key , main_dates , use_alt=use_alt)

        if len(df) == 0: 
            return cls()
        if len(df.index.names) > 1 or df.index.name: 
            df = df.reset_index()
        use_index = [f for f in cls.DEFAULT_INDEX if f in df.columns]
        assert 2 <= len(use_index) <= 3 , use_index
        if feature is not None:  
            df = df.loc[:,use_index + [f for f in feature if f not in use_index]]
        return cls.from_dataframe(df.set_index(use_index))
    
    @property
    def price_adjusted(self): 
        return getattr(self , '_price_adjusted' , False)

    @property
    def volume_adjusted(self): 
        return getattr(self , '_volume_adjusted' , False)

    def adjust_price(self , adjfactor = True , multiply : Any = 1 , divide : Any = 1 , 
                     price_feat = ['preclose' , 'close', 'high', 'low', 'open', 'vwap']):
        if self.price_adjusted: 
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
        DataBlock.save_dict({'avg' : self.avg , 'std' : self.std} , self.norm_path(key))

    @classmethod
    def load_path(cls , path : Path , dtype = None):
        if not path.exists(): 
            return None
        data = DataBlock.load_dict(path)
        return cls(data['avg'] , data['std'] , dtype)

    @classmethod
    def load_paths(cls , paths : Path | list[Path] , dtype = None):
        if not isinstance(paths , list): 
            paths = [paths]
        norms = [cls.load_path(path , dtype) for path in paths]
        return norms
    
    @classmethod
    def load_key(cls , key : str , predict = False , alias_search = True , dtype = None):
        path = cls.norm_path(key , predict , alias_search)
        return cls.load_path(path)

    @classmethod
    def load_keys(cls , keys : str | list[str] , predict = False , alias_search = True , dtype = None):
        if not isinstance(keys , list): 
            keys = [keys]
        return {key:cls.load_key(key , predict , alias_search , dtype) for key in keys}
    
    @classmethod
    def norm_path(cls , key : str , predict = False, alias_search = True):
        return norm_file_path(key , predict , alias_search)

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
    def load(cls , data_type_list : list[str] , 
             y_labels : list[str] | None = None , 
             factor_names : list[str] | None = None ,
             fit : bool = True , predict : bool = False , 
             dtype : str | Any = torch.float , 
             save_upon_loading : bool = True):
        
        assert fit or predict , (fit , predict)
        if not predict: 
            return cls.load_datas(data_type_list , y_labels , factor_names , False , dtype , save_upon_loading)
        elif not fit:
            return cls.load_datas(data_type_list , y_labels , factor_names , True  , dtype , save_upon_loading)
        else:
            hist_data = cls.load_datas(data_type_list , y_labels , factor_names , False , dtype , save_upon_loading)
            pred_data = cls.load_datas(data_type_list , y_labels , factor_names , True  , dtype , save_upon_loading)

            hist_data.y = hist_data.y.merge_others([pred_data.y])
            hist_data.secid , hist_data.date = hist_data.y.secid , hist_data.y.date
            for x_key in hist_data.x:
                hist_data.x[x_key] = hist_data.x[x_key].merge_others([pred_data.x[x_key]]).\
                    align_secid_date(hist_data.secid , hist_data.date)

            return hist_data

    @classmethod
    def load_datas(cls , data_type_list : list[str] , 
                   y_labels : list[str] | None = None , 
                   factor_names : list[str] | None = None ,
                   predict : bool = False , dtype : str | Any = torch.float , 
                   save_upon_loading : bool = True):
        '''
        load all x/y data if input_type is data or factor
        if predict is True, only load recent data
        '''
        if dtype is None: 
            dtype = torch.float
        if isinstance(dtype , str): 
            dtype = getattr(torch , dtype)

        if predict: 
            data = None
        else:
            last_date = DataBlock.last_data_date()
            data = cls.datacache_load(last_date , data_type_list , y_labels)

        if data is None:
            blocks = DataBlock.load_keys(['y' , *data_type_list], predict , dtype = dtype)
            norms  = DataBlockNorm.load_keys(['y' , *data_type_list], predict , dtype = dtype)

            y : DataBlock = blocks['y']
            x : dict[str,DataBlock] = {cls.abbr(key):val for key,val in blocks.items() if key != 'y'}
            norms = {cls.abbr(key):val for key,val in norms.items() if val is not None and key != 'y'}
            secid , date = y.secid , y.date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x' : x , 'y' : y , 'norms' : norms , 'secid' : secid , 'date' : date}
            if not predict and save_upon_loading: 
                cls.datacache_save(data , last_date or y.date[-1] , data_type_list)
            data = cls(**data)

        if factor_names:
            with Timer(f'Load {len(factor_names)} Factors'):
                from src.data.loader import FactorLoader
                add_x = FactorLoader(factor_names).load_block(data.date[0] , data.date[-1] , silent = True)
                data.x['factor'] = add_x.align_secid_date(data.secid , data.date)

        data.y.align_feature(y_labels)
        return data
    
    @staticmethod
    def abbr(data_type : str): 
        return data_type_abbr(data_type)

    @classmethod
    def datacache_key(cls , data_type_list : list[str]) -> str:
        if not data_type_list:
            return 'ds_y_only'
        cache_key_json_file = PATH.datacache.joinpath('cache_key.json')
        cache_key_json_file.touch(exist_ok=True)
        with open(cache_key_json_file , 'r') as f:
            try:
                cache_key_dict = json.load(f)
            except json.JSONDecodeError as e:
                print(f'cache_key.json is corrupted, reset it: {e}')
                cache_key_dict = {}
        for key , value in cache_key_dict.items():
            if value['type'] != 'dataset':
                continue
            if sorted(value['content']) == sorted(data_type_list):
                return key

        if len(data_type_list) < 5:
            new_key = 'ds_' + '+'.join(data_type_list)
        else:
            i = 0
            while True:
                new_key = f'ds_{len(data_type_list)}datas_{i:02d}'
                if new_key not in cache_key_dict:
                    break
                i += 1
        cache_key_dict.update({new_key : {'type' : 'dataset' , 'content' : data_type_list}})
        with open(cache_key_json_file , 'w') as f:
            json.dump(cache_key_dict , f)
        return new_key

    @classmethod
    def datacache_path(cls , date : int , data_type_list : list[str]) -> Path:
        data_cache_key = cls.datacache_key(data_type_list)
        return PATH.datacache.joinpath(data_cache_key , f'{date}.pt')

    @classmethod
    def datacache_load(cls , date : int | None , data_type_list : list[str] , y_labels : list[str] | None = None):
        if date is None:
            return None
        path = cls.datacache_path(date , data_type_list)
        if path is None or not path.exists():
            return None
        try:
            data = cls(**torch_load(path))
            if (np.isin(data_type_list , list(data.x.keys())).all() and
                (y_labels is None or np.isin(y_labels , list(data.y.feature)).all())):
                if not SILENT: 
                    print(f'Loading Module Data, Try \'{path}\', success!')
            else:
                if not SILENT: 
                    Logger.warning(f'Loading Module Data, Try \'{path}\', Incompatible, Load Raw blocks!')
                data = None
        except ModuleNotFoundError:
            '''can be caused by different package version'''
            Logger.warning(f'Loading Module Data, Try \'{path}\', Incompatible, Load Raw blocks!')
            data = None
        except Exception as e:
            raise e

        cls.datacache_purge_old(data_type_list)
        return data
    
    @classmethod
    def datacache_save(cls , data : dict , date : int , data_type_list : list[str]):
        if not data_type_list:
            return
        path = cls.datacache_path(date , data_type_list)
        path.parent.mkdir(exist_ok=True)
        torch.save(data , path , pickle_protocol = 4)

    @classmethod
    def datacache_purge_old(cls , data_type_list : list[str]):
        data_cache_key = cls.datacache_key(data_type_list)
        folder = PATH.datacache.joinpath(data_cache_key)
        dates = [int(path.stem) for path in folder.iterdir()]
        if len(dates) <= 1:
            return
        for path in folder.iterdir():
            if path.is_file() and int(path.stem) < max(dates):
                path.unlink()

    @classmethod
    def purge_all(cls):
        with open(PATH.datacache.joinpath('cache_key.json') , 'r') as f:
            cache_key_dict = json.load(f)
        for key , value in cache_key_dict.items():
            data_type_list = value['content']
            data_cache_key = key
            assert data_cache_key == cls.datacache_key(data_type_list) , (data_cache_key, cls.datacache_key(data_type_list))
            folder = PATH.datacache.joinpath(data_cache_key)
            files = list(folder.iterdir())
            if len(files) <= 1:
                continue
            files.sort(key = lambda x: int(x.stem))
            for file in files[:-1]:
                file.unlink()