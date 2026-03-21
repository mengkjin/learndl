import torch , json
import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.proj import PATH , Logger , Proj
from src.proj.func import torch_load , properties

from .data_block import DataBlock , DataBlockNorm , data_type_abbr

__all__ = ['ModuleData']

@dataclass(slots=True)
class ModuleData:
    '''load datas / norms / index'''
    x : dict[str,DataBlock]
    y : DataBlock
    norms : dict[str,DataBlockNorm]
    secid : np.ndarray
    date  : np.ndarray

    @property
    def empty_x(self):
        return len(self.x) == 0 or all([x.empty for x in self.x.values()])

    @property
    def shape(self):
        return properties.shape(self , ['x' , 'y' , 'secid' , 'date'])

    def copy(self):
        return deepcopy(self)

    def filter_dates(self , start_dt : int | None = None , end_dt : int | None = None , inplace = False):
        if start_dt is None and end_dt is None:
            return self
        if not inplace:
            self = self.copy()
        if start_dt is not None:
            self.date = self.date[self.date >= start_dt]
        if end_dt is not None:
            self.date = self.date[self.date <= end_dt]
        for x_key in self.x:
            self.x[x_key] = self.x[x_key].align_date(self.date , inplace = True)
        self.y = self.y.align_date(self.date , inplace = True)
        return self

    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        if secid is None:
            return self
        if not inplace:
            self = self.copy()
        if exclude:
            self.secid = self.secid[~np.isin(self.secid , secid)]
        else:
            self.secid = self.secid[np.isin(self.secid , secid)]
        for x_key in self.x:
            self.x[x_key] = self.x[x_key].align_secid(self.secid , inplace = True)
        self.y = self.y.align_secid(self.secid , inplace = True)
        return self

    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return self.date[(self.date >= start) & (self.date <= end)][::interval]
    
    @classmethod
    def load(cls , data_type_list : list[str] , 
             y_labels : list[str] | None = None , 
             factor_names : list[str] | None = None ,
             fit : bool = True , predict : bool = False , 
             factor_start_dt : int | None = None , factor_end_dt : int | None = None ,
             dtype : str | Any = torch.float , 
             save_upon_loading : bool = True):
        
        assert fit or predict , (fit , predict)
        if not predict: 
            data = cls.load_datas(data_type_list , y_labels , False , dtype , save_upon_loading)
        elif not fit:
            data = cls.load_datas(data_type_list , y_labels , True  , dtype , save_upon_loading)
        else:
            hist_data = cls.load_datas(data_type_list , y_labels , False , dtype , save_upon_loading)
            pred_data = cls.load_datas(data_type_list , y_labels , True  , dtype , save_upon_loading)

            hist_data.y = hist_data.y.merge_others([pred_data.y] , inplace = True)
            hist_data.secid , hist_data.date = hist_data.y.secid , hist_data.y.date
            for x_key in hist_data.x:
                hist_data.x[x_key] = hist_data.x[x_key].merge_others([pred_data.x[x_key]] , inplace = True).align_secid_date(hist_data.secid , hist_data.date , inplace = True)

            data = hist_data

        data.load_factor(factor_names , factor_start_dt , factor_end_dt)
        return data

    @classmethod
    def load_datas(cls , data_type_list : list[str] , 
                   y_labels : list[str] | None = None , 
                   predict : bool = False , dtype : str | Any = torch.float , 
                   save_upon_loading : bool = True , 
                   vb_level = 2):
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
            data = cls.datacache_load(last_date , data_type_list , y_labels , vb_level = vb_level)

        if data is None:
            block_title = f'{len(data_type_list) + 1} DataBlocks' if len(data_type_list) > 3 else f'DataBlock [{",".join(['y' , *data_type_list])}]'
            with Logger.Timer(f'Load {block_title} (predict={predict})' , vb_level = vb_level):
                blocks = {key:DataBlock.load_preprocess(key, predict , dtype = dtype , vb_level = vb_level) for key in ['y' , *data_type_list]}
            with Logger.Timer(f'Align {block_title} (predict={predict})' , vb_level = vb_level):
                blocks = DataBlock.blocks_align(blocks , vb_level = vb_level + 1)
            blocks = DataBlock.blocks_fillna(blocks)
            norms  = DataBlock.load_preprocess_norms(['y' , *data_type_list], predict , dtype = dtype)

            y : DataBlock = blocks['y']
            x : dict[str,DataBlock] = {cls.abbr(key):val for key,val in blocks.items() if key != 'y'}
            norms = {cls.abbr(key):val for key,val in norms.items() if val is not None and key != 'y'}
            secid = y.secid
            date = y.date

            assert all([xx.shape[:2] == y.shape[:2] == (len(secid),len(date)) for xx in x.values()])

            data = {'x' : x , 'y' : y , 'norms' : norms , 'secid' : secid , 'date' : date}
            if not predict and save_upon_loading: 
                cls.datacache_save(data , last_date or y.date[-1] , data_type_list)
            data = cls(**data)

        data.y.align_feature(y_labels , inplace = True)
        return data

    def load_factor(self , factor_names : list[str] | None , start_dt : int | None = None , end_dt : int | None = None , vb_level = 2):
        '''load factor data'''
        if not factor_names:
            return self
        factor_title = f'{len(factor_names)} Factors' if len(factor_names) > 1 else f'Factor [{factor_names[0]}]'
        start_dt = max(start_dt or self.date[0] , self.date[0])
        end_dt = min(end_dt or self.date[-1] , self.date[-1])
        with Logger.Timer(f'Load {factor_title} ({start_dt} - {end_dt})' , vb_level = vb_level):
            from src.data.loader import FactorLoader
            self.x['factor'] = FactorLoader(factor_names).load(start_dt , end_dt , vb_level = Proj.vb.inf).align_secid_date(self.secid , self.date , inplace = True)
        return self

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
                Logger.alert1(f'cache_key.json is corrupted, reset it: {e}')
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
    def datacache_load(cls , date : int | None , data_type_list : list[str] , y_labels : list[str] | None = None , vb_level = 2):
        if date is None:
            return None
        path = cls.datacache_path(date , data_type_list)
        if path is None or not path.exists():
            return None
        try:
            data = cls(**torch_load(path))
            if (np.isin(data_type_list , list(data.x.keys())).all() and
                (y_labels is None or np.isin(y_labels , list(data.y.feature)).all())):
                Logger.success(f'Loading Module Data, Try \'{path}\', success!' , vb_level = vb_level)
            else:
                Logger.alert1(f'Loading Module Data, Try \'{path}\', Incompatible, Load Raw blocks!')
                data = None
        except ModuleNotFoundError:
            '''can be caused by different package version'''
            Logger.alert1(f'Loading Module Data, Try \'{path}\', Incompatible, Load Raw blocks!')
            data = None
        except Exception as e:
            Logger.error(f'Failed to load Module Data: {e}')
            Logger.print_exc(e)
            raise e

        cls.datacache_purge_old(data_type_list)
        return data
    
    @classmethod
    def datacache_save(cls , data : dict , date : int , data_type_list : list[str]):
        if not data_type_list:
            return
        path = cls.datacache_path(date , data_type_list)
        path.parent.mkdir(exist_ok=True)
        torch.save(data , path , pickle_protocol = 5)

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