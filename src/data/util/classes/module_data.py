import torch
import numpy as np

from copy import deepcopy
from functools import partial
from typing import Any , Literal

from src.proj import Logger , Proj , CALENDAR , Dates
from src.proj.util import properties

from .data_block import DataBlock , DataBlockNorm , data_type_abbr
from .datacache import DataCache
from .special_dataset import SpecialDataSet

__all__ = ['ModuleData']

class ModuleData:
    '''load datas / norms / index'''
    def __init__(
        self , data_type_list : list[str] , y_labels : list[str] | None = None , use_data : Literal['fit' , 'predict' , 'both'] = 'fit' , * ,
        factor_names : list[str] | None = None , factor_start_dt : int | None = None , factor_end_dt : int | None = None , 
        filter_secid : str | None = None , filter_date : str | None = None ,
        indent : int = 1 , vb_level : Any = 2 , dtype = torch.float ,**kwargs
    ):
        self.data_type_list = sorted([self.abbr(data_type) for data_type in data_type_list])
        self.y_labels = y_labels
        self.use_data = use_data
        self.factor_names = factor_names
        self.factor_start_dt = factor_start_dt
        self.factor_end_dt = factor_end_dt

        self.datacache = DataCache(type = 'module_data' , data_type_list = self.data_type_list)
        
        self.indent = indent
        self.vb_level = Proj.vb(vb_level)

        if dtype is None: 
            dtype = torch.float
        if isinstance(dtype , str): 
            dtype = getattr(torch , dtype)
        self.dtype = dtype

        self.secid_filter = SecidFilter(filter_secid if self.use_data == 'fit' else None)
        self.date_filter = DateFilter(filter_date if self.use_data == 'fit' else None)
        self.enable_cache_save = self.enable_cache and filter_secid is None and filter_date is None

        self.kwargs = kwargs

        self.blocks : dict[str,DataBlock] = {}
        self.norms : dict[str,DataBlockNorm] = {}

    @property
    def PrePros(self):
        if not hasattr(self , '_prepros'):
            from src.data.preprocess import PrePros
            self._prepros = PrePros
        return self._prepros

    @property
    def load_keys(self):
        return ['y' , *self.data_type_list]

    @property
    def x(self) -> dict[str,DataBlock]:
        return {key:value for key,value in self.blocks.items() if key != 'y'}

    @property
    def y(self) -> DataBlock:
        return self.blocks['y'].align_feature(self.y_labels)

    @property
    def empty_x(self):
        return len(self.x) == 0 or all([x.empty for x in self.x.values()])

    @property
    def shape(self):
        return properties.shape(self , ['x' , 'y' , 'secid' , 'date'])

    @property
    def secid(self):
        return self.blocks['y'].secid

    @property
    def date(self):
        return self.blocks['y'].date

    @property
    def enable_cache(self):
        return self.use_data in ['fit' , 'both'] and self.datacache

    @property
    def loaded(self):
        if not hasattr(self , '_loaded'):
            self._loaded = False
        return self._loaded

    @property
    def block_title(self):
        return f'{len(self.load_keys)} DataBlocks' if len(self.load_keys) > 4 else f'DataBlock [{",".join(self.load_keys)}]'

    def __bool__(self):
        return not self.empty_x

    def copy(self):
        return deepcopy(self)

    def date_within(self , start : int , end : int , interval = 1) -> np.ndarray:
        return CALENDAR.slice(self.date , start , end)[::interval]

    def target_start_end(self):
        start = CALENDAR.td(CALENDAR.updated() , -366).td if self.use_data == 'predict' else 20070101
        end = DataBlock.last_data_date('y' , 'fit') if self.use_data == 'fit' else CALENDAR.updated()
        end = end or CALENDAR.updated()
        return start , end

    def load(self):
        '''
        load all relevant data of this module data, should be called before any other operations
        blocks: ['y' , *data_type_list] DataBlocks
        norms: ['y' , *data_type_list] DataBlockNorms (if exists)
        factor: factor_names DataBlock
        '''
        self.load_cache()
        self.extend_blocks()
        self.align_blocks()
        self.load_norms()
        self.save_cache()
        self.load_factor()
        DataBlock.blocks_ffill(self.blocks , exclude = ['y'])
        self._loaded = True
        return self

    def load_cache(self):
        if not self.enable_cache:
            return
        data , _ = self.datacache.load_data(self.vb_level)
        if data is not None:
            self.blocks , self.norms = data['blocks'] , data['norms']
            Logger.success(f'Loaded DataBlocks from cache {self.datacache.key} of {Dates(self.date)}' , vb_level = self.vb_level + 2)

    def extend_blocks(self):
        start , end = self.date_filter.filter_start_end(*self.target_start_end())
        date = CALENDAR.range(start , end)
        secid = None
        with Logger.Timer(f'Load {self.block_title} at {start}~{end}' , indent = self.indent , vb_level = self.vb_level + 1):
            for i , key in enumerate(self.load_keys):
                current_block = self.blocks.get(key , DataBlock())
                current_dates = current_block.valid_dates
                ext_dates = CALENDAR.diffs(date , current_dates)
                ext_block = self.load_one(key, dates = ext_dates , secid = secid)
                self.blocks[key] = DataBlock.merge([current_dates , ext_block] , inplace = True)
                if i == 0:
                    assert key == 'y' , f'y must be the first key'
                    secid = self.secid_filter(self.blocks[key].secid) # use the y_secid to align all other blocks in next step
                    date = self.date_filter(self.blocks[key].date) # use the y_date to align all other blocks in next step
        return self

    def align_blocks(self):
        if len(self.blocks) <= 1:
            return self
        with Logger.Timer(f'Align {self.block_title}' , indent = self.indent , vb_level = self.vb_level + 1):
            DataBlock.blocks_align(self.blocks , vb_level = self.vb_level + 2)
        index_lens = [block.shape[:2] for block in self.blocks.values()]
        if index_lens:
            assert all([lens == index_lens[0] for lens in index_lens]) , f'{[(name,block.shape) for name,block in self.blocks.items()]}'
        return self

    def load_one(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        if len(dates) == 0:
            return DataBlock()
        if key in self.PrePros.keys():
            return self.load_preprocess_block(key, dates = dates, secid = secid, vb_level = self.vb_level + 2 , **kwargs)
        elif key in SpecialDataSet.candidates:
            return self.load_special_block(key, dates = dates, secid = secid, vb_level = self.vb_level + 2 , **kwargs)
        else:
            raise ValueError(f'key [{key}] is not supported')

    def load_preprocess_block(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        type = 'predict' if self.use_data == 'predict' else 'fit'
        block = self.PrePros.get_processor(key , type = type).load(dates = dates , secid = secid, indent = self.indent + 1 , vb_level = self.vb_level + 2)
        return block

    def load_special_block(self , key : str , * , dates : np.ndarray , secid : np.ndarray | None = None , **kwargs):
        block = SpecialDataSet.load(key, dates = dates , secid = secid, dtype = self.dtype , vb_level = self.vb_level + 2)
        return block

    def load_norms(self):
        if self.norms:
            return
        self.norms.update(DataBlock.load_preprocess_norms(self.data_type_list , dtype = self.dtype))

    def load_factor(self):
        '''load factor data'''
        if not self.factor_names:
            return self
        factor_title = f'{len(self.factor_names)} Factors' if len(self.factor_names) > 1 else f'Factor [{self.factor_names[0]}]'
        start = max(self.factor_start_dt or self.date[0] , self.date[0])
        end = min(self.factor_end_dt or self.date[-1] , self.date[-1])
        with Logger.Timer(f'Load {factor_title} ({start} - {end})' , indent = self.indent , vb_level = self.vb_level + 2):
            from src.data.loader import FactorLoader
            self.blocks['factor'] = FactorLoader(self.factor_names).load(start , end , vb_level = 'never').align_secid_date(self.secid , self.date , inplace = True)
        return self

    def save_cache(self):
        if not self.enable_cache_save:
            return
        valid_end   = min(block.last_valid_date for block in self.blocks.values())
        valid_start = max(block.first_valid_date for block in self.blocks.values())
        old_metadata = self.datacache.load_metadata()
        old_valid_end   : int = old_metadata.get('valid_end'   , 19000101)
        old_valid_start : int = old_metadata.get('valid_start' , 99991231)
        if len(CALENDAR.range(old_valid_start , old_valid_end , 'td')) < len(CALENDAR.range(valid_start , valid_end , 'td')):
            metadata = {'valid_end' : int(valid_end) , 'valid_start' : int(valid_start)}
            blocks = {key:value for key,value in self.blocks.items() if key != 'factor'}
            self.datacache.save_data({'blocks' : blocks , 'norms' : self.norms} , vb_level = self.vb_level + 2 , **metadata)
            Logger.success(f'Saved DataBlocks to cache {self.datacache.key}' , vb_level = self.vb_level + 2)

    @staticmethod
    def abbr(data_type : str): 
        return data_type_abbr(data_type)

    def filter_dates(self , start : int | None = None , end : int | None = None , inplace = False):
        if start is None and end is None:
            return self
        if not inplace:
            self = self.copy()
        date = CALENDAR.slice(self.date , start , end)
        for block in self.blocks.values():
            block = block.align_date(date , inplace = True)
        return self

    def filter_secid(self , secid : np.ndarray | Any | None = None , exclude = False , inplace = False):
        if secid is None:
            return self
        if not inplace:
            self = self.copy()
        mask = np.isin(self.secid , secid)
        secid = self.secid[~mask] if exclude else self.secid[mask]
        for block in self.blocks.values():
            block = block.align_secid(secid , inplace = True)
        return self

class SecidFilter:
    def __init__(self , value : str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f'filtering secid for ModuleData: {value}')
            if value.startswith('random.'):
                self.filter = partial(self.random , num = int(value.split('.')[1]))
            elif value.startswith('first.'):
                self.filter = partial(self.first , num = int(value.split('.')[1]))
            elif value in ['csi300' , 'csi500' , 'csi1000']:
                self.filter = partial(self.benchmark , bm = value)
            else:
                raise ValueError(f'input.filter.secid {value} is not valid , should be random.200 , first.200 , csi300 , csi500 , csi1000')
        
    def __call__(self , secid : np.ndarray) -> np.ndarray:
        return self.filter(secid)

    def filter_blocks(self , blocks : dict[str,DataBlock]) -> dict[str,DataBlock]:
        if not blocks:
            return blocks
        secid = self.filter(blocks['y'].secid)
        for key,block in blocks.items():
            blocks[key] = block.align_secid(secid , inplace = True)
        return blocks

    @staticmethod
    def none(secid : np.ndarray) -> np.ndarray:
        return secid

    @staticmethod
    def random(secid : np.ndarray , num : int) -> np.ndarray:
        return np.random.choice(secid , num , replace = False)

    @staticmethod
    def first(secid : np.ndarray , num : int) -> np.ndarray:
        return secid[:num]

    @classmethod
    def Benchmark(cls):
        if not hasattr(cls , '_benchmark'):
            from src.res.factor.util.classes.benchmark import Benchmark
            cls._benchmark = Benchmark
        return cls._benchmark

    @classmethod
    def benchmark(cls , secid : np.ndarray , bm : str , date : int = 20200104) -> np.ndarray:
        return cls.Benchmark()(bm).get(date,True).secid

class DateFilter:
    def __init__(self , value : str | None):
        if value is None:
            self.filter = self.none
        else:
            Logger.alert1(f'filtering date for ModuleData: {value}')

            value = value.strip().replace('-', '~').replace(' ', '~')
            dates = value.split('~')
            assert len(dates) == 2 , f'input.filter.date {value} is not valid , should be yyyyMMdd~yyyyMMdd'
            self.filter = partial(self.slice , start = int(dates[0]) if dates[0] else None , end = int(dates[1]) if dates[1] else None)

        
    def __call__(self , date : np.ndarray) -> np.ndarray:
        return self.filter(date)

    def filter_blocks(self , blocks : dict[str,DataBlock]) -> dict[str,DataBlock]:
        if not blocks:
            return blocks
        date = self.filter(blocks['y'].date)
        for key,block in blocks.items():
            blocks[key] = block.align_date(date , inplace = True)
        return blocks

    def filter_start_end(self , start : int , end : int) -> tuple[int , int]:
        dates = self(CALENDAR.range(start , end))
        if len(dates) == 0:
            return 99991231 , 20070101
        return dates[0] , dates[-1]

    @staticmethod
    def none(date : np.ndarray) -> np.ndarray:
        return date

    @staticmethod
    def slice(date : np.ndarray , start : int | None = None , end : int | None = None) -> np.ndarray:
        return CALENDAR.slice(date , start , end)