"""Data Cache for All Data Types (Must be included in DataCache.possible_types)"""
from __future__ import annotations

import shutil , threading , torch , json

from typing import Any

from src.proj import PATH , Logger , BaseType
from src.proj.util.io.torch_load import torch_load

__all__ = ['DataCache']

class DataCache:
    """
    Data Cache for All Data Types (Must be included in DataCache.possible_types)
    Example: Cache a ModuleData

    self.module_data = ModuleData(data_type_list=['day' , '15m'])

    @property
    def enable_cache(self):
        return bool(self.datacache)  # and self.use_data in ['fit' , 'both']

    @cached_property
    def enable_cache_save(self):
        return self.enable_cache and self.config.filter_secid is None and self.config.filter_date is None

    def load_cache(self):
        if not self.enable_cache:
            return
        data, _ = self.datacache.load_data(self.vb_level)
        if data is not None:
            self.blocks, self.norms = data["blocks"], data["norms"]
            self.logger.success(f"Loaded DataBlocks from cache {self.datacache.key} of {Dates(self.date)}")
    def _save_cache(self):
        if not self.enable_cache_save:
            return
        valid_end = min(block.last_valid_date for block in self.blocks.values())
        valid_start = max(block.first_valid_date for block in self.blocks.values())
        old_metadata = self.datacache.load_metadata()
        old_valid_end: int = old_metadata.get("valid_end", 19000101)
        old_valid_start: int = old_metadata.get("valid_start", 99991231)
        if len(CALENDAR.range(old_valid_start, old_valid_end, "td")) < len(
            CALENDAR.range(valid_start, valid_end, "td")
        ):
            metadata = {"valid_end": int(valid_end), "valid_start": int(valid_start)}
            blocks = {
                key: value for key, value in self.blocks.items() if key != "factor"
            }
            self.datacache.save_data(
                {"blocks": blocks, "norms": self.norms},
                vb_level=self.vb_level + 2,
                **metadata,
            )
            self.logger.success(f"Saved DataBlocks to cache {self.datacache.key}")
    """
    metadata_file = PATH.datacache.joinpath('cache_metadata.json')
    possible_types : tuple[str,] = ('module_data' ,)
    locks_guard = threading.Lock()
    locks : dict[str, threading.Lock] = {}

    def __init__(self , type : str , **kwargs):
        self.type = type
        self.kwargs = kwargs
        self.content_name = list(kwargs.keys())[0]
        self.content_value = kwargs[self.content_name]

        self.key = self._get_key()

    def __bool__(self):
        return self.key is not None

    @property
    def path(self):
        """Get the cache file folder path"""
        if not self.key:
            return None
        return PATH.datacache.joinpath(self.key)

    def save_data(self , data : Any , * , vb_level : Any = 3 , **additional_metadata : Any):
        """Save the data to the cache file folder as 'data.pt' and update the metadata file"""
        if not self.key or not self.path:
            # if key is not created, the data is not saved (data_type_list is empty)
            return
        self.path.mkdir(parents = True , exist_ok = True)
        metadata_success = self._update_metadata(self.key , **additional_metadata)
        if not metadata_success:
            Logger.alert1(f'Failed to update metadata: {self.key}' , indent = 1 , vb_level = 2)
            return
        with self._get_lock(self.key):
            torch.save(data , self.path.joinpath('data.pt'))

    def load_data(self , vb_level : Any = 3) -> tuple[Any , dict[str, Any]]:
        """Load the data from the cache file folder as 'data.pt' and metadata from 'metadata.json'"""
        if not self.path or not self.path.joinpath('data.pt').exists():
            return None , {}
        try:
            with self._get_lock(self.key):
                metadata = self.load_metadata()
                data = torch_load(self.path.joinpath('data.pt'))
        except ModuleNotFoundError as e:
            '''can be caused by different package version'''
            Logger.alert2(f'ModuleNotFoundError {e} when loading {self.type.title()} CacheData, possibly you have change the code!' , indent = 1 , vb_level = 2)
            data , metadata = None , {}
            self._remove_cache_file(self.key)
        except Exception as e:
            Logger.error(f'Failed to load {self.type.title()} CacheData: {e}' , indent = 1 , vb_level = 2)
            Logger.print_exc(e)
            raise
        return data , metadata

    def load_metadata(self):
        """Load the cache specific metadata"""
        if not self.path:
            return {}
        return PATH.read_json(self.path.joinpath('metadata.json'))

    @classmethod
    def _get_lock(cls , key : str | None = None) -> threading.Lock:
        if key is None:
            key = 'all'
        with cls.locks_guard:
            if key not in cls.locks:
                cls.locks[key] = threading.Lock()
            return cls.locks[key]

    def _get_key(self) -> str | None:
        """Get the cache key for the cache file folder"""
        assert self.type in self.possible_types , f'{self.type} is not a valid type'
        metadata = self._all_metadata()
        with self._get_lock():
            for key , value in metadata.items():
                if not value['type'] == self.type:
                    continue
                if not all(self.kwargs[key] == value.get(key, None) for key in self.kwargs):
                    continue
                return key
            key = self._create_key(existing_keys = list(metadata.keys()))
            if not key:
                return None
            metadata[key] = {'type' : self.type , **self.kwargs}
            PATH.dump_json(metadata , self.metadata_file , overwrite = True)
            return key

    def _create_key(self , existing_keys : list[str]) -> str | None:
        """Create a new cache key for the cache file folder"""
        prefix = f'{self.type}_'
        if isinstance(self.content_value , str):
            key = self.content_value
        elif isinstance(self.content_value , (list , tuple)):
            if len(self.content_value) < 5:
                key = '+'.join(self.content_value)
            else:
                key = f'{self.content_name}({len(self.content_value)})'
        elif isinstance(self.content_value , dict):
            if len(self.content_value) < 5:
                key = ','.join([f'{k}={v}' for k , v in self.content_value.items()])
            else:
                key = f'{self.content_name}({len(self.content_value)})'
        else:
            key = str(self.content_value)
        if not key:
            return None
        else:
            key = f'{prefix}{key}'
        i = 0
        while key in existing_keys:
            i += 1
            key = f'{key}.{i}'
        return key

    @classmethod
    def _get_meta_file(cls , key : str):
        return PATH.datacache.joinpath(key , 'metadata.json')

    @classmethod
    def _all_metadata(cls) -> dict[str, dict[str, Any]]:
        """Load the universal metadata from the top level metadata file"""
        with cls._get_lock():
            if not cls.metadata_file.exists():
                return {}
            try:
                return PATH.read_json(cls.metadata_file)
            except Exception as e:
                Logger.alert2(f'Failed to load metadata: {e}')
                return {}

    @classmethod
    def _update_metadata(cls , key : str , **kwargs):
        """Update the universal metadata for a specific key"""
        metadata = cls._all_metadata()
        with cls._get_lock():
            metadata[key].update(kwargs)
            if not cls._try_save_metadata(metadata , cls.metadata_file):
                return False
            
        with cls._get_lock(key):
            return cls._try_save_metadata(metadata[key] , cls._get_meta_file(key))

    @classmethod
    def _try_save_metadata(cls , metadata : dict[str, Any] , path : BaseType.strPath) -> bool:
        """Remove the metadata for a specific key"""
        try:
            _ = json.dumps(metadata)
            PATH.dump_json(metadata , path , overwrite = True)
            return True
        except Exception as e:
            Logger.error(f'Failed to update metadata: {e}')
            Logger.print_exc(e)
            return False


    @classmethod
    def _remove_cache_file(cls , key : str | None = None):
        """Purge the data for a specific key"""
        if key is None:
            return
        if not cls._get_meta_file(key).exists():
            return
        PATH.datacache.joinpath(key , 'data.pt').unlink()
        PATH.datacache.joinpath(key , 'metadata.json').unlink()
        metadata = cls._all_metadata()
        with cls._get_lock():
            metadata.pop(key)
            cls._try_save_metadata(metadata , cls.metadata_file)

    @classmethod
    def purge_all(cls , confirm : bool = False):
        """eliminate all cache files and metadata"""
        if not confirm:
            raise ValueError('Purge all is not implemented')
        for path in PATH.datacache.iterdir():
            if path.is_file():
                path.unlink()
            if path.is_dir():
                shutil.rmtree(path)

    