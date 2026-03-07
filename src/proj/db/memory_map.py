import json
import os
import numpy as np
import torch

from pathlib import Path
from typing import Literal
from dataclasses import dataclass

@dataclass
class ArrayMeta:
    array_type: Literal['Tensor' , 'ndarray']
    dtype: str
    shape: tuple

    @classmethod
    def from_json(cls, json_path: str | Path) -> 'ArrayMeta':
        with open(json_path, 'r') as f:
            meta = json.load(f)
        return cls(array_type=meta['array_type'], dtype=meta['dtype'], shape=meta['shape'])

    def to_json(self, json_path: str | Path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

class ArrayMemoryMap:
    def __init__(self, path: str | Path):
        path = Path(path)
        assert not path.exists() or path.is_dir() , path
        self.path = path
        self.data_path = path.joinpath('data.bin')
        self.meta_path = path.joinpath('meta.json')
        self._full_mmap = None

    @classmethod
    def save(cls, array: np.ndarray | torch.Tensor , path: str | Path):
        """save arrays to memory map file (.bin) and metadata (.json)"""
        mmap = cls(path)
        mmap.path.mkdir(parents=True, exist_ok=True)
        array_type='Tensor' if torch.is_tensor(array) else 'ndarray'
        if torch.is_tensor(array):
            array = array.cpu().numpy()
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        dtype=str(array.dtype)
        shape=tuple(array.shape)

        with open(mmap.data_path, 'wb') as f:
            f.write(array.tobytes())

        metas = ArrayMeta(array_type,dtype,shape)
        metas.to_json(mmap.meta_path)

        return mmap

    def view(self , writable: bool = False) -> torch.Tensor | np.ndarray:
        """
        load array from memory map file (.bin) and metadata (.json)
        if copy is True, return a copy of the array
        if copy is False, return a view of the array
        """
        mode = 'r+' if writable else 'r'
        file_size = os.path.getsize(self.data_path)
        self._full_mmap = np.memmap(self.data_path, dtype='uint8', mode=mode, shape=(file_size,))
        metas = ArrayMeta.from_json(self.meta_path)
        result = np.ndarray(metas.shape, dtype=metas.dtype, buffer=self._full_mmap[:], order='C')
        if metas.array_type == 'Tensor':
            result = torch.from_numpy(result)

        return result

    @classmethod
    def load(cls, path: str | Path) -> torch.Tensor | np.ndarray:
        """
        load array from memory map file (.bin) and metadata (.json)
        """
        mmap = cls(path)
        metas = ArrayMeta.from_json(mmap.meta_path)
        
        dtype = np.dtype(metas.dtype)
        with open(mmap.data_path, 'rb') as f:
            f.seek(0)
            result = np.fromfile(f, dtype=dtype , count=np.prod(metas.shape)).reshape(metas.shape)
        if metas.array_type == 'Tensor':
            result = torch.from_numpy(result)
        return result

    @classmethod
    def load_tensor(cls, path: str | Path) -> torch.Tensor:
        """
        load tensor from memory map file (.bin) and metadata (.json)
        """
        data = cls.load(path)
        assert isinstance(data, torch.Tensor) , data
        return data

    @classmethod
    def load_ndarray(cls, path: str | Path) -> np.ndarray:
        """
        load ndarray from memory map file (.bin) and metadata (.json)
        """
        data = cls.load(path)
        assert isinstance(data, np.ndarray) , data
        return data

    def close(self):
        if self._full_mmap is not None:
            del self._full_mmap
            self._full_mmap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._full_mmap is not None:
            self._full_mmap.flush()
            self.close()
        
    def __del__(self):
        self.close()

    def __repr__(self):
        return f"MemoryMap(path={self.data_path}, meta={self.meta_path})"

    def __str__(self):
        return self.__repr__()