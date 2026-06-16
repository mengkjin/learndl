"""On-disk numpy/torch array storage via binary payload + JSON metadata."""
from __future__ import annotations
import json
import os
import shutil
import numpy as np

from pathlib import Path
from typing import Literal , TYPE_CHECKING , Any
from dataclasses import dataclass

from src.proj.core import strPath
from src.proj.log import Logger

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ['ArrayMemoryMap']

_STAGING_SUFFIX = '.staging'
_BACKUP_SUFFIX = '.old'

@dataclass
class ArrayMeta:
    """Serializable descriptor for a memory-mapped array (dtype, shape, tensor vs ndarray)."""

    array_type: Literal['Tensor' , 'ndarray']
    dtype: str
    shape: tuple

    @classmethod
    def from_json(cls, json_path: strPath) -> ArrayMeta:
        """Load metadata written by ``to_json``."""
        with open(json_path , encoding='utf-8') as f:
            meta = json.load(f)
        return cls(array_type=meta['array_type'], dtype=meta['dtype'], shape=meta['shape'])

    def to_json(self, json_path: strPath):
        """Persist ``array_type``, ``dtype``, and ``shape`` to JSON."""
        with open(json_path, 'w' , encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2)

class ArrayMemoryMap:
    """Read/write a single array as ``values/data.bin`` + ``values/meta.json`` + ``index.pt`` under a directory."""
    def __init__(self, path: strPath):
        """Open or prepare a map directory (created on ``save``)."""
        self.path = Path(path)
        assert not self.path.is_file() , f'{self.path} is a file'
        self._full_mmap = None

    def prepare(self):
        """prepare the mmap directory"""
        self.path.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def data_path(self) -> Path:
        return self.path.joinpath('values' , 'data.bin')
    @property
    def meta_path(self) -> Path:
        return self.path.joinpath('values' , 'meta.json')

    @classmethod
    def save(cls, values : np.ndarray | Tensor , path: strPath , **kwargs):
        """Write array bytes and metadata; tensors are copied to CPU numpy first.

        Builds the full bundle in a sibling ``*.staging`` directory, then publishes
        it atomically via directory rename so readers never observe partial writes.

        Returns:
            ``ArrayMemoryMap`` bound to ``path``.
        """
        import torch
        from src.proj.db.io.torch import torch_save

        live_path = Path(path)
        staging_path = live_path.with_name(f'{live_path.name}{_STAGING_SUFFIX}')
        if staging_path.exists():
            shutil.rmtree(staging_path)

        mmap = cls(staging_path).prepare()

        array_type='Tensor' if torch.is_tensor(values) else 'ndarray'
        if torch.is_tensor(values):
            values = values.cpu().numpy()
        if not values.flags.c_contiguous:
            values = np.ascontiguousarray(values)
        dtype=str(values.dtype)
        shape=tuple(values.shape)

        with open(mmap.data_path, 'wb') as f:
            f.write(values.tobytes())
            f.flush()
            os.fsync(f.fileno())

        metas = ArrayMeta(array_type,dtype,shape)
        metas.to_json(mmap.meta_path)

        if kwargs:
            for key, value in kwargs.items():
                torch_save(value, mmap.path.joinpath(f'{key}.pt'))

        cls._publish(live_path, staging_path)
        return cls(live_path)

    @staticmethod
    def _publish(live_path: Path, staging_path: Path) -> None:
        """Replace ``live_path`` with the completed ``staging_path`` bundle."""
        backup_path = live_path.with_name(f'{live_path.name}{_BACKUP_SUFFIX}')
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)

        live_path.parent.mkdir(parents=True, exist_ok=True)
        if live_path.exists():
            live_path.rename(backup_path)

        try:
            staging_path.rename(live_path)
        except Exception:
            if not live_path.exists() and backup_path.exists():
                backup_path.rename(live_path)
            raise
        finally:
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)

    def view(self , writable: bool = False):
        """Memory-map the binary file and return an ndarray or tensor view backed by the mmap.

        Args:
            writable: If True, open the file in read/write mode.
        """
        mode = 'r+' if writable else 'r'
        file_size = os.path.getsize(self.data_path)
        self._full_mmap = np.memmap(self.data_path, dtype='uint8', mode=mode, shape=(file_size,))
        metas = ArrayMeta.from_json(self.meta_path)
        result = np.ndarray(metas.shape, dtype=metas.dtype, buffer=self._full_mmap[:], order='C')
        if metas.array_type == 'Tensor':
            import torch
            result = torch.from_numpy(result)
        return result

    @classmethod
    def load(cls, path: strPath , values : bool = True , **kwargs) -> dict[str, Any]:
        """Load array into a new in-memory buffer (full read, not mmap)."""
        mmap = cls(path)
        results = {}
        if not mmap.path.exists():
            return results
        
        if values:
            metas = ArrayMeta.from_json(mmap.meta_path)
            
            dtype = np.dtype(metas.dtype)
            with open(mmap.data_path, 'rb') as f:
                f.seek(0)

                try: 
                    val = np.fromfile(f, dtype=dtype , count=np.prod(metas.shape))
                    val = val.reshape(metas.shape)
                except Exception as e:
                    Logger.error(f'Memeory Map {path} is messed up: ' , e)
                    raise
                
            if metas.array_type == 'Tensor':
                import torch
                val = torch.from_numpy(val)

            results['values'] = val

        for key, value in kwargs.items():
            if not value:
                continue
            from src.proj.db.io.torch import torch_load
            results[key] = torch_load(mmap.path.joinpath(f'{key}.pt'))
    
        return results

    @classmethod
    def load_metadata(cls, path: strPath) -> ArrayMeta:
        """Load array from file."""
        path = Path(path).joinpath('values' , 'meta.json')
        assert path.exists() , f'{path} does not exist'
        return ArrayMeta.from_json(path)

    @classmethod
    def load_component(cls, path: strPath , key : str) -> dict[str, Any] | None:
        """Load index from file."""
        from src.proj.db.io.torch import torch_load
        path = Path(path).joinpath(f'{key}.pt')
        if not path.exists():
            return None
        return torch_load(path)

    def close(self):
        """Drop the mmap view if ``view`` was used."""
        if self._full_mmap is not None:
            del self._full_mmap
            self._full_mmap = None

    def __enter__(self):
        """Context manager entry; returns ``self`` for use with ``view``."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Flush mmap on exit when applicable, then ``close``."""
        if self._full_mmap is not None:
            self._full_mmap.flush()
            self.close()
        
    def __del__(self):
        self.close()

    def __repr__(self):
        return f"MemoryMap(path={self.data_path}, meta={self.meta_path})"

    def __str__(self):
        return self.__repr__()