"""
ModelFile / ModelDict: The all-in-one model file/dict class for model training and inference.
"""

from __future__ import annotations
import torch

from pathlib import Path
from typing import Any
from src.proj.core import strPath
from src.proj.util import torch_load

__all__ = ['ModelDict' , 'ModelFile']

class ModelDict:
    """model dictionary for nn/boost models"""
    __slots__ = ['state_dict' , 'boost_head' , 'boost_dict']
    def __init__(self ,
                 state_dict  : dict[str,torch.Tensor] | None = None , 
                 boost_head : dict[str,Any] | None = None ,
                 boost_dict : dict[str,Any] | None = None) -> None:
        self.state_dict = state_dict
        self.boost_head = boost_head
        self.boost_dict = boost_dict

    def __repr__(self): return f'{self.__class__.__name__}(state_dict={self.state_dict},boost_head={self.boost_head},boost_dict={self.boost_dict})'

    def reset(self) -> None:
        """reset model dictionary"""
        self.state_dict = None
        self.boost_head = None
        self.boost_dict = None

    def save(self , path : strPath) -> None:
        """uniformly save model dictionary"""
        if isinstance(path , str): 
            path = Path(path)
        assert not path.exists() or path.is_dir() , f'{path} already exists or is not a directory'
        path.mkdir(parents=True,exist_ok=True)
        for key in self.__slots__:
            if (value := getattr(self , key)) is not None:
                torch.save(value , path.joinpath(f'{key}.pt'))

    @property
    def is_valid(self) -> bool:
        """check if model dictionary is valid"""
        if self.state_dict is not None:
            assert self.boost_dict is None 
        else:
            assert self.boost_head is None
        return True

class ModelFile:
    """model file for nn/boost models"""
    def __init__(self , model_path : Path | None) -> None:
        if model_path is None:
            model_path = Path('')
        self.model_path = model_path
    def __getitem__(self , key): return self.load(key)
    def __repr__(self): return f'{self.__class__.__name__}(path={self.model_path})'
    def load(self , key : str) -> Any:
        """load model dictionary"""
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.model_path.joinpath(f'{key}.pt')
        return torch_load(path , map_location='cpu') if path.exists() else None
    def exists(self) -> bool: 
        """check if model file exists"""
        return any([self.model_path.joinpath(f'{key}.pt').exists() for key in ModelDict.__slots__])
    def model_dict(self) -> ModelDict:
        """load model dictionary"""
        return ModelDict(**{key:self.load(key) for key in ModelDict.__slots__})