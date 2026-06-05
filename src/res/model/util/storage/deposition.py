from __future__ import annotations

import shutil

from typing import TypeVar, Any

from src.res.model.util.core import ModelDict, ModelPath

T = TypeVar('T')

class Deposition:
    """model saver"""
    def __init__(self , base_path_like : ModelPath | Any):
        if hasattr(base_path_like , 'base_path'):
            base_path = getattr(base_path_like , 'base_path')
        else:
            base_path = ModelPath(base_path_like)
        assert isinstance(base_path , ModelPath) , f'base_path should be a ModelPath object, but got {type(base_path)} (out from {base_path_like})'
        self.base_path = base_path

    def shrink_key(self , key : str) -> str:
        return key.replace('.' , '-').replace(' ' , '')

    def stack_model(self , model_dict : ModelDict , attempt_key : str , model_num , model_date , submodel = 'best'):
        assert attempt_key , 'attempt_key is required when stacking model'
        attempt_key = self.shrink_key(attempt_key)
        model_path = self.model_path(model_num , model_date , submodel)
        if attempt_key:
            model_path = model_path.joinpath(attempt_key.replace('.' , '-').replace(' ' , ''))
        model_dict.save(model_path)

    def dump_stacked_model(self , attempt_key : str , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        attempt_key = self.shrink_key(attempt_key)
        attempt_path = model_path.joinpath(attempt_key)
        assert attempt_path.exists() , f'attempt_path {attempt_path} does not exist'
        for path in attempt_path.iterdir():
            new_path = model_path.joinpath(path.name)
            new_path.unlink(missing_ok=True)
            path.rename(new_path)
        self.clear_stacked_models(model_num , model_date , submodel)

    def clear_stacked_models(self , model_num , model_date , submodel = 'best'):
        model_path = self.model_path(model_num , model_date , submodel)
        for path in model_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)

    def load_model(self , model_num , model_date , submodel = 'best'):
        return self.base_path.model_file(model_num , model_date , submodel)
    
    def exists(self , model_num , model_date , submodel = 'best'):
        return self.base_path.exists(model_num , model_date , submodel)
    
    def model_path(self , model_num , model_date , submodel = 'best'):
        """get model path of deposition giving model date / num / submodel"""
        return self.base_path.full_path(model_num , model_date , submodel)