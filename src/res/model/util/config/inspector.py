"""
Model configs inspector for the project, use it to check if the config files are correct
"""
from __future__ import annotations
from src.proj import PATH , Base
from typing import Any

__all__ = ['ModelConfigsInspector']

class ModelConfigsInspector(Base.BoundLogger):
    def __init__(self , * , indent : int = 0 , vb_level : Any = 1 , **kwargs):
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
        self.model_root = PATH.model
        self.config_root = PATH.conf

    def iter_configs(self):
        for self.current_path in self.model_root.rglob('*.yaml'):
            self.current_path_str = str(PATH.relative(self.current_path))
            yield self.current_path
        for self.current_path in self.config_root.rglob('*.yaml'):
            self.current_path_str = str(PATH.relative(self.current_path))
            yield self.current_path

    def inspect_key_values(self , warning_list : list[str] | dict[str , bool] | None = None):
        """
        warning_list is a list of strings or a dictionary of strings and whether to check full match
        """
        if warning_list is None:
            warning_list = {
                'ResetOptimizer' : False , 
                'lamb' : True , 
                'eps' : True , 
                'EarlyExitRetrain' : False ,
                'NanLossRetrain' : False ,
                'BatchDisplay' : False ,
                'ValidationConverge' : False ,
                'TrainConverge' : False ,
                'FitConverge' : False ,
                'CudaEmptyCache' : False ,
            }
        if not isinstance(warning_list , dict):
            warning_list = {warn: False for warn in warning_list}
        full_match_list = [warn for warn in warning_list if warning_list[warn]]
        partial_match_list = [warn for warn in warning_list if not warning_list[warn]]
        for path in self.iter_configs():
            config = PATH.read_yaml(path)
            self.inspect_object(config , full_match_list , full_match = True)
            self.inspect_object(config , partial_match_list , full_match = False)

    def inspect_object(self , obj , warning_list : list[str] , full_match : bool = False):
        if isinstance(obj , str):
            for warn in warning_list:
                if obj == warn:
                    self.logger.alert2(f'[{warn}] found in {self.current_path_str} (full_match)')
                elif not full_match and warn in obj:
                    self.logger.alert1(f'[{warn}] found in {self.current_path_str} within [{obj}] (partial_match)')
            return
        if isinstance(obj , dict):
            for key , value in obj.items():
                self.inspect_object(key , warning_list , full_match = full_match)
                self.inspect_object(value , warning_list , full_match = full_match)
            return
        if isinstance(obj , list):
            for item in obj:
                self.inspect_object(item , warning_list , full_match = full_match)
            return