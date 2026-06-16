"""
Model configs modifier for the project, use it to modify the config files
"""
from __future__ import annotations

from src.proj import PATH , Base
from typing import TypeAlias
from collections.abc import Callable

__all__ = ['ModelConfigModifier' , 'ModelConfigsBatchModifier']

DictLoader : TypeAlias = Callable[[], dict] | dict

class ModelConfigModifier(Base.BoundLogger):
    @classmethod
    def rename_ResetOptimizer(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        old_name = 'callbacks.ResetOptimizer'
        new_name = 'callbacks.LearnRateReset'
        default_value = {
            'num_reset': 2,
            'recover_level': 1.0,
            'speedup2x': True,
            'trigger': 40
        }
        if old_name in config:
            value = config.pop(old_name)
            cls.logger.success(f'{key} >> {old_name} has been removed')
        else:
            value = default_value
        if new_name not in config:
            config[new_name] = value
            cls.logger.success(f'{key} >> {new_name} has been added')

        for name in list(config.keys()):
            if 'ResetOptimizer' in name:
                new_name = name.replace('ResetOptimizer' , 'LearnRateReset')
                config[new_name] = config.pop(name)
                cls.logger.success(f'{key} >> {name} has been renamed to {new_name}')
        return config

    @classmethod
    def add_LearnRateReset(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        if 'LearnRateReset' not in config['train.callbacks']:
            config['train.callbacks'].append('LearnRateReset')
            cls.logger.success(f'{key} >> train.callbacks lacks LearnRateReset , added')
        return config

    @classmethod
    def rename_lamb(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith(('.model' , '.schedule')):
            return config
        if not isinstance(config , dict):
            config = config()
        if 'train.criterion.loss' not in config:
            return config
        for loss_name , loss_kwargs in config['train.criterion.loss'].items():
            if 'lamb' in loss_kwargs:
                assert 'alpha' not in loss_kwargs, f'{loss_name} already has alpha key : {loss_kwargs}'
                config['train.criterion.loss'][loss_name]['alpha'] = loss_kwargs.pop('lamb')
                cls.logger.success(f'{key} >> train.criterion.loss.{loss_name} has lamb key , renamed to alpha')
        return config

    @classmethod
    def remove_eps_callback_param(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith(('.model' , '.schedule')):
            return config
        if not isinstance(config , dict):
            config = config()
        for key in list(config.keys()):
            if key.startswith('callbacks.'):
                param = config[key]
                if 'eps' in param:
                    param.pop('eps')
                    cls.logger.success(f'{key} >> {param} has eps key , removed')
        return config

    @classmethod
    def replace_EarlyExitRetrain_with_BadAttemptRetrain(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        old_name = 'callbacks.EarlyExitRetrain'
        new_name = 'callbacks.BadAttemptRetrain'
        default_value = {
            'early_exit': 10,
            'min_ic': 0.05,
            'max_attempt': 4,
            'max_nan_redo': 4,
            'lr_multiplier': [1 , 0.1 , 10 , 0.01 , 100 , 1],
        }
        if old_name in config:
            value = config.pop(old_name)
            cls.logger.success(f'{key} >> {old_name} has been removed')
        else:
            value = default_value
        if new_name not in config:
            config[new_name] = value
            cls.logger.success(f'{key} >> {new_name} has been added')

        if 'BadAttemptRetrain' not in config['train.callbacks']:
            config['train.callbacks'].append('BadAttemptRetrain')
            cls.logger.success(f'{key} >> train.callbacks lacks BadAttemptRetrain , added')
        if 'EarlyExitRetrain' in config['train.callbacks']:
            config['train.callbacks'].remove('EarlyExitRetrain')
            cls.logger.success(f'{key} >> train.callbacks has EarlyExitRetrain , removed')
        return config

    @classmethod
    def change_BadAttemptRetrain_params(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        new_name = 'callbacks.BadAttemptRetrain'
        default_value = {
            'early_exit': 10,
            'min_ic': 0.05,
            'max_attempt': 4,
            'max_nan_redo': 4,
            'lr_multiplier': [1 , 0.1 , 10 , 0.01 , 100 , 1],
        }
        if new_name in config and not all(key in config[new_name] for key in default_value):
            for key , value in default_value.items():
                if key not in config[new_name]:
                    config[new_name][key] = value
            cls.logger.success(f'{key} >> {new_name} has been updated')
        return config

    @classmethod
    def change_EarlyStoppage_params(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        new_name = 'callbacks.EarlyStoppage'
        default_value = {
            'peak_patience': 20,
            'converge_patience': 5,
            'converge_dataset': 'valid',
        }
        if new_name in config and not all(key in config[new_name] for key in default_value):
            for key , value in default_value.items():
                if key not in config[new_name]:
                    config[new_name][key] = value
            cls.logger.success(f'{key} >> {new_name} has been updated')
        return config
    
    @classmethod
    def remove_NanLossRetrain(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        old_name = 'callbacks.NanLossRetrain'
        if old_name in config:
            config.pop(old_name)
            cls.logger.success(f'{key}.{old_name} has been removed')
         
        if 'NanLossRetrain' in config['train.callbacks']:
            config['train.callbacks'].remove('NanLossRetrain')
            cls.logger.success(f'{key} >> train.callbacks has NanLossRetrain , removed')
        return config

    @classmethod
    def remove_old_callbacks(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        old_names = ['BatchDisplay' , 'ValidationConverge' , 'TrainConverge' , 'FitConverge']

        for old_name in old_names:
            if f'callbacks.{old_name}' in config:
                config.pop(f'callbacks.{old_name}')
                cls.logger.success(f'{key} >> callbacks.{old_name} has been removed')
            if old_name in config['train.callbacks']:
                config['train.callbacks'].remove(old_name)
                cls.logger.success(f'{key} >> train.callbacks has {old_name} , removed')
        return config

    @classmethod
    def rename_CudaEmptyCache_to_MemoryOptimization(cls , key : str , config : DictLoader) -> DictLoader:
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        old_name = 'callbacks.CudaEmptyCache'
        new_name = 'callbacks.MemoryOptimization'
        default_value = {
            'batch_interval': 20,
        }
        if old_name in config:
            value = config.pop(old_name)
            cls.logger.success(f'{key}.{old_name} has been removed')
        else:
            value = default_value
        if new_name not in config:
            config[new_name] = value
            cls.logger.success(f'{key}.{new_name} has been added')

        for name in list(config.keys()):
            if 'CudaEmptyCache' in name:
                new_name = name.replace('CudaEmptyCache' , 'MemoryOptimization')
                config[new_name] = config.pop(name)
                cls.logger.success(f'{key} >> {name} has been renamed to {new_name}')
        return config

class ModelConfigsBatchModifier:
    def __init__(self):
        self.root = PATH.model
        
    def get_task_list(self) -> list[Callable[[str , DictLoader] , DictLoader]]:
        method_names = [name for name in dir(ModelConfigModifier) if not name.startswith('_')]
        for name in dir(Base.BoundLogger):
            if name in method_names:
                method_names.remove(name)
        return [getattr(ModelConfigModifier , name) for name in method_names]

    def load_config(self) -> dict:
        return PATH.read_yaml(self.current_path)

    def dump_config(self , config : DictLoader):
        if isinstance(config , dict):
            PATH.dump_yaml(config , self.current_path , overwrite = True)

    def batch_modify(self):
        for path in self.root.rglob('*.yaml'):
            self.current_path = path
            config = self.load_config
            for task in self.get_task_list():
                config = task(path.relative_to(self.root).as_posix() , config)
            if isinstance(config , dict):
                self.dump_config(config)