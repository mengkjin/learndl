from __future__ import annotations
from src.proj import PATH , Logger
from typing import Callable , TypeAlias

__all__ = ['ModelConfigModifier' , 'ModelConfigsBatchModifier']

DictLoader : TypeAlias = Callable[[] , dict] | dict

class ModelConfigModifier:
    @classmethod
    def rename_ResetOptimizer(cls , key : str , config : DictLoader):
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
            Logger.success(f'{key}.{old_name} has been removed')
        else:
            value = default_value
        if new_name not in config:
            config[new_name] = value
            Logger.success(f'{key}.{new_name} has been added')

        for name in config:
            if 'ResetOptimizer' in name:
                new_name = name.replace('ResetOptimizer' , 'LearnRateReset')
                config[new_name] = config.pop(name)
                Logger.success(f'{key}.{name} has been renamed to {new_name}')
        return config

    @classmethod
    def add_LearnRateReset(cls , key : str , config : DictLoader):
        if not key.endswith('.model'):
            return config
        if not isinstance(config , dict):
            config = config()
        if 'LearnRateReset' not in config['train.callbacks']:
            config['train.callbacks'].append('LearnRateReset')
            Logger.success(f'{key}.train.callbacks lacks LearnRateReset , added')
        return config

    @classmethod
    def rename_lamb(cls , key : str , config : DictLoader):
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
                Logger.success(f'{key}.train.criterion.loss.{loss_name} has lamb key , renamed to alpha')
        return config

    @classmethod
    def remove_eps_callback_param(cls , key : str , config : DictLoader):
        if not key.endswith(('.model' , '.schedule')):
            return config
        if not isinstance(config , dict):
            config = config()
        for key in config:
            if key.startswith('callbacks.'):
                param = config[key]
                if 'eps' in param:
                    param.pop('eps')
                    Logger.success(f'{key}.{param} has eps key , removed')
        return config

    @classmethod
    def replace_EarlyExitRetrain_with_BadAttemptRetrain(cls , key : str , config : DictLoader):
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
            'lr_multiplier': [1 , 0.1 , 10 , 0.01 , 100 , 1],
        }
        if old_name in config:
            value = config.pop(old_name)
            Logger.success(f'{key}.{old_name} has been removed')
        else:
            value = default_value
        if new_name not in config:
            config[new_name] = value
            Logger.success(f'{key}.{new_name} has been added')

        if 'BadAttemptRetrain' not in config['train.callbacks']:
            config['train.callbacks'].append('BadAttemptRetrain')
            Logger.success(f'{key}.train.callbacks lacks BadAttemptRetrain , added')
        if 'EarlyExitRetrain' in config['train.callbacks']:
            config['train.callbacks'].remove('EarlyExitRetrain')
            Logger.success(f'{key}.train.callbacks has EarlyExitRetrain , removed')
        return config

class ModelConfigsBatchModifier:
    def __init__(self):
        self.root = PATH.model
        self.task_list : list[Callable[[str , DictLoader] , DictLoader]] = [
            getattr(ModelConfigModifier , name) for name in dir(ModelConfigModifier) if not name.startswith('_')
        ]

    def load_config(self) -> dict:
        return PATH.read_yaml(self.current_path)

    def dump_config(self , config : DictLoader):
        if isinstance(config , dict):
            PATH.dump_yaml(config , self.current_path , overwrite = True)

    def batch_modify(self):
        for path in self.root.rglob('*.yaml'):
            self.current_path = path
            config = self.load_config
            for task in self.task_list:
                config = task(f'{path.parent.parent.stem}.{path.stem}' , config)
            if isinstance(config , dict):
                self.dump_config(config)