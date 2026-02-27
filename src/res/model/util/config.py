import os , random , shutil , torch
import numpy as np

from pathlib import Path
from typing import Any , Literal , Type

from src.proj import PATH , MACHINE , Logger , Proj
from src.proj.util import Device 
from src.proj.func import update_dict
from src.res.algo import AlgoModule
from src.res.factor.calculator import StockFactorHierarchy , FactorCalculator

from .abc import model_module_type , is_null_module_type
from .model_path import ModelPath
from .metrics import Metrics
from .storage import Checkpoint , Deposition

def schedule_path(base_path : ModelPath | None , name : str | None) -> Path | None:
    if not name:
        return None
    if base_path:
        path = base_path.conf('schedule').joinpath(f'{name}.yaml')
    else:
        schedule_path_0 = PATH.conf.joinpath('schedule').joinpath(f'{name}.yaml')
        schedule_path_1 = PATH.shared_schedule.joinpath(f'{name}.yaml')
        assert schedule_path_0.exists() or schedule_path_1.exists() , f'{name} does not exist in config/schedule or .local_resources/shared/schedule_model/schedule'
        assert not (schedule_path_0.exists() and schedule_path_1.exists()) , f'{name} exists in both config/schedule and .local_resources/shared/schedule_model/schedule'
        path = schedule_path_0 if schedule_path_0.exists() else schedule_path_1
    return path

def schedule_config(base_path : ModelPath | None , name : str | None):
    # schedule configs are used to override the train config
    if base_path:
        schedules = list(base_path.conf('schedule').glob('*.yaml'))
        assert len(schedules) <= 1 , f'multiple schedules found: {schedules}'
        name = schedules[0].stem if schedules else None

    path = schedule_path(base_path , name)
    p = PATH.read_yaml(path) if path and path.exists() else {}

    model_conf : dict[str,Any] = p.get('model' , {})
    model_conf = {f'{cfg}.{k}':v for cfg , cfg_params in model_conf.items() for k,v in cfg_params.items() if cfg in BaseConfig.CONFIG_LIST}
    algo_conf : dict[str,Any] = p.get('algo' , {})

    schedule_conf = {'model': model_conf, 'algo': algo_conf}
    return schedule_conf

def copy_file(source : Path | None , target : Path | None , overwrite = False):
    if source is None or not source.exists() or target is None or target == source or (target.exists() and not overwrite):
        return
    assert source.is_file() , f'{source} is not a file'
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source , target)

def striped_list(factors : list[str] | dict | str):
    if isinstance(factors , str): 
        return [factors.strip()]
    else:
        if isinstance(factors , dict): 
            factors = list(factors.values())
        return [ff for f in factors for ff in striped_list(f)]

class BaseConfig:
    CONFIG_LIST = ['env' , 'model' , 'input' , 'train' , 'callbacks' , 'conditional']
    REQUIRED_CONFIG_PARAM : dict[str,Any] = MACHINE.configs('model' , 'default' , 'required')
    OPTIONAL_CONFIG_PARAM : dict[str,Any] = MACHINE.configs('model' , 'default' , 'optional')

    def __init__(
        self , base_path : ModelPath | Path | str | None , * , 
        module : str | None = None , schedule_name : str | None = None , 
        override = None , **kwargs
    ):
        self.base_path = ModelPath(base_path)
        self.force_module = module
        self.schedule_name = schedule_name
        self.override = (override or {}) | kwargs
        self.load_params()
        self.override_params()
        self.check_validity()

    def __bool__(self): return True
    def __repr__(self): return f'{self.__class__.__name__}(base_path={self.base_path})'
    def __getitem__(self , key : str) -> Any:
        if key in self.REQUIRED_CONFIG_PARAM:
            return self.Param[key]
        else:
            return self.Param.get(key , self.OPTIONAL_CONFIG_PARAM[key])

    def __setitem__(self , key : str , value : Any):
        self.Param[key] = value

    def load_params(self):
        self.train_param : dict[str,Any] = {}
        for cfg , source , target in zip(self.CONFIG_LIST , self.source_conf_files() , self.target_conf_files()):
            p : dict[str,Any] = PATH.read_yaml(target) if target.exists() else PATH.read_yaml(source)
            self.train_param.update({f'{cfg}.{k}':v for k,v in p.items()})
        return self
    
    def override_params(self):
        self.schedule_conf = schedule_config(self.base_path , self.schedule_name)
        model_module_candidate = {
            'base_path': self.base_path.full_module_name,
            'force_module': str(self.force_module).lower().replace(' ' , '').replace('/' , '@') if self.force_module else None,
            'schedule_name': self.schedule_conf['model'].pop('model.module' , None),
        }
        assert sum(bool(v) for v in model_module_candidate.values()) <= 1 , \
            f'only one of base_path , force_module , schedule_name can be provided, but got {model_module_candidate}'
        model_modules = [v for v in model_module_candidate.values() if v]
        if model_modules:
            self['model.module'] = model_modules[0]
        
        # deal with short_test given short_test model path / override / should be short_test
        if self.base_path and self.base_path.is_short_test: 
            self.override['env.short_test'] = True
        else: 
            if (short_test := self.override.pop('short_test' , None)) is not None: 
                self.override['env.short_test'] = short_test
            if self.should_be_short_test and ('env.short_test' not in self.override): 
                self.override['env.short_test'] = True

        self.train_param.update(self.schedule_conf['model'])
        self.Param.update({k:v for k,v in self.override.items() if k in self.Param})
        
        if self.short_test:
            new_dict = {k:v for k,v in self.Param.get('conditional.short_test' , {}).items() if k not in self.override}
            update_dict(self.Param , new_dict)

        if self.model_module == 'transformer':
            new_dict = {k:v for k,v in self.Param.get('conditional.transformer' , {}).items() if k not in self.override}
            update_dict(self.Param , new_dict)

        return self

    def check_validity(self):
        if not self.base_path:
            full_name = self.generate_model_full_name()
            self.base_path.with_full_name(full_name)
        assert self.base_path , f'base_path is still not set after generating model full name {full_name}'
        assert self.base_path.module_type == self.module_type , f'module_type {self.module_type} is not the same as base_path.module_type {self.base_path.module_type}'
        assert self.base_path.model_module == self.model_module , f'model_module {self.model_module} is not the same as base_path.model_module {self.base_path.model_module}'

        if self.should_be_short_test and not self.short_test:
            Logger.alert1('Should be at server or short_test, but short_test is False now!')

        nn_category = AlgoModule.nn_category(self.model_module)
        if nn_category == 'tra': 
            assert self.sample_method != 'total_shuffle' , self.sample_method
        if nn_category == 'vae': 
            assert self.sample_method == 'sequential'    , self.sample_method

        nn_datatype = AlgoModule.nn_datatype(self.model_module)
        if nn_datatype:  
            self['input.data.types'] = nn_datatype

        if self.module_type != 'nn' or self.boost_head: 
            self['model.submodels'] = ['best']

        if self.module_type == 'factor':
            self['input.type'] = 'factor'
            self['input.factor.types'] = []
            self['input.sequence.lens'] = self['input.sequence.lens'] | {'factor': 1}

        if 'best' not in self.submodels: 
            self.submodels.insert(0 , 'best')

        if self.input_type != 'data' or self.module_type != 'nn':
            assert self.sample_method == 'sequential' , self.sample_method

        missing_required_keys = np.setdiff1d([*self.REQUIRED_CONFIG_PARAM.keys()] , [*self.Param.keys()]).tolist()
        if missing_required_keys:
            Logger.error(f'{missing_required_keys} are required but not in config files')
            raise ValueError(f'{missing_required_keys} are required but not in config files')

        redundant_keys = np.setdiff1d([*self.Param.keys()] , [*self.REQUIRED_CONFIG_PARAM.keys() , *self.OPTIONAL_CONFIG_PARAM.keys()]).tolist()
        if redundant_keys:
            Logger.alert1(f'{redundant_keys} in config files are not in default config params')
        
        return self

    def generate_model_full_name(self):
        mod_str = self.model_module.removeprefix(f'{self.module_type}@')
        if is_null_module_type(self.module_type):
            full_name = self.full_module_name
        else:
            if self['model.name']: 
                model_name = str(self['model.name'])
            else: 
                mod_str = self.model_module 
                head_str = 'boost' if self.boost_head else None
                if self.input_type == 'data':
                    data_str = '_'.join(self.input_data_types)
                else:
                    data_str = self.input_type
                model_name = '_'.join([s for s in [mod_str , head_str , data_str] if s])
            full_name = f'{self.full_module_name}@{model_name}'
        if self.short_test: 
            full_name = f'st@{full_name}'
        return full_name
    
    def generate_model_param(self , update_inplace = True):
        model_param = ModelParam(self.base_path , module = self.model_module , boost_head = self.boost_head , 
                                 short_test = self.short_test , schedule_name = self.schedule_name ,
                                 override = {k:v for k,v in self.override.items() if k not in self.Param.keys()}).expand()
        if update_inplace: 
            self.update_model_param(model_param)
        return model_param
    
    def update_model_param(self , model_param : 'ModelParam'):
        param = {k:v for k,v in model_param.Param.items() if k in self.Param}
        self.Param.update(param)
        return self

    def source_conf_files(self) -> list[Path]:
        return [PATH.conf.joinpath('model', f'{cfg}.yaml') for cfg in self.CONFIG_LIST]

    def target_conf_files(self) -> list[Path]:
        return [self.base_path.conf_file('model', cfg) for cfg in self.CONFIG_LIST]
    
    def copy_files(self , overwrite = False):
        for source , target in zip(self.source_conf_files() , self.target_conf_files()):
            copy_file(source , target , overwrite)

        copy_file(schedule_path(None , self.schedule_name) , 
                  schedule_path(self.base_path , self.schedule_name) , overwrite)

    @property
    def base_path(self):
        return self._base

    @base_path.setter
    def base_path(self , value : ModelPath):
        self._base = value

    @property
    def full_module_name(self):
        """get module_type@model_module out of model configs"""
        if not hasattr(self , '_full_module_name') or self._full_module_name is None:
            mod_str = str(self['model.module']).lower().replace(' ' , '').replace('/' , '@')
            module_type = model_module_type(mod_str)
            if mod_str.startswith(f'{module_type}@'):
                model_module = mod_str.removeprefix(f'{module_type}@')
            elif mod_str == module_type:
                model_module = self[f'model.module.{module_type}']
                assert model_module , f'model.module.{module_type} is empty!'
            else:
                model_module = mod_str
            assert model_module , f'model_module is empty after parsing for {mod_str}'
            assert '@' not in model_module , f'model_module {model_module} contains @'
            self._full_module_name = f'{module_type}@{model_module}'
        return self._full_module_name

    @property
    def module_type(self): 
        return self.full_module_name.split('@')[0]

    @property
    def model_module(self): 
        return self.full_module_name.split('@')[1]

    @model_module.setter
    def model_module(self , value : str):
        self['model.module'] = value.lower()
        self._full_module_name = None
    
    @property
    def model_name(self) -> str:
        return self.base_path.model_name

    @property
    def model_clean_name(self) -> str:
        return self.base_path.model_clean_name    

    @property
    def Param(self) -> dict[str,Any]: return self.train_param

    @property
    def should_be_short_test(self):
        return not self.base_path and not MACHINE.cuda_server

    @property
    def nn_category(self) -> str | None: 
        return AlgoModule.nn_category(self.model_module)

    @property
    def resumable(self) -> bool: 
        if not self.base_path:
            return False
        return all([self.base_path.conf(cfg).exists() for cfg in self.CONFIG_LIST])

    @property
    def short_test(self) -> bool: 
        return bool(self['env.short_test'])
    @short_test.setter
    def short_test(self , value : bool):
        self['env.short_test'] = value

    @property
    def random_seed(self) -> int | Any: 
        return self['env.random_seed']
    @property
    def mem_storage(self) -> bool: 
        return bool(self['env.mem_storage'])
    @property
    def precision(self) -> Any:
        prec = self['env.precision']
        return getattr(torch , prec) if isinstance(prec, str) else prec
    @property
    def beg_date(self) -> int: 
        return int(self['model.beg_date'])
    @property
    def end_date(self) -> int: 
        return int(self['model.end_date'])
    @property
    def labels(self) -> list[str]: 
        return self['model.labels']
    @property
    def submodels(self) -> list[str]: 
        return self['model.submodels']
    @property
    def input_type(self) -> Literal['data' , 'hidden' , 'factor' , 'combo']: 
        assert self['input.type'] in ['data' , 'hidden' , 'factor' , 'combo'] , self['input.type']
        return self['input.type']
    @property
    def input_filter_secid(self) -> str | None: 
        return self['input.filter.secid']
    @property
    def input_filter_date(self) -> str | None: 
        return self['input.filter.date']
    @property
    def input_data_types(self) -> list[str]: 
        if self.input_type == 'data' or (self.input_type == 'combo' and 'data' in self['input.combo.types']):
            data_types = self['input.data.types']
        else:
            data_types = []
        return str(data_types).split('+') if isinstance(data_types , str) else list(data_types)
    @input_data_types.setter
    def input_data_types(self , value : list[str]):
        self['input.data.types'] = value
    @property
    def input_data_prenorm(self) -> dict[str,Any]: 
        return self['input.data.prenorm']
    @property
    def input_hidden_types(self) -> list[str]: 
        if self.input_type == 'hidden' or (self.input_type == 'combo' and 'hidden' in self['input.combo.types']):
            hidden_types = self['input.hidden.types']
        else:
            hidden_types = []
        return str(hidden_types).split('+') if isinstance(hidden_types , str) else list(hidden_types)
    @property
    def input_factor_types(self) -> list[str]: 
        if self.input_type == 'factor' or (self.input_type == 'combo' and 'factor' in self['input.combo.types']):
            factors = self['input.factor.types']
        else:
            factors = []
        return str(factors).split('+') if isinstance(factors , str) else striped_list(factors)
    @property
    def input_factor_names(self) -> list[str]: 
        return [self.model_module] if self.module_type == 'factor' else []
    @property
    def input_combo_types(self) -> dict[str,list[str]]: 
        input_combos = self['input.combo.types']
        if self.input_type == 'combo' and not input_combos:
            raise ValueError('input.combo.types is empty when input_type is combo')
        combos = {k:v for k , v in self.input_keys_all.items() if k in input_combos}
        return combos
    @property
    def input_keys_all(self) -> dict[str,list[str]]: 
        return {
            'data' : self.input_data_types,
            'factor' : self.input_factor_types,
            'hidden' : self.input_hidden_types,
        }
    @property
    def input_keys_subkeys(self) -> dict[str,dict[str,str]]: 
        return {
            'data' : {tp:'.' for tp in self.input_data_types},
            'factor' : {tp:'.' for tp in self.input_factor_types},
            'hidden' : {tp:'.' for tp in self.input_hidden_types},
        }
    @property
    def window(self) -> int: 
        tw = self['model.window']
        tw = max(int(tw) if tw is not None else 0 , 0)
        return tw if tw > 0 else int(self[f'model.window.{self.module_type}'])
    @property
    def interval(self) -> int: 
        itv = self['model.interval']
        itv = max(int(itv) if itv is not None else 0 , 0)
        if itv > 0:
            return itv
        else:
            return int(self[f'model.interval.{self.module_type}'])
    @property
    def boost_head(self): 
        use_boost_head = bool(self['model.module.nn.boost_head'])
        if use_boost_head:
            return self['model.module.nn.boost_head.boost']
        else:
            return ''
    @property
    def boost_optuna(self) -> bool: 
        return bool(self['model.module.boost.optuna'])
    @property
    def boost_optuna_trials(self) -> int: 
        return int(self['model.module.boost.optuna.trials'])
    @property
    def seq_lens(self) -> dict[str,int]: 
        slens = dict(self['input.sequence.lens'])
        lens = {key:slens.get(key , slens.get(itp,1)) for itp,keys in self.input_keys_all.items() for key in keys}
        if self.module_type == 'factor':
            lens['factor'] = 1
        return lens
    @property
    def seq_steps(self) -> dict[str,int]: 
        ssteps = dict(self['input.sequence.steps'])
        steps = {key:ssteps.get(key , ssteps.get(itp,1)) for itp,keys in self.input_keys_all.items() for key in keys}
        if self.module_type == 'factor':
            steps['factor'] = 1
        return steps
    @property
    def fitting_step(self) -> int: 
        return int(self['train.fitting_step'])
    @property
    def train_ratio(self) -> float: 
        # valid ratio is 1 - train_ratio
        return float(self['train.dataloader.train_ratio'])
    @property
    def sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self['train.dataloader.sample_method']
    @property
    def shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self['train.dataloader.shuffle_option']
    @property
    def batch_size(self) -> int: 
        return int(self['train.batch_size'])
    @property
    def max_epoch(self) -> int: 
        return int(self['train.max_epoch'])
    @property
    def skip_horizon(self) -> int: 
        return int(self['train.skip_horizon'])
    @property
    def transfer_training(self) -> bool: 
        return self['train.trainer.transfer']
    @property
    def criterion_loss(self) -> dict[str,dict[str,Any]]: 
        kwargs = self['train.criterion.loss'] or {}
        assert len(kwargs) > 0 , f'{kwargs} should be not empty'
        for k,v in kwargs.items():
            if v is None:
                kwargs[k] = {}
            if 'lamb' not in v:
                kwargs[k]['lamb'] = 1.
        return {k:v for k,v in kwargs.items() if v['lamb'] != 0}
    @property
    def criterion_score(self) -> dict[str,dict[str,Any]]: 
        kwargs = self['train.criterion.score'] or {}
        assert len(kwargs) > 0 , f'{kwargs} should be not empty'
        return kwargs
    @property
    def criterion_multilosses(self) -> dict[str,dict[str,Any]]: 
        kwargs = self['train.criterion.multilosses'] or {}
        if kwargs:
            assert 'name' in kwargs , f'{kwargs} has no name'
            assert 'params' in kwargs , f'{kwargs} has no params'
            assert kwargs['name'] in ['ewa','hybrid','dwa','ruw','gls','rws'] , f'{kwargs['name']} must be one of ewa, hybrid, dwa, ruw, gls, rws'
            kwargs = {'name':kwargs['name'], 'params':kwargs['params'][kwargs['name']]}
        return kwargs
    @property
    def trainer_optimizer(self) -> dict[str,Any]: 
        return self['train.trainer.optimizer']
    @property
    def trainer_scheduler(self) -> dict[str,Any]: 
        return self['train.trainer.scheduler']
    @property
    def trainer_learn_rate(self) -> dict[str,Any]: 
        return self['train.trainer.learn_rate']
    @property
    def trainer_gradient_clip_value(self) -> float | None: 
        return self['train.trainer.gradient.clip_value']
    @property
    def factor_calculator(self) -> Type[FactorCalculator]: 
        assert self.module_type == 'factor' , f'{self.module_type} is not a factor module'
        return StockFactorHierarchy.get_factor(self.input_factor_names[0])
    @property
    def callbackes(self) -> list[str]: 
        return self['train.callbacks']
    @property
    def callback_kwargs(self) -> dict[str,dict]: 
        return {k.replace('callbacks.',''):v for k,v in self.Param.items() if k.startswith('callbacks.')}
    @property
    def try_cuda(self) -> bool: 
        return self.module_type == 'nn'
    @property
    def gc_collect_each_model(self) -> bool: 
        return self.module_type == 'nn'

class ModelParam:
    def __init__(self , base_path : ModelPath | Path | str | None , * , 
                 override : dict[str,Any] | None = None , module : str | None = None , 
                 boost_head : bool | str = False , short_test : bool | None = None , 
                 schedule_conf : dict[str,dict[str,Any]] | None = None, 
                 **kwargs):
        self.base_path = ModelPath(base_path)
        self.model_module = module
        self.boost_head = boost_head  
        self.short_test = short_test
        self.schedule_conf = schedule_conf or {'model': {}, 'algo': {}}
        self.override = (override or {}) | kwargs
        self.load_params()
        self.override_params()
        self.check_validity()

    def __repr__(self): 
        return f'{self.__class__.__name__}(model_name={self.model_name})'

    def target_conf_file(self) -> Path | None:
        if not self.base_path or self.base_path.is_null_model:
            return None
        else:
            return self.base_path.conf_file('param' , self.model_module)

    def source_conf_file(self) -> Path:
        path = PATH.conf.joinpath('algo' , self.module_type , f'{self.model_module}.yaml')
        if path.exists():
            return path
        else:
            return path.with_stem(f'default')

    def load_params(self):
        self.model_param : dict[str,Any] = {}
        conf_file = self.target_conf_file()
        if conf_file is None: 
            return self
        if not conf_file.exists():
            conf_file = self.source_conf_file()
        if not conf_file.exists():
            Logger.error(f'{conf_file} does not exist, and default.yaml does not exist either.')
        else:
            self.model_param.update(PATH.read_yaml(conf_file))
        return self
    
    def override_params(self):
        self.model_param.update(self.schedule_conf['algo'].get(f'{self.module_type}.{self.model_module}' , {}))
        self.model_param.update(self.override)
        return self

    def check_validity(self):
        if is_null_module_type(self.module_type):
            self.n_model = 1
        else:
            lens = [len(v) for v in self.Param.values() if isinstance(v , (list,tuple))]
            self.n_model = max(lens) if lens else 1
            if self.short_test: 
                self.n_model = min(1 , self.n_model)
            assert self.n_model <= 5 , self.n_model
        
        if self.model_module == 'tra':
            assert 'hist_loss_seq_len' in self.Param , f'{self.Param} has no hist_loss_seq_len'
            assert 'hist_loss_horizon' in self.Param , f'{self.Param} has no hist_loss_horizon'

        if self.boost_head:
            self.boost_head_param = ModelParam(self.base_path , module = self.boost_head , boost_head = False , schedule_conf = self.schedule_conf , **self.override)
        return self
    
    def copy_files(self , overwrite = False):
        if not self.base_path:
            return self

        if not self.base_path.is_null_model:
            copy_file(self.source_conf_file() , self.target_conf_file() , overwrite)
                
        if self.boost_head: 
            copy_file(
                PATH.conf.joinpath('algo' , 'boost' , f'{self.boost_head}.yaml') , 
                self.base_path.conf_file('param' , self.boost_head) , overwrite)
        return self

    def expand(self):
        self.params : list[dict[str,Any]] = []

        for mm in range(self.n_model):
            par = {k:v[mm % len(v)] if isinstance(v , (list, tuple)) else v for k,v in self.Param.items()}
            self.params.append(par)

        if self.boost_head: 
            self.boost_head_param.expand()
        return self

    def update_param_dict(self , param : dict[str,Any] , key : str , value : Any , delist = True , overwrite = False):
        if key in param.keys() and not overwrite: 
            return
        if delist and isinstance(value , (list , tuple)) and len(value) == 1: 
            value = value[0]
        if value is not None: 
            param[key] = value
       
    def update_data_param(self , x_data : dict, config : 'ModelConfig'):
        '''when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module'''
        if not x_data:
            return self
        
        keys = list(x_data.keys())
        input_dim = [x_data[mdt].shape[-1] for mdt in keys]
        inday_dim = [x_data[mdt].shape[-2] for mdt in keys]
        for param in self.params:
            self.update_param_dict(param , 'input_dim' , input_dim)
            self.update_param_dict(param , 'inday_dim' , inday_dim)
            if len(keys) == 1:
                value : int = (config.seq_lens | param.get('seqlens',{})).get(keys[0] , 1)
                self.update_param_dict(param , 'seq_len' , value)
        return self

    @property
    def base_path(self):
        return self._base

    @base_path.setter
    def base_path(self , value : ModelPath):
        self._base = value

    @property
    def model_name(self) -> str:
        return self.base_path.model_name

    @property
    def Param(self) -> dict[str,Any]: 
        return self.model_param

    @property
    def module_type(self): 
        return model_module_type(self.model_module)

    @property
    def model_module(self): 
        if not hasattr(self , '_model_module'):
            self._model_module = None
        return self._model_module if self._model_module else self.base_path.model_module

    @model_module.setter
    def model_module(self , value : str | None):
        self._model_module = value

    @property
    def boost_head(self) -> str:
        if not hasattr(self , '_boost_head'):
            self._boost_head = ''
        return self._boost_head

    @boost_head.setter
    def boost_head(self , value : bool | str | None):
        if not value or self.module_type != 'nn':
            self._boost_head = ''
        else:
            self._boost_head = value = 'lgbm' if value is True else value
            assert AlgoModule.is_valid(self._boost_head , 'boost') , f'{self._boost_head} is not a valid boost module'
    
    @property
    def max_num_output(self) -> int: 
        return max(self.Param.get('num_output' , [1]))

class ModelConfig(BaseConfig):
    def __init__(
        self , base_path : ModelPath | Path | str | None , * , 
        module : str | None = None , schedule_name : str | None = None ,
        override = None , stage = -1 , resume = -1 , selection = -1 , makedir = True , 
        start : int | None = None , end : int | None = None , 
        test_mode = False , **kwargs
    ):
        
        self.start  = int(start) if start is not None else None
        self.end    = int(end) if end is not None else None
        self.BaseConfig = BaseConfig(base_path , module = module , schedule_name = schedule_name , override = override , **kwargs)
        self.ModelParam = self.BaseConfig.generate_model_param()

        assert self.base_path , self.base_path

        if not test_mode:
            self.process_parser(stage , resume , selection)
            self.initialize_fitting()

        self.device = Device(try_cuda = self.try_cuda)
        assert self.BaseConfig.base_path is self.base_path , f'{self.BaseConfig.base_path} != {self.base_path}'
        assert self.ModelParam.base_path is self.ModelParam.base_path , f'{self.ModelParam.base_path} != {self.ModelParam.base_path}'

    def __repr__(self): return f'{self.__class__.__name__}(base_path={self.base_path})'

    @classmethod
    def initiate(cls , base_path : ModelPath | Path | str | None , * , vb_level = 2 , min_key_len = -1 , **kwargs):
        config = cls(base_path , **kwargs)
        config.print_out(vb_level = vb_level , min_key_len = min_key_len)
        return config

    @classmethod
    def default(cls , * , module = None , override = None , stage = 0, resume = 0 , selection = 0 , makedir = False):
        if module:
            override = override or {}
            override['model.module'] = module
        return cls(None, override = override , stage = stage , resume=resume , selection=selection,makedir=makedir)
    
    @classmethod
    def load_model(cls , model_name : ModelPath | Path | str , * , override = None , short_test : bool | None = None , 
                   stage = 2 , resume = 1 , selection = 0):
        ''' 
        load a existing model's config 
        stage is mostly irrelevant here, because mostly we are loading a model's config to pred
        '''
        model_path = ModelPath(model_name)
        assert model_path.base.exists() , f'{model_path.base} does not exist'
        return cls(model_path , override = override, short_test = short_test, stage = stage, resume = resume, selection = selection)

    def initialize_fitting(self):
        self.base_path.mkdir(model_nums = self.model_num_list , exist_ok=True)
        if 'fit' in self.stage_queue and not self.is_resuming:
            if self.base_path.base.exists(): 
                if not self.short_test and not self.base_path.is_null_model and self.BaseConfig.resumable:
                    raise Exception(f'{self.model_name} resumable , re-train has to delete folder manually')
                self.base_path.clear_model_path()
                Logger.alert1(f'{self.base_path} is cleared')
        self.BaseConfig.copy_files(overwrite = self.short_test)
        self.ModelParam.copy_files(overwrite = self.short_test)

    @property
    def base_path(self) -> ModelPath:
        return self.BaseConfig.base_path
    @property
    def is_null_model(self) -> bool:
        return self.base_path.is_null_model
    @property
    def full_module_name(self):
        return self.BaseConfig.full_module_name
    @property
    def Param(self) -> dict[str,Any]: 
        return self.BaseConfig.Param
    @property
    def model_name(self) -> str | Any: 
        return self.BaseConfig.model_name
    @property
    def model_param(self) -> list[dict[str,Any]]: 
        return self.ModelParam.params
    @property
    def model_num(self) -> int: 
        return self.ModelParam.n_model
    @property
    def short_test(self) -> bool:
        return self.base_path.is_short_test
    @property
    def model_num_list(self) -> list[int]: 
        return list(range(self.ModelParam.n_model))
    @property
    def boost_head_param(self) -> dict[str,Any]: 
        assert len(self.ModelParam.boost_head_param.params) == 1 , self.ModelParam.boost_head_param.params
        return self.ModelParam.boost_head_param.params[0]

    @property
    def beg_date(self) -> int: 
        beg_date = self.BaseConfig.beg_date
        if self.module_type == 'factor':
            beg_date = max(beg_date , self.factor_calculator.get_min_date())
        if self.start is not None:
            beg_date = max(beg_date , self.start)
        return beg_date
    @property
    def end_date(self) -> int: 
        end_date = self.BaseConfig.end_date
        if self.module_type == 'factor':
            end_date = min(end_date , self.factor_calculator.get_max_date())
        if self.end is not None:
            end_date = min(end_date , self.end)
        return end_date

    @property
    def stage_queue(self) -> list[Literal['data' , 'fit' , 'test']]:
        """stage queue for training"""
        return getattr(self , '_stage_queue' , [])

    @stage_queue.setter
    def stage_queue(self , value : list[Literal['data' , 'fit' , 'test']]):
        self._stage_queue = value

    @property
    def is_resuming(self) -> bool:
        return getattr(self , '_is_resuming' , False)

    @is_resuming.setter
    def is_resuming(self , value : bool):
        self._is_resuming = value
    
    def update(self, update = None , **kwargs):
        update = update or {}
        for k,v in update.items(): 
            setattr(self , k , v)
        for k,v in kwargs.items(): 
            setattr(self , k , v)
        return self
    
    def set_config_environment(self , manual_seed = None) -> None:
        self.set_random_seed(manual_seed if manual_seed else self.random_seed)
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self['env.allow_tf32']) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self['env.detect_anomaly'])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict) -> None:
        '''
        when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module
        do it in whenever x_data is changed
        '''
        if self.module_type == 'nn' and x_data: 
            self.ModelParam.update_data_param(x_data , self)
    
    def weight_scheme(self , stage : str , no_weight = False) -> str | None: 
        if stage == 'fit':
            stg = 'fit'
        else:
            # includes test / predict / extract
            stg = 'test'
        return None if no_weight else self.BaseConfig.Param[f'train.criterion.weight'].get(stg , 'equal')
    
    def init_utils(self):
        self.metrics = Metrics(self.module_type , self.nn_category , self.base_path ,
                               self.criterion_loss , self.criterion_score , self.criterion_multilosses)
        self.checkpoint = Checkpoint(self.mem_storage)
        self.deposition = Deposition(self.base_path)
        return self

    @staticmethod
    def set_random_seed(seed = None):
        if seed is None: 
            return NotImplemented
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def process_parser(self , stage = -1 , resume = -1 , selection = -1 , vb_level : int = 1):
        '''
        stage:     
            [-1] , if nn / boost then choose stage, else just data + test
            [ 0] , data + fit + test
            [ 1] , data + fit
            [ 2] , data + test
        resume:    
            [-1] , if not st and model_path(s) exists then choose
            [ 0] , no
            [ 1] , yes
        selection: 
            [-1] , choose if optional
            [ 0] , raw model name unless fitting and not resuming
            [1,2,3,...] , choose by model_index if is_resuming
        '''
        if self.base_path:
            vb_level = 99
        
        self.parser_stage(stage , vb_level)
        self.parser_resume(resume , vb_level)
        self.parser_select(selection , vb_level) 
        return self

    def parser_stage(self , value = -1 , vb_level : int = 1):
        """
        parser stage queue
        value:
            -1: choose if optional
            0: fit + test
            1: fit only
            2: test only
        """
        assert value in [-1 , 0 , 1 , 2] , f'stage must be -1, 0, 1, or 2, got {value}'
        stage_queue : list[Literal['data' , 'fit' , 'test']] = ['data' , 'fit' , 'test']

        if value == -1:
            if not self.base_path.is_null_model:
                value = 2
            else:
                Logger.note(f'--What stage would you want to run? 0: fit + test, 1: fit only , 2: test only')
                value = int(input(f'[0, fit & test] , [1, fit only] , [2, test only]'))
            
        match value:
            case 0:
                stage_queue = ['data' , 'fit' , 'test']
            case 1:
                stage_queue = ['data' , 'fit']
            case 2:
                stage_queue = ['data' , 'test']
            case _:
                raise ValueError(f'Invalid stage option: {value}')

        if self.base_path.is_null_model and 'fit' in stage_queue:
            stage_queue.remove('fit')

        self.stage_queue = stage_queue
        Logger.note('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]), stage_queue))) , 
                    color = 'lightblue' , vb_level = vb_level)

    def parser_resume(self , value = -1 , vb_level : int = 1):
        """
        parser resume flag
        value:
            -1: choose if optional
            0: no
            1: yes
        """
        assert value in [-1 , 0 , 1] , f'resume must be -1, 0, or 1, got {value}'
        if value == -1: 
            if self.short_test:
                value = 0
            else:
                candidates = self.base_path.find_resumable_candidates()
                if candidates:
                    Logger.note(f'Multiple model path of {self.model_clean_name} exists, input [0] to deny resuming , [1] to confirm resuming!')
                    value = int(input(f'[0, not resuming] , [1, resuming]'))
                else:
                    value = 0
                    
        match value:
            case 0:
                is_resuming = False
            case 1:
                is_resuming = True
            case _:
                raise ValueError(f'Invalid resume option: {value}')

        self.is_resuming = is_resuming
        Logger.note(f'Confirm Resume Training!' if is_resuming else 'Start Training New!' , vb_level = vb_level)

    def parser_select(self , value = -1 , vb_level : int = 1):
        '''
        parse model_name selection if model_name dirs exists
        value:

        -1: choose if optional
            if short_test or null_model or no candidates:
                don't change the base_path
            elif fit is in stage_queue:
                -1: choose (if model_name dirs exists, ask to choose one)
                0: raw model_name dir if is_resuming , create a new model_name dir otherwise
                1,2,3,...: choose one by model_index, start from 1 (if not is_resuming , raise error)
            elif test only:
                -1: choose (if more than 1 model_name dirs exists, ask to choose one)
                0: try to use the raw model_name dir
                1,2,3,...: choose one by model_index
        0: don't change the base_path
        '''
        assert value in [-1 , 0] , f'initial selection must be -1 or 0, got {value}'
        if value == -1:
            candidates = self.base_path.find_resumable_candidates()
            candidates_indices = sorted([mp.model_name_index for mp in candidates])
            if self.short_test or self.base_path.is_null_model or not candidates:
                value = 0
            elif 'fit' in self.stage_queue and not self.is_resuming:
                if self.base_path.model_name_index not in candidates_indices:
                    value = 0
                else:
                    value = int(np.setdiff1d(np.arange(1 , max(candidates_indices) + 1) , candidates_indices).min())
                    Logger.note(f'ModelPath(s) of {self.model_clean_name} exists, will create a new ModelPath with index {value} to Train New!' , vb_level = vb_level)
            else:
                if len(candidates_indices) == 1:
                    value = candidates_indices[0]
                else:
                    Logger.note(f'Multiple ModelPath of {self.model_clean_name} exists, input number to choose!' , vb_level = vb_level)
                    Logger.note(f'Options include: {candidates_indices}' , vb_level = vb_level)
                    value = int(input(f'Which Model to Resume?'))
                    assert value in candidates_indices , f'value {value} is not in candidates_indices {candidates_indices}'

        match value:
            case 0:
                ... # don't change the base_path
            case _:
                assert value > 0 , f'value {value} must be greater than 0'
                self.base_path.with_new_index(value)

        if not self.short_test and not self.is_resuming and not self.base_path.is_null_model and 'fit' in self.stage_queue and self.resumable:
            Logger.error(f'{self.base_path} resumable but choose not to resume! You have to start a new training or manually delete the existing model_name dir!')
            raise Exception(f'{self.base_path} resumable but choose not to resume!')
                        
    def print_out(self , color : str | None = None , vb_level : int = 2 , min_key_len : int = -1):
        info_strs : list[tuple[int , str , str]] = [] # indent , key , value
        
        info_strs.append((0 , 'Module' , f'{self.full_module_name}'))
        info_strs.append((0 , 'Model Name' , self.model_name))
        if self.base_path.is_null_model:
            info_strs.append((0 , 'Labels' , f'{self.labels}'))
            info_strs.append((0 , 'Period' , f'{self.beg_date} ~ {self.end_date}'))
        else:
            if self.module_type == 'boost':
                if self.boost_optuna:
                    info_strs.append((0 , 'Boost Params' , f'Optuna for {self.boost_optuna_trials} trials'))
                else:
                    info_strs.append((0 , 'Boost Params' , ''))
                    for k , v in self.ModelParam.Param.items():
                        info_strs.append((2 , k , f'{v}'))
            else:
                if self.boost_head:
                    info_strs.append((0 , 'Use Boost Head' , f'{self.boost_head}'))
                info_strs.append((0 , 'Model Params' , ''))
                for k , v in self.ModelParam.Param.items():
                    info_strs.append((1 , k , f'{v}'))
            info_strs.append((0 , 'Model Num' , f'{self.ModelParam.n_model}'))
            info_strs.append((0 , 'Inputs' , f'{self.input_type}'))
            if self.input_type == 'data':
                info_strs.append((1 , 'Data Types' , f'{self.input_data_types}'))
            elif self.input_type == 'hidden':
                info_strs.append((1 , 'Hidden Models' , f'{self.input_hidden_types}'))
            elif self.input_type == 'factor':
                info_strs.append((1 , 'Factor Types' , f'{self.input_factor_types}'))
                if self.input_factor_names:
                    info_strs.append((1 , 'Factor Names' , f'{self.input_factor_names}'))
            elif self.input_type == 'combo':
                info_strs.append((1 , 'Combo Types' , f'{self.input_combo_types}'))
            info_strs.append((0 , 'Labels' , f'{self.labels}'))
            info_strs.append((0 , 'Period' , f'{self.beg_date} ~ {self.end_date}'))
            info_strs.append((0 , 'Interval' , f'{self.interval} days'))
            info_strs.append((0 , 'Window' , f'{self.window} days'))
            info_strs.append((0 , 'Sampling' , f'{self.sample_method}'))
            info_strs.append((0 , 'Shuffling' , f'{self.shuffle_option}'))
            info_strs.append((0 , 'Random Seed' , f'{self.random_seed}'))
        info_strs.append((0 , 'Short Test' , f'{self.short_test}'))
        info_strs.append((0 , 'Stage Queue' , f'{self.stage_queue}'))
        info_strs.append((0 , 'Resuming' , f'{self.is_resuming}'))
        if self.is_resuming:
            info_strs.append((1 , 'Resume Test' , f'{Proj.Conf.Model.TRAIN.resume_test}'))
            info_strs.append((1 , 'Resume Perf' , f'{Proj.Conf.Model.TRAIN.resume_test_factor_perf}'))
            info_strs.append((1 , 'Resume FMP' , f'{Proj.Conf.Model.TRAIN.resume_test_fmp}'))
            info_strs.append((1 , 'Resume Account' , f'{Proj.Conf.Model.TRAIN.resume_test_fmp_account}'))

        Logger.stdout_pairs(info_strs , title = 'Train Config Initiated:' , color = color , vb_level = vb_level , min_key_len = min_key_len)
        return self
