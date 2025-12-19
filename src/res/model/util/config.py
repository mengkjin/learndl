import os , random , shutil , torch
import numpy as np

from pathlib import Path
from typing import Any , Literal

from src.proj import PATH , MACHINE , Logger
from src.basic import Device , ModelPath , ModelDBMapping , DB
from src.func import recur_update
from src.res.algo import AlgoModule

from .metrics import Metrics
from .storage import Checkpoint , Deposition

TRAIN_CONFIG_LIST = ['train' , 'env' , 'callbacks' , 'conditional' , 'model']
TYPE_MODULE_TYPES = Literal['nn' , 'booster' , 'db' , 'factor']

def conf_dir(base_path : ModelPath | Path | str | None , *args):
    return ModelPath(base_path).conf(*args)

def conf_path(base_path : ModelPath | Path | str | None , *args):
    return ModelPath(base_path).conf(*args).with_suffix('.yaml')

def schedule_path(base_path : ModelPath | Path | str | None , name : str):
    base_path = ModelPath(base_path)
    if base_path:
        path = base_path.conf('schedule').joinpath(f'{name}.yaml')
    else:
        schedule_path_0 = PATH.conf.joinpath('schedule').joinpath(f'{name}.yaml')
        schedule_path_1 = PATH.shared_schedule.joinpath(f'{name}.yaml')
        assert schedule_path_0.exists() or schedule_path_1.exists() , f'{name} does not exist in config/schedule or .local_resources/shared/schedule_model/schedule'
        assert not (schedule_path_0.exists() and schedule_path_1.exists()) , f'{name} exists in both config/schedule and .local_resources/shared/schedule_model/schedule'
        path = schedule_path_0 if schedule_path_0.exists() else schedule_path_1
    return path

def conf_copy(source : Path , target : Path , overwrite = False):
    if source.is_dir():
        if not overwrite: 
            assert not target.exists() or len([v for v in target.iterdir()]) == 0 , target
        shutil.copytree(source , target , dirs_exist_ok = True)
    else:
        if not overwrite: 
            assert not target.exists() , target
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source , target)

def striped_list(factors : list[str] | dict | str):
    if isinstance(factors , str): 
        return [factors.strip()]
    else:
        if isinstance(factors , dict): 
            factors = list(factors.values())
        return [ff for f in factors for ff in striped_list(f)]

def conf_mod_type(module : str) -> TYPE_MODULE_TYPES:
    if module.startswith('db@'):
        return 'db'
    elif module.startswith('factor@'):
        return 'factor'
    else:
        return AlgoModule.module_type(module)

def schedule_config(base_path : ModelPath | Path | None , name : str | None):
    # schedule configs are used to override the train config
    base_path = ModelPath(base_path)
    if base_path:
        schedules = list(base_path.conf('schedule').glob('*.yaml'))
        assert len(schedules) <= 1 , f'multiple schedules found: {schedules}'
        name = schedules[0].stem if schedules else None
    p : dict[str,Any] = {}
    if name: 
        p.update(PATH.read_yaml(schedule_path(base_path , name)))
    return p

class TrainParam:
    def __init__(self , base_path : ModelPath | Path | str | None , override = None , schedule_name : str | None = None , **kwargs):
        self.base_path = ModelPath(base_path)
        self.model_name = self.base_path.name
        self.override = (override or {}) | kwargs
        self.schedule_name = schedule_name
        self.load_param().check_validity()

    def __bool__(self): return True
    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name})'

    def reset_base_path(self , base_path : Path | ModelPath | str | None):
        self.base_path = ModelPath(base_path)
        self.model_name = self.base_path.name
        return self

    @property
    def model_root_path(self) -> Path:
        return PATH.null_model if self.module_type in ['db' , 'factor'] else PATH.model

    @property
    def model_base_path(self):
        # assert self.base_path , f'{self.base_path} is None'
        return self.base_path

    @property
    def Param(self) -> dict[str,Any]: return self.train_param

    def load_param(self):
        self.train_param : dict[str,Any] = {}
        for cfg in TRAIN_CONFIG_LIST:
            p : dict[str,Any] = PATH.read_yaml(conf_path(self.base_path , 'train', cfg))
            self.train_param.update({f'{cfg}.{k}':v for k,v in p.items()})

        self.update_schedule_param()
        self.special_adjustment()
        self.make_model_name()
        return self
    
    def update_schedule_param(self):
        schedule_conf : dict[str,Any] = schedule_config(self.base_path , self.schedule_name).get('train' , {})
        for cfg in TRAIN_CONFIG_LIST:
            self.train_param.update({f'{cfg}.{k}':v for k,v in schedule_conf.get(cfg , {}).items()})
        return self
    
    @property
    def should_be_short_test(self):
        return not self.base_path and not MACHINE.server

    def special_adjustment(self):
        if 'verbosity'  in self.override: 
            self.override['env.verbosity']  = self.override.pop('verbosity')
        if 'short_test' in self.override: 
            self.override['env.short_test'] = self.override.pop('short_test')
        if 'module'     in self.override: 
            self.override['model.module']   = self.override.pop('module')
        if '_ShortTest' in self.model_name: 
            self.override['env.short_test'] = True
        
        if self.should_be_short_test and ('env.short_test' not in self.override): 
            self.override['env.short_test'] = True
        for override_key in self.override:
            assert override_key in self.Param.keys() , override_key
        self.Param.update(self.override)

        if self.short_test:
            new_dict = {k:v for k,v in self.Param.get('conditional.short_test' , {}).items() if k not in self.override}
            recur_update(self.Param , new_dict)

        if self.model_module == 'transformer':
            new_dict = {k:v for k,v in self.Param.get('conditional.transformer' , {}).items() if k not in self.override}
            recur_update(self.Param , new_dict)
        return self
    
    def make_model_name(self):
        if self.model_name: 
            return self
        if self.Param['model.name']: 
            model_name = str(self.Param['model.name'])
        elif self.module_type in ['db' , 'factor']:
            model_name = self.model_module
        else: 
            mod_str = self.model_module 
            head_str = 'booster' if self.model_booster_head else None
            if self.model_input_type == 'data':
                data_str = '+'.join(self.model_data_types)
            else:
                data_str = self.model_input_type
            model_name = '_'.join([s for s in [mod_str , head_str , data_str] if s])
        if self.short_test: 
            model_name += '_ShortTest'
        self.model_name = model_name
        return self
    
    def check_validity(self):
        if self.should_be_short_test and not self.short_test:
            Logger.warning('Beware! Should be at server or short_test, but short_test is False now!')

        nn_category = AlgoModule.nn_category(self.model_module)
        if nn_category == 'tra': 
            assert self.train_sample_method != 'total_shuffle' , self.train_sample_method
        if nn_category == 'vae': 
            assert self.train_sample_method == 'sequential'    , self.train_sample_method

        nn_datatype = AlgoModule.nn_datatype(self.model_module)
        if nn_datatype:  
            self.Param['model.data.types'] = nn_datatype

        if self.module_type != 'nn' or self.model_booster_head: 
            self.Param['model.submodels'] = ['best']

        if self.module_type == 'factor':
            self.Param['model.input_type'] = 'factor'
            self.Param['model.factor.types'] = []
            self.Param['model.sequence.lens']['factor'] = 1

        if 'best' not in self.model_submodels: 
            self.model_submodels.insert(0 , 'best')

        if self.model_input_type != 'data' or self.module_type != 'nn':
            assert self.train_sample_method == 'sequential' , self.train_sample_method

        return self
    
    def generate_model_param(self , update_inplace = True , **kwargs):
        module = self.model_booster_type if self.module_type == 'booster' else self.model_module
        assert isinstance(module , str) , (self.model_module , module)
        model_param = ModelParam(self.base_path , module , self.model_booster_head , self.verbosity , self.short_test , self.schedule_name , **kwargs).expand()
        if update_inplace: 
            self.update_model_param(model_param)
        return model_param
    
    def update_model_param(self , model_param : 'ModelParam'):
        param = {k:v for k,v in model_param.Param.items() if k in self.Param}
        self.Param.update(param)
        return self
    
    def copy_to(self , where : Path | ModelPath | str , overwrite = False):
        if self.base_path:
            assert self.model_name == ModelPath(where).name , f'{self.model_name} != {ModelPath(where).name}'
        
        conf_copy(conf_dir(None , 'train') , conf_dir(where , 'train') , overwrite)
        if self.schedule_name:
            conf_copy(
                schedule_path(None , self.schedule_name) , 
                schedule_path(where , self.schedule_name) , overwrite)

    @classmethod
    def guess_module(cls , base_path : Path | ModelPath | None) -> str:
        return str(PATH.read_yaml(conf_path(base_path , 'train', 'model'))['module']).lower()
    
    @staticmethod
    def get_module_type(module : str) -> TYPE_MODULE_TYPES:
        return conf_mod_type(module)
    
    @property
    def module_type(self) -> TYPE_MODULE_TYPES: 
        return conf_mod_type(self.model_module)

    @property
    def nn_category(self) -> str | None: 
        return AlgoModule.nn_category(self.model_module)

    @property
    def resumeable(self) -> bool: 
        assert self.model_name , f'{self} has model_name None'
        model_path = ModelPath(self.model_name)
        return all([model_path.conf(cfg_key).exists() for cfg_key in TRAIN_CONFIG_LIST])

    @property
    def short_test(self) -> bool: 
        return bool(self.Param['env.short_test'])
    @short_test.setter
    def short_test(self , value : bool):
        self.Param['env.short_test'] = value

    @property
    def verbosity(self) -> int: 
        v = int(self.Param['env.verbosity'])
        return min(max(v , 0) , 10)
    @property
    def random_seed(self) -> Any: 
        return self.Param['env.random_seed']
    @property
    def mem_storage(self) -> bool: 
        return bool(self.Param['env.mem_storage'])
    @property
    def precision(self) -> Any:
        prec = self.Param['env.precision']
        return getattr(torch , prec) if isinstance(prec, str) else prec
    @property
    def beg_date(self) -> int: 
        return int(self.Param['model.beg_date'])
    @property
    def end_date(self) -> int: 
        return int(self.Param['model.end_date'])
    @property
    def model_rslt_path(self) -> Path: 
        return self.model_base_path.rslt()
    @property
    def model_submodels(self) -> list: 
        return self.Param['model.submodels']
    @property
    def model_module(self): 
        return str(self.Param['model.module']).lower()
    @property
    def model_input_type(self) -> Literal['db' , 'data' , 'hidden' , 'factor' , 'combo']: 
        if self.module_type == 'db':
            return 'db'
        assert self.Param['model.input_type'] in ['data' , 'hidden' , 'factor' , 'combo'] , self.Param['model.input_type']
        return self.Param['model.input_type']
    @property
    def model_labels(self) -> list[str]: 
        return self.Param['model.labels']
    @property
    def model_data_types(self) -> list[str]: 
        if self.module_type == 'db':
            data_types = []
        elif self.model_input_type == 'data' or (self.model_input_type == 'combo' and 'data' in self.Param['model.combo.types']):
            data_types = self.Param['model.data.types']
        else:
            data_types = []
        return str(data_types).split('+') if isinstance(data_types , str) else list(data_types)
    @property
    def model_data_prenorm(self) -> dict[str,Any]: 
        return self.Param.get('model.data.prenorm',{})
    @property
    def model_hidden_types(self) -> list[str]: 
        if self.module_type == 'db':
            hidden_types = []
        elif self.model_input_type == 'hidden' or (self.model_input_type == 'combo' and 'hidden' in self.Param['model.combo.types']):
            hidden_types = self.Param['model.hidden.types']
        else:
            hidden_types = []
        return str(hidden_types).split('+') if isinstance(hidden_types , str) else list(hidden_types)
    @property
    def model_factor_types(self) -> list[str]: 
        if self.module_type == 'db':
            factors = []
        elif self.model_input_type == 'factor' or (self.model_input_type == 'combo' and 'factor' in self.Param['model.combo.types']):
            factors = self.Param['model.factor.types']
        else:
            factors = []
        return str(factors).split('+') if isinstance(factors , str) else striped_list(factors)
    @property
    def model_combo_types(self) -> dict[str,list[str]]: 
        input_combos = self.Param.get('model.combo.types' , [])
        if self.model_input_type == 'combo' and not input_combos:
            raise ValueError('model.combo.types is empty when input_type is combo')
        combos = {k:v for k , v in self.model_all_input_keys.items() if k in input_combos}
        return combos
    @property
    def model_all_input_keys(self) -> dict[str,list[str]]: 
        return {
            'data' : self.model_data_types,
            'factor' : self.model_factor_types,
            'hidden' : self.model_hidden_types,
            'db' : [self.model_module] if self.model_input_type == 'db' else [],
        }
    @property
    def model_train_window(self) -> int: 
        if (train_window := int(self.Param.get('model.train_window', 0))) > 0:
            return train_window
        else:
            return int(self.Param.get(f'model.train_window.{self.module_type}' , 240))
    @property
    def model_interval(self) -> int: 
        if (interval := int(self.Param.get('model.interval', 0))) > 0:
            return interval
        elif self.module_type == 'db':
            return int(self.Param.get(f'model.interval.{self.module_type}', 2400))
        else:
            return int(self.Param.get(f'model.interval.{self.module_type}', 120))
    @property
    def model_booster_head(self) -> Any: 
        return self.Param['model.booster_head']
    @property
    def model_booster_type(self) -> str:
        if self.model_module in ['booster' , 'hidden_aggregator']:
            assert AlgoModule.is_valid(self.Param['model.booster_type'] , 'booster') , self.Param['model.booster_type']
            return self.Param['model.booster_type']
        elif AlgoModule.is_valid(self.model_module, 'booster'):
            return self.model_module
        else:
            return 'not_a_booster'
    @property
    def model_booster_optuna(self) -> bool: 
        return bool(self.Param.get('model.booster_optuna'))
    @property
    def model_booster_optuna_n_trials(self) -> int: 
        return int(self.Param.get('model.booster_optuna_n_trials',10))
    @property
    def model_sequence_lens(self) -> dict[str,int]: 
        return dict(self.Param.get('model.sequence.lens',{}))
    @property
    def model_sequence_steps(self) -> dict[str,int]: 
        return dict(self.Param.get('model.sequence.steps',{}))
    def seq_lens(self) -> dict[str,int]: 
        slens = self.model_sequence_lens
        lens = {key:slens.get(key , slens.get(itp,1)) for itp,keys in self.model_all_input_keys.items() for key in keys}
        if self.module_type == 'factor':
            lens['factor'] = 1
        return lens
    def seq_steps(self) -> dict[str,int]: 
        ssteps = self.model_sequence_steps
        steps = {key:ssteps.get(key , ssteps.get(itp,1)) for itp,keys in self.model_all_input_keys.items() for key in keys}
        if self.module_type == 'factor':
            steps['factor'] = 1
        return steps
    @property
    def train_data_step(self) -> int: 
        return int(self.Param['train.data_step'])
    @property
    def train_train_ratio(self) -> float: 
        # valid ratio is 1 - train_ratio
        return float(self.Param['train.dataloader.train_ratio'])
    @property
    def train_sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self.Param['train.dataloader.sample_method']
    @property
    def train_shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self.Param['train.dataloader.shuffle_option']
    @property
    def train_batch_size(self) -> int: 
        return int(self.Param['train.batch_size'])
    @property
    def train_max_epoch(self) -> int: 
        return int(self.Param['train.max_epoch'])
    @property
    def train_skip_horizon(self) -> int: 
        return int(self.Param['train.skip_horizon'])
    @property
    def train_trainer_transfer(self) -> bool: 
        return self.Param['train.trainer.transfer']
    @property
    def train_criterion_loss(self) -> str: 
        return self.Param['train.criterion.loss']
    @property
    def train_criterion_score(self) -> str: 
        return self.Param['train.criterion.score']
    @property
    def train_criterion_penalty(self) -> dict[Any,Any]: 
        return self.Param['train.criterion.penalty']
    @property
    def train_multilosses_type(self) -> str: 
        return self.Param['train.multilosses.type']
    @property
    def train_multilosses_param(self) -> dict: 
        return self.Param[f'train.multilosses.param.{self.train_multilosses_type}']
    @property
    def train_trainer_optimizer(self) -> dict[str,Any]: 
        return self.Param['train.trainer.optimizer']
    @property
    def train_trainer_scheduler(self) -> dict[str,Any]: 
        return self.Param['train.trainer.scheduler']
    @property
    def train_trainer_learn_rate(self) -> dict[str,Any]: 
        return self.Param['train.trainer.learn_rate']
    @property
    def train_trainer_gradient_clip_value(self) -> Any: 
        return self.Param['train.trainer.gradient.clip_value']
    @property
    def input_factor_names(self) -> list[str]: 
        if self.module_type == 'factor':
            return [self.model_module.removeprefix('factor@')]
        else:
            return []
    @property
    def factor_factor_calculator(self): 
        assert self.module_type == 'factor' , f'{self.module_type} is not a factor module'
        from src.res.factor.calculator import StockFactorHierarchy
        return StockFactorHierarchy.get_factor(self.input_factor_names[0])

    @property
    def callbacks(self) -> dict[str,dict]: 
        return {k.replace('callbacks.',''):v for k,v in self.Param.items() if k.startswith('callbacks.')}

    @property
    def try_cuda(self) -> bool: 
        return self.module_type == 'nn'

    @property
    def gc_collect_each_model(self) -> bool: 
        return self.module_type == 'nn'

class ModelParam:
    def __init__(self , base_path : ModelPath | Path | str | None , module : str , 
                 booster_head : Any = False , verbosity = 2 , short_test : bool | None = None , 
                 schedule_name : str | None = None,
                 **kwargs):
        self.base_path = ModelPath(base_path)
        self.model_name = self.base_path.name
        self.module = module.lower()
        self.booster_head = booster_head if self.module_type == 'nn' else None    
        self.short_test = short_test
        self.verbosity = verbosity
        self.schedule_name = schedule_name
        self.override = kwargs
        self.load_param().check_validity()

    def __repr__(self): 
        return f'{self.__class__.__name__}(model_name={self.model_name})'

    @property
    def model_base_path(self):
        assert self.base_path is not None , f'{self} has base_path None'
        return self.base_path

    def reset_base_path(self , base_path : Path | ModelPath | str | None):
        self.base_path = ModelPath(base_path)
        self.model_name = self.base_path.name
        return self

    @property
    def Param(self) -> dict[str,Any]: 
        return self.model_param

    @property
    def module_type(self) -> TYPE_MODULE_TYPES: 
        return conf_mod_type(self.module)

    @property
    def mod_type_dir(self) -> str:
        return 'boost' if self.module_type == 'booster' else self.module_type

    def conf_file(self , base : Path | ModelPath | str | None | Literal['self'] = 'self'):
        if base == 'self': 
            base = self.base_path
        if self.module_type == 'db':
            return conf_path(base , 'registry' , 'db_models_mapping')
        elif self.module_type == 'factor':
            return Path('')
        else:
            return conf_path(base , self.mod_type_dir , self.module)

    def load_param(self):
        self.model_param : dict[str,Any] = {}
        if self.module_type != 'factor':
            p : dict[str,Any] = PATH.read_yaml(self.conf_file())
            self.model_param.update(p)
        self.update_schedule_param()
        self.special_adjustment()
        return self
    
    def update_schedule_param(self):
        schedule_conf : dict[str,Any] = schedule_config(self.base_path , self.schedule_name).get(self.mod_type_dir , {})
        self.model_param.update(schedule_conf.get(self.module , {}))
        return self
    
    def special_adjustment(self):
        self.model_param['verbosity'] = self.verbosity
        self.model_param.update(self.override)
        return self

    def check_validity(self):
        assert self.module_type == 'nn' or not self.booster_head , f'{self.module} is not a valid module'

        if self.module_type in ['db' , 'factor']:
            self.n_model = 1
        else:
            lens = [len(v) for v in self.Param.values() if isinstance(v , (list,tuple))]
            self.n_model = max(lens) if lens else 1
            if self.short_test: 
                self.n_model = min(1 , self.n_model)
            assert self.n_model <= 5 , self.n_model
        
        if self.module == 'tra':
            assert 'hist_loss_seq_len' in self.Param , f'{self.Param} has no hist_loss_seq_len'
            assert 'hist_loss_horizon' in self.Param , f'{self.Param} has no hist_loss_horizon'

        if self.booster_head:
            assert AlgoModule.is_valid(self.booster_head , 'booster') , self.booster_head
            self.booster_head_param = ModelParam(self.base_path , self.booster_head , False , self.verbosity , **self.override)
        return self
    
    def get(self , key : str , default = None) -> Any:
        return self.Param.get(key , default)
    
    def copy_to(self , where : Path | ModelPath | str , overwrite = False):
        if self.base_path:
            assert self.model_name == ModelPath(where).name , \
                f'{self.model_name} != {ModelPath(where).name}'

        if self.module_type != 'factor':
            conf_copy(self.conf_file(None) , self.conf_file() , overwrite)
            if self.booster_head: 
                conf_copy(
                    conf_path(None , 'booster' , self.booster_head) , 
                    conf_path(where , 'booster' , self.booster_head) , overwrite)
        return self

    def expand(self):
        self.params : list[dict[str,Any]] = []
        if self.module_type == 'db':
            self.params.append(self.model_param.copy())
            return self

        for mm in range(self.n_model):
            par = {k:v[mm % len(v)] if isinstance(v , (list, tuple)) else v for k,v in self.Param.items()}
            self.params.append(par)

        if self.booster_head: 
            self.booster_head_param.expand()
        return self
       
    def update_data_param(self , x_data : dict, config : 'TrainConfig'):
        '''when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module'''
        if self.module_type == 'db' or not x_data:
            return self
        
        keys = list(x_data.keys())
        input_dim = [x_data[mdt].shape[-1] for mdt in keys]
        inday_dim = [x_data[mdt].shape[-2] for mdt in keys]
        for param in self.params:
            self.update_param_dict(param , 'input_dim' , input_dim)
            self.update_param_dict(param , 'inday_dim' , inday_dim)
            if len(keys) == 1:
                value : int = (config.seq_lens() | param.get('seqlens',{})).get(keys[0] , 1)
                self.update_param_dict(param , 'seq_len' , value)
        return self
    
    @staticmethod
    def update_param_dict(param : dict[str,Any] , key : str , value : Any , delist = True , overwrite = False):
        if key in param.keys() and not overwrite: 
            return
        if delist and isinstance(value , (list , tuple)) and len(value) == 1: 
            value = value[0]
        if value is not None: 
            param[key] = value
    
    @property
    def max_num_output(self) -> int: 
        return max(self.Param.get('num_output' , [1]))

    @property
    def db_mapping(self) -> ModelDBMapping:
        assert self.module_type == 'db' , f'{self.module_type} is not a db module , cannot use db_mapping'
        if not hasattr(self , '_db_mapping'):
            self._db_mapping = ModelDBMapping.from_dict(self.module , self.Param)
        return self._db_mapping

class TrainConfig(TrainParam):
    def __init__(
        self , base_path : ModelPath | Path | str | None , override = None, schedule_name : str | None = None ,
        stage = -1 , resume = -1 , selection = -1 , makedir = True , start : int | None = None , end : int | None = None , **kwargs
    ):
        
        self.start  = int(start) if start is not None else None
        self.end    = int(end) if end is not None else None
        self.Train  = TrainParam(base_path , override, schedule_name , **kwargs)
        self.Model  = self.Train.generate_model_param()

        self.process_parser(stage , resume , selection)
        assert self.Train.model_base_path , self.Train.model_name
        if not base_path and makedir: 
            self.makedir()

        self.device = Device(try_cuda = self.try_cuda)

    @classmethod
    def default(cls , module = None , override = None , stage = 0, resume = 0 , selection = 0 , makedir = False):
        if module:
            override = override or {}
            override['model.module'] = module
        return cls(None, override , stage = stage,resume=resume , selection=selection,makedir=makedir)
        
    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name})'
    
    @classmethod
    def load_model(cls , model_name : str | ModelPath | Path , override = None , stage = 2):
        ''' 
        load a existing model's config 
        stage is mostly irrelevant here, because mostly we are loading a model's config to pred
        '''
        model_path = ModelPath(model_name)
        assert model_path.base.exists() , f'{model_path.base} does not exist'
        return cls(model_path , override = override, stage = stage)

    def makedir(self):
        if 'fit' in self.stage_queue and not self.is_resuming:
            if self.model_base_path.base.exists(): 
                if not self.short_test and self.Train.resumeable:
                    raise Exception(f'{self.model_name} resumeable, re-train has to delete folder manually')
                shutil.rmtree(self.model_base_path.base)
            self.model_base_path.mkdir(model_nums = self.model_num_list , exist_ok=True)
            self.Train.copy_to(self.model_base_path , overwrite = self.short_test)
            self.Model.copy_to(self.model_base_path , overwrite = self.short_test)

    
    @property
    def Param(self) -> dict[str,Any]: 
        return self.Train.Param
    @property
    def model_base_path(self) -> ModelPath: 
        return self.Train.model_base_path
    @property
    def model_name(self) -> str | Any: 
        return self.Train.model_name
    @property
    def model_param(self) -> list[dict[str,Any]]: 
        return self.Model.params
    @property
    def model_num(self) -> int: 
        return self.Model.n_model
    @property
    def model_num_list(self) -> list[int]: 
        return list(range(self.Model.n_model))
    @property
    def booster_head_param(self) -> dict[str,Any]: 
        assert len(self.Model.booster_head_param.params) == 1 , self.Model.booster_head_param.params
        return self.Model.booster_head_param.params[0]
    @property
    def db_mapping(self) -> ModelDBMapping:
        return self.Model.db_mapping
    @property
    def beg_date(self) -> int: 
        beg_date = self.Train.beg_date
        if self.module_type == 'db':
            beg_date = max(beg_date , DB.min_date(self.db_mapping.src , self.db_mapping.key))
        elif self.module_type == 'factor':
            beg_date = max(beg_date , self.factor_factor_calculator.get_min_date())
        if self.start is not None:
            beg_date = max(beg_date , self.start)
        return beg_date
    @property
    def end_date(self) -> int: 
        end_date = self.Train.end_date
        if self.module_type == 'db':
            end_date = min(end_date , DB.max_date(self.db_mapping.src , self.db_mapping.key))
        elif self.module_type == 'factor':
            end_date = min(end_date , self.factor_factor_calculator.get_max_date())
        if self.end is not None:
            end_date = min(end_date , self.end)
        return end_date
    @property
    def model_labels(self) -> list[str]: 
        return self.Train.model_labels[:self.Model.max_num_output]

    def update(self, update = None , **kwargs):
        update = update or {}
        for k,v in update.items(): 
            setattr(self , k , v)
        for k,v in kwargs.items(): 
            setattr(self , k , v)
        return self
    
    def get(self , key , default = None) -> Any:
        return self.Param[key] if key in self.Train.Param else getattr(self , key , default)
    
    def set_config_environment(self , manual_seed = None) -> None:
        self.set_random_seed(manual_seed if manual_seed else self.random_seed)
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self.Param['env.allow_tf32']) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self.Param['env.detect_anomaly'])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict) -> None:
        '''
        when x_data is known , use it to fill some params(seq_len , input_dim , inday_dim , etc.) of nn module
        do it in whenever x_data is changed
        '''
        if self.module_type == 'nn' and x_data: 
            self.Model.update_data_param(x_data , self)
    
    def weight_scheme(self , stage : str , no_weight = False) -> str | None: 
        stg = stage if stage == 'fit' else 'test'
        return None if no_weight else self.Train.Param[f'train.criterion.weight.{stg}']
    
    def init_utils(self):
        self.metrics = Metrics(self.module_type , self.nn_category ,
                               self.train_criterion_loss , self.train_criterion_score , self.train_criterion_penalty ,
                               self.train_multilosses_type , self.train_multilosses_param)
        self.checkpoint = Checkpoint(self.mem_storage)
        self.deposition = Deposition(self.model_base_path)
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

    def parser_stage(self , value = -1 , verbose = True):
        """
        parser stage queue
        value:
            -1: choose
            0: fit + test
            1: fit only
            2: test only
        """
        stage_queue : list[Literal['data' , 'fit' , 'test']] = ['data' , 'fit' , 'test']
        if self.module_type in ['db' , 'factor']:
            stage_queue = ['data' , 'test']
        else:
            if value < 0:
                Logger.info(f'--What stage would you want to run? 0: fit + test, 1: fit only , 2: test only')
                value = int(input(f'[0,fit+test] , [1,fit] , [2,test]'))
            if value > 0:
                stage_queue = ['data' , stage_queue[value]]
            elif value < 0:
                raise Exception(f'Error input : {value}')
        if verbose:
            Logger.info('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]), stage_queue))))
        self._stage_queue = stage_queue

    def parser_resume(self , value = -1 , verbose = True):
        """
        parser resume flag
        value:
            -1: choose
            0: no
            1: yes
        """
        model_name = self.model_name
        assert model_name is not None , f'{self} has model_name None'
        candidate_name = sorted([m.name for m in self.model_root_path.iterdir() if m.name.split('.')[0] == model_name]) 
        if value < 0:
            if not candidate_name:
                value = 0
            else:
                Logger.info(f'--Multiple model path of {model_name} exists, input [yes] to resume training, or start a new one!')
                user_input = input(f'Confirm resume training [{model_name}]? [yes/no] : ')
                value = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
        self.is_resuming = value > 0 
        if verbose:
            Logger.info(f'--Confirm Resume Training!' if self.is_resuming else '--Start Training New!')

    def parser_select(self , value = -1 , verbose = True):
        '''
        parse model_name selection if model_name dirs exists
        value:
            if no model_name dir exists:
                skip this section
            elif fit is in stage_queue:
                -1: choose (if model_name dirs exists, ask to choose one)
                0: raw model_name dir if is_resuming , create a new model_name dir otherwise
                1,2,3,...: choose one by number, start from 1 (if not is_resuming , raise error)
            elif test only:
                -1: choose (if more than 1 model_name dirs exists, ask to choose one)
                0: try to use the raw model_name dir
                1,2,3,...: choose one by number, start from 1
        '''
        model_name = self.model_name
        assert model_name is not None , f'{self} has model_name None'
        candidate_name = sorted([m.name for m in self.model_root_path.iterdir() 
                                 if m.name == model_name or m.name.startswith(f'{model_name}.')])
        if self.short_test or self.module_type in ['db' , 'factor'] or not candidate_name:
            ...
        elif 'fit' in self.stage_queue:
            if value < 0 or (value == 0 and self.is_resuming and model_name not in candidate_name):
                Logger.info(f'--Model dirs of {model_name} exists, input number to choose!')
                if self.is_resuming:
                    Logger.info(f'    0: use the raw model_name [{model_name}] to resume training!')
                else:
                    Logger.info(f'    0: create a new model_name dir!')
                for i , mn in enumerate(candidate_name):
                    Logger.info(f'    {i+1}: [{mn}]')
                value = int(input(f'which one to use? '))
            if value < 0:
                raise Exception(f'value {value} is out of range , must be 0 ~ {len(candidate_name)}')
            elif value == 0: 
                if self.is_resuming:
                    if model_name not in candidate_name:
                        Logger.error(f'The raw model_name [{model_name}] does not exist! You have to start a new training or manually delete the existing model_name dir!')
                        raise Exception(f'the raw model_name [{model_name}] does not exist!')
                    if self.verbosity > 0:
                        Logger.info(f'Input 0 to use the raw model_name [{model_name}] to resume training!')
                else:
                    if model_name in candidate_name:
                        model_name += '.'+str(max([1]+[int(model.split('.')[-1])+1 for model in candidate_name[1:]]))
                    Logger.info(f'Input 0 to create a new model_name dir! New model_name is {model_name}')
            else:
                model_name = candidate_name[value-1]
                if not self.is_resuming:
                    Logger.error(f'You are not resuming, cannot choose model_name {model_name}! You have to start a new training or manually delete the existing model_name dir!')
                    raise Exception(f'you are not resuming, cannot choose model_name {model_name}')
        elif 'test' in self.stage_queue:
            if len(candidate_name) == 1 and candidate_name[0] == model_name:
                value = 0 # use the raw model_name dir
            elif value < 0 or (value == 0 and model_name not in candidate_name):
                Logger.info(f'--Model dirs of {model_name} exists, input number to choose!')
                Logger.info(f'    0: try to use raw model_name [{model_name}]!')
                for i , mn in enumerate(candidate_name):
                    Logger.info(f'    {i+1}: [{mn}]')
                value = int(input(f'which one to use? '))
            if value < 0:
                raise Exception(f'value {value} is out of range , must be 0 ~ {len(candidate_name)}')
            elif value == 0: 
                if model_name not in candidate_name:
                    Logger.error(f'The raw model_name [{model_name}] does not exist! You have to fit one or select a different model_name!')
                    raise Exception(f'the raw model_name [{model_name}] does not exist!')
            else:
                model_name = candidate_name[value-1]
        else:
            raise Exception(f'Invalid stage queue: {self.stage_queue}')

        if verbose:
            Logger.info(f'--Model_name is set to {model_name}!')  
        self.Train.reset_base_path(model_name)
        self.Model.reset_base_path(model_name)

    def process_parser(self , stage = -1 , resume = -1 , selection = -1 , verbose = True):
        if self.model_base_path:
            if resume == -1 or resume == 1:
                resume = 1
            else:
                raise ValueError(f'resume must be -1 or 1 when base_path is not None , got {resume}')
            if selection == -1 or selection == 0:
                selection = 0
            else:
                raise ValueError(f'selection must be -1 or 0 when base_path is not None , got {selection}')
            verbose = False
        
        self.parser_stage(stage , verbose)
        self.parser_resume(resume , verbose)
        self.parser_select(selection , verbose) 
        return self

    def print_out(self):
        info_strs = []
        info_strs.append(f'Model Name   : {self.model_name}')
        if self.module_type in ['db' , 'factor']:
            info_strs.append(f'Model Labels : {self.model_labels}')
            info_strs.append(f'Model Period : {self.beg_date} ~ {self.end_date}')
            info_strs.append(f'Resuming     : {self.is_resuming}')
            info_strs.append(f'Verbosity    : {self.verbosity}')
            
        else:
            info_strs.append(f'Model Module : {self.model_module}')
            if self.module_type == 'booster':
                if self.model_module != self.model_booster_type:
                    info_strs.append(f'  --> Booster Type   : {self.model_booster_type}')
                if self.model_booster_optuna:
                    info_strs.append(f'  --> Booster Params :  Optuna for {self.model_booster_optuna_n_trials} trials')
                else:
                    info_strs.append(f'  --> Booster Params :')
                    for k , v in self.Model.Param.items():
                        info_strs.append(f'    --> {k} : {v}')
            else:
                if self.model_booster_head:
                    info_strs.append(f'  --> Use Booster Head : {self.model_booster_head}')
                info_strs.append(f'  --> Model Params :')
                for k , v in self.Model.Param.items():
                    info_strs.append(f'    --> {k} : {v}')
            info_strs.append(f'Model Num    : {self.Model.n_model}')
            info_strs.append(f'Model Inputs : {self.model_input_type}')
            if self.model_input_type == 'data':
                info_strs.append(f'  --> Data Types : {self.model_data_types}')
            elif self.model_input_type == 'hidden':
                info_strs.append(f'  --> Hidden Models : {self.model_hidden_types}')
            elif self.model_input_type == 'factor':
                info_strs.append(f'  --> Factor Types : {self.model_factor_types}')
                if self.input_factor_names:
                    info_strs.append(f'  --> Factor Names : {self.input_factor_names}')
            elif self.model_input_type == 'combo':
                info_strs.append(f'  --> Combo Types : {self.model_combo_types}')
            info_strs.append(f'Model Labels : {self.model_labels}')
            info_strs.append(f'Model Period : {self.beg_date} ~ {self.end_date}')
            info_strs.append(f'Interval     : {self.model_interval} days')
            info_strs.append(f'Train Window : {self.model_train_window} days')
            info_strs.append(f'Sampling     : {self.train_sample_method}')
            info_strs.append(f'Shuffling    : {self.train_shuffle_option}')
            info_strs.append(f'Random Seed  : {self.random_seed}')
            info_strs.append(f'Stage Queue  : {self.stage_queue}')
            info_strs.append(f'Resuming     : {self.is_resuming}')
            info_strs.append(f'Verbosity    : {self.verbosity}')

        
        Logger.stdout('\n'.join(info_strs))
        return self

    @property
    def stage_queue(self) -> list[Literal['data' , 'fit' , 'test']]:
        """
        stage queue for training
        if module_type is db, fit stage is not included
        """
        if not hasattr(self , '_stage_queue'):
            return []
        else:
            return self._stage_queue
