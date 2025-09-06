import argparse , os , random , shutil , torch
import numpy as np

from pathlib import Path
from typing import Any , Literal , Optional

from src.proj import PATH , MACHINE , Logger
from src.basic import Device , ModelPath
from src.func import recur_update
from src.res.algo import AlgoModule

from .metrics import Metrics
from .storage import Checkpoint , Deposition

TRAIN_CONFIG_LIST = ['train' , 'env' , 'callbacks' , 'conditional' , 'model']

def conf_dir(base_path : ModelPath | Path | None , *args):
    base_path = ModelPath(base_path)
    f = base_path.conf if base_path else PATH.conf.joinpath
    path = f(*args)
    return path

def conf_path(base_path : ModelPath | Path | None , *args):
    base_path = ModelPath(base_path)
    f = base_path.conf if base_path else PATH.conf.joinpath
    path = f(*args).with_suffix('.yaml')
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

def conf_mod_type(module : str):
    return AlgoModule.module_type(module)

def schedule_config(base_path : ModelPath | Path | None , name : str | None):
    # schedule configs are used to override the train config
    base_path = ModelPath(base_path)
    if base_path:
        schedules = list(base_path.conf('schedule').glob('*.yaml'))
        assert len(schedules) <= 1 , f'multiple schedules found: {schedules}'
        name = schedules[0].stem if schedules else None
    p : dict[str,Any] = {}
    if name: p.update(PATH.read_yaml(conf_path(base_path , 'schedule', name)))
    return p

class TrainParam:
    def __init__(self , base_path : ModelPath | Path | None , override = None , schedule_name : str | None = None):
        self.base_path = ModelPath(base_path)
        self.override = override or {}
        self.model_name = self.base_path.name
        self.schedule_name = schedule_name
        self.load_param().check_validity()

    def __bool__(self): return True
    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name})'

    @property
    def model_base_path(self):
        # assert self.base_path
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
        if 'verbosity'  in self.override: self.override['env.verbosity']  = self.override.pop('verbosity')
        if 'short_test' in self.override: self.override['env.short_test'] = self.override.pop('short_test')
        if 'module'     in self.override: self.override['model.module']   = self.override.pop('module')
        if self.should_be_short_test and ('env.short_test' not in self.override): self.override['env.short_test'] = True
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
        if self.model_name: return self
        if self.Param['model.name']: 
            self.model_name = str(self.Param['model.name'])
        else: 
            mod_str = self.model_module 
            head_str = 'booster' if self.model_booster_head else None
            if self.model_input_type == 'data':
                data_str = '+'.join(self.model_data_types)
            else:
                data_str = self.model_input_type
            self.model_name = '_'.join([s for s in [mod_str , head_str , data_str] if s])
        if self.short_test: self.model_name += '_ShortTest'
        return self
    
    def check_validity(self):
        if self.should_be_short_test and not self.short_test:
            Logger.warning('Beware! Should be at server or short_test, but short_test is False now!')

        nn_category = AlgoModule.nn_category(self.model_module)
        if nn_category == 'tra': assert self.train_sample_method != 'total_shuffle' , self.train_sample_method
        if nn_category == 'vae': assert self.train_sample_method == 'sequential'    , self.train_sample_method

        nn_datatype = AlgoModule.nn_datatype(self.model_module)
        if nn_datatype:  self.Param['model.data.types'] = nn_datatype


        if self.module_type != 'nn' or self.model_booster_head: 
            self.Param['model.submodels'] = ['best']
        if 'best' not in self.model_submodels: 
            self.model_submodels.insert(0 , 'best')

        if self.model_input_type != 'data' or self.module_type != 'nn':
            assert self.train_sample_method == 'sequential' , self.train_sample_method

        return self
    
    def generate_model_param(self , update_inplace = True , **kwargs):
        module = self.model_module if self.module_type == 'nn' else self.model_booster_type
        assert isinstance(module , str) , (self.model_module , module)
        model_param = ModelParam(self.base_path , module , self.model_booster_head , self.verbosity , self.short_test , self.schedule_name , **kwargs).expand()
        if update_inplace: self.update_model_param(model_param)
        return model_param
    
    def update_model_param(self , model_param : 'ModelParam'):
        param = {k:v for k,v in model_param.Param.items() if k in self.Param}
        self.Param.update(param)
        return self
    
    def copy_to(self , target_dir : Path | ModelPath , override = False):
        if self.base_path: raise Exception(f'Only copy TrainParam with from None base_path: {self.base_path.base}')
        target_dir = ModelPath(target_dir)

        source = conf_dir(None , 'train')
        target = conf_dir(target_dir , 'train')
        conf_copy(source , target , override)

        self.base_path , self.model_name = target_dir , target_dir.name

        if self.schedule_name:
            source = conf_path(None , 'schedule' , self.schedule_name) 
            target = conf_path(target_dir , 'schedule' , self.schedule_name)
            conf_copy(source , target , override)

    @classmethod
    def guess_module(cls , base_path : Path | ModelPath | None):
        return str(PATH.read_yaml(conf_path(base_path , 'train', 'model'))['module']).lower()
    
    @staticmethod
    def get_module_type(module : str):
        return conf_mod_type(module)
    
    @property
    def module_type(self): return conf_mod_type(self.model_module)
    @property
    def nn_category(self): return AlgoModule.nn_category(self.model_module)
    @property
    def resumeable(self): 
        assert self.model_name
        model_path = ModelPath(self.model_name)
        return all([model_path.conf(cfg_key).exists() for cfg_key in TRAIN_CONFIG_LIST])

    @property
    def short_test(self): return bool(self.Param['env.short_test'])
    @short_test.setter
    def short_test(self , value : bool):
        self.Param['env.short_test'] = value
    @property
    def verbosity(self): return int(self.Param['env.verbosity'])
    @property
    def random_seed(self): return self.Param['env.random_seed']
    @property
    def mem_storage(self): return bool(self.Param['env.mem_storage'])
    @property
    def precision(self) -> Any:
        prec = self.Param['env.precision']
        return getattr(torch , prec) if isinstance(prec, str) else prec

    @property
    def beg_date(self): return int(self.Param['model.beg_date'])
    @property
    def end_date(self): return int(self.Param['model.end_date'])
    @property
    def model_rslt_path(self): return self.model_base_path.rslt()
    @property
    def model_submodels(self) -> list: return self.Param['model.submodels']
    @property
    def model_module(self): return str(self.Param['model.module']).lower()
    @property
    def model_input_type(self) -> Literal['data' , 'hidden' , 'factor']: 
        assert self.Param['model.input_type'] in ['data' , 'hidden' , 'factor'] , self.Param['model.input_type']
        return self.Param['model.input_type']
    @property
    def model_labels(self) -> list[str]: return self.Param['model.labels']
    @property
    def model_data_types(self) -> list[str]: 
        if isinstance(self.Param['model.data.types'] , str):
            return str(self.Param['model.data.types']).split('+')
        else:
            return list(self.Param['model.data.types'])
    @model_data_types.setter
    def model_data_types(self , value : list[str]):
        self.Param['model.data.types'] = value
    @property
    def model_data_prenorm(self) -> dict[str,Any]: return self.Param['model.data.prenorm']
    @property
    def model_hidden_types(self) -> list[str]: return self.Param['model.hidden.types']
    @property
    def model_factor_types(self) -> list[str]: 
        factors = self.Param.get('model.factor.types' , [])
        def striped_list(factors : list[str] | dict | str):
            if isinstance(factors , str): 
                return [factors.strip()]
            else:
                if isinstance(factors , dict): factors = list(factors.values())
                return [ff for f in factors for ff in striped_list(f)]
        return striped_list(factors)
    @property
    def model_train_window(self): 
        if (train_window := int(self.Param.get('model.train_window', 0))) > 0:
            return train_window
        else:
            return int(self.Param[f'model.train_window.{self.module_type}'])
    @property
    def model_interval(self): 
        if (interval := int(self.Param.get('model.interval', 0))) > 0:
            return interval
        else:
            return int(self.Param[f'model.interval.{self.module_type}'])
    @property
    def model_booster_head(self): return self.Param['model.booster_head']
    @property
    def model_booster_type(self):
        if self.model_module in ['booster' , 'hidden_aggregator']:
            assert AlgoModule.is_valid(self.Param['model.booster_type'] , 'booster') , self.Param['model.booster_type']
            return self.Param['model.booster_type']
        elif AlgoModule.is_valid(self.model_module, 'booster'):
            return self.model_module
        else:
            return 'not_a_booster'
    @property
    def model_booster_optuna(self): return bool(self.Param.get('model.booster_optuna'))
    @property
    def model_booster_optuna_n_trials(self): return int(self.Param.get('model.booster_optuna_n_trials',10))
    @property
    def train_data_step(self): return int(self.Param['train.data_step'])
    @property
    def train_train_ratio(self): return float(self.Param['train.dataloader.train_ratio'])
    @property
    def train_sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self.Param['train.dataloader.sample_method']
    @property
    def train_shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self.Param['train.dataloader.shuffle_option']
    @property
    def train_batch_size(self): return int(self.Param['train.batch_size'])
    @property
    def train_max_epoch(self): return int(self.Param['train.max_epoch'])
    @property
    def train_skip_horizon(self): return int(self.Param['train.skip_horizon'])
    @property
    def train_trainer_transfer(self) -> bool: return self.Param['train.trainer.transfer']
    @property
    def train_criterion_loss(self) -> Any: return self.Param['train.criterion.loss']
    @property
    def train_criterion_score(self) -> Any: return self.Param['train.criterion.score']
    @property
    def train_criterion_penalty(self) -> dict[Any,Any]: return self.Param['train.criterion.penalty']
    @property
    def train_multilosses_type(self) -> Any: return self.Param['train.multilosses.type']
    @property
    def train_multilosses_param(self) -> dict: return self.Param[f'train.multilosses.param.{self.train_multilosses_type}']
    @property
    def train_trainer_optimizer(self) -> dict[str,Any]: return self.Param['train.trainer.optimizer']
    @property
    def train_trainer_scheduler(self) -> dict[str,Any]: return self.Param['train.trainer.scheduler']
    @property
    def train_trainer_learn_rate(self) -> dict[str,Any]: return self.Param['train.trainer.learn_rate']
    @property
    def train_trainer_gradient_clip_value(self) -> Any: return self.Param['train.trainer.gradient.clip_value']

    @property
    def callbacks(self) -> dict[str,dict]: 
        return {k.replace('callbacks.',''):v for k,v in self.Param.items() if k.startswith('callbacks.')}
        
class ModelParam:
    def __init__(self , base_path : Optional[Path | ModelPath] , module : str , 
                 booster_head : Any = False , verbosity = 2 , short_test : bool | None = None , 
                 schedule_name : str | None = None,
                 **kwargs):
        self.base_path = ModelPath(base_path)
        self.model_name = self.base_path.name
   
        self.module = module.lower()
        self.booster_head = booster_head
        self.short_test = short_test
        self.verbosity = verbosity
        self.schedule_name = schedule_name
        self.override = kwargs
        self.load_param().check_validity()

    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name})'

    @property
    def model_base_path(self):
        assert self.base_path is not None
        return self.base_path

    @property
    def Param(self) -> dict[str,Any]: return self.model_param

    @property
    def module_type(self): return conf_mod_type(self.module)

    @property
    def mod_type_dir(self):
        return 'boost' if self.module_type == 'booster' else self.module_type

    def conf_file(self , base : Path | ModelPath | None | Literal['self'] = 'self'):
        if base == 'self': base = self.base_path
        return conf_path(base , self.mod_type_dir , self.module)

    def load_param(self):
        self.model_param : dict[str,Any] = {}
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

        lens = [len(v) for v in self.Param.values() if isinstance(v , (list,tuple))]
        self.n_model = max(lens) if lens else 1
        if self.short_test: self.n_model = min(1 , self.n_model)
        assert self.n_model <= 5 , self.n_model
        
        if self.module == 'tra':
            assert 'hist_loss_seq_len' in self.Param
            assert 'hist_loss_horizon' in self.Param

        if self.booster_head:
            assert AlgoModule.is_valid(self.booster_head , 'booster') , self.booster_head
            self.booster_head_param = ModelParam(self.base_path , self.booster_head , False , self.verbosity , **self.override)
        return self
    
    def get(self , key : str , default = None):
        return self.Param.get(key , default)
    
    def copy_to(self , target_dir : Path | ModelPath , override = False):
        if self.base_path: raise Exception(f'Only copy TrainParam with from None base_path: {self.base_path.base}')

        target_dir = ModelPath(target_dir)
        source = self.conf_file(None)
        target = self.conf_file(target_dir)
        conf_copy(source , target , override)

        self.base_path , self.model_name = target_dir , target_dir.name

        if self.booster_head: self.booster_head_param.copy_to(target_dir , override = override)
        return self

    def expand(self):
        self.params : list[dict[str,Any]] = []
        for mm in range(self.n_model):
            par = {k:v[mm % len(v)] if isinstance(v , (list, tuple)) else v for k,v in self.Param.items()}
            self.params.append(par)

        if self.booster_head: self.booster_head_param.expand()
        return self
       
    def update_data_param(self , x_data : dict):
        '''when x_data is know , use it to fill some params(seq_len , input_dim , inday_dim , etc.)'''
        if not x_data: return self
        keys = list(x_data.keys())
        input_dim = [x_data[mdt].shape[-1] for mdt in keys]
        inday_dim = [x_data[mdt].shape[-2] for mdt in keys]
        for param in self.params:
            self.update_param_dict(param , 'input_dim' , input_dim)
            self.update_param_dict(param , 'inday_dim' , inday_dim)
            if len(keys) == 1: self.update_param_dict(param , 'seq_len' , param.get('seqlens',{}).get(keys[0]))
        return self
    
    @staticmethod
    def update_param_dict(param , key : str , value , delist = True , overwrite = False):
        if key in param.keys() and not overwrite: return
        if delist and isinstance(value , (list , tuple)) and len(value) == 1: value = value[0]
        if value is not None: param.update({key : value})
    
    @property
    def max_num_output(self): return max(self.Param.get('num_output' , [1]))

class TrainConfig(TrainParam):
    def __init__(self , base_path : Path | ModelPath | None , override = None, schedule_name : str | None = None):
        self.resume_training: bool  = False
        self.stage_queue: list      = []

        self.device     = Device()
        self.logger     = Logger()

        self.Train = TrainParam(base_path , override, schedule_name)
        self.Model = self.Train.generate_model_param()
        
    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name})'
        
    def resume_old(self , old_path : Path | ModelPath , override = None):
        self.Train = TrainParam(old_path , override)
        self.Model = self.Train.generate_model_param()

    def start_new(self , new_path : Path | ModelPath):
        new_path = ModelPath(new_path.name)
        assert not self.Train.resumeable or self.short_test , f'{new_path.base} has to be delete manually'

        new_path.mkdir(model_nums = self.model_num_list , exist_ok=True)
        self.Train.copy_to(new_path , override = self.short_test)
        self.Model.copy_to(new_path , override = self.short_test)
        return self

    @classmethod
    def load(cls , base_path : Optional[Path] = None , do_parser = False , 
             override = None , schedule_name = None , makedir = True , **kwargs):
        '''load config yaml to get default/giving params'''
        config = cls(base_path , override, schedule_name)
        if do_parser: config.process_parser(config.parser_args(**kwargs))

        model_path = ModelPath(config.model_name)
        if config.resume_training:
            config.resume_old(model_path , override)
        elif 'fit' in config.stage_queue and makedir:
            config.start_new(model_path)
            
        return config
    
    @classmethod
    def load_model(cls , model_name : str | ModelPath | Path , override = None):
        return cls.load(ModelPath(model_name).base , override = override)
    
    @property
    def Param(self) -> dict[str,Any]: return self.Train.Param
    @property
    def model_base_path(self): return self.Train.model_base_path
    @property
    def model_name(self) -> str|Any: return self.Train.model_name
    @property
    def model_param(self): return self.Model.params
    @property
    def model_num(self): return self.Model.n_model
    @property
    def model_num_list(self) -> list[int]: return list(range(self.Model.n_model))
    @property
    def booster_head_param(self): 
        assert len(self.Model.booster_head_param.params) == 1 , self.Model.booster_head_param.params
        return self.Model.booster_head_param.params[0]

    def update(self, update = None , **kwargs):
        update = update or {}
        for k,v in update.items(): setattr(self , k , v)
        for k,v in kwargs.items(): setattr(self , k , v)
        return self

    def reload(self , base_path : Path | None = None , do_parser = False , override = None , **kwargs):
        new_config = self.load(base_path,do_parser,override,**kwargs)
        self.__dict__ = new_config.__dict__
        return self
    
    def get(self , key , default = None):
        return self.Param[key] if key in self.Train.Param else getattr(self , key , default)
    
    def set_config_environment(self , manual_seed = None):
        self.set_random_seed(manual_seed if manual_seed else self.random_seed)
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self.Param['env.allow_tf32']) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self.Param['env.detect_anomaly'])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict):
        if self.module_type == 'nn' and x_data: self.Model.update_data_param(x_data)
    
    def weight_scheme(self , stage : str , no_weight = False) -> Optional[str]: 
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
        if seed is None: return NotImplemented
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 重复加载libiomp5md.dll https://zhuanlan.zhihu.com/p/655915099
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    @classmethod
    def parser_args(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        parser = argparse.ArgumentParser(description='manual to this script')
        parser.add_argument(f'--stage', type=int, default = stage)
        parser.add_argument(f'--resume', type=int, default = resume)
        parser.add_argument(f'--checkname', type=int, default = checkname)
        args , _ = parser.parse_known_args()
        return args

    def parser_stage(self , value = -1):
        if value < 0:
            Logger.info(f'--What stage would you want to run? 0: fit + test, 1: fit only , 2: test only')
            value = int(input(f'[0,fit+test] , [1,fit] , [2,test]'))
        stage_queue = ['data' , 'fit' , 'test']
        if value > 0:
            stage_queue = ['data' , stage_queue[value]]
        elif value < 0:
            raise Exception(f'Error input : {value}')
        Logger.info('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]), stage_queue))))
        self.stage_queue = stage_queue

    def parser_resume(self , value = -1):
        '''ask if resume training when candidate names exists'''
        model_name = self.model_name
        assert model_name is not None
        candidate_name = sorted([m.name for m in PATH.model.iterdir() if m.name.split('.')[0] == model_name])
        if len(candidate_name) > 0 and 'fit' in self.stage_queue:
            if value < 0:
                Logger.info(f'--Multiple model path of {model_name} exists, input [yes] to resume training, or start a new one!')
                user_input = input(f'Confirm resume training [{model_name}]? [yes/no] : ')
                value = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
            self.resume_training = value > 0 
            Logger.info(f'--Confirm Resume Training!' if self.resume_training else '--Start Training New!')
        else:
            self.resume_training = False

    def parser_select(self , value = -1):
        '''
        checkname confirmation
        Confirm the model_name if multifple model_name dirs exists.
        if train:
            if zero or (resume and single): do it
            elif resume and multiple: ask to choose one
            elif not resume and single/multiple: ask to make new folder
        else:
            if multiple: ask to choose one
            elif zero: raise
        '''
        model_name = self.model_name
        assert model_name is not None
        candidate_name = sorted([m.name for m in PATH.model.iterdir() 
                                 if m.name == model_name or m.name.startswith(f'{model_name}.')])
        if self.short_test:
            ...
        elif 'fit' in self.stage_queue and candidate_name:
            if self.resume_training and len(candidate_name) == 1:
                model_name = candidate_name[0]
            elif self.resume_training:
                if value < 0:
                    Logger.info(f'--Attempting to resume but multiple models exist, input number to choose')
                    [Logger.info(f'{i} : {PATH.model}/{model}') for i , model in enumerate(candidate_name)]
                    value = int(input('which one to use? '))
                model_name = candidate_name[value]
            else:
                if value < 0:
                    Logger.info(f'--Model dirs of {model_name} exists, input [yes] to add a new directory!')
                    user_input = input(f'Add a new folder of [{model_name}]? [yes/no] : ').lower()
                    value = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
                if value == 0: raise Exception(f'--Model dirs of [{model_name}] exists!')
                model_name += '.'+str(max([1]+[int(model.split('.')[-1])+1 for model in candidate_name[1:]]))

        elif 'fit' not in self.stage_queue and 'test' in self.stage_queue:
            assert len(candidate_name) > 0 , f'no models of {model_name} while you want to test'
            if len(candidate_name) == 1:
                model_name = candidate_name[0]
            else:
                if value < 0:
                    Logger.info(f'--Attempting to test while multiple models exists, input number to choose')
                    [Logger.info(f'{i} : {PATH.model}/{model}') for i , model in enumerate(candidate_name)]
                    value = int(input('which one to use? '))
                model_name = candidate_name[value]

        Logger.info(f'--Model_name is set to {model_name}!')  
        self.Train.model_name = model_name

    def process_parser(self , par_args = None):
        par_args = par_args or {}
        with Logger.EnclosedMessage(' parser training args '):
            self.parser_stage(getattr(par_args , 'stage' , -1))
            self.parser_resume(getattr(par_args , 'resume' , -1))
            self.parser_select(getattr(par_args , 'checkname' , -1)) 
        return self

    def print_out(self):
        info_strs = []
        info_strs.append(f'Model Name   : {self.model_name}')
        info_strs.append(f'Model Module : {self.model_module}')
        if self.module_type == 'booster':
            if self.model_module != self.model_booster_type:
                info_strs.append(f'  -->  Booster Type   : {self.model_booster_type}')
            if self.model_booster_optuna:
                info_strs.append(f'  -->  Booster Params :  Optuna for {self.model_booster_optuna_n_trials} trials')
            else:
                info_strs.append(f'  -->  Booster Params :')
                for k , v in self.Model.Param.items():
                    info_strs.append(f'    -->  {k} : {v}')
        else:
            if self.model_booster_head:
                info_strs.append(f'  -->  Use Booster Head : {self.model_booster_head}')
            info_strs.append(f'  -->  Model Params :')
            for k , v in self.Model.Param.items():
                info_strs.append(f'    -->  {k} : {v}')
        info_strs.append(f'Model Num    : {self.Model.n_model}')
        info_strs.append(f'Model Inputs : {self.model_input_type}')
        if self.model_input_type == 'data':
            info_strs.append(f'  -->  Data Types : {self.model_data_types}')
        elif self.model_input_type == 'hidden':
            info_strs.append(f'  -->  Hidden Models : {self.model_hidden_types}')
        elif self.model_input_type == 'factor':
            info_strs.append(f'  -->  Factor Types : {self.model_factor_types}')
        info_strs.append(f'Model Labels : {self.model_labels}')
        info_strs.append(f'Model Period : {self.beg_date} ~ {self.end_date}')
        info_strs.append(f'Interval     : {self.model_interval} days')
        info_strs.append(f'Train Window : {self.model_train_window} days')
        info_strs.append(f'Sampling     : {self.train_sample_method}')
        info_strs.append(f'Shuffling    : {self.train_shuffle_option}')
        info_strs.append(f'Random Seed  : {self.random_seed}')

        with Logger.EnclosedMessage(' model info '): 
            Logger.print('\n'.join(info_strs))
