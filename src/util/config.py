import argparse , itertools , os , random , shutil , socket
import numpy as np
import torch

from dataclasses import dataclass , field
from typing import Any , ClassVar , Literal , Optional

from ..func.basic import pretty_print_dict , recur_update , Filtered
from ..environ import DIR

@dataclass    
class ModelParam:
    config_path : str
    module      : str
    Param       : dict         = field(default_factory=dict)
    n_model     : int          = 0
    params      : list[dict]   = field(default_factory=list)

    model_yaml  : ClassVar[str] = 'model_{}.yaml'
    default_yaml: ClassVar[str] = 'model_default.yaml'
    inday_dims  : ClassVar[dict] = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}

    def __post_init__(self) -> None:
        source_dir = DIR.conf if self.config_path == 'default' else self.config_path
        source_base = self.model_yaml.format(self.module.lower())
        if not os.path.exists(f'{source_dir}/{source_base}'): source_base = self.default_yaml
        self.Param = DIR.read_yaml(f'{source_dir}/{source_base}')
        assert isinstance(self.Param , dict)
        for key , value in self.Param.items():
            if isinstance(value , (list,tuple)): 
                self.n_model = max(self.n_model , len(value))
            else:
                self.Param[key] = [value]
        assert self.n_model <= 4 , self.n_model

    def __getitem__(self , key : str):
        return self.Param[key]
    
    def get(self , key : str , default = None):
        return self.Param.get(key , default)
    
    def copy_to(self , target_dir , exist_ok = False):
        source_dir = DIR.conf if self.config_path == 'default' else self.config_path
        target_base = self.model_yaml.format(self.module.lower())
        source_base = target_base if os.path.exists(f'{source_dir}/{target_base}') else self.default_yaml
        os.makedirs(target_dir, exist_ok = True)
        if not exist_ok: assert not os.path.exists(f'{target_dir}/{target_base}')
        shutil.copyfile(f'{source_dir}/{source_base}' , f'{target_dir}/{target_base}')

    def expand(self , base_path):
        self.params = [{'path':f'{base_path}/{mm}' , **{k:v[mm % len(v)] for k,v in self.Param.items()}} for mm in range(self.n_model)]

    @classmethod
    def load(cls , base_path , model_num : int | None = None):
        Param = DIR.read_yaml(f'{base_path}/{cls.model_yaml}')
        assert isinstance(Param , dict)
        n_model = 1
        for key , value in Param.items():
            if isinstance(value , (list,tuple)): 
                n_model = max(n_model , len(value))
            else:
                Param[key] = [value]
        param = [{'path':f'{base_path}/{mm}' , **{k:v[mm % len(v)] for k,v in Param.items()}} for mm in range(n_model)]
        if model_num is not None: param = param[model_num]
        return param

    def update_data_param(self , x_data : dict):
        '''when x_data is know , use it to fill some params(seq_len , input_dim , inday_dim , etc.)'''
        if not x_data: return self
        keys = list(x_data.keys())
        input_dim = [x_data[mdt].shape[-1] for mdt in keys]
        inday_dim = [x_data[mdt].shape[-2] for mdt in keys]
        for param in self.params:
            self.update_param_dict(param , 'input_dim' , input_dim)
            self.update_param_dict(param , 'inday_dim' , inday_dim)
            if len(keys) == 1: self.update_param_dict(param , 'seq_len'   , param.get('seqlens',{}).get(keys[0]))
        return self
    
    @staticmethod
    def update_param_dict(param , key : str , value , delist = True , overwrite = False):
        if key in param.keys() and not overwrite: return
        if delist and isinstance(value , (list , tuple)) and len(value) == 1: value = value[0]
        if value is not None: param.update({key : value})
    
    @property
    def max_num_output(self):
        return max(self.Param.get('num_output' , [1]))
    
@dataclass    
class TrainParam:
    config_path : str
    spec_adjust : bool = True
    configs     : dict = field(default_factory=dict)
    train_param : dict = field(default_factory=dict)
    model_name  : str | None = None
    override    : dict | None = None

    train_yaml  : ClassVar[str] = 'train_param.yaml'

    def __post_init__(self) -> None:
        source_dir = DIR.conf if self.config_path == 'default' else self.config_path
        source_base = self.train_yaml
        Param : dict = DIR.read_yaml(f'{source_dir}/{source_base}')
        if self.override: Param.update(self.override)
        if socket.gethostname() != 'mengkjin-server': Param['short_test'] = True

        if self.spec_adjust:
            if Param['short_test'] and Param.get('on_short_test'): 
                recur_update(Param , Param['on_short_test'])
            if Param['model_module'].lower() == 'transformer' and Param.get('on_transformer'):
                recur_update(Param , Param['on_transformer'])

        assert 'best' in Param['model_types']
        Param['tra_model'] = Param['tra_switch'] and Param['model_module'].lower().startswith('tra')
        assert not (Param['tra_model'] and Param['train_param']['dataloader']['sample_method'] == 'total_shuffle')

        if self.model_name is None:
            self.model_name = '_'.join([Param['model_module'].lower() , Param['model_data_type']])
            if Param['short_test']: self.model_name += '_ShortTest'

        self.train_param = Param['train_param']
        self.configs = {k:v for k,v in Param.items() if k != 'train_param'}

    def __getitem__(self , key : str):
        return self.configs[key]
    
    def copy_to(self , target_dir , exist_ok = False):
        source_dir = DIR.conf if self.config_path == 'default' else self.config_path
        target_base = source_base = self.train_yaml
        os.makedirs(target_dir, exist_ok = True)
        if not exist_ok: assert not os.path.exists(f'{target_dir}/{target_base}')
        shutil.copyfile(f'{source_dir}/{source_base}' , f'{target_dir}/{target_base}')

    @property
    def model_base_path(self) -> str: return f'{DIR.model}/{self.model_name}'
    @property
    def resumeable(self) -> bool: return os.path.exists(f'{self.model_base_path}/{self.train_yaml}')
    @property
    def model_module(self) -> str: return self.configs['model_module'].lower()

@dataclass
class TrainConfig:
    short_test: bool        = False
    model_name: str | None  = None
    model_module: str       = ''
    model_data_type: str    = 'day' 
    model_data_prenorm: dict= field(default_factory=dict)
    model_types: list       = field(default_factory=list)
    labels: list            = field(default_factory=list)
    callbacks: dict[str,Any]= field(default_factory=dict)
    beg_date: int           = 20170103
    end_date: int           = 99991231
    input_span: int         = 2400
    interval: int           = 120
    max_epoch: int          = 200
    verbosity: int          = 2
    batch_size: int         = 10000
    input_step_day: int     = 5
    skip_horizon: int       = 20

    mem_storage: bool       = True
    random_seed: int | None = None
    allow_tf32: bool        = True
    detect_anomaly: bool    = False
    precision: Any          = torch.float # double , bfloat16

    tra_switch:  bool       = True
    tra_param: dict         = field(default_factory=dict)
    tra_model: bool         = False
    buffer_type: str | None = 'tra'
    buffer_param: dict      = field(default_factory=dict)
    
    on_short_test: dict     = field(default_factory=dict)
    on_transformer: dict     = field(default_factory=dict)

    resume_training: bool   = True
    stage_queue: list       = field(default_factory=list)
    
    _TrainParam: Optional[TrainParam] = None
    _ModelParam: Optional[ModelParam] = None

    def __post_init__(self):
        if isinstance(self.precision , str): self.precision = getattr(torch , self.precision)
        self.stage_queue = ['data' , 'fit' , 'test']
        if not self.tra_model or self.buffer_type != 'tra': self.buffer_type = None
        assert socket.gethostname() == 'mengkjin-server' or self.short_test

    def __getitem__(self , k): return self.__dict__[k]

    def update(self, update = {} , **kwargs):
        for k,v in update.items(): setattr(self , k , v)
        for k,v in kwargs.items(): setattr(self , k , v)
        return self

    def reload(self , config_path = 'default' , do_parser = False , par_args = {} , override = None):
        new_config = self.load(config_path,do_parser,par_args,override)
        self.__dict__ = new_config.__dict__
        return self
    
    def get(self , key , default = None):
        return getattr(self , key , default)
    
    def set_config_environment(self , manual_seed = None):
        self.set_random_seed(manual_seed if manual_seed else self.get('random_seed'))
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self.allow_tf32) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self.detect_anomaly)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict):
        self.Model.update_data_param(x_data)
    
    @classmethod
    def load(cls , config_path = 'default' , do_parser = False , par_args = {} , override = None , makedir = True):
        '''load config yaml to get default/giving params'''
        model_name = None if config_path == 'default' else os.path.basename(config_path)
        _TrainParam = TrainParam(config_path , model_name = model_name , override = override)
        _ModelParam = ModelParam(config_path , _TrainParam.model_module)

        config = cls(**_TrainParam.configs , _TrainParam = _TrainParam , _ModelParam = _ModelParam)
        if do_parser: config.process_parser(cls.parser_args(par_args))

        model_path = config.model_base_path
        if config_path != 'default':
            assert config_path == model_path , (config_path , model_path)
            _TrainParam = TrainParam(model_path , model_name = model_name , override = override)
            _ModelParam = ModelParam(model_path , _TrainParam.model_module)
            config_resume = cls(**_TrainParam.configs , _TrainParam = _TrainParam , _ModelParam = _ModelParam)
            config.update(config_resume.__dict__)
        elif 'fit' in config.stage_queue and makedir:
            if config.Train.resumeable and not config.short_test:
                raise Exception(f'{model_path} has to be delete manually')
            [os.makedirs(f'{model_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]
            config.Train.copy_to(model_path , exist_ok=config.short_test)
            config.Model.copy_to(model_path , exist_ok=config.short_test)
        
        config.Model.expand(config.model_base_path)
        return config
    
    def model_path(self , model_date , model_num , model_type , base_path = None):
        '''get model path of deposition giving model date/type/base_path/num'''
        if base_path is None:
            model_dir = f'{self.model_base_path}/{model_num}'
        else:
            model_dir = f'{base_path}/{model_num}'
        return '{}/{}.{}.pt'.format(model_dir , model_date , model_type)
    
    def model_iter(self , stage , model_date_list):
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(model_date_list , self.model_num_list))
        if self.resume_training and stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(self.model_path(model_date = model_date , model_num = model_num , model_type = 'best')):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = Filtered(new_iter , ~models_trained)
        return new_iter
    
    @property
    def Model(self) -> ModelParam: 
        assert self._ModelParam is not None
        return self._ModelParam
    @property
    def Train(self) -> TrainParam: 
        assert self._TrainParam is not None
        return self._TrainParam
    @property
    def model_base_path(self) -> str: return self.Train.model_base_path
    @property
    def model_param(self) -> list[dict]: return self.Model.params
    @property
    def model_num(self) -> int: return self.Model.n_model
    @property
    def train_param(self) -> dict: return self.Train.train_param
    @property
    def model_num_list(self) -> list[int]: return list(range(self.Model.n_model))
    @property
    def data_type_list(self) -> list[str]: return self.model_data_type.split('+')
    @property
    def sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self.train_param.get('dataloader',{}).get('sample_method' , 'sequential')
    @property
    def train_ratio(self) -> float:  return self.train_param.get('dataloader',{}).get('train_ratio',0.8)
    @property
    def shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self.train_param.get('dataloader',{}).get('shuffle_option','static')
    @property
    def clip_value(self) -> float: return self.train_param['trainer']['gradient'].get('clip_value' , None)
    def weight_scheme(self , stage : str , no_weight = False) -> Optional[str]: 
        weight_dict = self.train_param.get('criterion',{}).get('weight',{})
        return None if no_weight else weight_dict.get(stage.lower() , 'equal')
    
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
    def parser_args(cls , kwargs : dict , description='manual to this script'):
        parser = argparse.ArgumentParser(description=description)
        for arg in ['stage' , 'resume' , 'checkname']:
            parser.add_argument(f'--{arg}', type=int, default = kwargs.get(arg , -1))
        args , _ = parser.parse_known_args()
        return args

    def parser_stage(self , value = -1):
        if value < 0:
            print(f'--What stage would you want to run? 0: fit + test, 1: fit only , 2: test only')
            value = int(input(f'[0,fit+test] , [1,fit] , [2,test]'))
        stage_queue = ['data' , 'fit' , 'test']
        if value > 0:
            stage_queue = ['data' , stage_queue[value]]
        elif value < 0:
            raise Exception(f'Error input : {value}')
        print('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]), stage_queue))))
        self.stage_queue = stage_queue

    def parser_resume(self , value = -1):
        '''ask if resume training when candidate names exists'''
        model_name = self.Train.model_name
        assert model_name is not None
        candidate_name = [model for model in [self.model_name] if os.path.exists(f'{DIR.model}/{model}')] + \
                [model for model in os.listdir(DIR.model) if model.startswith(model_name + '.')]  
        if len(candidate_name) > 0 and 'fit' in self.stage_queue:
            if value < 0:
                print(f'--Multiple model path of {self.model_name} exists, input [yes] to resume training, or start a new one!')
                user_input = input(f'Confirm resume training [{self.model_name}]? [yes/no] : ')
                value = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
            self.resume_training = value > 0 
            print(f'--Confirm Resume Training!' if self.resume_training else '--Start Training New!')
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
        model_name = self.Train.model_name
        assert model_name is not None
        candidate_name = [model for model in [model_name] if os.path.exists(f'{DIR.model}/{model}')] + \
                [model for model in os.listdir(DIR.model) if model.startswith(model_name + '.')] 
        if self.short_test:
            ...
        elif 'fit' in self.stage_queue and candidate_name:
            if self.resume_training and len(candidate_name) == 1:
                model_name = candidate_name[0]
            elif self.resume_training:
                if value < 0:
                    print(f'--Attempting to resume but multiple models exist, input number to choose')
                    [print(str(i) + ' : ' + f'{DIR.model}/{model}') for i , model in enumerate(candidate_name)]
                    value = int(input('which one to use? '))
                model_name = candidate_name[value]
            else:
                if value < 0:
                    print(f'--Model dirs of {model_name} exists, input [yes] to add a new directory!')
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
                    print(f'--Attempting to test while multiple models exists, input number to choose')
                    [print(str(i) + ' : ' + f'{DIR.model}/{model}') for i , model in enumerate(candidate_name)]
                    value = int(input('which one to use? '))
                model_name = candidate_name[value]

        print(f'--Model_name is set to {model_name}!')  
        self.model_name = model_name
        self.Train.model_name = model_name

    def process_parser(self , par_args = {}):
        self.parser_stage(getattr(par_args , 'stage' , -1))
        self.parser_resume(getattr(par_args , 'resume' , -1))
        self.parser_select(getattr(par_args , 'checkname' , -1)) 
        return self

    def print_out(self):
        subset = [
            'random_seed' , 'model_name' , 'model_module' , 'model_data_type' , 
            'beg_date' , 'end_date' , 'sample_method' , 'shuffle_option' ,
        ]
        pretty_print_dict({k:self.get(k) for k in subset})
        # pretty_print_dict(self.train_param)
        pretty_print_dict(self.Model.Param)
        
        