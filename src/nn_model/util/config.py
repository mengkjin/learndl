import argparse , os , random , shutil , socket , torch
import numpy as np

from dataclasses import dataclass , field
from typing import Any , ClassVar , Literal , Optional

from ..nn import get_nn_category , get_nn_datatype
from ...func import pretty_print_dict , recur_update
from ...env import PATH , BOOSTER_MODULE

def check_config_validity(config : 'TrainConfig'):
    assert socket.gethostname() == 'mengkjin-server' or config.short_test , socket.gethostname()

    if 'best' not in config.model_types:
        config.model_types.insert(0 , 'best')

    nn_category = get_nn_category(config.model_module)
    samp_method = config.sample_method

    nn_datatype = get_nn_datatype(config.model_module)
    
    if nn_category == 'tra':
        assert samp_method != 'total_shuffle' , samp_method
    elif nn_category == 'vae':
        assert samp_method == 'sequential' , samp_method

    if nn_datatype:
        config.model_data_type = nn_datatype
        config.Train.configs['model_data_type'] = nn_datatype

def check_model_param_validity(model_param : 'ModelParam'):
    if model_param.module == 'tra':
        assert 'hist_loss_seq_len' in model_param.Param
        assert 'hist_loss_horizon' in model_param.Param

@dataclass    
class ModelParam:
    config_path : str
    module      : str
    Param       : dict         = field(default_factory=dict)
    n_model     : int          = 0
    params      : list[dict]   = field(default_factory=list)

    MODEL_YAML  : ClassVar[str] = '{}.yaml'
    DEFAULT_YAML: ClassVar[str] = 'default.yaml'
    INDAY_DIMS  : ClassVar[dict] = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}

    def __post_init__(self) -> None:
        self.Param = PATH.read_yaml(self.source_path())
        assert isinstance(self.Param , dict)
        for key , value in self.Param.items():
            if isinstance(value , (list,tuple)): 
                self.n_model = max(self.n_model , len(value))
            else:
                self.Param[key] = [value]
        assert self.n_model <= 4 , self.n_model
        check_model_param_validity(self)

    def __getitem__(self , key : str):
        return self.Param[key]
    
    def get(self , key : str , default = None):
        return self.Param.get(key , default)
    
    def copy_to(self , target_dir , exist_ok = False):
        source_path = self.source_path()
        target_path = self.target_path(target_dir)
        os.makedirs(target_dir, exist_ok = True)
        if not exist_ok: assert not os.path.exists(target_path)
        shutil.copyfile(source_path , target_path)

    def source_path(self):
        module_base = self.MODEL_YAML.format(self.module.lower())
        source_dir  = f'{PATH.conf}/model' if self.config_path == 'default' else self.config_path
        source_base = module_base if os.path.exists(f'{source_dir}/{module_base}') else self.DEFAULT_YAML
        return f'{source_dir}/{source_base}'
    
    def target_path(self , target_dir):
        module_base = self.MODEL_YAML.format(self.module.lower())
        return f'{target_dir}/{module_base}'

    def expand(self , base_path):
        self.params = [{'path':f'{base_path}/{mm}' , **{k:v[mm % len(v)] for k,v in self.Param.items()}} for mm in range(self.n_model)]

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
    def max_num_output(self): return max(self.Param.get('num_output' , [1]))
    
@dataclass    
class TrainParam:
    config_path : str
    spec_adjust : bool = True
    configs     : dict = field(default_factory=dict)
    train_param : dict = field(default_factory=dict)
    model_name  : Optional[str]  = None
    override    : Optional[dict] = None
    is_booster  : bool = False

    TRAIN_YAML  : ClassVar[str] = 'train_param.yaml'

    def __post_init__(self) -> None:
        source_dir = PATH.conf if self.config_path == 'default' else self.config_path
        source_base = self.TRAIN_YAML
        Param : dict = PATH.read_yaml(f'{source_dir}/{source_base}')
        self.is_booster = Param['model_module'] in BOOSTER_MODULE
        if self.override: Param.update(self.override)
        if self.is_booster: Param['model_types'] = ['best']
        if socket.gethostname() != 'mengkjin-server': Param['short_test'] = True

        if self.spec_adjust:
            if Param['short_test'] and Param.get('on_short_test'): 
                recur_update(Param , Param['on_short_test'])
            if Param['model_module'].lower() == 'transformer' and Param.get('on_transformer'):
                recur_update(Param , Param['on_transformer'])

        if self.model_name is None:
            if Param['model_name']:
                self.model_name = str(Param['model_name'])
            else:
                self.model_name = '_'.join([Param['model_module'].lower() , Param['model_data_type']])
            if Param['short_test']: 
                self.model_name += '_ShortTest'

        self.train_param = Param['train_param']
        self.configs = {k:v for k,v in Param.items() if k != 'train_param'}
        if self.model_name: self.configs['model_name'] = self.model_name

    def __getitem__(self , key : str):
        return self.configs[key]
    
    def copy_to(self , target_dir , exist_ok = False):
        source_dir = PATH.conf if self.config_path == 'default' else self.config_path
        target_base = source_base = self.TRAIN_YAML
        os.makedirs(target_dir, exist_ok = True)
        if not exist_ok: assert not os.path.exists(f'{target_dir}/{target_base}')
        shutil.copyfile(f'{source_dir}/{source_base}' , f'{target_dir}/{target_base}')

    @property
    def model_base_path(self) -> str: return f'{PATH.model}/{self.model_name}'
    @property
    def resumeable(self) -> bool: return os.path.exists(f'{self.model_base_path}/{self.TRAIN_YAML}')
    @property
    def model_module(self) -> str: return self.configs['model_module'].lower()
    @classmethod
    def guess_module(cls) -> str:
        return PATH.read_yaml(f'{PATH.conf}/{cls.TRAIN_YAML}')['model_module'].lower()

@dataclass
class TrainConfig:
    short_test: bool            = False
    model_name: Optional[str]   = None
    model_module: str           = ''
    model_data_type: str        = 'day' 
    model_data_prenorm: dict    = field(default_factory=dict)
    model_types: list[str]      = field(default_factory=list)
    labels: list                = field(default_factory=list)
    callbacks: dict[str,Any]    = field(default_factory=dict)
    beg_date: int               = 20170103
    end_date: int               = 99991231
    input_span: int             = 2400
    interval: int               = 120
    max_epoch: int              = 200
    verbosity: int              = 2
    batch_size: int             = 10000
    input_step_day: int         = 5
    skip_horizon: int           = 20

    mem_storage: bool           = True
    random_seed: Optional[int]  = None
    allow_tf32: bool            = True
    detect_anomaly: bool        = False
    precision: Any              = torch.float # double , bfloat16

    # special model : tra , lgbm
    lgbm_ensembler: bool        = False

    on_short_test: dict         = field(default_factory=dict)
    on_transformer: dict        = field(default_factory=dict)

    resume_training: bool       = True
    stage_queue: list           = field(default_factory=list)
    
    _TrainParam: Optional[TrainParam] = None
    _ModelParam: Optional[ModelParam] = None

    def __post_init__(self):
        if isinstance(self.precision , str): self.precision = getattr(torch , self.precision)
        self.stage_queue = ['data' , 'fit' , 'test']
        check_config_validity(self)

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
        if not self.is_booster: self.Model.update_data_param(x_data)
    
    @classmethod
    def load(cls , config_path : Optional[str] = 'default' , 
             do_parser = False , par_args = {} , override = None , makedir = True):
        '''load config yaml to get default/giving params'''
        if config_path is None: config_path = 'default'
        model_name = None if config_path == 'default' else os.path.basename(config_path)
        _TrainParam = TrainParam(config_path , model_name = model_name , override = override)
        _ModelParam = ModelParam(config_path , _TrainParam.model_module)

        config = cls(**_TrainParam.configs , _TrainParam = _TrainParam , _ModelParam = _ModelParam)
        if do_parser: config.process_parser(cls.parser_args(par_args))

        base_path = config.model_base_path
        if config.resume_training:
            _TrainParam = TrainParam(base_path , model_name = config.model_name , override = override)
            _ModelParam = ModelParam(base_path , _TrainParam.model_module)
            config_resume = cls(**_TrainParam.configs , _TrainParam = _TrainParam , _ModelParam = _ModelParam)
            config.update(config_resume.__dict__)
        elif 'fit' in config.stage_queue and makedir:
            if config.Train.resumeable and not config.short_test:
                raise Exception(f'{base_path} has to be delete manually')
            [os.makedirs(f'{base_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]
            config.Train.copy_to(base_path , exist_ok=config.short_test)
            config.Model.copy_to(base_path , exist_ok=config.short_test)
        
        config.Model.expand(config.model_base_path)
        return config
    
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
    @property
    def is_booster(self): return self.Train.is_booster
    
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
        candidate_name = [model for model in [model_name] if os.path.exists(f'{PATH.model}/{model}')] + \
                [model for model in os.listdir(PATH.model) if model.startswith(model_name + '.')]
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
        candidate_name = [model for model in [model_name] if os.path.exists(f'{PATH.model}/{model}')] + \
                [model for model in os.listdir(PATH.model) if model.startswith(model_name + '.')] 
        if self.short_test:
            ...
        elif 'fit' in self.stage_queue and candidate_name:
            if self.resume_training and len(candidate_name) == 1:
                model_name = candidate_name[0]
            elif self.resume_training:
                if value < 0:
                    print(f'--Attempting to resume but multiple models exist, input number to choose')
                    [print(str(i) + ' : ' + f'{PATH.model}/{model}') for i , model in enumerate(candidate_name)]
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
                    [print(str(i) + ' : ' + f'{PATH.model}/{model}') for i , model in enumerate(candidate_name)]
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
            'random_seed' , 'model_name' , 'model_module' , 'model_data_type' , 'model_types' , 'labels' ,
            'beg_date' , 'end_date' , 'sample_method' , 'shuffle_option' , 'lgbm_ensembler'
        ]
        pretty_print_dict({k:self.get(k) for k in subset})
        # pretty_print_dict(self.train_param)
        pretty_print_dict(self.Model.Param)

    @staticmethod
    def guess_module() -> str: return TrainParam.guess_module()

    @staticmethod
    def get_config_path(model_name : str): 
        return f'{PATH.model}/{model_name}'