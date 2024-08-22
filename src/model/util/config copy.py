import argparse , os , random , shutil , socket , torch
import numpy as np

from dataclasses import dataclass , field
from pathlib import Path
from typing import Any , ClassVar , Literal , Optional

from ..nn import get_nn_category , get_nn_datatype
from ...basic import PATH
from ...func import pretty_print_dict , recur_update

TRAIN_YAML = 'train_param.yaml'
MODEL_YAML   = '{}.yaml'
DEFAULT_YAML = 'default.yaml'

def get_module_type(module : str):
    if module in ['booster' , 'lgbm' , 'ada' , 'xgboost' , 'catboost']:
        return 'booster'
    elif module in ['hidden_aggregator']:
        return 'aggregator'
    else:
        return 'nn'
    
def get_booster_type(config : 'TrainConfig'):
    if config.model_module in ['booster' , 'hidden_aggregator', ]:
        return config['model.booster_type']
    elif config.model_module in ['lgbm' , 'ada' , 'xgboost' , 'catboost']:
        return config.model_module
    else:
        return False

def check_config_validity(config : 'TrainConfig'):
    assert socket.gethostname() == 'mengkjin-server' or config.short_test or \
        (isinstance(config.Train.override,dict) and config.Train.override.get('short_test') == False), socket.gethostname()

    if 'best' not in config['model.types']: config['model.types'].insert(0 , 'best')

    nn_category = get_nn_category(config.model_module)
    samp_method = config.sample_method

    nn_datatype = get_nn_datatype(config.model_module)
    
    if nn_category == 'tra':
        assert samp_method != 'total_shuffle' , samp_method
    elif nn_category == 'vae':
        assert samp_method == 'sequential' , samp_method

    if nn_datatype:
        config.Train.Param['data.type'] = nn_datatype

def check_model_param_validity(model_param : 'ModelParam'):
    if model_param.module == 'tra':
        assert 'hist_loss_seq_len' in model_param.Param
        assert 'hist_loss_horizon' in model_param.Param
@dataclass    
class TrainParam:
    config_path : str | Path
    spec_adjust : bool = True
    Param : dict[str,Any] = field(default_factory=dict)
    model_name  : Optional[str] = None
    override    : Optional[dict] = None

    def __post_init__(self) -> None:
        source_dir = PATH.conf if self.config_path == 'default' else self.config_path
        source_base = TRAIN_YAML
        Param : dict = PATH.read_yaml(f'{source_dir}/{source_base}')
        
        if socket.gethostname() != 'mengkjin-server': Param['short_test'] = True
        if self.override: Param.update(self.override)

        if self.spec_adjust:
            if Param['short_test'] and Param.get('conditional.short_test'): 
                recur_update(Param , Param['conditional.short_test'])
            if Param['model.module'].lower() == 'transformer' and Param.get('conditional.transformer'):
                recur_update(Param , Param['conditional.transformer'])

        if self.model_name is None:
            if Param['model.name']:
                self.model_name = str(Param['model.name'])
            else:
                self.model_name = '_'.join([Param['model.module'].lower() , Param['data.type']])
            if Param['short_test']: 
                self.model_name += '_ShortTest'

        self.Param = Param
        if not self.is_nn: Param['model.types'] = ['best']

    def __getitem__(self , key : str):
        return self.Param[key]
    
    def update_data_param(self , model_param : 'ModelParam'):
        for k,v in model_param.Param.items():
            if k in self.Param: self.Param[k] = v
    
    def copy_to(self , target_dir , exist_ok = False):
        source_dir = PATH.conf if self.config_path == 'default' else self.config_path
        target_base = source_base = TRAIN_YAML
        os.makedirs(target_dir, exist_ok = True)
        if not exist_ok: assert not os.path.exists(f'{target_dir}/{target_base}')
        if f'{source_dir}/{source_base}' != f'{target_dir}/{target_base}':
            shutil.copyfile(f'{source_dir}/{source_base}' , f'{target_dir}/{target_base}')

    @classmethod
    def guess_module(cls , config_path : str) -> str:
        if not config_path: config_path = f'{PATH.conf}/{TRAIN_YAML}'
        if not config_path.endswith(TRAIN_YAML): config_path = f'{config_path}/{TRAIN_YAML}'
        return PATH.read_yaml(config_path)['model.module'].lower()
    @property
    def model_base_path(self) -> str: return f'{PATH.model}/{self.model_name}'
    @property
    def resumeable(self) -> bool: return os.path.exists(f'{self.model_base_path}/{TRAIN_YAML}')
    @property
    def model_module(self) -> str: return self.Param['model.module'].lower()
    @property
    def is_nn(self) -> bool: return get_module_type(self.model_module) == 'nn'
    @property
    def callbacks(self) -> dict[str,dict]: 
        return {k.replace('callbacks.',''):v for k,v in self.Param.items() if k.startswith('callbacks.')}
        
@dataclass    
class ModelParam:
    config_path : str | Path
    module      : str
    Param       : dict         = field(default_factory=dict)
    n_model     : int          = 0
    params      : list[dict]   = field(default_factory=list)

    INDAY_DIMS : ClassVar[dict[str,int]] = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}

    def __post_init__(self) -> None:
        self.Param = PATH.read_yaml(self.source_path())
        assert isinstance(self.Param , dict)
        for key , value in self.Param.items():
            if isinstance(value , (list,tuple)): 
                self.n_model = max(self.n_model , len(value))
            #else:
            #    self.Param[key] = [value]
        assert self.n_model <= 5 , self.n_model
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
        if source_path != target_path:
            shutil.copyfile(source_path , target_path)

    def source_path(self):
        module_base = MODEL_YAML.format(self.module.lower())
        source_dir  = f'{PATH.conf}/model' if self.config_path == 'default' else self.config_path
        source_base = module_base if os.path.exists(f'{source_dir}/{module_base}') else DEFAULT_YAML
        return f'{source_dir}/{source_base}'
    
    def target_path(self , target_dir):
        module_base = MODEL_YAML.format(self.module.lower())
        return f'{target_dir}/{module_base}'

    def expand(self , base_path):
        params = []
        for mm in range(self.n_model):
            par = {'path':f'{base_path}/{mm}'}
            for k,v in self.Param.items():
                if isinstance(v , (list, tuple)):
                    par[k] = v[mm % len(v)]
                else:
                    par[k] = v
            params.append(par)
        self.params = params
        #self.params = [{'path':f'{base_path}/{mm}' , **{k:v[mm % len(v)] for k,v in self.Param.items()}} for mm in range(self.n_model)]

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


class TrainConfig:
    def __init__(self , train_param : TrainParam , model_param : ModelParam):
        self.Train : TrainParam = train_param
        self.Model : ModelParam = model_param
        
        self.resume_training: bool  = False
        self.stage_queue: list      = []

        self.Train.update_data_param(self.Model)
        check_config_validity(self)

    @property
    def model_name(self) -> str|Any: return self.Train.model_name
    @property
    def callbacks(self) -> dict[str,dict]: return self.Train.callbacks
    @property
    def short_test(self) -> bool: return self.Train['short_test']
    @property
    def model_base_path(self) -> str: return self.Train.model_base_path
    @property
    def model_param(self) -> list[dict]: return self.Model.params
    @property
    def model_num(self) -> int: return self.Model.n_model
    @property
    def model_num_list(self) -> list[int]: return list(range(self.Model.n_model))
    @property
    def model_module(self) -> str: return self.Train.model_module
    @property
    def model_rslt_path(self) -> str: return f'{self.model_base_path}/detailed_analysis'
    @property
    def data_type_list(self) -> list[str]: return self['data.type'].split('+')

    @property
    def precision(self) -> Any: 
        if isinstance(self.Train['precision'] , str): 
            return getattr(torch , self.Train['precision'])
        else:
            return self.Train['precision']
        
    @property
    def sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self.Train['train.dataloader.sample_method']
    @property
    def train_ratio(self) -> float:  
        return self.Train['train.dataloader.train_ratio']
    @property
    def shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self.Train['train.dataloader.shuffle_option']

    @property
    def is_nn(self): return self.Train.is_nn
    @property
    def nn_category(self): return get_nn_category(self.model_module)

    def __getitem__(self , k): return self.Train[k]

    def update(self, update = {} , **kwargs):
        for k,v in update.items(): setattr(self , k , v)
        for k,v in kwargs.items(): setattr(self , k , v)
        return self

    def reload(self , config_path = 'default' , do_parser = False , par_args = {} , override = None):
        new_config = self.load(config_path,do_parser,par_args,override)
        self.__dict__ = new_config.__dict__
        return self
    
    def get(self , key , default = None):
        return self[key] if key in self.Train.Param else getattr(self , key , default)
    
    def set_config_environment(self , manual_seed = None):
        self.set_random_seed(manual_seed if manual_seed else self.get('random_seed'))
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self['allow_tf32']) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self['detect_anomaly'])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict):
        if self.is_nn: self.Model.update_data_param(x_data)
    
    @classmethod
    def load(cls , config_path : Optional[str | Path] = 'default' , 
             do_parser = False , par_args = {} , override = None , makedir = True):
        '''load config yaml to get default/giving params'''
        if config_path is None: config_path = 'default'
        model_name = None if config_path == 'default' else os.path.basename(config_path)
        train_param = TrainParam(config_path , model_name = model_name , override = override)
        model_param = ModelParam(config_path , train_param.model_module)

        config = cls(train_param , model_param)
        if do_parser: config.process_parser(cls.parser_args(par_args))

        base_path = config.model_base_path
        if base_path != config_path and config.resume_training:
            train_param = TrainParam(base_path , model_name = config.model_name , override = override)
            model_param = ModelParam(base_path , train_param.model_module)
            config_resume = cls(train_param , model_param)
            config.update(config_resume.__dict__)
        elif 'fit' in config.stage_queue and makedir:
            if config.Train.resumeable and not config.short_test:
                raise Exception(f'{base_path} has to be delete manually')
            [os.makedirs(f'{base_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]
            config.Train.copy_to(base_path , exist_ok=config.short_test)
            config.Model.copy_to(base_path , exist_ok=config.short_test)
            
        os.makedirs(config.model_rslt_path , exist_ok=True)
        config.Model.expand(config.model_base_path)
        return config
    
    def weight_scheme(self , stage : str , no_weight = False) -> Optional[str]: 
        stg = stage if stage == 'fit' else 'test'
        return None if no_weight else self.Train[f'train.criterion.weight.{stg}']

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
        self.Train.model_name = model_name

    def process_parser(self , par_args = {}):
        self.parser_stage(getattr(par_args , 'stage' , -1))
        self.parser_resume(getattr(par_args , 'resume' , -1))
        self.parser_select(getattr(par_args , 'checkname' , -1)) 
        return self

    def print_out(self):
        subset = [
            'model_name' , 'model_module' , 'model.types' , 'model.booster_type' ,
            'model.booster_ensembler' , 'data.type' , 'data.labels' ,
            'random_seed' , 'beg_date' , 'end_date' , 'sample_method' , 'shuffle_option' , 
        ]
        pretty_print_dict({k:self.get(k) for k in subset})
        pretty_print_dict(self.Model.Param)

    @staticmethod
    def guess_module(config_path : str) -> str: 
        return TrainParam.guess_module(config_path)

    @staticmethod
    def get_config_path(model_name : str): 
        return f'{PATH.model}/{model_name}'
    