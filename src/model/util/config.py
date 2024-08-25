import argparse , os , random , shutil , torch
import numpy as np

from pathlib import Path
from typing import Any , Literal , Optional

from ..boost import VALID_BOOSTERS
from ..nn import get_nn_category , get_nn_datatype
from ...basic import PATH , THIS_IS_SERVER
from ...func import pretty_print_dict , recur_update

class TrainParam:
    def __init__(self , config_path : Optional[Path] , model_name  : Optional[str] = None , override = {}):
        self.config_path = config_path
        self.model_name = model_name

        self.load_param(**override).special_adjustment().make_model_name().check_validity()

    @property
    def Param(self) -> dict[str,Any]: return self.train_param
    def __getitem__(self , key : str): return self.Param[key]
    def __setitem__(self , key : str , value : Any): self.Param[key] = value

    def load_param(self , **kwargs):
        self.train_param : dict[str,Any] = PATH.read_yaml(self.source_path)
        self.train_param.update(kwargs)
        return self

    def special_adjustment(self):
        if not THIS_IS_SERVER: self['short_test'] = True
        if self.short_test and self.Param.get('conditional.short_test'): 
            recur_update(self.Param , self['conditional.short_test'])

        if self.model_module == 'transformer' and self.Param.get('conditional.transformer'):
            recur_update(self.Param , self['conditional.transformer'])
        return self
    
    def make_model_name(self):
        if self.model_name: return self
        if self['model.name']: self.model_name = str(self['model.name'])
        else: self.model_name = '_'.join([self.model_module , self['data.types']])
        if self.short_test: self.model_name += '_ShortTest'
        return self
    
    def check_validity(self):
        assert THIS_IS_SERVER or self.short_test , f'must be at server or short_test'

        if 'best' not in self.model_types: self.model_types.insert(0 , 'best')

        nn_category = get_nn_category(self.model_module)
        nn_datatype = get_nn_datatype(self.model_module)
        
        if nn_category == 'tra': assert self.sample_method != 'total_shuffle' , self.sample_method
        if nn_category == 'vae': assert self.sample_method == 'sequential'    , self.sample_method

        if nn_datatype:              self['data.types'] = nn_datatype
        if self.module_type != 'nn': self['model.types'] = ['best']
        if 'best' not in self.model_types: self.model_types.insert(0 , 'best')
        return self
    
    def generate_model_param(self , update_inplace = False , **kwargs):
        model_param = ModelParam(self.config_path , self.model_module , self.booster_head , verbosity = self.verbosity , **kwargs)
        if update_inplace: self.update_model_param(model_param)
        return model_param
    
    def update_model_param(self , model_param : 'ModelParam'):
        param = {k:v for k,v in model_param.Param.items() if k in self.Param}
        self.Param.update(param)
        return self
    
    def copy_to(self , target_dir : Path , exist_ok = False):
        target_dir.mkdir(exist_ok=True)
        target = self.target_config_path(target_dir)
        if not exist_ok: assert not target.exists()
        if self.source_path != target: shutil.copyfile(self.source_path , target)

    @classmethod
    def guess_module(cls , config_path : Path | None) -> str:
        return PATH.read_yaml(cls.source_config_path(config_path))['model.module'].lower()
    
    @classmethod
    def get_module_type(cls , module : str):
        if module in ['booster' , *VALID_BOOSTERS]:
            return 'booster'
        elif module in ['hidden_aggregator']:
            return 'aggregator'
        else:
            return 'nn'
    
    @staticmethod
    def source_config_path(parent : Path | None , name : str = 'default.yaml'):
        if not parent: parent = PATH.conf_train
        return parent.joinpath(name)
    
    @staticmethod
    def target_config_path(parent : Path , name : str = 'train.yaml'):
        return parent.joinpath(name)
    
    @property
    def source_path(self): return self.source_config_path(self.config_path)
    @property
    def short_test(self): return bool(self['short_test'])
    @property
    def resumeable(self): return self.target_config_path(self.model_base_path).exists()
    @property
    def model_base_path(self): 
        assert self.model_name, 'model_name must not be none'
        return PATH.model.joinpath(self.model_name)
    @property
    def model_rslt_path(self): return self.model_base_path.joinpath('detailed_analysis')
    @property
    def model_types(self) -> list: return self['model.types']
    @property
    def model_module(self): return str(self['model.module']).lower()
    @property
    def model_data_types(self): return str(self['data.types'])
    @property
    def model_data_labels(self) -> list[str]: return self['data.labels']
    @property
    def model_data_hiddens(self) -> list[str]: return self['data.hiddens']
    @property
    def module_type(self): return self.get_module_type(self.model_module)
    @property
    def verbosity(self): return int(self['verbosity'])
    @property
    def train_ratio(self): return float(self['train.dataloader.train_ratio'])
    @property
    def sample_method(self) -> Literal['total_shuffle' , 'sequential' , 'both_shuffle' , 'train_shuffle']: 
        return self['train.dataloader.sample_method']
    @property
    def shuffle_option(self) -> Literal['static' , 'init' , 'epoch']: 
        return self['train.dataloader.shuffle_option']
    @property
    def callbacks(self) -> dict[str,dict]: 
        return {k.replace('callbacks.',''):v for k,v in self.Param.items() if k.startswith('callbacks.')}
    @property
    def random_seed(self): return self.Param.get('random_seed')
    @property
    def data_type_list(self) -> list[str]: return self.model_data_types.split('+')
    @property
    def precision(self) -> Any: 
        return getattr(torch , self['precision']) if isinstance(self['precision'] , str) else self['precision']
    @property
    def nn_category(self): return get_nn_category(self.model_module)
    @property
    def booster_type(self):
        if self.model_module in ['booster' , 'hidden_aggregator', ]:
            assert self['model.booster_type'] in VALID_BOOSTERS , self['model.booster_type']
            return self['model.booster_type']
        elif self.model_module in VALID_BOOSTERS:
            return self.model_module
        elif self.booster_head:
            assert self.booster_head in VALID_BOOSTERS , self.booster_head
            return self.booster_head
        else:
            return False
    @property
    def booster_head(self): return self['model.booster_head']
        
class ModelParam:
    INDAY_DIMS = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}

    def __init__(self , config_path : Optional[Path] , module : str , booster_head : Any = False , clip_n : int = -1 , 
                 verbosity = 2 , **kwargs):
        self.config_path = config_path
        self.module = module
        self.booster_head = booster_head
        self.clip_n = clip_n
        self.verbosity = verbosity
        self.override = kwargs
        self.load_param().check_validity()

    @property
    def Param(self) -> dict[str,Any]: return self.model_param
    def __getitem__(self , key : str): return self.Param[key]
    def __setitem__(self , key : str , value : Any): self.Param[key] = value

    def load_param(self):
        self.model_param : dict[str,Any] = PATH.read_yaml(self.source_path)
        self.model_param['verbosity'] = self.verbosity
        self.model_param.update(self.override)
        return self

    def check_validity(self):
        self.module = self.module.lower()
        assert TrainParam.get_module_type(self.module) == 'nn'  or (not self.booster_head) , self.module

        lens = [len(v) for v in self.Param.values() if isinstance(v , (list,tuple))]
        self.n_model = max(lens) if lens else 1
        if self.clip_n > 0: self.n_model = min(self.clip_n , self.n_model)
        assert self.n_model <= 5 , self.n_model
        
        if self.module == 'tra':
            assert 'hist_loss_seq_len' in self.Param
            assert 'hist_loss_horizon' in self.Param

        if self.booster_head:
            assert self.booster_head in VALID_BOOSTERS , self.booster_head
            self.booster_head_param = ModelParam(self.config_path , self.booster_head , False , 1 , self.verbosity , **self.override)
        return self

    @staticmethod
    def source_config_path(parent : Path | None , module : str , name : str = 'default.yaml'):
        if not parent: parent = PATH.conf_nn if TrainParam.get_module_type(module) == 'nn' else PATH.conf_boost
        return parent.joinpath(name)
    
    @staticmethod
    def target_config_path(target_dir : Path , name : str):
        return target_dir.joinpath(name)
    
    def get(self , key : str , default = None):
        return self.Param.get(key , default)
    
    def copy_to(self , target_dir : Path , exist_ok = False):
        target_dir.mkdir(exist_ok=True)
        target = self.target_config_path(target_dir , f'{self.module}.yaml')
        
        if not exist_ok: assert not target.exists()
        if self.source_path != target: shutil.copyfile(self.source_path , target)

    @property
    def source_path(self):
        path = self.source_config_path(self.config_path , self.module , f'{self.module}.yaml')
        if not path.exists(): path = path.with_name('default.yaml')
        return path

    def expand(self , base_path : Path):
        params = []
        for mm in range(self.n_model):
            par = {'path':base_path.joinpath(str(mm))}
            for k,v in self.Param.items():
                par[k] = v[mm % len(v)] if isinstance(v , (list, tuple)) else v
            params.append(par)
        self.params : list[dict[str,Any]] = params
        if self.booster_head: self.booster_head_param.expand(base_path)
       
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


class TrainConfig(TrainParam):
    def __init__(self , config_path : Optional[Path] , model_name  : Optional[str] = None , override = {}):
        self.Train = TrainParam(config_path , model_name , override)
        self.Model = self.Train.generate_model_param(update_inplace = True)
        
        self.resume_training: bool  = False
        self.stage_queue: list      = []

    @property
    def Param(self) -> dict[str,Any]: return self.Train.Param
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

    def update(self, update : dict = {} , **kwargs):
        for k,v in update.items(): setattr(self , k , v)
        for k,v in kwargs.items(): setattr(self , k , v)
        return self

    def reload(self , config_path : Path | None = None , do_parser = False , par_args = {} , override = {}):
        new_config = self.load(config_path,do_parser,par_args,override)
        self.__dict__ = new_config.__dict__
        return self
    
    def get(self , key , default = None):
        return self[key] if key in self.Train.Param else getattr(self , key , default)
    
    def set_config_environment(self , manual_seed = None):
        self.set_random_seed(manual_seed if manual_seed else self.random_seed)
        torch.set_default_dtype(self.precision)
        torch.backends.cuda.matmul.__setattr__('allow_tf32' ,self['allow_tf32']) #= self.allow_tf32
        torch.autograd.anomaly_mode.set_detect_anomaly(self['detect_anomaly'])
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    def update_data_param(self , x_data : dict):
        if self.module_type == 'nn': self.Model.update_data_param(x_data)
    
    @classmethod
    def load(cls , config_path : Optional[Path] = None , 
             do_parser = False , par_args = {} , override = {} , makedir = True):
        '''load config yaml to get default/giving params'''
        model_name = config_path.name if config_path else None
        config = cls(config_path , model_name , override)
        if do_parser: config.process_parser(cls.parser_args(par_args))

        base_path = config.model_base_path
        if base_path != config_path and config.resume_training:
            config_resume = cls(base_path , config.model_name , override)
            config.update(config_resume.__dict__)
        elif 'fit' in config.stage_queue and makedir:
            if config.Train.resumeable and not config.short_test:
                raise Exception(f'{base_path} has to be delete manually')
            [base_path.joinpath(str(mm)).mkdir(parents=True,exist_ok=True) for mm in config.model_num_list]
            config.copy_to(base_path)
            
        config.model_rslt_path.mkdir(exist_ok=True)
        config.Model.expand(config.model_base_path)
        return config
    
    def weight_scheme(self , stage : str , no_weight = False) -> Optional[str]: 
        stg = stage if stage == 'fit' else 'test'
        return None if no_weight else self.Train[f'train.criterion.weight.{stg}']
    
    def copy_to(self , target_dir : Path):
        self.Train.copy_to(target_dir , exist_ok=self.short_test)
        self.Model.copy_to(target_dir , exist_ok=self.short_test)

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
        model_name = self.model_name
        assert model_name is not None
        candidate_name = [model for model in [model_name] if self.get_config_path(model).exists()] + \
                [model.name for model in PATH.model.iterdir() if model.name.startswith(model_name + '.')]
        if len(candidate_name) > 0 and 'fit' in self.stage_queue:
            if value < 0:
                print(f'--Multiple model path of {model_name} exists, input [yes] to resume training, or start a new one!')
                user_input = input(f'Confirm resume training [{model_name}]? [yes/no] : ')
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
        model_name = self.model_name
        assert model_name is not None
        candidate_name = [model for model in [model_name] if self.get_config_path(model).exists()] + \
                [model.name for model in PATH.model.iterdir() if model.name.startswith(model_name + '.')]
        if self.short_test:
            ...
        elif 'fit' in self.stage_queue and candidate_name:
            if self.resume_training and len(candidate_name) == 1:
                model_name = candidate_name[0]
            elif self.resume_training:
                if value < 0:
                    print(f'--Attempting to resume but multiple models exist, input number to choose')
                    [print(f'{i} : {PATH.model}/{model}') for i , model in enumerate(candidate_name)]
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
                    [print(f'{i} : {PATH.model}/{model}') for i , model in enumerate(candidate_name)]
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
            'model.booster_head' , 'data.types' , 'data.labels' ,
            'random_seed' , 'beg_date' , 'end_date' , 'sample_method' , 'shuffle_option' , 
        ]
        pretty_print_dict({k:self.get(k) for k in subset})
        pretty_print_dict(self.Model.Param)

    @staticmethod
    def get_config_path(model_name : str): 
        return PATH.model.joinpath(model_name)
    