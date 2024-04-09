import argparse , os , random , shutil , yaml

import numpy as np
import torch

from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass , field
from typing import ClassVar , Optional

from ..func.basic import pretty_print_dict
from ..environ import DIR

class TrainConfig(Namespace):
    config_train  = 'train_param.yaml'
    config_model  = 'model_{}.yaml'
    default_model = 'model_default.yaml'

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.update(kwargs)

    def __getitem__(self , k):
        return self.__dict__[k]

    def update(self, updater = {} , **kwargs):
        for k,v in updater.items(): setattr(self , k , v)
        for k,v in kwargs.items():  setattr(self , k , v)
        return self

    def reload(self , config_path = 'default' , par_args = Namespace() , do_process = False , override = None):
        if config_path is not None:
            new_config = self.load(config_path,par_args,do_process,override)
            self.__dict__ = new_config.__dict__
        return self
    
    def get(self , key , default = None):
        return getattr(self,key , default)
    
    def set_config_environment(self , manual_seed = None):
        self.set_random_seed(manual_seed if manual_seed else self.get('random_seed'))
        torch.set_default_dtype(getattr(torch , self.precision))
        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.autograd.set_detect_anomaly(self.detect_anomaly) # type:ignore
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    @staticmethod
    def read_yaml(yaml_file):
        with open(yaml_file ,'r') as f:
            d = yaml.load(f , Loader = yaml.FullLoader)
        return d
    
    @classmethod
    def load(cls , config_path = 'default' , par_args = Namespace() , do_process = False , override = None):
        """
        1. namespace type of config
        2. Ask what process would anyone want to run : 0 : train & test(default) , 1 : train only , 2 : test only
        3. Ask if model_name and model_base_path should be changed if old dir exists
        """
        config = cls.load_config_path(config_path)
        config = cls.process_parser(config_path , config , par_args , do_process)

        assert isinstance(config.model_param , ModelParam)
        if override is not None: config = config.update(override)

        if config_path != config.model_base_path and os.path.exists(f'{config.model_base_path}/{cls.config_train}'):
            config_resume = cls.load_config_path(config.model_base_path)
            config.update(config_resume)
        else:
            os.makedirs(config.model_base_path, exist_ok = True)
            
            shutil.copyfile(f'{DIR.conf}/{cls.config_train}' , f'{config.model_base_path}/{cls.config_train}')
            config.model_param.copy(config.model_base_path)
        [os.makedirs(f'{config.model_base_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]

        
        config.model_param.expand(config.model_base_path , config.resume_training)
        return config
    
    @classmethod
    def load_config_path(cls , config_path = 'default' , adjust = True):
        if config_path == 'default': config_path = DIR.conf
        config_dict = cls.read_yaml(f'{config_path}/{cls.config_train}')
        config = cls(**config_dict)

        # model_name
        if config_path != DIR.conf:
            config.model_name      = os.path.basename(config_path)
            if config.short_test: config.model_name += '_ShortTest'
            config.model_base_path = config_path
        elif config.model_name is None:
            config.model_name = '_'.join([config.model_module , config.model_data_type])
            if config.short_test: config.model_name += '_ShortTest'
            config.model_base_path = f'{DIR.model}/{config.model_name}'
        else:
            config.model_base_path = f'{DIR.model}/{config.model_name}'

        if adjust and 'special_config' in config.keys():
            if 'short_test' in config.special_config.keys() and config.short_test: 
                config.update(config.special_config['short_test'])
            if 'transformer' in config.special_config.keys() and config.model_module.lower() == 'transformer':
                config.train_param['trainer'].update(config.special_config['transformer']['trainer'])
            delattr(config , 'special_config')

        config.model_param = ModelParam(config_path , config.model_module)
        config.model_num_list = config.model_param.num_list
        config.data_type_list = config.model_data_type.split('+')
            
        # check conflicts:
        assert 'best' in config.output_types
        config.tra_model = config.tra_switch and config.model_module.lower().startswith('tra')
        assert not (config.tra_model and config.train_param['dataloader']['sample_method'] == 'total_shuffle')

        return config

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
    def parser_args(cls , input = {} , description='manual to this script'):
        parser = argparse.ArgumentParser(description=description)
        for arg in ['process' , 'checkname' , 'resume']:
            parser.add_argument(f'--{arg}', type=int, default = input.get(arg , -1))
        args , _ = parser.parse_known_args()
        return args

    @classmethod
    def process_parser(cls , config_path , config , par_args , do_process = False):
        for k , v in par_args.__dict__.items(): 
            assert k not in config.keys() , k
            setattr(config , k , v)

        if do_process:
            # process_confirmation
            process = getattr(config , 'process' , -1)
            if process < 0:
                print(f'--What process would you want to run? 0: all, 1: train only (default), 2: test only')
                process = int(input(f'[0,all] , [1,train] , [2,test]'))
            process_dict = ['data' , 'train' , 'test']
            if process == 0:
                config.process_queue = process_dict
            elif process > 0:
                config.process_queue = ['data' , process_dict[process]]
            else:
                raise Exception(f'Error input : {process}')
            print('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]),config.process_queue))))

            if config_path == 'default':
                candidate_name = []
            else:
                candidate_name = [model for model in [config.model_name] if os.path.exists(f'{DIR.model}/{model}')] + \
                        [model for model in os.listdir(DIR.model) if model.startswith(config.model_name + '.')]  
            # ask if resume training
            resume = getattr(config , 'resume' , -1)
            if len(candidate_name) > 0:
                if 'train' in config.process_queue and resume < 0:
                    print(f'--Multiple model path of {config.model_name} exists, input [yes] to resume training, or start a new one!')
                    user_input = input(f'Confirm resume training [{config.model_name}]? [yes/no] : ')
                    resume = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
                config.resume_training = resume > 0 
                print(f'--Confirm Resume Training!' if config.resume_training else '--Start Training New!')
            else:
                config.resume_training = True

            # checkname confirmation
            # Confirm the model_name if multifple model_name dirs exists.
            # If include train: check if dir of model_name exists, if so ask to continue with a sequential one
            # If test only :    check if model_name exists multiple dirs, if so ask to use the raw one or a select one
            checkname = getattr(config , 'checkname' , -1)
            if len(candidate_name) > 0 and checkname == 0:
                raise Exception(f'--Directories of [{config.model_name}] exists!')
            elif len(candidate_name) > 0:
                if 'train' in config.process_queue and config.resume_training:
                    if checkname < 0:
                        print(f'--Attempting to resume but multiple models exists, input number to choose')
                        [print(str(i) + ' : ' + f'{DIR.model}/{model}') for i , model in enumerate(candidate_name)]
                        config.model_name = candidate_name[int(input('which one to use? '))]
                    else:
                        config.model_name = candidate_name[0]
                elif 'train' in config.process_queue:
                    if checkname < 0:
                        print(f'--Multiple model path of {config.model_name} exists, input [yes] to add a new directory!')
                        user_input = input(f'Add a new folder of [{config.model_name}]? [yes/no] : ').lower()
                        checkname = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
                    if checkname:
                        config.model_name += '.'+str(max([1]+[int(model.split('.')[-1])+1 for model in candidate_name[1:]]))
                    else:
                        raise Exception(f'--Directories of [{config.model_name}] exists!')
                elif 'test' in config.process_queue:
                    if checkname < 0:
                        print(f'--Attempting to resume but multiple models exists, input number to choose')
                        [print(str(i) + ' : ' + f'{DIR.model}/{model}') for i , model in enumerate(candidate_name)]
                        config.model_name = candidate_name[int(input('which one to use? '))]
                    else:
                        config.model_name = candidate_name[0]
            config.model_base_path = f'{DIR.model}/{config.model_name}'
            print(f'--Model_name is set to {config.model_name}!')  
        else:
            config.process_queue = ['data' , 'train' , 'test']
            config.resume_training = True
        return config
    
    def subset(self , keys = None):
        if keys is None: keys = self.items()
        return {k:self.get(k) for k in keys}

    def print_out(self):
        subset = [
            'random_seed' , 'verbosity' , 'precision' , 'batch_size' , 
            'model_name' , 'model_module' , 'model_num' , 'model_data_type' , 'labels' ,
            'beg_date' , 'end_date' , 'interval' , 'input_step_day' , 'test_step_day' , 
        ]
        pretty_print_dict({k:self.get(k) for k in subset})
        pretty_print_dict(self.train_param)
        pretty_print_dict(self.model_param.Param)

@dataclass    
class ModelParam:
    config_path : str
    module      : str
    Param       : dict         = field(default_factory=dict)
    num_list    : list[int]    = field(default_factory=list)
    params      : list[dict]   = field(default_factory=list)

    model_yaml  : ClassVar[str] = 'model_{}.yaml'
    default_yaml: ClassVar[str] = 'model_default.yaml'
    target_base : ClassVar[str] = 'model_params.pt'

    def __post_init__(self) -> None:
        path = f'{self.config_path}/{self.model_yaml.format(self.module.lower())}'
        if not os.path.exists(path):
            path = f'{self.config_path}/{self.default_yaml}'

        with open(path ,'r') as f:
            self.Param = yaml.load(f , Loader = yaml.FullLoader)

        n_model = 1
        for value in self.Param.values():
            if isinstance(value , (list,tuple)):
                n_model = max(n_model , len(value))
        self.num_list = list(range(n_model))
        assert len(self.num_list) <= 3 , len(self.num_list)

    def __getitem__(self , pos : int):
        return self.params[pos]
    
    def copy(self , target_dir):
        path = self.model_yaml.format(self.module.lower())
        if os.path.exists(f'{self.config_path}/{path}'):
            shutil.copyfile(f'{self.config_path}/{path}' , f'{target_dir}/{path}')
        else:
            shutil.copyfile(f'{self.config_path}/{self.default_yaml}' , f'{target_dir}/{path}')

    def expand(self , base_path , resume_training):
        if resume_training and os.path.exists(f'{base_path}/{self.target_base}'):
            params = torch.load(f'{base_path}/{self.target_base}')
        else:
            params = []
            for mm in self.num_list:
                dict_mm = {k:(v[mm % len(v)] if isinstance(v,(tuple,list)) else v) 
                            for k,v in self.Param.items()}
                dict_mm.update({'path':f'{base_path}/{mm}'}) 
                params.append(dict_mm) 
        self.params = params

    @classmethod
    def load(cls , base_path , model_num : int | None = None):
        model_param = torch.load(f'{base_path}/{cls.target_base}')
        if model_num is not None:
            model_param = model_param[model_num]
        return model_param

    def save(self , base_path):
        torch.save(self.params , f'{base_path}/{self.target_base}')   

    def data_related(self , x_data = {} , data_type_list = None):
        # when x_data is know , use it to fill some params
        if data_type_list is None: data_type_list = list(x_data.keys())

        inday_dim_dict = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}
        input_dim , inday_dim = [] , []
        for mdt in data_type_list:
            x = x_data.get(mdt)
            input_dim.append(x.shape[-1] if x else 6)
            inday_dim.append(x.shape[-2] if x else inday_dim_dict.get(mdt , 1))
        if len(data_type_list) > 1:
            filler = {'input_dim':tuple(input_dim), 'inday_dim':tuple(inday_dim)}
        elif len(data_type_list) == 1:
            filler = {'input_dim':input_dim[0] , 'inday_dim':inday_dim[0]}
        else:
            filler = {'input_dim':1, 'inday_dim':1 }

        if hasattr(self , 'params'):
            for param in self.params: param.update(filler)

        return self
        
        