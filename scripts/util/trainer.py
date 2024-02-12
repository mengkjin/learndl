import random , os , shutil
import numpy as np
import torch
import argparse
from types import SimpleNamespace
from copy import deepcopy
from .environ import *

use_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class TrainConfig(SimpleNamespace):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    def update(self, updater = {} , **kwargs) -> None:
        for k,v in updater.items(): setattr(self , k , v)
        for k,v in kwargs.items():  setattr(self , k , v)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def replace(self , new_config):
        for k in self.keys(): delattr(self,k)
        for k,v in new_config.items(): setattr(self,k,v)
    def reload(self , config_path = ''):
        if os.path.exists(config_path):
            reload_base_path = os.path.dirname(config_path)
            reload_name = os.path.basename(reload_base_path)
            raw_config_dict = train_config(_load_raw_config(config_path) , reload_name = reload_name , reload_base_path = reload_base_path)
            key_list = list(self.keys())
            for k in key_list: delattr(self,k)
            for k,v in raw_config_dict.items(): setattr(self,k,v)
    def get(self , key , default = None):
        return getattr(self,key,default)
    def get_dict(self , keys = None):
        if keys is None: keys = self.items()
        return {k:self.get(k) for k in keys}

def trainer_parser(input = {} , default = -1 , description='manual to this script'):
    parser = argparse.ArgumentParser(description=description)
    for arg in ['process' , 'rawname' , 'resume' , 'anchoring']:
        parser.add_argument(f'--{arg}', type=int, default = default if input.get(arg) is None else input.get(arg))
    return parser

def set_trainer_environment(config , manual_random_seed = None):
    _set_random_seed(manual_random_seed if manual_random_seed is not None else config.TRAIN_PARAM['dataloader']['random_seed'])
    torch.set_default_dtype(getattr(torch , config.precision))
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def _set_random_seed(seed = None):
    if seed is None: 
        return NotImplemented
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _load_raw_config(config_files = None):
    config_files = ['train'] if config_files is None else config_files
    config = get_config(config_files)
    if 'special_config' in config.keys() and 'short_test' in config['special_config'].keys(): 
        if config['short_test']: config.update(config['special_config']['short_test'])
        del config['special_config']['short_test']
    if 'special_config' in config.keys() and 'transformer' in config['special_config'].keys():
        if (config['model_module'].lower() == 'transformer' or
            (config['model_module'].lower() in ['generalrnn'] and 
             'transformer' in config['MODEL_PARAM']['type_rnn'])):
            config['TRAIN_PARAM']['trainer'].update(config['special_config']['transformer']['trainer'])
        del config['special_config']['transformer']
    return config

def train_config(config = None , parser = SimpleNamespace() , do_process = False , 
                 reload_name = None , reload_base_path = None , config_files = None , override_config = {}):
    """
    1. namespace type of config
    2. Ask what process would anyone want to run : 0 : train & test(default) , 1 : train only , 2 : test only , 3 : copy to instance only
    3. Ask if model_name and model_base_path should be changed if old dir exists
    """
    if config is None:
        raw_config = _load_raw_config(config_files)
    else:
        raw_config = deepcopy(config)
    config = TrainConfig(device = use_device)
    config.update(raw_config)
    config.update(override_config)

    config.model_data_type = config.model_datatype[config.model_module]
    config.data_type_list  = config.model_data_type.split('+')
    config.model_num_list  = list(range(config.model_num))
    
    # model_name
    name_element = [config.model_module ,config.model_data_type, config.model_nickname]
    config.model_name = '_'.join([x for x in name_element if x is not None])
    config.model_base_path = f'./model/{config.model_name}'
    config.instance_path   = f'./instance/{config.model_name}'
        
    # parser
    for k , v in parser.__dict__.items(): setattr(config , k , v)
    
    if do_process:
        # process_confirmation
        process = getattr(parser , 'process' , -1)
        if process < 0:
            print(f'--What process would you want to run? 0: all, 1: train only (default), 2: test only , 3: copy to instance')
            process = int(input(f'[0,all] , [1,train] , [2,test] , [3,instance]: '))
        process_dict = ['data' , 'train' , 'test' , 'instance']
        if process == 0:
            config.process_queue = process_dict
        elif process > 0:
            config.process_queue = ['data' , process_dict[process]]
        else:
            raise Exception(f'Error input : {process}')
        print('--Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]),config.process_queue))))

        # resume training
        # ask if resume training, since unexpected end of training may happen
        resume = getattr(parser , 'resume' , -1)
        candidate_name = [x for x in [config.model_name] if os.path.exists(f'./model/{x}')] + \
            [x for x in os.listdir(f'./model') if x.startswith(config.model_name + '.')]   
        if 'train' in config.process_queue and resume < 0 and len(candidate_name) > 0:
            print(f'--Multiple model path of {config.model_name} exists, input [yes] to resume training, or start a new one!')
            resume = 1 if input(f'Confirm resume training [{config.model_name}]? [yes/no] : ').lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
        config.resume_training = resume > 0 and len(candidate_name) > 0
        print(f'--Confirm Resume Training!' if config.resume_training else '--Start Training New!')

        # rawname_confirmation
        # Confirm the model_name and model_base_path if multifple model_name dirs exists.
        # If include train: check if dir of model_name exists, if so ask to remove the old ones or continue with a sequential one
        # If test only :    check if model_name exists multiple dirs, if so ask to use the raw one or a select one
        rawname = getattr(parser , 'rawname' , -1)
        if 'train' in config.process_queue:
            if rawname < 0 and config.resume_training and len(candidate_name) > 0:
                if len(candidate_name) > 1:
                    print(f'--Attempting to resume but multiple models exists, input number to choose')
                    [print(str(i) + ' : ' + f'./model/{fn}') for i , fn in enumerate(candidate_name)]
                    config.model_name = candidate_name[int(input('which one to use? '))]
                else:
                    config.model_name = candidate_name[0]
            elif rawname < 0 and len(candidate_name) > 0:
                print(f'--Multiple model path of {config.model_name} exists, input [yes] to confirm deletion, or a new directory will be made!')
                if input(f'Delete all old dirs of [{config.model_name}]? [yes/no] : ').lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1']: 
                    [shutil.rmtree(f'./model/{d}') for d in candidate_name]
                    print(f'--Directories of [{config.model_name}] deletion Confirmed!')
                else:
                    if os.path.exists(config.model_base_path):
                        config.model_name += '.'+str(max([1]+[int(d.split('.')[-1])+1 for d in candidate_name[1:]]))
                        print(f'--A new directory [{config.model_name}] will be made!')
            config.model_base_path = f'./model/{config.model_name}'
            os.makedirs(config.model_base_path, exist_ok = True)
            [os.makedirs(f'{config.model_base_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]
            if config.resume_training and os.path.exists(f'{config.model_base_path}/config_train.yaml'):
                config_resume = _load_raw_config(f'{config.model_base_path}/config_train.yaml')
                config.update(config_resume)
                config.model_data_type = config.model_datatype[config.model_module]
                config.data_type_list  = config.model_data_type.split('+')
                config.model_num_list  = list(range(config.model_num))
            else:
                shutil.copyfile(f'./configs/config_train.yaml', f'{config.model_base_path}/config_train.yaml')
        elif 'test' in config.process_queue:
            if rawname < 0 and config.resume_training and len(candidate_name) > 0:
                if len(candidate_name) > 1:
                    print(f'--Attempting to resume but multiple models exists, input number to choose')
                    [print(str(i) + ' : ' + f'./model/{fn}') for i , fn in enumerate(candidate_name)]
                    config.model_name = candidate_name[int(input('which one to use? '))]
                else:
                    config.model_name = candidate_name[0]
            config.model_base_path = f'./model/{config.model_name}'

        print(f'--Model_name is set to {config.model_name}!')  
    else:
        config.process_queue = ['data' , 'train' , 'test' , 'instance']
        config.resume_training = True

    if reload_name is not None: config.model_name = reload_name
    if reload_base_path is not None: config.model_base_path = reload_base_path
        
    config.train_params = deepcopy(config.TRAIN_PARAM)
    if 'best' not in config.train_params['output_types']:
        config.train_params['output_types'] = ['best'] + config.train_params['output_types']
    config.output_types = config.train_params['output_types']
    
    config.compt_params = deepcopy(config.COMPT_PARAM)

    if config.resume_training and os.path.exists(f'{config.model_base_path}/model_params.pt'):
        config.model_params = torch.load(f'{config.model_base_path}/model_params.pt')
    else:
        config.model_params = []
        for mm in config.model_num_list:
            dict_mm = {k:(v[mm % len(v)] if isinstance(v,(tuple,list)) else v) for 
                       k,v in config.MODEL_PARAM.items()}
            dict_mm.update({'path':f'{config.model_base_path}/{mm}'}) 
            config.model_params.append(deepcopy(dict_mm))

    config.tra_model = config.tra_switch and config.model_module.lower().startswith('tra')
    
    return config

class Device:
    def __init__(self , device = None) -> None:
        if device is None: device = use_device
        self.device = device
    def __call__(self, *args):
        if len(args) == 0: return None
        args = self._to(args)
        return args[0] if len(args) == 1 else args
    def _to(self , x):
        if isinstance(x , (list,tuple)):
            return type(x)(self._to(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:self._to(v) for k,v in x.items()}
        else:
            return x.to(self.device) if x is not None else None
    def torch_nans(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs).fill_(torch.nan)
    def torch_zeros(self,*args , **kwargs):
        return torch.zeros(*args , device = self.device , **kwargs)
    def torch_ones(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs)
    def torch_arange(self,*args,**kwargs):
        return torch.arange(*args , device = self.device , **kwargs)
    def print_cuda_memory(self):
        print(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
              f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')
