import argparse , os , random , shutil , yaml

import numpy as np
import torch

from argparse import Namespace
from copy import deepcopy

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

    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def items(self):
        return self.__dict__.items()
    
    def replace(self , new_config):
        self.__dict__ = new_config.__dict__
        return self

    def reload(self , config_path = 'default' , par_args = Namespace() , do_process = False , override = None):
        if config_path is not None:
            new_config = self.load(config_path,par_args,do_process,override)
            self.replace(new_config)
        return self
    
    def get(self , key , default = None):
        return getattr(self,key , default)
    
    def subset(self , keys = None):
        if keys is None: keys = self.items()
        return {k:self.get(k) for k in keys}
    
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
        if override is not None: config = config.update(override)

        cls.process_parser(config , par_args , do_process)
        if config.resume_training and os.path.exists(f'{config.model_base_path}/model_params.pt'):
            config.model_params = torch.load(f'{config.model_base_path}/model_params.pt')
        else:
            config.model_params = []
            for mm in config.model_num_list:
                dict_mm = {k:(v[mm % len(v)] if isinstance(v,(tuple,list)) else v) for 
                        k,v in config.MODEL_PARAM.items()}
                dict_mm.update({'path':f'{config.model_base_path}/{mm}'}) 
                config.model_params.append(deepcopy(dict_mm))       
        
        return config
    
    @classmethod
    def load_config_path(cls , config_path = 'default' , adjust = False):
        if config_path == 'default': config_path = DIR.conf
        config = cls.read_yaml(f'{config_path}/{cls.config_train}')

        module = config['model_module'].lower()
        model_param_path = f'{config_path}/{cls.config_model.format(module)}'
        if os.path.exists(model_param_path):
            config['MODEL_PARAM'] = cls.read_yaml(model_param_path)
        else:
            config['MODEL_PARAM'] = cls.read_yaml(f'{config_path}/{cls.default_model}')
        
        if adjust:
            if 'special_config' in config.keys() and 'short_test' in config['special_config'].keys(): 
                if config['short_test']: config.update(config['special_config']['short_test'])
                del config['special_config']['short_test']
            if 'special_config' in config.keys() and 'transformer' in config['special_config'].keys():
                if (config['model_module'].lower() == 'transformer' or
                    (config['model_module'].lower() in ['generalrnn'] and 
                    'transformer' in config['MODEL_PARAM']['type_rnn'])):
                    config['TRAIN_PARAM']['trainer'].update(config['special_config']['transformer']['trainer'])
                del config['special_config']['transformer']

        config.data_type_list = config.model_data_type.split('+')
        config.model_num_list = list(range(config.model_num))
        
        # model_name
        if config.model_name is None:
            config.model_name = '_'.join([config.model_module , config.model_data_type])
            config.model_base_path = f'{DIR.model}/{config.model_name}'

        if config_path != 'default':
            config.model_base_path = config_path
            config.model_name      = os.path.basename(config.model_base_path)
        elif config.model_name is None:
            config.model_name = '_'.join([config.model_module , config.model_data_type])
            config.model_base_path = f'{DIR.model}/{config.model_name}'
        else:
            config.model_base_path = f'{DIR.model}/{config.model_name}'
            
        config.train_params = deepcopy(config.TRAIN_PARAM)

        # check conflicts:
        config.tra_model = config.tra_switch and config.model_module.lower().startswith('tra')
        assert 'best' in config.output_types
        assert not (config.tra_model and config.train_params['dataloader']['sample_method'] == 'total_shuffle')

        return cls(**config)

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
    def process_parser(cls , config , par_args , do_process = False):
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

            # resume training
            # ask if resume training, since unexpected end of training may happen
            resume = getattr(config , 'resume' , -1)
            candidate_name = [model for model in [config.model_name] if os.path.exists(f'{DIR.model}/{model}')] + \
                [model for model in os.listdir(DIR.model) if model.startswith(config.model_name + '.')]   
            if 'train' in config.process_queue and resume < 0 and len(candidate_name) > 0:
                print(f'--Multiple model path of {config.model_name} exists, input [yes] to resume training, or start a new one!')
                resume = 1 if input(f'Confirm resume training [{config.model_name}]? [yes/no] : ').lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0
            config.resume_training = resume > 0 and len(candidate_name) > 0
            print(f'--Confirm Resume Training!' if config.resume_training else '--Start Training New!')

            # checkname confirmation
            # Confirm the model_name and model_base_path if multifple model_name dirs exists.
            # If include train: check if dir of model_name exists, if so ask to continue with a sequential one
            # If test only :    check if model_name exists multiple dirs, if so ask to use the raw one or a select one
            checkname = getattr(config , 'checkname' , -1)
            if 'train' in config.process_queue:
                if checkname < 0 and config.resume_training and len(candidate_name) > 0:
                    if len(candidate_name) > 1:
                        print(f'--Attempting to resume but multiple models exists, input number to choose')
                        [print(str(i) + ' : ' + f'{DIR.model}/{model}') for i , model in enumerate(candidate_name)]
                        config.model_name = candidate_name[int(input('which one to use? '))]
                    else:
                        config.model_name = candidate_name[0]
                elif checkname < 0 and len(candidate_name) > 0:
                    print(f'--Multiple model path of {config.model_name} exists, input [yes] to add a new directory will be made!')
                    user_input = input(f'Add a new folder of [{config.model_name}]? [yes/no] : ').lower()
                    checkname = 1 if user_input.lower() in ['' , 'yes' , 'y' ,'t' , 'true' , '1'] else 0

                if checkname and len(candidate_name) > 0:
                    config.model_name += '.'+str(max([1]+[int(model.split('.')[-1])+1 for model in candidate_name[1:]]))
                    config.model_base_path = f'{DIR.model}/{config.model_name}'
                else:
                    raise Exception(f'--Directories of [{config.model_name}] exists!')

                if config.resume_training and os.path.exists(f'{config.model_base_path}/{cls.config_train}'):
                    config_resume = cls.load_config_path(config.model_base_path)
                    config.update(config_resume)
                else:
                    os.makedirs(config.model_base_path, exist_ok = True)
                    shutil.copyfile(f'{DIR.conf}/{cls.config_train}' , f'{config.model_base_path}/{cls.config_train}')
                    model_param_path = cls.config_model.format(config.model_module.lower())
                    if os.path.exists(model_param_path):
                        shutil.copyfile(f'{DIR.conf}/{model_param_path}' , f'{config.model_base_path}/{model_param_path}')
                    else:
                        shutil.copyfile(f'{DIR.conf}/{cls.default_model}' , f'{config.model_base_path}/{model_param_path}')
                [os.makedirs(f'{config.model_base_path}/{mm}' , exist_ok = True) for mm in config.model_num_list]

            elif 'test' in config.process_queue:
                if checkname < 0 and config.resume_training and len(candidate_name) > 0:
                    if len(candidate_name) > 1:
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