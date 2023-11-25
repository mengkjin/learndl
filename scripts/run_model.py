#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# ./run_model.py --process=0 --rawname=1 --resume=0 --anchoring=0
'''
1.TRA
https://arxiv.org/pdf/2106.12950.pdf
https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TRA/src/model.py
1.1 HIST
https://arxiv.org/pdf/2110.13716.pdf
https://github.com/Wentao-Xu/HIST
2.Lightgbm
https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/plot_example.py
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html
3.other factors
'''
import argparse
import torch
import torch.nn as nn
import numpy as np
import itertools , random , os, shutil , gc , time , h5py , yaml

from torch.optim.swa_utils import AveragedModel , update_bn
from my_utils import lr_cosine_scheduler , Mydataset , multiloss_calculator , versatile_storage
from gen_data import load_trading_data
from environ import get_logger , get_config
from tqdm import tqdm
from scipy import stats
from copy import deepcopy

# from globalvars import *
from function import *
from mymodel import *
# from audtorch.metrics.functional import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TIME_RECODER = False
logger = get_logger()
config = get_config()

torch.set_default_dtype(getattr(torch , config['PRECISION']))
torch.backends.cuda.matmul.allow_tf32 = config['ALLOW_TF32']
torch.autograd.set_detect_anomaly(config['DETECT_ANOMALY'])

storage_model  = versatile_storage(config['STORAGE_TYPE'])
storage_loader = versatile_storage(config['STORAGE_TYPE'])

class ShareNames_conctroller():
    """
    1. Assign variables into shared namespace.
    2. Ask what process would anyone want to run : 0 : train & test(default) , 1 : train only , 2 : test only , 3 : copy to instance only
    3. Ask if model_name and model_base_path should be changed if old dir exists
    """
    def __init__(self):
        self.assign_variables(if_process = True , if_rawname = True)
        
    def assign_variables(self , if_process = False , if_rawname = False):
        ShareNames.max_epoch       = config['MAX_EPOCH']
        ShareNames.batch_size      = config['BATCH_SIZE']
        ShareNames.precision       = getattr(torch , config['PRECISION'])
        ShareNames.allow_tf32      = config['ALLOW_TF32']
        
        ShareNames.model_module    = config['MODEL_MODULE']
        ShareNames.model_data_type = config['MODEL_DATATYPE'][ShareNames.model_module]
        ShareNames.model_nickname  = config['MODEL_NICKNAME']
        
        ShareNames.model_num_list  = list(range(config['MODEL_NUM']))
        ShareNames.data_type_list  = ShareNames.model_data_type.split('+')
        
        ShareNames.model_name      = self._model_name()
        ShareNames.model_base_path = f'./model/{ShareNames.model_name}'
        ShareNames.instance_path   = f'./instance/{ShareNames.model_name}'
        ShareNames.batch_dir       = {k:f'./data/{k}_batch_path' for k in ['train' , 'valid' , 'test']}
        
        if if_process  : self._process_confirmation()
        if if_rawname  : self._rawname_confirmation()
        
        ShareNames.train_params = deepcopy(config['TRAIN_PARAM'])
        ShareNames.compt_params = deepcopy(config['COMPT_PARAM'])
        ShareNames.raw_model_params = deepcopy(config['MODEL_PARAM'])
        ShareNames.model_params = self._load_model_param()
        ShareNames.output_types = ShareNames.train_params['output_types']

    def _model_name(self):
        name_element = [
            ShareNames.model_module ,
            ShareNames.model_data_type , 
            ShareNames.model_nickname
        ]
        return '_'.join([x for x in name_element if x is not None])
                          
    def _load_model_param(self):
        """
        Load and return model_params of each model_num , or save one for later use
        """
        try:
            model_params = torch.load(f'{ShareNames.model_base_path}/model_params.pt')
        except:
            model_params = []
            for mm in ShareNames.model_num_list:
                dict_mm = {'path':f'{ShareNames.model_base_path}/{mm}'}
                dict_mm.update({k:(v[mm % len(v)] if isinstance(v,list) else v) for k,v in ShareNames.raw_model_params.items()})
                model_params.append(dict_mm)
        return model_params
        
    def _process_confirmation(self):
        if ShareNames.process < 0:
            logger.critical(f'What process would you want to run? 0: all (default), 1: train only, 2: test only , 3: copy to instance')
            promt_text = f'[0,all] , [1,train] , [2,test] , [3,instance]: '
            _text , _cond = ask_for_confirmation(promt_text , proceed_condition = lambda x:False)
            key = _text[0]
        else:
            key = str(ShareNames.process)

        if key in ['' , '0' , 'all']:
            ShareNames.process_queue = ['data' , 'train' , 'test' , 'instance']
        elif key in ['1' , 'train']:
            ShareNames.process_queue = ['data' , 'train']
        elif key in ['2' , 'test']:
            ShareNames.process_queue = ['data' , 'test' , 'instance']
        elif key in ['3' , 'instance']:
            ShareNames.process_queue = ['data' , 'instance']
        else:
            raise Exception(f'Error input : {key}')
        logger.critical('Process Queue : {:s}'.format(' + '.join(map(lambda x:(x[0].upper() + x[1:]),ShareNames.process_queue))))
                
    def _rawname_confirmation(self , recurrent = 1):
        """
        Confirm the model_name and model_base_path if multifple model_name dirs exists.
        If include train: check if dir of model_name exists, if so ask to remove the old ones or continue with a sequential one
        If test only :    check if model_name exists multiple dirs, if so ask to use the raw one or the last one(default)
        Also ask if resume training, since unexpected end of training may happen
        """
        if_rawname = None if (ShareNames.rawname < 0) else (ShareNames.rawname > 0)
        if_resume  = None if (ShareNames.resume < 0)  else (ShareNames.resume > 0)
        
        if 'train' in ShareNames.process_queue:
            if os.path.exists(ShareNames.model_base_path) == False:
                if_rawname = True
                if_resume = False
              
            if if_resume is None:
                logger.critical(f'[{ShareNames.model_base_path}] exists, input [yes] to resume training, or start a new one!')
                promt_text = f'Confirm resume training [{ShareNames.model_name}]? [yes/no] : '
                _text , _cond = ask_for_confirmation(promt_text ,  recurrent = recurrent)
                if_resume = all([_t.lower() in ['' , 'yes' , 'y'] for _t in _text])
            
            if if_resume:
                logger.critical(f'Resume training {ShareNames.model_name}!') 
                file_appendix = sorted([int(x.split('.')[-1]) for x in os.listdir(f'./model') if x.startswith(ShareNames.model_name + '.')])
                if if_rawname is None and len(file_appendix) > 0:
                    logger.critical(f'Multiple model path of {ShareNames.model_name} exists, input [yes] to confirm using the raw one, or [no] the latest!')
                    promt_text = f'Use the raw one? [yes/no] : '
                    _text , _cond = ask_for_confirmation(promt_text ,  recurrent = recurrent)
                    if_rawname = all([_t.lower() in ['' , 'yes' , 'y'] for _t in _text])
                    
                if if_rawname or len(file_appendix) == 0:
                    logger.critical(f'model_name is still {ShareNames.model_name}!') 
                else:
                    ShareNames.model_name = f'{ShareNames.model_name}.{file_appendix[-1]}'
                    ShareNames.model_base_path = f'./model/{ShareNames.model_name}'
                    logger.critical(f'model_name is now {ShareNames.model_name}!')
            else:
                if if_rawname is None:
                    logger.critical(f'[{ShareNames.model_base_path}] exists, input [yes] to confirm deletion, or a new directory will be made!')
                    promt_text = f'Confirm Deletion of all old directories with model name [{ShareNames.model_name}]? [yes/no] : '
                    _text , _cond = ask_for_confirmation(promt_text ,  recurrent = recurrent)
                    if_rawname = all([_t.lower() in ['' , 'yes' , 'y'] for _t in _text])

                if if_rawname:
                    rmdir([f'./model/{d}' for d in os.listdir(f'./model') if d.startswith(ShareNames.model_name)])
                    logger.critical(f'Directories of [{ShareNames.model_name}] deletion Confirmed!')
                else:
                    ShareNames.model_name += '.'+str(max([1]+[int(d.split('.')[-1])+1 for d in os.listdir(f'./model') if d.startswith(ShareNames.model_name+'.')]))
                    ShareNames.model_base_path = f'./model/{ShareNames.model_name}'
                    logger.critical(f'A new directory [{ShareNames.model_name}] will be made!')

                os.makedirs(ShareNames.model_base_path, exist_ok = True)
                [os.makedirs(f'{ShareNames.model_base_path}/{mm}' , exist_ok = True) for mm in ShareNames.model_num_list]
                for copy_filename in ['configs/config_train.yaml']:
                    shutil.copyfile(f'./{copy_filename}', f'{ShareNames.model_base_path}/{os.path.basename(copy_filename)}')
                    
        elif 'test' in ShareNames.process_queue:
            file_appendix = sorted([int(x.split('.')[-1]) for x in os.listdir(f'./model') if x.startswith(ShareNames.model_name + '.')])
            if if_rawname is None and len(file_appendix) > 0:
                logger.critical(f'Multiple model path of {ShareNames.model_name} exists, input [yes] to confirm using the raw one, or [no] the latest!')
                promt_text = f'Use the raw one? [yes/no] : '
                _text , _cond = ask_for_confirmation(promt_text ,  recurrent = recurrent)
                if_rawname = all([_t.lower() in ['' , 'yes' , 'y'] for _t in _text])

            if if_rawname or len(file_appendix) == 0:
                logger.critical(f'model_name is still {ShareNames.model_name}!') 
            else:
                ShareNames.model_name = f'{ShareNames.model_name}.{file_appendix[-1]}'
                ShareNames.model_base_path = f'./model/{ShareNames.model_name}'
                logger.critical(f'model_name is now {ShareNames.model_name}!')
                
        ShareNames.resume_training = if_resume
                
class model_controller():
    """
    A class to control the whole process of training , includes:
    1. Display controls: tqdm , once , step
    2. Parameters: train_params , compt_params , model_data_type
    3. Data : class of train_data
    3. loop status: model , round , attempt , epoch
    4. file path: model , lastround , transfer(last model date)
    5. text: model , round , attempt , epoch , exit , stat , time , trainer
    """
    def __init__(self , **kwargs):
        self.model_info = dict()
        self.model_info['global_start_time'] = time.ctime()
        self.model_info['config'] = config

        self.init_time = time.time()
        
        self.display = {
            'tqdm' : True if config['VERBOSITY'] >= 10 else False ,
            'once' : True if config['VERBOSITY'] <=  2 else False ,
            'step' : [10,5,5,3,3,1][min(config['VERBOSITY'] // 2 , 5)],
        }
        self.process_time = {}
        self.shared_ctrl = ShareNames_conctroller()
        
    def main_process(self):
        """
        Main process of load_data + train + test + instance
        """
        for process_name in ShareNames.process_queue:
            self.SetProcessName(process_name)
            self.__getattribute__(f'model_process_{process_name.lower()}')()
            rmdir([v for v in ShareNames.batch_dir.values()] , remake_dir = True)
    
    def SetProcessName(self , key = 'data'):
        ShareNames.process_name = key.lower()
        self.model_count = 0
        self.epoch_count = 0
        if 'data' in vars(self) : self.data.reset_dataloaders()
        if ShareNames.process_name == 'data': 
            pass
        elif ShareNames.process_name == 'train': 
            self.f_loss    = loss_function(ShareNames.train_params['criterion']['loss'])
            self.f_metric  = metric_function(ShareNames.train_params['criterion']['metric'])
            self.f_penalty = {k:[penalty_function(k),v] for k,v in ShareNames.train_params['criterion']['penalty'].items() if v > 0.}
        elif ShareNames.process_name == 'test':
            self.f_metric  = metric_function(ShareNames.train_params['criterion']['metric'])
            self.ic_by_date , self.ic_by_model = None , None
        elif ShareNames.process_name == 'instance':
            self.ic_by_date , self.ic_by_model = None , None
        else:
            raise Exception(f'KeyError : {key}')
        
    def model_process_data(self):
        """
        Main process of loading basic data
        """
        self.data_time = time.time()
        logger.critical(f'Start Process [Load Data]!')
        self.data = ModelData()
        logger.critical('Finish Process [Load Data]! Cost {:.1f}Secs'.format(time.time() - self.data_time))
        
    def model_process_train(self):
        """
        Main process of training
        1. loop over model(model_date , model_num)
        2. loop over round(if necessary) , attempt(if converge too soon) , epoch(most prevailing loops)
        """
        self.model_info['train_start_time'] = time.ctime()
        self.train_time = time.time()
        logger.critical(f'Start Process [Train Model]!')
        self.printer('model_specifics')
        logger.error(f'Start Training Models!')
        torch.save(ShareNames.model_params , f'{ShareNames.model_base_path}/model_params.pt')    
        for model_date , model_num in self.ModelIter():
            self.model_date , self.model_num = model_date , model_num
            self.ModelPreparation('train')
            self.TrainModel()
        total_time = time.time() - self.train_time
        train_process = 'Finish Process [Train Model]! Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
            total_time / 3600 , total_time / 60 / max(self.model_count , 1) , total_time / max(self.epoch_count , 1))
        logger.critical(train_process)
        self.model_info['train_process'] = train_process

    def model_process_test(self):
        self.model_info['test_start_time'] = time.ctime()
        self.test_time = time.time()
        logger.critical(f'Start Process [Test Model]!')        
        logger.warning('Each Model Date Testing Mean Rank_ic:')
        self.test_result_model_num = np.repeat(ShareNames.model_num_list,len(ShareNames.output_types))
        self.test_result_output_type = np.tile(ShareNames.output_types,len(ShareNames.model_num_list))
        logger.info('{: <11s}'.format('Models') + ('{: >8d}'*len(self.test_result_model_num)).format(*self.test_result_model_num))
        logger.info('{: <11s}'.format('Output') + ('{: >8s}'*len(self.test_result_model_num)).format(*self.test_result_output_type))
        for model_date , model_num in self.ModelIter():
            self.model_date , self.model_num = model_date , model_num
            self.ModelPreparation('test')
            self.TestModel()
        self.ModelResult()
        test_process = 'Finish Process [Test Model]! Cost {:.1f} Secs'.format(time.time() - self.test_time)
        logger.critical(test_process)
        self.model_info['test_process'] = test_process

    def model_process_instance(self):
        if ShareNames.anchoring < 0:
            logger.critical(f'Do you want to copy the model to instance?')
            promt_text = f'[yes/else no]: '
            _text , _cond = ask_for_confirmation(promt_text , timeout = -1)
            anchoring = all([_t.lower() in ['yes','y'] for _t in _text])
        else:
            anchoring = ShareNames.anchoring > 0
        if anchoring == 0:
            logger.critical(f'Will not copy to instance!')
            return
        else:
            self.instance_time = time.time()
            logger.critical(f'Start Process [Copy to Instance]!')        
            if os.path.exists(ShareNames.instance_path): 
                logger.critical(f'Old instance {ShareNames.instance_path} exists , remove manually first to override!')
                logger.critical(f'The command can be "rm -r {ShareNames.instance_path}"')
                return
            else:
                shutil.copytree(ShareNames.model_base_path , ShareNames.instance_path)
                
        logger.warning('Copy from model to instance finished , Start going forward')
        self.InstanceStart()
        for model_date , model_num in self.ModelIter():
            self.model_date , self.model_num = model_date , model_num
            self.ModelPreparation('instance')
            self.TestModel()
            self.StorePreds()
        self.ModelResult()
        logger.critical('Finish Process [Copy to Instance]! Cost {:.1f} Secs'.format(time.time() - self.instance_time))  
        
    def print_vars(self):
        print(vars(self))

    def ModelIter(self):
        model_iter = itertools.product(ShareNames.model_date_list , ShareNames.model_num_list)
        if ShareNames.resume_training and (ShareNames.process_name == 'train'):
            models_trained = [os.path.exists(f'{ShareNames.model_base_path}/{mn}/{md}.pt') for md,mn in model_iter]
            if models_trained[0] == 0: 
                models_trained[:] = False
            else:
                resume_point = -1 if all(models_trained) else (np.where(np.array(models_trained) == 0)[0][0] - 1)
                models_trained[resume_point:] = False
            model_iter = FilteredIterator(itertools.product(ShareNames.model_date_list , ShareNames.model_num_list), iter(models_trained == 0))
        return model_iter
    
    def ModelPreparation(self , process , last_n = 30 , best_n = 5):
        assert process in ['train' , 'test' , 'instance']
        _start_time = time.time()
        param = ShareNames.model_params[self.model_num]
        
        # variable updates for train_params
        if process in ['train' , 'instance']:
            if 'hidden_orthogonality' in self.f_penalty.keys(): self.f_penalty['hidden_orthogonality'][1] = 1 * (param.get('hidden_as_factors') == True)
        
        path_prefix = '{}/{}'.format(param.get('path') , self.model_date)
        path = {k:f'{path_prefix}.{k}.pt' for k in ShareNames.output_types} #['best','swalast','swabest']
        path.update({f'src_model.{k}':[] for k in ShareNames.output_types})
        if 'swalast' in ShareNames.output_types: 
            path['lastn'] = [f'{path_prefix}.lastn.{i}.pt' for i in range(last_n)]
        if 'swabest' in ShareNames.output_types: 
            path['bestn'] = [f'{path_prefix}.bestn.{i}.pt' for i in range(best_n)]
            path['bestn_ic'] = [-10000. for i in range(best_n)]
        
        if ShareNames.train_params['transfer'] and self.model_date > ShareNames.model_date_list[0]:
            path['transfer'] = '{}/{}.best.pt'.format(param.get('path') , max([d for d in ShareNames.model_date_list if d < self.model_date])) 
            
        self.Param = param
        self.path = path
        self.time_recoder(_start_time , ['ModelPreparation' , process])
    
    def TrainModel(self):
        self.TrainModelStart()
        while self.cond.get('loop_status') != 'model':
            self.NewLoop()
            self.TrainerInit()
            self.TrainEpoch()
            self.LoopCondition()
        self.TrainModelEnd()
        gc.collect() , torch.cuda.empty_cache()
    
    def TestModel(self):
        self.TestModelStart()
        self.Forecast()
        self.TestModelEnd()
        gc.collect() , torch.cuda.empty_cache()
        
    def TrainModelStart(self):
        """
        Reset model specific variables
        """
        _start_time = time.time()
        
        self._init_variables('model')
        self.nanloss_life = ShareNames.train_params['trainer']['nanloss']['retry']
        
        self.text['model'] = '{:s} #{:d} @{:4d}'.format(ShareNames.model_name , self.model_num , self.model_date)

        if (self.data.dataloader_param != (self.model_date , self.Param['seqlens'])):
            self.data.new_train_dataloader(self.model_date , self.Param['seqlens']) 
            self.time[1] = time.time()
            self.printer('train_dataloader')
            
        self.time_recoder(_start_time , ['TrainModelStart'])
            
    def TrainModelEnd(self):
        """
        Do necessary things of ending a model(model_data , model_num)
        """
        _start_time = time.time()
        
        storage_model.del_path(self.path.get('rounds') , self.path.get('lastn') , self.path.get('bestn'))
        if ShareNames.process_name == 'train' : self.model_count += 1
        self.time[2] = time.time()
        self.printer('model_end')
        
        self.time_recoder(_start_time , ['TrainModelEnd'])
        
    def NewLoop(self):
        """
        Reset and loop variables giving loop_status
        """
        _start_time = time.time()
        
        self._init_variables(self.cond.get('loop_status'))
        self.epoch_i += 1
        self.epoch_all += 1
        self.epoch_count += 1
        if self.cond.get('loop_status') in ['attempt' , 'round']:
            self.attempt_i += 1
            self.text['attempt'] = f'FirstBite' if self.attempt_i == 0 else f'Retrain#{self.attempt_i}'
        if self.cond.get('loop_status') in ['round']:
            self.round_i += 1
            self.text['round'] = 'Round{:2d}'.format(self.round_i)
            
        self.time_recoder(_start_time , ['NewLoop'])
        
    def TrainerInit(self):
        """
        Initialize net , optimizer , scheduler if loop_status in ['round' , 'attempt']
        net : 1. Create an instance of f'My{ShareNames.model_module}' or inherit from 'lastround'/'transfer'
              2. In transfer mode , p_late and p_early with be trained with different lr's. If not net.parameters are trained by same lr
        optimizer : Adam or SGD
        scheduler : Cosine or StepLR
        """
        _start_time = time.time()

        if self.cond.get('loop_status') == 'epoch': return
        self.net       = self.load_model('train')
        self.max_round = self.net.max_round() if 'max_round' in self.net.__dir__() else 1
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler() 
        self.multiloss = self.load_multiloss()

        self.time_recoder(_start_time , ['TrainerInit'])
        
    def TrainEpoch(self):
        """
        Iterate train and valid dataset, calculate loss/metrics , update values
        If nan loss occurs, turn to _deal_nanloss
        """
        _start_time = time.time()
        loss_train , loss_valid , ic_train , ic_valid = [] , [] , [] , []
        clip_value = ShareNames.train_params['trainer']['gradient'].get('clip_value')
        
        if self.display.get('tqdm'):
            iter_train , iter_valid = tqdm(self.data.dataloaders['train']) , tqdm(self.data.dataloaders['valid'])
            disp_train = lambda x:iter_train.set_description(f'Ep#{self.epoch_i:3d} train loss:{np.mean(x):.5f}')
            disp_valid = lambda x:iter_valid.set_description(f'Ep#{self.epoch_i:3d} valid ic:{np.mean(x):.5f}')
        else:
            iter_train , iter_valid = self.data.dataloaders['train'] , self.data.dataloaders['valid']
            disp_train = disp_valid = lambda x:0

        self.time_recoder(_start_time , ['TrainEpoch' , 'assign_loader'])
        _start_time = time.time()

        self.net.train()
        _start_time_1 = time.time()
        for i , (x , y) in enumerate(iter_train):
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'train' , 'fetch'])
            self.optimizer.zero_grad()
            _start_time_1 = time.time()
            pred , hidden = self.net(x)
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'train' , 'forward'])
            _start_time_1 = time.time()
            loss , metric = self._loss_and_metric(y , pred , 'train' , hidden = hidden)
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'train' , 'loss'])
            _start_time_1 = time.time()
            loss.backward()
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'train' , 'backward'])
            _start_time_1 = time.time()
            if clip_value is not None : nn.utils.clip_grad_value_(self.net.parameters(), clip_value = clip_value)
            self.optimizer.step()

            loss_train.append(loss.item()) , ic_train.append(metric)
            disp_train(loss_train)
            _start_time_1 = time.time()
        if np.isnan(sum(loss_train)): return self._deal_nanloss()
        self.loss_list['train'].append(np.mean(loss_train)) , self.ic_list['train'].append(np.mean(ic_train))
        
        self.time_recoder(_start_time , ['TrainEpoch' , 'train_epochs'])
        _start_time = time.time()

        self.net.eval()     
        _start_time_1 = time.time()  
        for i , (x , y) in enumerate(iter_valid):
            # print(torch.cuda.memory_allocated(DEVICE) / 1024**3 , torch.cuda.memory_reserved(DEVICE) / 1024**3)
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'valid' , 'fetch'])
            _start_time_1 = time.time()
            pred , _ = self.net(x)
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'valid' , 'forward'])
            _start_time_1 = time.time()
            loss , metric = self._loss_and_metric(y , pred , 'valid')
            self.time_recoder(_start_time_1 , ['TrainEpoch' , 'valid' , 'loss'])
            _start_time_1 = time.time()
            loss_valid.append(loss) , ic_valid.append(metric)
            disp_valid(ic_valid)
            _start_time_1 = time.time()
        self.loss_list['valid'].append(np.mean(loss_valid)) , self.ic_list['valid'].append(np.mean(ic_valid))
        self.lr_list.append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()
        self.reset_scheduler()

        self.time_recoder(_start_time , ['TrainEpoch' , 'valid_epochs'])

    def LoopCondition(self):
        """
        Update condition of continuing training epochs , restart attempt if early exit , proceed to next round if convergence , reset round if nan loss
        """
        _start_time = time.time()

        if self.cond['nan_loss']:
            logger.error(f'Initialize a new model to retrain! Lives remaining {self.nanloss_life}')
            self._init_variables('model')
            self.cond['loop_status'] = 'round'
            return
            
        valid_ic = self.ic_list['valid'][-1]
        
        save_targets = [] 
        if valid_ic > self.ic_attempt_best: 
            self.epoch_attempt_best  = self.epoch_i 
            self.ic_attempt_best = valid_ic
            
        if valid_ic > self.ic_round_best:
            self.ic_round_best = valid_ic
            self.path['src_model.best']  = [self.path['best']]
            save_targets.append(self.path['best'])

        if 'swalast' in ShareNames.output_types:
            self.path['lastn'] = self.path['lastn'][1:] + self.path['lastn'][:1]
            save_targets.append(self.path['lastn'][-1])
            
            p_valid = self.path['lastn'][-len(self.ic_list['valid']):]
            arg_max = np.argmax(self.ic_list['valid'][-len(p_valid):])
            arg_swa = (lambda x:x[(x>=0) & (x<len(p_valid))])(min(5,len(p_valid)//3)*np.arange(-5,3)+arg_max)[-5:]
            self.path['src_model.swalast'] = [p_valid[i] for i in arg_swa]
            
        if 'swabest' in ShareNames.output_types:
            arg_min = np.argmin(self.path['bestn_ic'])
            if valid_ic > self.path['bestn_ic'][arg_min]:
                self.path['bestn_ic'][arg_min] = valid_ic
                save_targets.append(self.path['bestn'][arg_min])
                if self.path['bestn'][arg_min] not in self.path['src_model.swabest']: self.path['src_model.swabest'].append(self.path['bestn'][arg_min])
            
        storage_model.save_model_state(self.net , save_targets)
        self.printer('epoch_step')
        self.time_recoder(_start_time , ['LoopCondition' , 'assess'])
        _start_time = time.time()
        
        self.cond['terminate'] = {k:self._terminate_cond(k,v) for k , v in ShareNames.train_params['terminate'].get('overall' if self.max_round <= 1 else 'round').items()}
        if any(self.cond.get('terminate').values()):
            self.text['exit'] = {
                'max_epoch'      : 'Max Epoch' , 
                'early_stop'     : 'EarlyStop' ,
                'tv_converge'    : 'T&V Convg' , 
                'train_converge' : 'Tra Convg' , 
                'valid_converge' : 'Val Convg' ,
            }[[k for k,v in self.cond.get('terminate').items() if v][0]] 
            if (self.epoch_i < ShareNames.train_params['trainer']['retrain'].get('min_epoch' if self.max_round <= 1 else 'min_epoch_round') - 1 and 
                self.attempt_i < ShareNames.train_params['trainer']['retrain']['attempts'] - 1):
                self.cond['loop_status'] = 'attempt'
                self.printer('new_attempt')
            elif self.round_i < self.max_round - 1:
                self.cond['loop_status'] = 'round'
                self.save_model('best')
                self.printer('new_round')
            else:
                self.cond['loop_status'] = 'model'
                self.save_model(ShareNames.output_types)
        else:
            self.cond['loop_status'] = 'epoch'

        _start_time = time.time()
        self.time_recoder(_start_time , ['LoopCondition' , 'confirm_status'])
        
            
    def TestModelStart(self):
        """
        Reset model specific variables
        """
        self._init_variables('model')        
        if (self.data.dataloader_param != (self.model_date , self.Param['seqlens'])):
            self.data.new_test_dataloader(self.model_date , self.Param['seqlens'])
            
        if self.model_num == 0:
            ic_date_0 = np.zeros((len(self.data.model_test_dates) , len(self.test_result_model_num)))
            ic_model_0 =  np.zeros((1 , len(self.test_result_model_num)))
            self.ic_by_date = ic_date_0 if self.ic_by_date is None else np.concatenate([self.ic_by_date , ic_date_0])
            self.ic_by_model = ic_model_0 if self.ic_by_model is None else np.concatenate([self.ic_by_model , ic_model_0])
                
    def Forecast(self):
        if not os.path.exists(self.path['best']): self.TrainModel()
        
        #self.y_pred = cuda(torch.zeros(self.data.stock_n,len(self.data.model_test_dates),self.data.labels_n,len(ShareNames.output_types)).fill_(np.nan))
        self.y_pred = cuda(torch.zeros(self.data.stock_n,len(self.data.model_test_dates),len(ShareNames.output_types)).fill_(np.nan))
        for oi , okey in enumerate(ShareNames.output_types):
            self.net = self.load_model('test' , okey)
            self.net.eval()

            if self.display.get('tqdm'):
                iter_test = tqdm(self.data.dataloaders['test'])
                disp_test = lambda x:iter_test.set_description(f'Date#{x[0]:3d} :{np.mean(x[1]):.5f}')
            else:
                iter_test = self.data.dataloaders['test']
                disp_test = lambda x:0

            m_test = []         
            with torch.no_grad():
                for i , (x , y) in enumerate(iter_test):
                    stock_pos = np.where(self.data.test_nonnan_sample[:,i])[0]
                    for batch_j in torch.utils.data.DataLoader(np.arange(len(y)) , batch_size = ShareNames.batch_size):
                        x_j = tuple([xx[batch_j] for xx in x]) if isinstance(x , tuple) else x[batch_j]
                        output , _ = self.net(x_j)
                        self.y_pred[stock_pos[batch_j],i,oi] = output.select(-1,0).detach()
                    metric = self.f_metric(y.select(-1,0) , self.y_pred[stock_pos,i,oi]).item()
                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                    m_test.append(metric) 
                    disp_test((i , m_test))
                    
            self.ic_by_date[-len(self.data.model_test_dates):,self.model_num*len(ShareNames.output_types) + oi] = torch.tensor(m_test).nan_to_num(0).cpu().numpy()   
        self.y_pred = self.y_pred.cpu().numpy()
        
    def TestModelEnd(self):
        """
        Do necessary things of ending a model(model_data , model_num)
        """
        if self.model_num == ShareNames.model_num_list[-1]:
            self.ic_by_model[-1,:] = np.nanmean(self.ic_by_date[-len(self.data.model_test_dates):,],axis = 0)
            logger.info('{: <11d}'.format(self.model_date)+('{:>8.4f}'*len(self.test_result_model_num)).format(*self.ic_by_model[-1,:]))
        #if False:
        #    df = pd.DataFrame(self.y_pred.T, index = self.data.model_test_dates, columns = self.data.index_stock.astype(str))
        #    with open(f'{ShareNames.instance_path}/{ShareNames.model_name}_fac{self.model_num}.csv', 'a') as f:
        #        df.to_csv(f , mode = 'a', header = f.tell()==0, index = True)

    def ResultOutput(self):
        out_dict = {
            '0_start':self.model_info.get('global_start_time'),
            '1_basic':'+'.join([
                'short' if config['SHORTTEST'] else 'long' ,
                config['STORAGE_TYPE'] , config['PRECISION']
            ]),
            '2_model':''.join([
                ShareNames.model_module , '_' , ShareNames.model_data_type ,
                '(x' , str(config['MODEL_NUM']) , ')'
            ]),
            '3_time':'-'.join([str(config['BEG_DATE']),str(config['END_DATE'])]),
            '4_typeNN':'+'.join(list(set(config['MODEL_PARAM']['type_rnn']))),
            '5_train':self.model_info.get('train_process'),
            '6_test':self.model_info.get('test_process'),
            '7_result':self.model_info.get('test_ic_sum'),
        }

        out_path = f'./results/model_results.yaml'
        if os.path.exists(out_path):
            out_type = 'a'
        else:
            os.makedirs(os.path.dirname(out_path) , exist_ok=True)
            out_type = 'w'
        
        with open(out_path , out_type) as f:
            yaml.dump(out_dict , f)


    def StorePreds(self):
        assert ShareNames.process_name == 'instance'
        if self.model_num == 0:
            self.y_pred_models = []
            gc.collect()
        self.y_pred_models.append(self.y_pred)
        if self.model_num == ShareNames.model_num_list[-1]:
            self.y_pred_models = np.concatenate(self.y_pred_models,axis=-1).transpose(1,0,2)
            # idx = np.array(np.meshgrid(self.data.model_test_dates , self.data.index_stock)).T.reshape(-1,2)
            mode = 'r+' if os.path.exists(f'{ShareNames.instance_path}/{ShareNames.model_name}.h5') else 'w'
            with h5py.File(f'{ShareNames.instance_path}/{ShareNames.model_name}.h5' , mode = mode) as f:
                for di in range(len(self.data.model_test_dates)):
                    arr , row = self.y_pred_models[di] , self.data.index_stock 
                    arr , row = arr[np.isnan(arr).all(axis=1) == 0] , row[np.isnan(arr).all(axis=1) == 0]
                    col = [f'{mn}.{o}' for mn,o in zip(self.test_result_model_num,self.test_result_output_type)]
                    if str(self.data.model_test_dates[di]) in f.keys():
                        del f[str(self.data.model_test_dates[di])]
                    g = f.create_group(str(self.data.model_test_dates[di]))
                    g.create_dataset('arr' , data=arr , compression='gzip')
                    g.create_dataset('row' , data=row , compression='gzip')
                    g.create_dataset('col' , data=col , compression='gzip')     
  
    def ModelResult(self):
        # date ic writed down
        _step = (1 if ShareNames.process_name == 'instance' else self.data.test_step)
        _dates_list = ShareNames.test_full_dates[::_step]
        for model_num in ShareNames.model_num_list:
            df = {'dates' : _dates_list}
            for oi , okey in enumerate(ShareNames.output_types):
                df.update({f'rank_ic.{okey}' : self.ic_by_date[:,model_num*len(ShareNames.output_types) + oi], 
                           f'cum_ic.{okey}' : np.nancumsum(self.ic_by_date[:,model_num*len(ShareNames.output_types) + oi])})
            df = pd.DataFrame(df , index = map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , _dates_list.astype(str)))
            df.to_csv(ShareNames.model_params[model_num]['path'] + f'/{ShareNames.model_name}_ic_by_date_{model_num}.csv')

        # model ic presentation
        add_row_key   = ['AllTimeAvg' , 'AllTimeSum' , 'Std'      , 'TValue'   , 'AnnIR']
        add_row_fmt   = ['{:>8.4f}'   , '{:>8.2f}'   , '{:>8.4f}' , '{:>8.2f}' , '{:>8.4f}']
        ic_mean   = np.nanmean(self.ic_by_date , axis = 0)
        ic_sum    = np.nansum(self.ic_by_date , axis = 0) 
        ic_std    = np.nanstd(self.ic_by_date , axis = 0)
        ic_tvalue = ic_mean / ic_std * (len(self.ic_by_date)**0.5) # 10 days return predicted
        ic_annir  = ic_mean / ic_std * ((240 / 10)**0.5) # 10 days return predicted
        add_row_value = (ic_mean , ic_sum , ic_std , ic_tvalue , ic_annir)
        df = pd.DataFrame(np.concatenate([self.ic_by_model , np.stack(add_row_value)]) , 
                          index = [str(d) for d in ShareNames.model_date_list] + add_row_key , 
                          columns = [f'{mn}.{o}' for mn,o in zip(self.test_result_model_num,self.test_result_output_type)])
        df.to_csv(f'{ShareNames.model_base_path}/{ShareNames.model_name}_ic_by_model.csv')
        for i in range(len(add_row_key)):
            logger.info('{: <11s}'.format(add_row_key[i]) + (add_row_fmt[i]*len(self.test_result_model_num)).format(*add_row_value[i]))
    
        self.model_info['test_ic_sum'] = {k:v for k,v in zip(df.columns , ic_sum.tolist())}

    def InstanceStart(self):
        exec(open(f'{ShareNames.instance_path}/globalvars.py').read())
        self.shared_ctrl.assign_variables()
        for mm in range(len(ShareNames.model_params)): ShareNames.model_params[mm].update({'path':f'{ShareNames.instance_path}/{mm}'})
    
    def printer(self , key):
        """
        Print out status giving display conditions and looping conditions
        """
        _detail_print = (self.display.get('once') == 0 or self.model_count <= max(ShareNames.model_num_list))
        if key == 'model_specifics':
            logger.warning('Model Parameters:')
            logger.info(f'Basic Parameters : ')
            print(f'STORAGE [{config["STORAGE_TYPE"]}] | DEVICE [{DEVICE}] | PRECISION [{ShareNames.precision}] | BATCH_SIZE [{ShareNames.batch_size}].') 
            print(f'NAME [{ShareNames.model_name}] | MODULE [{ShareNames.model_module}] | DATATYPE [{ShareNames.model_data_type}] | MODEL_NUM [{len(ShareNames.model_num_list)}].')
            print(f'BEG_DATE [{config["BEG_DATE"]}] | END_DATE [{ShareNames.test_full_dates[-1]}] | ' +
                  f'INTERVAL [{config["INTERVAL"]}] | INPUT_STEP_DAY [{config["INPUT_STEP_DAY"]}] | TEST_STEP_DAY [{config["TEST_STEP_DAY"]}].') 
            logger.info(f'MODEL_PARAM : ')
            pretty_print_dict(ShareNames.raw_model_params)
            logger.info(f'TRAIN_PARAM : ')
            pretty_print_dict(ShareNames.train_params)
            logger.info(f'COMPT_PARAM : ')
            pretty_print_dict(ShareNames.compt_params)
        elif key == 'model_end':
            self.text['epoch'] = 'Ep#{:3d}'.format(self.epoch_all)
            self.text['stat']  = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(self.ic_list['train'][-1],self.ic_list['valid'][-1],self.ic_round_best)
            self.text['time']  = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format((self.time[2]-self.time[0])/60 , (self.time[2]-self.time[1])/(self.epoch_all+1))
            sdout = self.text['model'] + '|' + self.text['round'] + ' ' + self.text['attempt'] + ' ' +\
            self.text['epoch'] + ' ' + self.text['exit'] + '|' + self.text['stat'] + '|' + self.text['time']
            logger.warning(sdout)
        elif key == 'epoch_step':
            self.text['trainer'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, max{: .4f}, best{: .4f}, lr{:.1e}'.format(
                self.loss_list['train'][-1] , self.ic_list['train'][-1] , self.ic_list['valid'][-1] , self.ic_attempt_best , self.ic_round_best , self.lr_list[-1])
            if self.epoch_i % self.display.get('step') == 0:
                sdout = ' '.join([self.text['attempt'],'Ep#{:3d}'.format(self.epoch_i),':', self.text['trainer']])
                logger.info(sdout) if _detail_print else logger.debug(sdout) 
        elif key == 'reset_learn_rate':
            speedup = ShareNames.train_params['trainer']['learn_rate']['reset']['speedup2x']
            sdout = 'Reset learn rate and scheduler at the end of epoch {} , effective at epoch {}'.format(self.epoch_i , self.epoch_i+1 , ', and will speedup2x' * speedup)
            logger.info(sdout) if _detail_print else logger.debug(sdout) 
        elif key == 'new_attempt':
            sdout = ' '.join([self.text['attempt'],'Epoch #{:3d}'.format(self.epoch_i),':',self.text['trainer'],', Next attempt goes!'])
            logger.info(sdout) if _detail_print else logger.debug(sdout) 
        elif key == 'new_round':
            sdout = self.text['round'] + ' ' + self.text['exit'] + ': ' + self.text['trainer'] + ', Next round goes!'
            logger.info(sdout) if _detail_print else logger.debug(sdout)
        elif key == 'train_dataloader':
            sdout = ' '.join([self.text['model'],'LoadData Cost {:>6.1f}Secs'.format(self.time[1]-self.time[0])])  
            logger.info(sdout) if _detail_print else logger.debug(sdout)
        else:
            raise Exception(f'KeyError : {key}')        
            
    def _init_variables(self , key = 'model'):
        """
        Reset variables of 'model' , 'round' , 'attempt' start
        """
        if key == 'epoch' : return
        assert key in ['model' , 'round' , 'attempt'] , f'KeyError : {key}'

        self.epoch_i = -1
        self.epoch_attempt_best = -1
        self.ic_attempt_best = -10000.
        self.loss_list = {'train' : [] , 'valid' : []}
        self.ic_list   = {'train' : [] , 'valid' : []}
        self.lr_list   = []
        
        if key in ['model' , 'round']:
            self.attempt_i = -1
            self.ic_round_best = -10000.
        
        if key in ['model']:
            self.round_i = -1
            self.epoch_all = -1
            self.time = np.ones(10) * time.time()
            self.text = {k : '' for k in ['model','round','attempt','epoch','exit','stat','time','trainer']}
            self.cond = {'terminate' : {} , 'nan_loss' : False , 'loop_status' : 'round'}
            
    def _loss_and_metric(self, labels , pred , key , **kwargs):
        """
        Calculate loss(with gradient), metric
        Inputs : 
            cal_options : 'l'for loss , 'm' as metric , 'p' for penalty (add to l) , (1,1,1) as default
            kwargs : other inputs used in calculating loss , penalty and metric
        Possible Methods :
        loss:    pearsonr , mse , ccc
        penalty: none , hidden_orthogonality
        metric:  pearsonr , rankic , mse , ccc
        """
        assert key in ['train' , 'valid'] , key
        if labels.shape != pred.shape:
            # if more labels than output
            assert labels.shape[:-1] == pred.shape[:-1] , (labels.shape , pred.shape)
            labels = labels.transpose(0,-1)[:pred.shape[-1]].transpose(0,-1)
            
        if key == 'train':
            if self.Param['num_output'] > 1:
                loss = self.f_loss(labels , pred , dim = 0)[:self.Param['num_output']]
                loss = self.multiloss.calculate_multi_loss(loss , self.net.get_multiloss_params())
            else:
                loss    = self.f_loss(labels.select(-1,0) , pred.select(-1,0))
            metric  = self.f_metric(labels.select(-1,0) , pred.select(-1,0)).item()
            penalty = sum([w * f(**kwargs) for k,(f,w) in self.f_penalty.items() if w != 0])
            loss = loss + penalty  
        else:
            metric  = self.f_metric(labels.select(-1,0) , pred.select(-1,0)).item()
            loss    = 0.
        return loss , metric
    
    def _deal_nanloss(self):
        """
        Deal with nan loss, life -1 and change nan_loss condition to True
        """
        logger.error(f'{self.text["model"]} Attempt{self.attempt_i}, epoch{self.epoch_i} got nan loss!')
        if self.nanloss_life > 0:
            self.nanloss_life -= 1
            self.cond['nan_loss'] = True
        else:
            raise Exception('Nan loss life exhausted, possible gradient explosion/vanish!')
    
    def _terminate_cond(self , key , arg):
        """
        Whether terminate condition meets
        """
        if key == 'early_stop':
            return self.epoch_i - self.epoch_attempt_best >= arg
        elif key == 'train_converge':
            return list_converge(self.loss_list['train'] , arg.get('min_epoch') , arg.get('eps'))
        elif key == 'valid_converge':
            return list_converge(self.ic_list['valid'] , arg.get('min_epoch') , arg.get('eps'))
        elif key == 'tv_converge':
            return (list_converge(self.loss_list['train'] , arg.get('min_epoch') , arg.get('eps')) and
                    list_converge(self.ic_list['valid'] , arg.get('min_epoch') , arg.get('eps')))
        elif key == 'max_epoch':
            return self.epoch_i >= min(arg , ShareNames.max_epoch) - 1
        else:
            raise Exception(f'KeyError : {key}')
    
    def save_model(self , key = 'best'):
        assert isinstance(key , (list,tuple,str))
        _start_time = time.time()
        if isinstance(key , (list,tuple)):
            [self.save_model(k) for k in key]
        else:
            assert key in ['best' , 'swalast' , 'swabest']
            if key == 'best':
                model_state = storage_model.load(self.path['best'])
                if self.round_i < self.max_round - 1:
                    if 'rounds' not in self.path.keys():
                        self.path['rounds'] = ['{}/{}.round.{}.pt'.format(self.Param.get('path') , self.model_date , r) for r in range(self.max_round - 1)]
                    # self.path[f'round.{self.round_i}'] = '{}/{}.round.{}.pt'.format(self.Param.get('path') , self.model_date , self.round_i)
                    storage_model.save(model_state , self.path['rounds'][self.round_i])
                storage_model.save(model_state , self.path['best'] , to_disk = True)
            else:
                p_exists = storage_model.valid_paths(self.path[f'src_model.{key}'])
                if len(p_exists) == 0:
                    print(key , self.path[f'bestn'] , self.path[f'bestn_ic'] , self.path[f'src_model.{key}'])
                    raise Exception(f'Model Error')
                else:
                    model = self.swa_model(p_exists)
                    storage_model.save_model_state(model , self.path[key] , to_disk = True) 
        self.time_recoder(_start_time , ['save_model'])
    
    def load_model(self , process , key = 'best'):
        assert process in ['train' , 'test']
        _start_time = time.time()
        net = globals()[f'My{ShareNames.model_module}'](**self.Param)
        if process == 'train':           
            if self.round_i > 0:
                model_path = self.path['rounds'][self.round_i-1]
            elif 'transfer' in self.path.keys():
                model_path = self.path['transfer']
            else:
                model_path = -1
            if os.path.exists(model_path): net = storage_model.load_model_state(net , model_path , from_disk = True)
            if 'training_round' in net.__dir__(): net.training_round(self.round_i)
        else:
            net = storage_model.load_model_state(net , self.path[key] , from_disk = True)
        net = cuda(net)
        self.time_recoder(_start_time , ['load_model'])
        return net
    
    def swa_model(self , model_path_list = []):
        net = globals()[f'My{ShareNames.model_module}'](**self.Param)
        swa_net = AveragedModel(net)
        for p in model_path_list:
            swa_net.update_parameters(storage_model.load_model_state(net , p))
        swa_net = cuda(swa_net)
        update_bn(self.data.dataloaders['train'] , swa_net)
        return swa_net.module
    
    def load_optimizer(self , new_opt_kwargs = None , new_lr_kwargs = None):
        if new_opt_kwargs is None:
            opt_kwargs = ShareNames.train_params['trainer']['optimizer']
        else:
            opt_kwargs = deepcopy(ShareNames.train_params['trainer']['optimizer'])
            opt_kwargs.update(new_opt_kwargs)
        
        if new_lr_kwargs is None:
            lr_kwargs = ShareNames.train_params['trainer']['learn_rate']
        else:
            lr_kwargs = deepcopy(ShareNames.train_params['trainer']['learn_rate'])
            lr_kwargs.update(new_lr_kwargs)

        base_lr = lr_kwargs['base'] * lr_kwargs['ratio']['attempt'][:self.attempt_i+1][-1] * lr_kwargs['ratio']['round'][:self.round_i+1][-1]
        if 'transfer' in self.path.keys():
            # define param list to train with different learn rate
            p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in self.net.named_parameters() if 'encoder' in x.split('.')[:3]]
            p_dec = [p for x,p in self.net.named_parameters() if 'encoder' not in x.split('.')[:3]]
            self.net_param_gourps = [{'params': p_dec , 'lr': base_lr , 'lr_param' : base_lr},
                                     {'params': p_enc , 'lr': base_lr * lr_kwargs['ratio']['transfer'] , 'lr_param': base_lr * lr_kwargs['ratio']['transfer']}]
        else:
            self.net_param_gourps = [{'params': [p for p in self.net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

        optimizer = {
            'Adam': torch.optim.Adam ,
            'SGD' : torch.optim.SGD ,
        }[opt_kwargs['name']](self.net_param_gourps , **opt_kwargs['param'])
        return optimizer
    
    def load_scheduler(self , new_shd_kwargs = None):
        if new_shd_kwargs is None:
            shd_kwargs = ShareNames.train_params['trainer']['scheduler']
        else:
            shd_kwargs = deepcopy(ShareNames.train_params['trainer']['scheduler'])
            shd_kwargs.update(new_shd_kwargs)

        if shd_kwargs['name'] == 'cos':
            scheduler = lr_cosine_scheduler(self.optimizer, **shd_kwargs['param'])
        elif shd_kwargs['name'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **shd_kwargs['param'])
        elif shd_kwargs['name'] == 'cycle':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, max_lr=[pg['lr_param'] for pg in self.optimizer.param_groups],cycle_momentum=False,mode='triangular2',**shd_kwargs['param'])

        return scheduler
    
    def reset_scheduler(self):
        rst_kwargs = ShareNames.train_params['trainer']['learn_rate']['reset']
        if rst_kwargs['num_reset'] <= 0 or (self.epoch_i + 1) < rst_kwargs['trigger']: return

        trigger_intvl = rst_kwargs['trigger'] // 2 if rst_kwargs['speedup2x'] else rst_kwargs['trigger']
        if (self.epoch_i + 1 - rst_kwargs['trigger']) % trigger_intvl != 0: return
        
        trigger_times = ((self.epoch_i + 1 - rst_kwargs['trigger']) // trigger_intvl) + 1
        if trigger_times > rst_kwargs['num_reset']: return
        
        # confirm reset : change back optimizor learn rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr_param']  * rst_kwargs['recover_level']
        
        # confirm reset : reassign scheduler
        if rst_kwargs['speedup2x']:
            shd_kwargs = deepcopy(ShareNames.train_params['trainer']['scheduler'])
            for k in np.intersect1d(list(shd_kwargs['param'].keys()),['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']):
                shd_kwargs['param'][k] //= 2
        else:
            shd_kwargs = None
        self.scheduler = self.load_scheduler(shd_kwargs)
        self.printer('reset_learn_rate')
        
    def load_multiloss(self):
        multiloss = None
        if self.Param['num_output'] > 1:
            multiloss = multiloss_calculator(multi_type = ShareNames.train_params['multitask']['type'])
            multiloss.reset_multi_type(self.Param['num_output'] , **ShareNames.train_params['multitask']['param_dict'][multiloss.multi_type])
        return multiloss

    def time_recoder(self , start_time , keys , init_length = 100):
        if TIME_RECODER:
            if isinstance(keys , (list , tuple)): k = '/'.join(keys)
            if self.process_time.get(k) is None: 
                self.process_time[k] = {
                    'value' : np.zeros(init_length) , 
                    'index' : -1 , 
                    'length' : init_length , 
                }
            d = self.process_time[k]
            d['index'] += 1
            if d['length'] <= d['index']: 
                d['value'] = np.append(d['value'] , np.zeros(init_length))
                d['length'] += init_length
            d['value'][d['index']] = time.time() - start_time
    
    def print_time_recorder(self):
        if TIME_RECODER:
            keys = list(self.process_time.keys())
            num_calls = [self.process_time[k]['index']+1 for k in keys]
            total_time = [self.process_time[k]['value'].sum() for k in keys]
            tb = pd.DataFrame({'keys':keys , 'num_calls': num_calls, 'total_time': total_time})
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))
    
class FilteredIterator:
    def __init__(self, iterable, condition):
        self.iterable = iterable
        self.condition = condition

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item

class ModelData():
    """
    A class to store relavant training data , includes:
    1. Parameters: train_params , compt_params , model_data_type
    2. Datas: x_data , y_data , norm_param , index_stock , index_date
    3. Dataloader : yield x , y of training samples , create new ones if necessary
    """
    def __init__(self):     
        self.x_data , self.y_data , self.norm_param , (self.index_stock , self.index_date) = load_trading_data(ShareNames.model_data_type , ShareNames.precision)
        self.stock_n , self.all_day_len = self.y_data.shape[:2]
        self.labels_n = self.y_data.shape[-1] if any([smp['num_output'] > 1 for smp in ShareNames.model_params]) else 1
        self.feat_dims = {mdt:v.shape[-1] for mdt,v in self.x_data.items()}
        if len(ShareNames.data_type_list) > 1: 
            [smp.update({'input_dim':tuple([self.feat_dims[mdt] for mdt in ShareNames.data_type_list])}) for smp in ShareNames.model_params]
        else:
            [smp.update({'input_dim':self.feat_dims[ShareNames.data_type_list[0]]}) for smp in ShareNames.model_params]
        #self.x_feat_dim = {mdt:v.shape[-1] for mdt,v in self.x_data.items()}
        #self.input_dim = [self.x_feat_dim[mdt] for mdt in ShareNames.data_type_list if mdt in config['DATATYPE']['trade']]
        #self.factor_dim = [self.x_feat_dim[mdt] for mdt in ShareNames.data_type_list if mdt in config['DATATYPE']['factor']]
        #if len(self.input_dim) > 0: [smp.update({'input_dim':self.input_dim[0]}) for smp in ShareNames.model_params]
        #if len(self.factor_dim) > 0: [smp.update({'factor_dim':self.factor_dim[0]}) for smp in ShareNames.model_params]
        
        self.input_step = config['INPUT_STEP_DAY']
        self.test_step  = config['TEST_STEP_DAY']

        ShareNames.model_date_list = self.index_date[(self.index_date >= config['BEG_DATE']) & (self.index_date <= config['END_DATE'])][::config['INTERVAL']]
        ShareNames.test_full_dates = self.index_date[(self.index_date > config['BEG_DATE']) & (self.index_date <= config['END_DATE'])]
        self.reset_dataloaders()
    
    def reset_dataloaders(self):        
        """
        Reset dataloaders and dataloader_param
        """
        self.dataloaders = {}
        self.dataloader_param = ()
        gc.collect() , torch.cuda.empty_cache()
    
    def new_train_dataloader(self , model_date , seqlens):
        """
        Create train/valid dataloaders
        """
        assert ShareNames.process_name in ['train' , 'instance']
        self.dataloader_param = (model_date , seqlens)
        self.i_train , self.i_valid , self.ii_train , self.ii_valid = None , None , None , None
        self.y_train , self.y_valid , self.train_nonnan_sample = None , None , None
        gc.collect() , torch.cuda.empty_cache()
        
        seqlens = {mdt:(seqlens[mdt] if mdt in seqlens.keys() else 1) for mdt in ShareNames.data_type_list}
        self.seq0 = max(seqlens.values())
        self.seq = {mdt:self.seq0 + seqlens[mdt] if seqlens[mdt] <= 0 else seqlens[mdt] for mdt in ShareNames.data_type_list}
        model_date_col = (self.index_date < model_date).sum()    
        d0 , d1 = max(0 , model_date_col - 15 - config['INPUT_SPAN']) , max(0 , model_date_col - 15)
        self.day_len  = d1 - d0
        self.step_len = self.day_len // self.input_step
        self.lstepped = np.arange(0 , self.day_len , self.input_step)[:self.step_len]
        
        data_func = lambda x:torch.nn.functional.pad(x[:,d0:d1] , (0,0,0,0,0,self.seq0-self.input_step,0,0) , value=np.nan)
        x = {k:data_func(v) for k,v in self.x_data.items()}
        y = data_func(self.y_data).squeeze(2)[:,:,:self.labels_n]

        self._train_nonnan_sample(x , y)
        self._train_tv_split()
        self._train_y_data(y)
        self._train_dataloader(x)
        x , y = None , None
        self.i_train , self.i_valid , self.ii_train , self.ii_valid = None , None , None , None
        self.y_train , self.y_valid , self.train_nonnan_sample = None , None , None
        gc.collect() , torch.cuda.empty_cache()
        
    def new_test_dataloader(self , model_date , seqlens):
        """
        Create test dataloaders
        """
        assert ShareNames.process_name in ['test' , 'instance']
        self.dataloader_param = (model_date , seqlens)
        
        self.x_test , self.y_test = None , None
        gc.collect() , torch.cuda.empty_cache()
        
        seqlens = {mdt:(seqlens[mdt] if mdt in seqlens.keys() else 1) for mdt in ShareNames.data_type_list}
        self.seq0 = max(seqlens.values())
        self.seq = {mdt:self.seq0 + seqlens[mdt] if seqlens[mdt] <= 0 else seqlens[mdt] for mdt in ShareNames.data_type_list}
        
        if model_date == ShareNames.model_date_list[-1]:
            next_model_date = config['END_DATE'] + 1
        else:
            next_model_date = ShareNames.model_date_list[ShareNames.model_date_list > model_date][0]
        _step = (1 if ShareNames.process_name == 'instance' else self.test_step)
        _dates_list = ShareNames.test_full_dates[::_step]
        self.model_test_dates = _dates_list[(_dates_list > model_date) * (_dates_list <= next_model_date)]
        d0 , d1 = np.where(self.index_date == self.model_test_dates[0])[0][0] , np.where(self.index_date == self.model_test_dates[-1])[0][0] + 1
        self.day_len  = d1 - d0
        self.step_len = (self.day_len // _step) + (0 if self.day_len % _step == 0 else 1)
        self.lstepped = np.arange(0 , self.day_len , _step)[:self.step_len]
        
        data_func = lambda x:x[:,d0 - self.seq0 + 1:d1]
        x = {k:data_func(v) for k,v in self.x_data.items()}
        y = data_func(self.y_data).squeeze(2)[:,:,:self.labels_n]
        
        self._test_nonnan_sample(x , y)
        self._test_y_data(y)
        self._test_dataloader(x)
        x , y = None , None
        self.x_test = None
        gc.collect() , torch.cuda.empty_cache()
        
    def _train_nonnan_sample(self , x , y):
        """
        return non-nan sample position (with shape of stock_n * step_len)
        """
        nansamp = y[:,self.lstepped + self.seq0 - 1].isnan().sum(-1)
        for mdt in ShareNames.data_type_list:
            for i in range(self.seq[mdt]): nansamp += x[mdt][:,(self.seq0 - self.seq[mdt] + i):][:,self.lstepped].isnan().sum((2,3))
            if mdt in config['DATATYPE']['trade']: nansamp += (x[mdt][:,self.lstepped + self.seq0 - 1][:,:,-1] == 0).sum(-1)
        self.train_nonnan_sample = (nansamp == 0)
            
    def _train_tv_split(self):
        """
        update index of train/valid sub-samples of flattened all-samples(with in 0:stock_n * step_len - 1)
        """
        ii_stock_wise = np.arange(self.stock_n * self.step_len)[self.train_nonnan_sample.flatten()]
        ii_time_wise  = np.arange(self.stock_n * self.step_len).reshape(self.step_len , self.stock_n).transpose().flatten()[ii_stock_wise]
        train_samples = int(len(ii_stock_wise) * ShareNames.train_params['dataloader']['train_ratio'])
        random.seed(ShareNames.train_params['dataloader']['random_seed'])
        if ShareNames.train_params['dataloader']['random_tv_split']:
            random.shuffle(ii_stock_wise)
            ii_train , ii_valid = ii_stock_wise[:train_samples] , ii_stock_wise[train_samples:]
        else:
            early_samples = ii_time_wise < sorted(ii_time_wise)[train_samples]
            ii_train , ii_valid = ii_stock_wise[early_samples] , ii_stock_wise[early_samples == 0]
        random.shuffle(ii_train) , random.shuffle(ii_valid)
        self.ii_train , self.ii_valid = ii_train , ii_valid
    
    def _train_y_data(self , y):
        """
        update position (stock_i , date_i) of and normalized (maybe include w) train/valid ydata
        """
        # init i (row , col position) and y (labels) matrix
        i_tv = torch.zeros(self.stock_n , self.step_len , 2 , dtype = int) # i_row (sec) , i_col_x (end)
        i_tv[:,:,0] = torch.tensor(np.arange(self.stock_n , dtype = int)).reshape(-1,1) 
        i_tv[:,:,1] = torch.tensor(self.lstepped + self.seq0 - 1)
        i_tv = i_tv.reshape(-1,i_tv.shape[-1])
        self.i_train , self.i_valid = (i_tv[self.ii_train] , i_tv[self.ii_valid])
        
        y_tv = torch.zeros(self.stock_n , self.step_len , self.labels_n)
        y_tv[:] = y[:,self.lstepped + self.seq0 - 1].nan_to_num(0)
        y_tv[self.train_nonnan_sample == 0] = np.nan
        y_tv , w_tv = tensor_standardize_and_weight(y_tv , dim = 0)
        y_tv , w_tv = y_tv.reshape(-1,y_tv.shape[-1]) , w_tv.reshape(-1,w_tv.shape[-1]) 
        self.y_train , self.y_valid = (y_tv[self.ii_train] , y_tv[self.ii_valid])
        # self.w_train , self.w_valid = (w_tv[self.ii_train] , w_tv[self.ii_valid])
        
    def _train_dataloader(self , x):
        """
        1. if model_data_type == 'day' , update dataloaders dict(dict.key = ['train' , 'valid']), by using a oneshot method
        2. update dataloaders dict(set_name = ['train' , 'valid']), save batch_data to './model/{model_name}/{set_name}_batch_data' and later load them
        """
        if ShareNames.model_data_type == 'day' and False:
            mdt = 'day'
            x_tv = self._norm_x(torch.cat([x[mdt][:,self.lstepped + i] for i in range(self.seq[mdt])] , dim=2) , mdt)
            x_tv = x_tv.reshape(-1 , self.seq[mdt] , self.feat_dims[mdt])
            x_train , x_valid = x_tv[self.ii_train] , x_tv[self.ii_valid]
            num_worker = min(os.cpu_count() , ShareNames.compt_params['num_worker'])
            self.dataloaders['train'] = self.dataloader_oneshot((x_train , self.y_train) , num_worker , ShareNames.compt_params['cuda_first'])
            self.dataloaders['valid'] = self.dataloader_oneshot((x_valid , self.y_valid) , num_worker , ShareNames.compt_params['cuda_first'])
        else:
            storage_loader.del_group('train')
            set_iter = [('train' , self.i_train , self.y_train) , ('valid' , self.i_valid , self.y_valid)]
            for set_name , set_i , set_y in set_iter:
                batch_sampler = torch.utils.data.BatchSampler(range(len(set_i)) , ShareNames.batch_size , drop_last = False)
                batch_file_list = []
                for batch_num , batch_pos in enumerate(batch_sampler):
                    batch_file_list.append(ShareNames.batch_dir[set_name] + f'/{set_name}.{batch_num}.pt')
                    i0 , i1 , batch_y , batch_x = set_i[batch_pos , 0] , set_i[batch_pos , 1] , set_y[batch_pos] , []
                    for mdt in ShareNames.data_type_list:
                        batch_x.append(self._norm_x(torch.cat([x[mdt][i0,i1+i+1-self.seq[mdt]] for i in range(self.seq[mdt])],dim=1),mdt))
                    batch_x = batch_x[0] if len(batch_x) == 1 else tuple(batch_x)
                    storage_loader.save((batch_x, batch_y), batch_file_list[-1] , group = 'train')
                self.dataloaders[set_name] = self.dataloader_saved(batch_file_list)

    def _test_nonnan_sample(self , x , y):
        """
        return non-nan sample position (with shape of stock_n * day_len)
        """
        nansamp = y[:,self.lstepped + self.seq0 - 1].isnan().sum(-1)
        for mdt in ShareNames.data_type_list:
            for i in range(self.seq[mdt]): nansamp += x[mdt][:,(self.seq0 - self.seq[mdt] + i):][:,self.lstepped].isnan().sum((2,3))
            if mdt in config['DATATYPE']['trade']: nansamp += (x[mdt][:,self.lstepped + self.seq0 - 1][:,:,-1] == 0).sum(-1)
        self.test_nonnan_sample = (nansamp == 0)
    
    def _test_y_data(self , y):
        """
        update normalized (maybe include w) test ydata
        """
        y_test = torch.zeros(self.stock_n , self.step_len , self.labels_n)
        y_test[:] = y[:,self.lstepped + self.seq0 - 1].nan_to_num(0)
        y_test[self.test_nonnan_sample == 0] = np.nan
        self.y_test , _ = tensor_standardize_and_weight(y_test , dim = 0)
    
    def _test_dataloader(self , x):
        """
        1. if model_data_type == 'day' , update dataloaders dict(dict.key = ['test']), by using a oneshot method (seperate dealing by TEST_INTERVAL days)
        2. update dataloaders dict(set_name = ['test']), save batch_data to './model/{model_name}/{set_name}_batch_data' and later load them
        """
        if ShareNames.model_data_type == 'day' and False:
            mdt = 'day'
            x_test = self._norm_x(torch.cat([x[mdt][:,i+self.lstepped] for i in range(self.seq[mdt])],dim=2) , mdt)
            self.dataloaders['test'] = self.dataloader_oneshot((x_test , self.y_test) , 0 , ShareNames.compt_params['cuda_first'] , 1) # iter over col(date)
        else:
            storage_loader.del_group('test')
            batch_sampler = [(np.where(self.test_nonnan_sample[:,i])[0] , self.lstepped[i]) for i in range(self.step_len)] # self.test_nonnan_sample.permute(1,0)
            batch_file_list = []
            for batch_num , batch_pos in enumerate(batch_sampler):
                batch_file_list.append(ShareNames.batch_dir['test'] + f'/test.{batch_num}.pt')
                i0 , i1 , batch_y , batch_x = batch_pos[0] , batch_pos[1] + self.seq0 - 1 , self.y_test[batch_pos[0] , batch_num] , []
                for mdt in ShareNames.data_type_list:
                    batch_x.append(self._norm_x(torch.cat([x[mdt][i0,i1+i+1-self.seq[mdt]] for i in range(self.seq[mdt])],dim=1),mdt))
                batch_x = batch_x[0] if len(batch_x) == 1 else tuple(batch_x)
                storage_loader.save((batch_x, batch_y), batch_file_list[-1] , group = 'test')
            self.dataloaders['test'] = self.dataloader_saved(batch_file_list)
        
    def _norm_x(self , x , key):
        """
        return panel_normalized x
        1.for ts-cols , divide by the last value, get seq-mormalized x
        2.for seq-mormalized x , normalized by history avg and std
        """
        if key in config['DATATYPE']['trade']:
            x /= x.select(-2,-1).unsqueeze(-2)
            x -= self.norm_param[key]['avg'][-x.shape[-2]:]
            x /= self.norm_param[key]['std'][-x.shape[-2]:] + 1e-4
        else:
            pass
        return x
    
    class dataloader_oneshot:
        """
        class of oneshot dataloader
        """
        def __init__(self, data , num_worker = 0 , cuda_first = True , batch_by_axis = None):
            if cuda_first: data = cuda(data)
            self.batch_by_axis = batch_by_axis
            if self.batch_by_axis is None:
                self.dataset = Mydataset(*data)  
                self.dataloader = torch.utils.data.DataLoader(self.dataset , batch_size = ShareNames.batch_size , num_workers = (1 - cuda_first)*num_worker)
            else:
                self.x , self.y = data
                
        def __iter__(self):
            if self.batch_by_axis is None:
                for batch_data in self.dataloader: 
                    yield cuda(batch_data)
            else:
                for batch_i in range(self.y.shape[self.batch_by_axis]):
                    x , y = self.x.select(self.batch_by_axis , batch_i) , self.y.select(self.batch_by_axis , batch_i)
                    if y.dim() == 1:
                        valid_row = y.isnan() == 0
                    elif y.dim() == 2:
                        valid_row = y.isnan().sum(-1) == 0
                    else:
                        valid_row = y.isnan().sum(list(range(y.dim()))[1:]) == 0
                    batch_data = (x[valid_row] , y[valid_row])
                    yield cuda(batch_data)
                
    class dataloader_saved:
        """
        class of saved dataloader , retrieve batch_data from './model/{model_name}/{set_name}_batch_data'
        """
        def __init__(self, batch_file_list):
            self.batch_file_list = batch_file_list
        def __iter__(self):
            for batch_file in self.batch_file_list: 
                yield cuda(storage_loader.load(batch_file))
                
def cuda(x):
    if isinstance(x , (list,tuple)):
        return type(x)(map(cuda , x))
    else:
        return x.to(DEVICE)

def loss_function(key):
    """
    loss function , metric should * -1.
    """
    assert key in ('mse' , 'pearson' , 'ccc')
    def decorator(func , key):
        def wrapper(*args, **kwargs):
            v = func(*args, **kwargs)
            if key != 'mse':  
                v = torch.exp(-v)
            return v
        return wrapper
    func = globals()[key]
    return decorator(func , key)

def metric_function(key):
    assert key in ('mse' , 'pearson' , 'ccc' , 'spearman')
    def decorator(func , key , item_only = False):
        def wrapper(*args, **kwargs):
            with torch.no_grad():
                v = func(*args, **kwargs)
            if key == 'mse' : v = -v
            return v
        return wrapper
    func = globals()[key]
    return decorator(globals()[key] , key)
    
def penalty_function(key):
    _cat_tensor = lambda x:(torch.cat(x,dim=-1) if isinstance(x,(tuple,list)) else x)
    def _none(**kwargs):
        return 0.
    def _hidden_orthogonality(**kwargs):
        _cat_tensor = lambda x:(torch.cat(x,dim=-1) if isinstance(x,(tuple,list)) else x)
        return _cat_tensor(kwargs.get('hidden')).T.corrcoef().triu(1).nan_to_num().square().sum()
    return locals()[f'_{key}']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--process",     type=int, default=-1)
    parser.add_argument("--rawname",     type=int, default=-1)
    parser.add_argument("--resume",      type=int, default=-1)
    parser.add_argument("--anchoring",   type=int, default=-1)
    ShareNames = parser.parse_args()

    Controller = model_controller()
    Controller.main_process()
    Controller.ResultOutput()
    Controller.print_time_recorder()
