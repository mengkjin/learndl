#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --process=0 --rawname=1 --resume=0 --anchoring=0
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
from tqdm import tqdm
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel , update_bn
from ..scripts.util.environ import get_logger , get_config , cuda , DEVICE
from ..scripts.util.basic import FilteredIterator , lr_cosine_scheduler , versatile_storage
from ..scripts.util.multiloss import multiloss_calculator 
from ..scripts.data_util.ModelData import ModelData
from ..scripts.function.basic import *
from ..scripts.function.algos import sinkhorn
from ..scripts.nn.My import *
# from audtorch.metrics.functional import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
TIME_RECODER = False #False # dict()
logger = get_logger()
CONFIG = get_config()

torch.set_default_dtype(getattr(torch , CONFIG['PRECISION']))
torch.backends.cuda.matmul.allow_tf32 = CONFIG['ALLOW_TF32']
torch.autograd.set_detect_anomaly(CONFIG['DETECT_ANOMALY'])

storage_model  = versatile_storage(CONFIG['STORAGE_TYPE'])

class ShareNames_conctroller():
    """
    1. Assign variables into shared namespace.
    2. Ask what process would anyone want to run : 0 : train & test(default) , 1 : train only , 2 : test only , 3 : copy to instance only
    3. Ask if model_name and model_base_path should be changed if old dir exists
    """
    def __init__(self):
        self.assign_variables(if_process = True , if_rawname = True)
        
    def assign_variables(self , if_process = False , if_rawname = False):
        ShareNames.max_epoch       = CONFIG['MAX_EPOCH']
        ShareNames.batch_size      = CONFIG['BATCH_SIZE']
        ShareNames.precision       = CONFIG['PRECISION']
        ShareNames.allow_tf32      = CONFIG['ALLOW_TF32']
        
        ShareNames.model_module    = CONFIG['MODEL_MODULE']
        ShareNames.model_data_type = CONFIG['MODEL_DATATYPE'][ShareNames.model_module]
        ShareNames.model_nickname  = CONFIG['MODEL_NICKNAME']
        
        ShareNames.model_num_list  = list(range(CONFIG['MODEL_NUM']))
        ShareNames.data_type_list  = ShareNames.model_data_type.split('+')
        
        ShareNames.model_name      = self._model_name()
        ShareNames.model_base_path = f'./model/{ShareNames.model_name}'
        ShareNames.instance_path   = f'./instance/{ShareNames.model_name}'
        
        if if_process  : self._process_confirmation()
        if if_rawname  : self._rawname_confirmation()
        
        ShareNames.train_params = deepcopy(CONFIG['TRAIN_PARAM'])
        ShareNames.compt_params = deepcopy(CONFIG['COMPT_PARAM'])
        ShareNames.raw_model_params = deepcopy(CONFIG['MODEL_PARAM'])
        ShareNames.model_params = self._load_model_param()
        ShareNames.output_types = ShareNames.train_params['output_types']
        ShareNames.TRA_model = (CONFIG['TRA_switch'] == True) and CONFIG['MODEL_MODULE'].startswith('TRA')
        ShareNames.weight_train = CONFIG['WEIGHT_TRAIN']
        ShareNames.weight_test  = CONFIG['WEIGHT_TEST']

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
        self.init_time = time.time()
        self.display = {
            'tqdm' : True if CONFIG['VERBOSITY'] >= 10 else False ,
            'once' : True if CONFIG['VERBOSITY'] <=  2 else False ,
            'step' : [10,5,5,3,3,1][min(CONFIG['VERBOSITY'] // 2 , 5)],
        }
        self.shared_ctrl = ShareNames_conctroller()
        
    def main_process(self):
        """
        Main process of load_data + train + test + instance
        """
        for process_name in ShareNames.process_queue:
            self.SetProcessName(process_name)
            self.__getattribute__(f'model_process_{process_name.lower()}')()
    
    def SetProcessName(self , key = 'data'):
        ShareNames.process_name = key.lower()
        self.model_count = 0
        self.epoch_count = 0
        if 'data' in vars(self) : self.data.reset_dataloaders()
        if ShareNames.process_name == 'data': 
            pass
        elif ShareNames.process_name == 'train': 
            self.f_loss    = loss_function(ShareNames.train_params['criterion']['loss'])
            self.f_score   = score_function(ShareNames.train_params['criterion']['score'])
            self.f_penalty = {k:penalty_function(k,v) for k,v in ShareNames.train_params['criterion']['penalty'].items()}
        elif ShareNames.process_name == 'test':
            self.f_loss    = lambda x:None 
            self.f_score   = score_function(ShareNames.train_params['criterion']['score'])
            self.f_penalty = {}
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
        self.data = ModelData(ShareNames.model_data_type , CONFIG)
        # retrieve data
        ShareNames.data_type_list  = self.data.data_type_list
        ShareNames.model_date_list = self.data.model_date_list
        ShareNames.test_full_dates = self.data.test_full_dates
        if len(ShareNames.data_type_list) > 1: 
            [smp.update({'input_dim':tuple([self.data.feat_dims[mdt] for mdt in ShareNames.data_type_list])}) for smp in ShareNames.model_params]
        else:
            [smp.update({'input_dim':self.data.feat_dims[ShareNames.data_type_list[0]]}) for smp in ShareNames.model_params]

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
        with process_timer('ModelPreparation' , process):
            param = ShareNames.model_params[self.model_num]

            # In a new model , alters the penalty function's lamb
            if 'hidden_orthogonality' in self.f_penalty.keys():
                self.f_penalty['hidden_orthogonality']['cond'] = (param.get('hidden_as_factors') == True) or ShareNames.TRA_model
            if 'tra_ot_penalty' in self.f_penalty.keys(): 
                self.f_penalty['tra_ot_penalty']['cond'] = ShareNames.TRA_model

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
        with process_timer('TrainModelStart'):
            self._init_variables('model')
            self.nanloss_life = ShareNames.train_params['trainer']['nanloss']['retry']
            self.text['model'] = '{:s} #{:d} @{:4d}'.format(ShareNames.model_name , self.model_num , self.model_date)
            if (self.data.dataloader_param != (self.model_date , self.Param['seqlens'])):
                self.data.create_dataloader(ShareNames.process_name , 'train' , self.model_date , self.Param['seqlens']) 
                self.time[1] = time.time()
                self.printer('train_dataloader')
            
    def TrainModelEnd(self):
        """
        Do necessary things of ending a model(model_data , model_num)
        """
        with process_timer('TrainModelEnd'):
            storage_model.del_path(self.path.get('rounds') , self.path.get('lastn') , self.path.get('bestn'))
            if ShareNames.process_name == 'train' : self.model_count += 1
            self.time[2] = time.time()
            self.printer('model_end')
        
    def NewLoop(self):
        """
        Reset and loop variables giving loop_status
        """
        with process_timer('NewLoop'):
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
        
    def TrainerInit(self):
        """
        Initialize net , optimizer , scheduler if loop_status in ['round' , 'attempt']
        net : 1. Create an instance of f'My{ShareNames.model_module}' or inherit from 'lastround'/'transfer'
              2. In transfer mode , p_late and p_early with be trained with different lr's. If not net.parameters are trained by same lr
        optimizer : Adam or SGD
        scheduler : Cosine or StepLR
        """
        with process_timer('TrainerInit'):
            if self.cond.get('loop_status') == 'epoch': return
            self.load_model('train')
            self.max_round = self.net.max_round() if 'max_round' in self.net.__dir__() else 1
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler() 
            self.multiloss = self.load_multiloss()
        
    def TrainEpoch(self):
        """
        Iterate train and valid dataset, calculate loss/score , update values
        If nan loss occurs, turn to _deal_nanloss
        """
        with process_timer('TrainEpoch/assign_loader'):
            loss_train , loss_valid , ic_train , ic_valid = [] , [] , [] , []
            clip_value = ShareNames.train_params['trainer']['gradient'].get('clip_value')
            
            if self.display.get('tqdm'):
                iter_train , iter_valid = tqdm(self.data.dataloaders['train']) , tqdm(self.data.dataloaders['valid'])
                disp_train = lambda x:iter_train.set_description(f'Ep#{self.epoch_i:3d} train loss:{np.mean(x):.5f}')
                disp_valid = lambda x:iter_valid.set_description(f'Ep#{self.epoch_i:3d} valid ic:{np.mean(x):.5f}')
            else:
                iter_train , iter_valid = self.data.dataloaders['train'] , self.data.dataloaders['valid']
                disp_train = disp_valid = lambda x:0

        with process_timer('TrainEpoch/train_epochs'):
            self.net.train()
            for _ , batch_data in enumerate(iter_train):
                x = self.modifier['inputs'](batch_data['x'] , batch_data , self.data)
                self.optimizer.zero_grad()
                with process_timer('TrainEpoch/train/forward'):
                    pred , hidden = self.net(x)
                with process_timer('TrainEpoch/train/loss'):
                    penalty_kwargs = {'net' : self.net , 'hidden' : hidden , 'label' : batch_data['y']}
                    metric = self.metric_calculator(batch_data['y'] , pred , 'train' , weight = batch_data['w'] , **penalty_kwargs)
                    metric = self.modifier['metric'](metric, batch_data, self.data)
                with process_timer('TrainEpoch/train/backward'):
                    metric['loss'].backward()
                with process_timer('TrainEpoch/train/step'):
                    if clip_value is not None : nn.utils.clip_grad_value_(self.net.parameters(), clip_value = clip_value)
                    self.optimizer.step()
                self.modifier['update'](None , batch_data , self.data)
                loss_train.append(metric['loss'].item()) , ic_train.append(metric['score'])
                disp_train(loss_train)

            if np.isnan(sum(loss_train)): return self._deal_nanloss()
            self.loss_list['train'].append(np.mean(loss_train)) , self.ic_list['train'].append(np.mean(ic_train))
        
        with process_timer('TrainEpoch/valid_epochs'):
            self.net.eval()     
            for _ , batch_data in enumerate(iter_valid):
                x = self.modifier['inputs'](batch_data['x'] , batch_data , self.data)
                # print(torch.cuda.memory_allocated(DEVICE) / 1024**3 , torch.cuda.memory_reserved(DEVICE) / 1024**3)
                with process_timer('TrainEpoch/valid/forward'):
                    pred , _ = self.net(x)
                with process_timer('TrainEpoch/valid/loss'):
                    metric = self.metric_calculator(batch_data['y'] , pred , 'valid' , weight = batch_data['w'])
                    metric = self.modifier['metric'](metric, batch_data, self.data)
                self.modifier['update'](None , batch_data , self.data)
                loss_valid.append(metric['loss']) , ic_valid.append(metric['score'])
                disp_valid(ic_valid)

            self.loss_list['valid'].append(np.mean(loss_valid)) , self.ic_list['valid'].append(np.mean(ic_valid))
            self.lr_list.append(self.scheduler.get_last_lr()[0])
            self.scheduler.step()
            self.reset_scheduler()

    def LoopCondition(self):
        """
        Update condition of continuing training epochs , restart attempt if early exit , proceed to next round if convergence , reset round if nan loss
        """
        with process_timer('LoopCondition/assess'):
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
        
        with process_timer('LoopCondition/confirm_status'):
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
            
    def TestModelStart(self):
        """
        Reset model specific variables
        """
        self._init_variables('model')
        dataloader_param = (ShareNames.process_name , 'test' , self.model_date , self.Param['seqlens'])   
        if (self.data.dataloader_param != dataloader_param):
            self.data.create_dataloader(*dataloader_param)
            
        if self.model_num == 0:
            ic_date_0 = np.zeros((len(self.data.model_test_dates) , len(self.test_result_model_num)))
            ic_model_0 =  np.zeros((1 , len(self.test_result_model_num)))
            self.ic_by_date = ic_date_0 if self.ic_by_date is None else np.concatenate([self.ic_by_date , ic_date_0])
            self.ic_by_model = ic_model_0 if self.ic_by_model is None else np.concatenate([self.ic_by_model , ic_model_0])
                
    def Forecast(self):
        if not os.path.exists(self.path['best']): self.TrainModel()
        
        #self.y_pred = cuda(torch.zeros(len(self.data.index[0]),len(self.data.model_test_dates),self.data.labels_n,len(ShareNames.output_types)).fill_(np.nan))
        self.y_pred = cuda(torch.zeros(len(self.data.index[0]),
                                       len(self.data.model_test_dates),
                                       len(ShareNames.output_types)).fill_(np.nan))
        iter_dates = np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
        assert self.data.dataloaders['test'].__len__() == len(iter_dates)
        for oi , okey in enumerate(ShareNames.output_types):
            self.load_model('test' , okey)
            self.net.eval()
            
            if self.display.get('tqdm'):
                iter_test = tqdm(self.data.dataloaders['test'])
                disp_test = lambda x:iter_test.set_description(f'Date#{x[0]:3d} :{np.mean(x[1]):.5f}')
            else:
                iter_test = self.data.dataloaders['test']
                disp_test = lambda x:0

            m_test = []    
            with torch.no_grad():
                for i , batch_data in enumerate(iter_test):
                    """
                    x , y , w , nonnan = tuple([batch_data[k] for k in ['x','y','w','nonnan']])
                    pred = torch.zeros_like(y).fill_(np.nan)
                    for batch_j in torch.utils.data.DataLoader(np.arange(len(y)) , batch_size = ShareNames.batch_size):
                        nnj = batch_j[nonnan[batch_j]]
                        x_j = tuple([xx[nnj] for xx in x]) if isinstance(x , tuple) else x[nnj]
                        pred[nnj,0] = self.net(x_j)[0].select(-1,0).detach()
                    
                    if i >= len(self.data.early_test_dates):
                        self.y_pred[:,i-len(self.data.early_test_dates),oi] = pred[:,0]
                        w = None if w is None else w.select(-1,0)[nonnan]
                        score = self.f_score(y.select(-1,0)[nonnan] , pred.select(-1,0)[nonnan] , w).item()
                        m_test.append(score) 
                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                    """
                    nonnan = batch_data['nonnan']
                    pred = torch.zeros_like(batch_data['y']).fill_(np.nan)
                    for batch_j in torch.utils.data.DataLoader(torch.arange(len(nonnan)).to(nonnan.device) , batch_size = ShareNames.batch_size):
                        nnj = batch_j[nonnan[batch_j]]
                        batch_nnj = subset(batch_data , nnj)
                        x = self.modifier['inputs'](batch_nnj['x'] , batch_nnj , self.data)
                        pred_nnj = self.net(x)[0].detach()
                        pred[nnj,0] = pred_nnj[:,0]
                        self.modifier['update'](None , batch_nnj , self.data)
                    
                    if i >= len(self.data.early_test_dates):
                        self.y_pred[:,i-len(self.data.early_test_dates),oi] = pred[:,0]
                        y = batch_data['y'].select(-1,0)[nonnan]
                        pred = pred.select(-1,0)[nonnan]
                        w = None if batch_data['w'] is None else batch_data['w'].select(-1,0)[nonnan]
                        score = self.f_score(y , pred , w).item()
                        m_test.append(score) 
                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                    
                    disp_test((i-len(self.data.early_test_dates) , m_test))
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
        #    df = pd.DataFrame(self.y_pred.T, index = self.data.model_test_dates, columns = self.data.secid.astype(str))
        #    with open(f'{ShareNames.instance_path}/{ShareNames.model_name}_fac{self.model_num}.csv', 'a') as f:
        #        df.to_csv(f , mode = 'a', header = f.tell()==0, index = True)

    def ResultOutput(self):
        out_dict = {
            '0_start':self.model_info.get('global_start_time'),
            '1_basic':'+'.join([
                'short' if CONFIG['SHORTTEST'] else 'long' ,
                CONFIG['STORAGE_TYPE'] , CONFIG['PRECISION']
            ]),
            '2_model':''.join([
                ShareNames.model_module , '_' , ShareNames.model_data_type ,
                '(x' , str(CONFIG['MODEL_NUM']) , ')'
            ]),
            '3_time':'-'.join([str(CONFIG['BEG_DATE']),str(CONFIG['END_DATE'])]),
            '4_typeNN':'+'.join(list(set(CONFIG['MODEL_PARAM']['type_rnn']))),
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
            # idx = np.array(np.meshgrid(self.data.model_test_dates , self.data.sec_id)).T.reshape(-1,2)
            mode = 'r+' if os.path.exists(f'{ShareNames.instance_path}/{ShareNames.model_name}.h5') else 'w'
            with h5py.File(f'{ShareNames.instance_path}/{ShareNames.model_name}.h5' , mode = mode) as f:
                for di in range(len(self.data.model_test_dates)):
                    arr , row = self.y_pred_models[di] , self.data.sec_id 
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
        for mm in range(len(ShareNames.model_params)): 
            ShareNames.model_params[mm].update({'path':f'{ShareNames.instance_path}/{mm}'})
    
    def printer(self , key):
        """
        Print out status giving display conditions and looping conditions
        """
        _detail_print = (self.display.get('once') == 0 or self.model_count <= max(ShareNames.model_num_list))
        if key == 'model_specifics':
            logger.warning('Model Parameters:')
            logger.info(f'Basic Parameters : ')
            print(f'STORAGE [{CONFIG["STORAGE_TYPE"]}] | DEVICE [{DEVICE}] | PRECISION [{ShareNames.precision}] | BATCH_SIZE [{ShareNames.batch_size}].') 
            print(f'NAME [{ShareNames.model_name}] | MODULE [{ShareNames.model_module}] | DATATYPE [{ShareNames.model_data_type}] | MODEL_NUM [{len(ShareNames.model_num_list)}].')
            print(f'BEG_DATE [{CONFIG["BEG_DATE"]}] | END_DATE [{ShareNames.test_full_dates[-1]}] | ' +
                  f'INTERVAL [{CONFIG["INTERVAL"]}] | INPUT_STEP_DAY [{CONFIG["INPUT_STEP_DAY"]}] | TEST_STEP_DAY [{CONFIG["TEST_STEP_DAY"]}].') 
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
            
    def metric_calculator(self, labels , pred , key , weight = None , **penalty_kwargs):
        """
        Calculate loss(with gradient), score
        Inputs : 
            kwargs : other inputs used in calculating loss , penalty and score
        Possible Methods :
        loss:    pearsonr , mse , ccc
        penalty: hidden_orthogonality , tra_ot_penalty
        score:  pearsonr , rankic , mse , ccc
        """
        assert key in ['train' , 'valid'] , key
        if labels.shape != pred.shape:
            # if more labels than output
            assert labels.shape[:-1] == pred.shape[:-1] , (labels.shape , pred.shape)
            labels = labels.transpose(0,-1)[:pred.shape[-1]].transpose(0,-1)

        weight0 = None if weight is None else weight.select(-1,0)  
        if key == 'train':
            if self.Param['num_output'] > 1:
                loss = self.f_loss(labels , pred , weight , dim = 0)[:self.Param['num_output']]
                loss = self.multiloss.calculate_multi_loss(loss , self.net.get_multiloss_params())
            else:
                loss = self.f_loss(labels.select(-1,0) , pred.select(-1,0) , weight0)
            score  = self.f_score(labels.select(-1,0) , pred.select(-1,0) , weight0).item()

            for k,d in self.f_penalty.items():
                if d['lamb'] > 0 and d['cond']: loss = loss + d['lamb'] * d['func'](**penalty_kwargs)  
        else:
            score  = self.f_score(labels.select(-1,0) , pred.select(-1,0) , weight0).item()
            loss    = 0.
        return {'loss' : loss , 'score' : score}
    
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
        with process_timer('save_model'):
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
    
    def load_model(self , process , key = 'best'):
        assert process in ['train' , 'test']
        with process_timer('load_model'):
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
            self.net = net
            # default : none modifier
            # input : (inputs/metric/update , batch_data , self.data)
            # output : new_inputs/new_metric/None 
            self.modifier = {'inputs': lambda x,b,d:x, 'metric': lambda x,b,d:x, 'update': lambda x,b,d:None}
            if 'modifier_inputs' in self.net.__dir__(): self.modifier['inputs'] = lambda x,b,d:self.net.modifier_inputs(x,b,d)
            if 'modifier_metric' in self.net.__dir__(): self.modifier['metric'] = lambda x,b,d:self.net.modifier_metric(x,b,d)
            if 'modifier_update' in self.net.__dir__(): self.modifier['update'] = lambda x,b,d:self.net.modifier_update(x,b,d)

    
    def swa_model(self , model_path_list = []):
        net = globals()[f'My{ShareNames.model_module}'](**self.Param)
        swa_net = AveragedModel(net)
        for p in model_path_list:
            swa_net.update_parameters(storage_model.load_model_state(net , p))
        swa_net = cuda(swa_net)
        update_bn(self.update_bn_loader(self.data.dataloaders['train']) , swa_net)
        return swa_net.module
    
    def update_bn_loader(self , loader):
        for data in loader: yield [data['x'] , data['y'] , data['w']]
    
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
    
class process_timer:
    def __init__(self , *args):
        if isinstance(TIME_RECODER , dict):
            self.key = '/'.join(args)
            if self.key not in TIME_RECODER.keys():
                TIME_RECODER[self.key] = []
    def __enter__(self):
        if isinstance(TIME_RECODER , dict):
            self.start_time = time.time()
    def __exit__(self, type, value, trace):
        if isinstance(TIME_RECODER , dict):
            time_cost = time.time() - self.start_time
            TIME_RECODER[self.key].append(time_cost)

def loss_function(key):
    """
    loss function , pearson/ccc should * -1.
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

def score_function(key):
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
    
def penalty_function(key , param):
    def _none(**kwargs):
        return 0.
    def _hidden_orthogonality(**kwargs):
        hidden = kwargs['hidden']
        if hidden.shape[-1] == 1:
            return 0
        if isinstance(hidden,(tuple,list)):
            hidden = torch.cat(hidden,dim=-1)
        return hidden.T.corrcoef().triu(1).nan_to_num().square().sum()
    def _tra_ot_penalty(**kwargs):
        net = kwargs['net']
        if net.training and net.probs is not None and net.num_states > 1:
            pred , label = kwargs['hidden'] , kwargs['label']
            square_error = (pred - label).square()
            square_error -= square_error.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            P = sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
            lamb = (param['rho'] ** net.global_steps)
            reg = net.probs.log().mul(P).sum(dim=-1).mean()
            net.global_steps += 1
            return - lamb * reg
        else:
            return 0
        
    return {'lamb': param['lamb'] , 'cond' : True , 'func' : locals()[f'_{key}']}

def print_time_recorder():
    if isinstance(TIME_RECODER , dict):
        keys = list(TIME_RECODER.keys())
        num_calls = [len(TIME_RECODER[k]) for k in keys]
        total_time = [np.sum(TIME_RECODER[k]) for k in keys]
        tb = pd.DataFrame({'keys':keys , 'num_calls': num_calls, 'total_time': total_time})
        tb['avg_time'] = tb['total_time'] / tb['num_calls']
        print(tb.sort_values(by=['total_time'],ascending=False))

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
    print_time_recorder()
