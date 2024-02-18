#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --process=0 --rawname=1 --resume=0 --anchoring=0
'''
1.1 HIST
https://arxiv.org/pdf/2110.13716.pdf
https://github.com/Wentao-Xu/HIST
2.Lightgbm
https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/plot_example.py
https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_tree.html
3.other factors
'''
import torch
import torch.nn as nn
import numpy as np
import itertools , os, shutil , gc , time , yaml
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel , update_bn
from scripts.util.environ import get_logger
from scripts.util.basic import process_timer , FilteredIterator , lr_cosine_scheduler , versatile_storage
from scripts.util.multiloss import multiloss_calculator 
from scripts.util.trainer import trainer_parser , train_config , set_trainer_environment , Device
from scripts.data_util.ModelData import ModelData
from scripts.function.basic import *
from scripts.function.metric import loss_function,score_function,penalty_function
from scripts.nn import model
# from audtorch.metrics.functional import *

do_timer = True
logger   = get_logger()

class RunModel():
    """
    A class to control the whole process of training , includes:
    1. Parameters: train_params , compt_params , model_data_type
    2. Data : class of train_data
    3. loop status: model , round , attempt , epoch
    4. file path: model , lastround , transfer(last model date)
    5. text: model , round , attempt , epoch , exit , stat , time , trainer
    """
    def __init__(self , timer = False , storage_type = 'mem' , device = None , **kwargs):
        self.model_info = {'init_time' : time.time()}
        self.ptimer     = process_timer(False)
        self.storage    = versatile_storage(storage_type)
        self.device     = Device(device)
        
    def main_process(self):
        """
        Main process of load_data + train + test + instance
        """
        for process_name in config.process_queue:
            self.process_setname(process_name)
            self.__getattribute__(f'process_{process_name.lower()}')()
    
    def process_setname(self , key = 'data'):
        self.process_name = key.lower()
        self.model_count = 0
        self.epoch_count = 0
        if self.process_name == 'data': 
            pass
        elif self.process_name in ['train' , 'test' , 'instance']: 
            self.data.reset_dataloaders()
            self.metric_function = self.new_metricfunc(config.train_params['criterion'])
        else:
            raise Exception(f'KeyError : {key}')
        
    def process_data(self):
        """
        Main process of loading basic data
        """
        self.model_info['data_time'] = time.time()
        logger.critical(f'Start Process [Load Data]!')
        self.data = ModelData(config.model_data_type , config)
        # retrieve from data object
        filler = self.model_params_filler(self.data.x_data , self.data.data_type_list)
        for smp in config.model_params:  smp.update(filler)
        logger.critical('Finish Process [Load Data]! Cost {:.1f}Secs'.format(time.time() - self.model_info['data_time']))
        
    def process_train(self):
        """
        Main process of training
        1. loop over model(model_date , model_num)
        2. loop over round(if necessary) , attempt(if converge too soon) , epoch(most prevailing loops)
        """
        self.model_info['train_time'] = time.time()
        logger.critical(f'Start Process [Train Model]!')
        torch.save(config.model_params , f'{config.model_base_path}/model_params.pt')    
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation('train')
            self.model_train()
        total_time = time.time() - self.model_info['train_time']
        _str = 'Finish Process [Train Model]! Cost {:.1f} Hours, {:.1f} Min/model, {:.1f} Sec/Epoch'.format(
            total_time / 3600 , total_time / 60 / max(self.model_count , 1) , total_time / max(self.epoch_count , 1))
        self.model_info['train_process'] = _str #type:ignore
        logger.critical(self.model_info['train_process'])

    def process_test(self):
        self.model_info['test_time'] = time.time()
        logger.critical(f'Start Process [Test Model]!')        
        logger.warning(f'Each Model Date Testing Mean Score({config.train_params["criterion"]["score"]}):')
        self.test_result_model_num = np.repeat(config.model_num_list,len(config.output_types))
        self.test_result_output_type = np.tile(config.output_types,len(config.model_num_list))
        logger.info('{: <11s}'.format('Models') + ('{: >8d}'*len(self.test_result_model_num)).format(*self.test_result_model_num))
        logger.info('{: <11s}'.format('Output') + ('{: >8s}'*len(self.test_result_model_num)).format(*self.test_result_output_type))
        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation('test')
            self.model_test()
            self.save_model_preds()
        self.model_result()
        _str = 'Finish Process [Test Model]! Cost {:.1f} Secs'.format(time.time()-self.model_info['test_time'])
        self.model_info['test_process'] = _str # type: ignore
        logger.critical(self.model_info['test_process'])

    def process_instance(self):
        self.model_info['instance_time'] = time.time()
        if config.anchoring < 0:
            _text , _cond = ask_for_confirmation(f'Do you want to copy the model to instance?[yes/else no]: ' , timeout = -1)
            anchoring = all([_t.lower() in ['yes','y'] for _t in _text])
        else:
            anchoring = config.anchoring > 0

        if anchoring:
            self.model_info['instance_time'] = time.time()
            logger.critical(f'Start Process [Copy to Instance]!')        
            if os.path.exists(config.instance_path): 
                logger.critical(f'Old instance {config.instance_path} exists , remove manually first to override!')
                logger.critical(f'The command can be "rm -r {config.instance_path}"')
                return
            else:
                shutil.copytree(config.model_base_path , config.instance_path)
        else:
            logger.critical(f'Will not copy to instance!')
            return
        self.test_result_model_num = np.repeat(config.model_num_list,len(config.output_types))
        self.test_result_output_type = np.tile(config.output_types,len(config.model_num_list))
        logger.warning('Copy from model to instance finished , Start going forward')

        # load the exact config in the instance folder
        config.reload(config_path = f'{config.instance_path}/config_train.yaml')

        for model_date , model_num in self.model_iter():
            self.model_date , self.model_num = model_date , model_num
            self.model_preparation('instance')
            self.model_test()
            self.save_model_preds()
        self.model_result()
        logger.critical('Finish Process [Copy to Instance]! Cost {:.1f} Secs'.format(time.time() - self.model_info['instance_time']))  

    def model_iter(self):
        new_iter = list(itertools.product(self.data.model_date_list , config.model_num_list))
        if config.resume_training and self.process_name == 'train':
            models_trained = np.full(len(new_iter) , True)
            for i,(model_date,model_num) in enumerate(new_iter):
                if not os.path.exists(f'{config.model_base_path}/{model_num}/{model_date}.best.pt'):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = FilteredIterator(new_iter , models_trained == 0)
        return new_iter
    
    def model_preparation(self , process , last_n = 30 , best_n = 5):
        assert process in ['train' , 'test' , 'instance']
        with self.ptimer('model_preparation' , process):
            param = config.model_params[self.model_num]

            self.prediction = [] if config.get('output_prediction') or self.process_name == 'instance' else None
            # In a new model , alters the penalty function's lamb
            self.metric_function = self.update_metricfunc(self.metric_function , param , config)

            path_prefix = '{}/{}'.format(param.get('path') , self.model_date)
            path = {'target'      : {op_type:f'{path_prefix}.{op_type}.pt' for op_type in config.output_types} , 
                    'source'      : {op_type:[] for op_type in (config.output_types + ['rounds'])} , # del at each model train
                    'candidate'   : {op_type:None for op_type in (config.output_types + ['transfer'])} , # not del at each model train
                    'performance' : {op_type:None for op_type in config.output_types}}
            if 'best'    in config.output_types:
                path['candidate']['best'] = f'{path_prefix}.best.pt'
            if 'swalast' in config.output_types: 
                path['source']['swalast'] = [f'{path_prefix}.lastn.{i}.pt' for i in range(last_n)]
            if 'swabest' in config.output_types: 
                path['source']['swabest']      = [f'{path_prefix}.bestn.{i}.pt' for i in range(best_n)] 
                path['candidate']['swabest']   = path['source']['swabest']
                path['performance']['swabest'] = [-10000. for i in range(best_n)]
            if config.train_params['transfer'] and self.model_date > self.data.model_date_list[0]:
                last_model_date = max([d for d in self.data.model_date_list if d < self.model_date])
                path['candidate']['transfer'] = '{}/{}.best.pt'.format(param.get('path') , last_model_date)
                
            self.param , self.path = param , path
    
    def model_train(self):
        self.TrainModelStart()
        while self.cond.get('loop_status') != 'model':
            self.NewLoop()
            self.TrainerInit()
            self.TrainEpoch()
            self.LoopCondition()
        self.TrainModelEnd()
        gc.collect() 
        torch.cuda.empty_cache()
    
    def model_test(self):
        self.model_test_start()
        self.model_forecast()
        self.model_test_end()
        gc.collect() 
        torch.cuda.empty_cache()
        
    def _init_variables(self , key = 'model'):
        """
        Reset variables of 'model' , 'round' , 'attempt' start
        """
        if key == 'epoch' : return
        assert key in ['model' , 'round' , 'attempt'] , f'KeyError : {key}'

        self.epoch_i = -1
        self.epoch_attempt_best = -1
        self.score_attempt_best = -10000.
        self.loss_list  = {'train' : [] , 'valid' : []}
        self.score_list = {'train' : [] , 'valid' : []}
        self.lr_list    = []
        
        if key in ['model' , 'round']:
            self.attempt_i = -1
            self.score_round_best = -10000.
        
        if key in ['model']:
            self.round_i = -1
            self.epoch_all = -1
            self.tick = np.ones(10) * time.time()
            self.text = {k : '' for k in ['model','round','attempt','epoch','exit','stat','time','trainer']}
            self.cond = {'terminate' : {} , 'nan_loss' : False , 'loop_status' : 'round'}

    def TrainModelStart(self):
        """
        Reset model specific variables
        """
        with self.ptimer('TrainModelStart'):
            self._init_variables('model')
            self.nanloss_life = config.train_params['trainer']['nanloss']['retry']
            self.text['model'] = '{:s} #{:d} @{:4d}'.format(config.model_name , self.model_num , self.model_date)
            dataloader_param = self.data.get_dataloader_param('train' , namespace=self)   
            if (self.data.dataloader_param != dataloader_param):
                self.data.create_dataloader(*dataloader_param) 
                self.tick[1] = time.time()
                self._prints('train_dataloader')
            
    def TrainModelEnd(self):
        """
        Do necessary things of ending a model(model_data , model_num)
        """
        with self.ptimer('TrainModelEnd'):
            self.storage.del_path(*[list(v) for v in self.path['source'].values()])
            if self.process_name == 'train' : self.model_count += 1
            self.tick[2] = time.time()
            self._prints('model_end')

    def NewLoop(self):
        """
        Reset and loop variables giving loop_status
        """
        with self.ptimer('NewLoop'):
            self._init_variables(self.cond.get('loop_status' , ''))
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
        net : 1. Create an instance of f'{config.model_module}' or inherit from 'lastround'/'transfer'
              2. In transfer mode , p_late and p_early with be trained with different lr's. If not net.parameters are trained by same lr
        optimizer : Adam or SGD
        scheduler : Cosine or StepLR
        """
        with self.ptimer('TrainerInit'):
            if self.cond.get('loop_status') == 'epoch': return
            self.load_model('train')
            self.max_round: int = getattr(self.net , 'max_round' , lambda x=0:1)()
            self.optimizer = self.load_optimizer()
            self.scheduler = self.load_scheduler() 
            self.multiloss = self.new_multiloss(config.train_params['multitask'] , self.param['num_output'])

    def TrainEpoch(self):
        """
        Iterate train and valid dataset, calculate loss/score , update values
        If nan loss occurs, turn to _deal_nanloss
        """

        """
        # with modifier
        with self.ptimer('TrainEpoch/train_epochs'):
            self.net.train()
            clip_value = config.train_params['trainer']['gradient'].get('clip_value')
            iterator = self.data.dataloaders['train']
            _loss , _score = np.full(len(iterator),np.nan) , np.full(len(iterator),np.nan)
            for i , batch_data in enumerate(iterator):
                x = self.modifier['inputs'](batch_data['x'] , batch_data , self.data)
                self.optimizer.zero_grad()
                with self.ptimer('TrainEpoch/train/forward'):
                    pred , hidden = self.net(x)
                with self.ptimer('TrainEpoch/train/loss'):
                    penalty_kwargs = {'net' : self.net , 'hidden' : hidden , 'label' : batch_data['y']}
                    metric = self.model_metric(batch_data['y'] , pred , 'train' , weight = batch_data['w'] , penalty_kwargs=penalty_kwargs)
                    metric = self.modifier['metric'](metric, batch_data, self.data)
                with self.ptimer('TrainEpoch/train/backward'):
                    (metric['loss'] + metric['penalty']).backward()
                with self.ptimer('TrainEpoch/train/step'):
                    if clip_value is not None : nn.utils.clip_grad_value_(self.net.parameters(), clip_value = clip_value)
                    self.optimizer.step()
                self.modifier['update'](None , batch_data , self.data)
                _loss[i] , _score[i] = metric['loss_item'] , metric['score']
                if getattr(iterator , 'progress_bar'): iterator.display(f'Ep#{self.epoch_i:3d} train loss:{np.mean(_loss[:i+1]):.5f}')
            if np.isnan(sum(_loss)): return self._deal_nanloss()
            self.loss_list['train'].append(np.mean(_loss)) , self.score_list['train'].append(np.mean(_score))

        with self.ptimer('TrainEpoch/valid_epochs'):
            self.net.eval()     
            iterator = self.data.dataloaders['valid']
            _loss , _score = np.full(len(iterator),np.nan) , np.full(len(iterator),np.nan)
            for i , batch_data in enumerate(iterator):
                x = self.modifier['inputs'](batch_data['x'] , batch_data , self.data)
                # self.device.print_cuda_memory()
                with self.ptimer('TrainEpoch/valid/forward'):
                    pred , _ = self.net(x)
                with self.ptimer('TrainEpoch/valid/loss'):
                    metric = self.model_metric(batch_data['y'] , pred , 'valid' , weight = batch_data['w'])
                    metric = self.modifier['metric'](metric, batch_data, self.data)
                self.modifier['update'](None , batch_data , self.data)
                _loss[i] , _score[i] = metric['loss_item'] , metric['score']
                if getattr(iterator , 'progress_bar'): iterator.display(f'Ep#{self.epoch_i:3d} valid ic:{np.mean(_score[:i+1]):.5f}')
            self.loss_list['valid'].append(np.mean(_loss)) , self.score_list['valid'].append(np.mean(_score))
        """
        with self.ptimer('TrainEpoch/train_epochs'):
            self.net.train() # type:ignore
            iterator = self.data.dataloaders['train']
            _loss , _score = torch.ones(len(iterator)).fill_(torch.nan), torch.ones(len(iterator)).fill_(torch.nan)
            for i , batch_data in enumerate(iterator):
                
                outputs = self.model_forward('train' , batch_data)
                metrics = self.model_metric('train' , outputs , batch_data , valid_sample = None)
                self.model_backward('train' , metrics)

                _loss[i] , _score[i] = metrics['loss_item'] , metrics['score']
                if getattr(iterator , 'progress_bar'): iterator.display(f'Ep#{self.epoch_i:3d} train loss:{_loss[:i+1].mean():.5f}')
            if _loss.isnan().any(): return self._deal_nanloss()
            self.loss_list['train'].append(_loss.mean()) 
            self.score_list['train'].append(_score.mean())
        
        with self.ptimer('TrainEpoch/valid_epochs'):
            self.net.eval()  # type:ignore
            iterator = self.data.dataloaders['valid']
            _loss , _score = torch.ones(len(iterator)).fill_(torch.nan), torch.ones(len(iterator)).fill_(torch.nan)
            for i , batch_data in enumerate(iterator):
                # self.device.print_cuda_memory()
                outputs = self.model_forward('valid' , batch_data)
                metrics = self.model_metric('valid' , outputs , batch_data , valid_sample = None)

                _loss[i] , _score[i] = metrics['loss_item'] , metrics['score']
                if getattr(iterator , 'progress_bar'): iterator.display(f'Ep#{self.epoch_i:3d} valid ic:{_score[:i+1].mean():.5f}')
            self.loss_list['valid'].append(_loss.mean()) 
            self.score_list['valid'].append(_score.mean())

        self.lr_list.append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()
        self.reset_scheduler()

    def LoopCondition(self):
        """
        Update condition of continuing training epochs , restart attempt if early exit , proceed to next round if convergence , reset round if nan loss
        """
        with self.ptimer('LoopCondition/assess'):
            if self.cond['nan_loss']:
                logger.error(f'Initialize a new model to retrain! Lives remaining {self.nanloss_life}')
                self._init_variables('model')
                self.cond['loop_status'] = 'round'
                return
                
            valid_score = self.score_list['valid'][-1]
            
            save_targets = [] 
            if valid_score > self.score_attempt_best: 
                self.epoch_attempt_best = self.epoch_i 
                self.score_attempt_best = valid_score
                
            if valid_score > self.score_round_best:
                self.score_round_best = valid_score
                save_targets.append(self.path['target']['best'])

            if 'swalast' in config.output_types:
                self.path['source']['swalast'] = self.path['source']['swalast'][1:] + self.path['source']['swalast'][:1]
                save_targets.append(self.path['source']['swalast'][-1])
                
                p_valid = self.path['source']['swalast'][-len(self.score_list['valid']):]
                arg_max = np.argmax(self.score_list['valid'][-len(p_valid):])
                arg_swa = (lambda x:x[(x>=0) & (x<len(p_valid))])(min(5,len(p_valid)//3)*np.arange(-5,3)+arg_max)[-5:]
                self.path['candidate']['swalast'] = [p_valid[i] for i in arg_swa]
                
            if 'swabest' in config.output_types:
                arg_min = np.argmin(self.path['performance']['swabest'])
                if valid_score > self.path['performance']['swabest'][arg_min]:
                    self.path['performance']['swabest'][arg_min] = valid_score
                    save_targets.append(self.path['candidate']['swabest'][arg_min])
                
            self.save_model(paths = save_targets)
            # self.storage.save_model_state(self.net , save_targets)
            self._prints('epoch_step')
        
        with self.ptimer('LoopCondition/confirm_status'):
            self.text['exit'] , self.cond['terminate'] = self._terminate_cond() #type: ignore
            if self.text['exit']:
                if (self.epoch_i < config.train_params['trainer']['retrain'].get('min_epoch' if self.max_round <= 1 else 'min_epoch_round') - 1 and 
                    self.attempt_i < config.train_params['trainer']['retrain']['attempts'] - 1):
                    self.cond['loop_status'] = 'attempt'
                    self._prints('new_attempt')
                elif self.round_i < self.max_round - 1:
                    self.cond['loop_status'] = 'round'
                    self.save_model(key = 'best')
                    self._prints('new_round')
                else:
                    self.cond['loop_status'] = 'model'
                    # print(self.net.get_probs())
                    self.save_model(key = config.output_types)
            else:
                self.cond['loop_status'] = 'epoch'
            
    def model_test_start(self):
        """
        Reset model specific variables
        """
        self._init_variables('model')
        dataloader_param = self.data.get_dataloader_param('test' , namespace=self)   
        self.data.create_dataloader(*dataloader_param)
            
        if self.model_num == 0:
            score_date  = np.zeros((len(self.data.model_test_dates) , len(self.test_result_model_num)))
            score_model = np.zeros((1 , len(self.test_result_model_num)))
            self.score_by_date  = np.concatenate([getattr(self,'score_by_date' ,np.empty((0,len(self.test_result_model_num)))) , score_date])
            self.score_by_model = np.concatenate([getattr(self,'score_by_model',np.empty((0,len(self.test_result_model_num)))) , score_model])
                
    def model_forecast(self):
        if not os.path.exists(self.path['target']['best']): self.model_train()
        
        with self.ptimer('model_test/forecast') , torch.no_grad():
            iter_dates = np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
            l0 , l1 = len(self.data.early_test_dates) , len(self.data.model_test_dates)
            assert self.data.dataloaders['test'].__len__() == len(iter_dates)
            for oi , op_type in enumerate(config.output_types):
                self.load_model('test' , op_type)
                self.net.eval() # type:ignore
                iterator = self.data.dataloaders['test']
                test_score = np.full(len(iter_dates),np.nan)
                for i , batch_data in enumerate(iterator):
                    valid_sample=torch.where(batch_data['nonnan'])[0]
                    if len(valid_sample) == 0: continue
                    #batch_data_nn = subset(batch_data , torch.where(batch_data['nonnan'])[0])
                    """
                    pred = torch.full_like(batch_data['y'], fill_value=torch.nan)
                    for batch_j in torch.utils.data.DataLoader(self.device.torch_arange(len(nonnan)) , batch_size = config.batch_size):
                        nnj = nonnan[batch_j]
                        batch_nnj = subset(batch_data , nnj)
                        if hasattr(self.net , 'dynamic_data_assign'): self.net.dynamic_data_assign(batch_nnj , self.data)
                        # x = self.modifier['inputs'](batch_nnj['x'] , batch_nnj , self.data)
                        pred_nnj = self.net(batch_nnj['x'])[0].detach()
                        pred[nnj,0] = pred_nnj[:,0]
                        # self.modifier['update'](None , batch_nnj , self.data)
                    """
                    #batch_data_nn = subset(batch_data , torch.where(batch_data['nonnan'])[0])
                    #x , y , w = batch_data_nn['x'] , batch_data_nn['y'] , batch_data_nn['w']
                    #if hasattr(self.net , 'dynamic_data_assign'): self.net.dynamic_data_assign(batch_data_nn , self.data)
                    outputs = self.model_forward('test' , batch_data)

                    assert not outputs[0].isnan().any()
                    assert iter_dates[i] == self.data.y_date[batch_data['i'][0,1]] , (iter_dates[i] , self.data.y_date[batch_data['i'][0,1]])
                    
                    if i < l0: continue # before this date is warmup stage
                    metrics = self.model_metric('test' , outputs, batch_data , valid_sample = None)
                    test_score[i] = metrics['score']

                    if hasattr(self , 'prediction'):
                        batch_index , batch_pred = batch_data['i'].cpu() , outputs[0].detach().cpu()
                        assert isinstance(self.prediction , list)
                        self.prediction.append({
                            'secid' : self.data.y_secid[batch_index[:,0]] , 
                            'date'  : self.data.y_date[batch_index[:,1]] ,
                            'model' : f'{self.model_num}/{op_type}' ,
                            'values': batch_pred[:,0]
                        })

                    if (i + 1) % 20 == 0 : torch.cuda.empty_cache()
                    if getattr(iterator , 'progress_bar'): iterator.display(f'Date {iter_dates[i]}:{test_score[l0:i+1].mean():.5f}')
                self.score_by_date[-l1:,self.model_num*len(config.output_types) + oi] = np.nan_to_num(test_score[-l1:])
        
    def model_test_end(self):
        """
        Do necessary things of ending a model(model_data , model_num)
        """
        if self.model_num == config.model_num_list[-1]:
            self.score_by_model[-1,:] = np.nanmean(self.score_by_date[-len(self.data.model_test_dates):,],axis = 0)
            logger.info('{: <11d}'.format(self.model_date)+('{:>8.4f}'*len(self.test_result_model_num)).format(*self.score_by_model[-1,:]))

    def save_model_preds(self):
        if not hasattr(self , 'prediction') and self.process_name != 'instance': return NotImplemented
        assert isinstance(self.prediction , list)
        path_output = f'{config.instance_path}/{config.model_name}_allpreds.csv'
        for op_data in self.prediction:
            df = pd.concat([pd.DataFrame(op_data_alist) for op_data_alist in op_data])
            df = df.pivot_table('values' , ['secid' , 'date'] , ['model'])
            with open(path_output , 'a') as f:
                df.to_csv(f , mode = 'a', header = f.tell()==0, index = True)   
  
    def model_result(self):
        # date ic writed down
        date_step = (1 if self.process_name == 'instance' else self.data.kwarg.test_step_day)
        date_list = self.data.test_full_dates[::date_step]
        for model_num in config.model_num_list:
            df = pd.DataFrame({'dates' : date_list} , index = map(lambda x:f'{x[:4]}-{x[4:6]}-{x[6:]}' , date_list.astype(str))) #type:ignore
            for oi , op_type in enumerate(config.output_types):
                df[f'score.{op_type}'] = self.score_by_date[:,model_num*len(config.output_types) + oi]
                df[f'cum_score.{op_type}'] = np.nancumsum(self.score_by_date[:,model_num*len(config.output_types) + oi])
            df.to_csv(config.model_params[model_num]['path'] + f'/{config.model_name}_score_by_date_{model_num}.csv')

        # model ic presentation
        add_row_key   = ['AllTimeAvg' , 'AllTimeSum' , 'Std'      , 'TValue'   , 'AnnIR']
        add_row_fmt   = ['{:>8.4f}'   , '{:>8.2f}'   , '{:>8.4f}' , '{:>8.2f}' , '{:>8.4f}']
        score_mean   = np.nanmean(self.score_by_date , axis = 0)
        score_sum    = np.nansum(self.score_by_date  , axis = 0) 
        score_std    = np.nanstd(self.score_by_date  , axis = 0)
        score_tvalue = score_mean / score_std * (len(self.score_by_date)**0.5) # 10 days return predicted
        score_annir  = score_mean / score_std * ((240 / 10)**0.5) # 10 days return predicted
        add_row_value = (score_mean , score_sum , score_std , score_tvalue , score_annir)
        df = pd.DataFrame(np.concatenate([self.score_by_model , np.stack(add_row_value)]) , 
                          index   = [str(d) for d in self.data.model_date_list] + add_row_key , 
                          columns = [f'{mn}.{o}' for mn,o in zip(self.test_result_model_num,self.test_result_output_type)])
        df.to_csv(f'{config.model_base_path}/{config.model_name}_score_by_model.csv')
        for i in range(len(add_row_key)):
            logger.info('{: <11s}'.format(add_row_key[i]) + (add_row_fmt[i]*len(self.test_result_model_num)).format(*add_row_value[i]))
        self.model_info['test_score_sum'] = {k:v for k,v in zip(df.columns , score_sum.tolist())}  # type:ignore
    
    def summary(self):
        out_dict = {
            '0_start' : time.ctime(self.model_info.get('init_time')),
            '1_basic' :'+'.join(['short' if config.short_test else 'long' , config.storage_type , config.precision]),
            '2_model' :''.join([config.model_module , '_' , config.model_data_type , '(x' , str(config.model_num) , ')']),
            '3_time'  :'-'.join([str(config.beg_date),str(config.end_date)]),
            '4_train' :self.model_info.get('train_process'),
            '5_test'  :self.model_info.get('test_process'),
            '6_result':self.model_info.get('test_score_sum'),
        }
        out_path = f'./results/model_results.yaml'
        os.makedirs(os.path.dirname(out_path) , exist_ok=True)
        with open(out_path , 'a' if os.path.exists(out_path) else 'w') as f:
            yaml.dump(out_dict , f)
        app.ptimer.print()

    """     
    def model_metric(self, label , pred , key , weight = None , valid_sample = None , penalty_kwargs={} , **kwargs):
        '''
        Calculate loss(with gradient), score
        Inputs : 
            kwargs : other inputs used in calculating loss , penalty and score
        Possible Methods :
        loss:    pearsonr , mse , ccc
        penalty: hidden_orthogonality , tra_ot_penalty
        score:  pearson , spearman , mse , ccc
        '''

        assert key in ['train' , 'valid' , 'test'] , key
        if label.shape != pred.shape: # if more label than output
            assert label.shape[:-1] == pred.shape[:-1] , (label.shape , pred.shape)
            label = label.transpose(0,-1)[:pred.shape[-1]].transpose(0,-1)
        if valid_sample is not None:
            label , pred = label[valid_sample] , pred[valid_sample]
            if weight is not None: weight = weight[valid_sample]
        score_dim = lambda x:None if x is None else x.select(-1,0)

        if key == 'train':
            if self.param['num_output'] > 1:
                loss = self.metric_function['loss'](label , pred , weight , dim = 0)[:self.param['num_output']]
                loss = self.multiloss.calculate_multi_loss(loss , self.net.get_multiloss_params())
            else:
                loss = self.metric_function['loss'](score_dim(label) , score_dim(pred) , score_dim(weight))
            
            penalty = 0.
            for _pen_dict in self.metric_function['penalty'].values():
                if _pen_dict['lamb'] > 0 and _pen_dict['cond']: 
                    penalty = penalty + _pen_dict['lamb'] * _pen_dict['func'](**penalty_kwargs)  
            loss_item = loss.item()
        else:
            loss_item = loss = penalty = 0.

        score = self.metric_function['score'][key](score_dim(label) , score_dim(pred) , score_dim(weight)).item()
        return {'loss' : loss , 'loss_item' : loss_item , 'score' : score , 'penalty' : penalty}
    """
    
    def model_forward(self , key , batch_data):
        with self.ptimer(f'{key}/forward'):
            if hasattr(self.net , 'dynamic_data_assign'): getattr(self.net , 'dynamic_data_assign')(batch_data , self.data)
            return self.net(batch_data['x']) #type:ignore

    def model_metric(self, key , outputs_of_net, batch_data , valid_sample = None, **kwargs):
        # outputs_of_net = self.net(batch_data['x'])
        # valid_sample = torch.where(batch_data['nonnan'])[0]
        with self.ptimer(f'{key}/loss'):
            pred , hidden  = outputs_of_net
            label , weight = batch_data['y'] , batch_data['w']
            penalty_kwargs = {}
            if key == 'train': penalty_kwargs.update({'net' : self.net , 'hidden' : hidden , 'label' : label})
            metrics = self.calculate_metrics(
                key , self.metric_function , label=label , pred=pred , weight = weight ,
                multiloss = self.multiloss , net = self.net ,
                valid_sample = valid_sample , penalty_kwargs = penalty_kwargs)
            return metrics
        
    def model_backward(self, key , metrics):
        if key.lower() != 'train': return NotImplemented
        clip_value = config.train_params['trainer']['gradient'].get('clip_value')
        with self.ptimer(f'{key}/backward'):
            self.optimizer.zero_grad()
            (metrics['loss'] + metrics['penalty']).backward()
            if clip_value is not None : nn.utils.clip_grad_value_(self.net.parameters(), clip_value = clip_value) #type:ignore
            self.optimizer.step()
    
    def _prints(self , key):
        """
        Print out status giving display conditions and looping conditions
        """
        printer = [logger.info] if (config.verbosity > 2 or self.model_count < config.model_num) else [logger.debug]
        sdout   = None
        if key == 'model_end':
            self.text['epoch'] = 'Ep#{:3d}'.format(self.epoch_all)
            self.text['stat']  = 'Train{: .4f} Valid{: .4f} BestVal{: .4f}'.format(self.score_list['train'][-1],self.score_list['valid'][-1],self.score_round_best)
            self.text['time']  = 'Cost{:5.1f}Min,{:5.1f}Sec/Ep'.format((self.tick[2]-self.tick[0])/60 , (self.tick[2]-self.tick[1])/(self.epoch_all+1))
            sdout = self.text['model'] + '|' + self.text['round'] + ' ' + self.text['attempt'] + ' ' + \
                    self.text['epoch'] + ' ' + self.text['exit'] + '|' + self.text['stat'] + '|' + self.text['time']
            printer = [logger.warning]
        elif key == 'epoch_step':
            self.text['trainer'] = 'loss {: .5f}, train{: .5f}, valid{: .5f}, max{: .4f}, best{: .4f}, lr{:.1e}'.format(
                self.loss_list['train'][-1] , self.score_list['train'][-1] , self.score_list['valid'][-1] , 
                self.score_attempt_best , self.score_round_best , self.lr_list[-1])
            if self.epoch_i % [10,5,5,3,3,1][min(config.verbosity // 2 , 5)] == 0:
                sdout = ' '.join([self.text['attempt'],'Ep#{:3d}'.format(self.epoch_i),':', self.text['trainer']])
        elif key == 'reset_learn_rate':
            sdout = 'Reset learn rate and scheduler at the end of epoch {} , effective at epoch {}'.format(
                self.epoch_i , self.epoch_i+1 , ', and will speedup2x' * config.train_params['trainer']['learn_rate']['reset']['speedup2x'])
        elif key == 'new_attempt':
            sdout = ' '.join([self.text['attempt'],'Epoch #{:3d}'.format(self.epoch_i),':',self.text['trainer'],', Next attempt goes!'])
        elif key == 'new_round':
            sdout = self.text['round'] + ' ' + self.text['exit'] + ': ' + self.text['trainer'] + ', Next round goes!'
        elif key == 'train_dataloader':
            sdout = ' '.join([self.text['model'],'LoadData Cost {:>6.1f}Secs'.format(self.tick[1]-self.tick[0])])  
        else:
            raise Exception(f'KeyError : {key}')
        
        for prt in printer:
            if sdout is not None: prt(sdout)        

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

    def _terminate_cond(self):
        """
        Whether terminate condition meets
        """
        term_dict = config.train_params['terminate'].get('overall')
        if self.max_round > 1 and config.train_params['terminate'].get('round') is not None:
            term_dict = config.train_params['terminate'].get('round')
        term_cond = {}
        exit_text = None
        for key , arg in term_dict.items():
            if key == 'max_epoch':
                term_cond[key] = self.epoch_i >= min(arg , config.max_epoch) - 1
                if term_cond[key] and exit_text is None: exit_text = 'Max Epoch'
            elif key == 'early_stop':
                term_cond[key] = self.epoch_i - self.epoch_attempt_best >= arg
                if term_cond[key] and exit_text is None: exit_text = 'EarlyStop'
            elif key == 'tv_converge':
                term_cond[key] = (list_converge(self.loss_list['train']  , arg.get('min_epoch') , arg.get('eps')) and
                             list_converge(self.score_list['valid'] , arg.get('min_epoch') , arg.get('eps')))
                if term_cond[key] and exit_text is None: exit_text = 'T & V Cvg'
            elif key == 'train_converge':
                term_cond[key] = list_converge(self.loss_list['train']  , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text is None: exit_text = 'Train Cvg'
            elif key == 'valid_converge':
                term_cond[key] = list_converge(self.score_list['valid'] , arg.get('min_epoch') , arg.get('eps'))
                if term_cond[key] and exit_text is None: exit_text = 'Valid Cvg'
            else:
                raise Exception(f'KeyError : {key}')
        return exit_text , term_cond
    
    def save_model(self , key = None , paths = None , savable_net = None):
        if key is None or paths is None: return NotImplemented
        if savable_net is None:
            savable_net = self.net
            if hasattr(savable_net,'dynamic_data_unlink'):  savable_net = getattr(savable_net , 'dynamic_data_unlink')()

        with self.ptimer('save_model'):
            if isinstance(paths , (str,list,tuple)):
                self.storage.save_model_state(savable_net , paths , to_disk=False)
            if isinstance(key , (list,tuple)):
                [self.save_model(k , savable_net = savable_net) for k in key]
            elif isinstance(key , str):
                assert key in ['best' , 'swalast' , 'swabest']
                p_exists = self.storage.valid_paths(self.path['candidate'][key])
                if len(p_exists) == 0: print(key , self.path['candidate'][key] , self.path['performance'][key])
                if key == 'best':
                    savable_net = self.storage.load(p_exists[0])
                    if self.round_i < self.max_round - 1:
                        if not self.path['source']['rounds']:
                            self.path['source']['rounds'] = ['{}/{}.round.{}.pt'.format(self.param.get('path'),self.model_date,r) for r in range(self.max_round-1)]
                        self.storage.save_model_state(savable_net , self.path['source']['rounds'][self.round_i])
                else:
                    savable_net = self.load_swa_model(p_exists)
                self.storage.save_model_state(savable_net , self.path['target'][key] , to_disk = True) 
    
    def load_model(self , process , key = 'best'):
        assert process in ['train' , 'test' , 'instance']
        with self.ptimer('load_model'):
            if process == 'train':           
                if self.round_i > 0:
                    model_path = self.path['source']['rounds'][self.round_i-1]
                elif self.path['candidate'].get('transfer'):
                    if not config.train_params['transfer']: raise Exception('get transfer')
                    model_path = self.path['candidate']['transfer']
                else:
                    model_path = -1
            else:
                model_path = self.path['target'][key]

            self.net = self.device(self.new_model(config.model_module , self.param , self.storage.load(model_path , from_disk = True))) # type:ignore
            if hasattr(self.net , ''): getattr(self.net, 'training_round')(self.round_i)

            """
            # default : none modifier
            # input : (inputs/metric/update , batch_data , self.data)
            # output : new_inputs/new_metric/None 
            self.modifier = {'inputs': lambda x,b,d:x, 'metric': lambda x,b,d:x, 'update': lambda x,b,d:None}
            if 'modifier_inputs' in self.net.__dir__(): self.modifier['inputs'] = lambda x,b,d:self.net.modifier_inputs(x,b,d)
            if 'modifier_metric' in self.net.__dir__(): self.modifier['metric'] = lambda x,b,d:self.net.modifier_metric(x,b,d)
            if 'modifier_update' in self.net.__dir__(): self.modifier['update'] = lambda x,b,d:self.net.modifier_update(x,b,d)
            """
            
    def load_swa_model(self , model_path_list = []):
        if len(model_path_list) == 0: raise Exception('empty swa input')
        net = self.new_model(config.model_module , self.param)
        swa_net = self.device(AveragedModel(net))
        for p in model_path_list:
            swa_net.update_parameters(self.storage.load_model_state(net , p)) # type: ignore
        update_bn(self._swa_update_bn_loader(self.data.dataloaders['train']) , swa_net) # type: ignore
        return swa_net.module # type: ignore
    
    def _swa_update_bn_loader(self , loader):
        for data in loader: yield [data['x'] , data['y'] , data['w']]
    
    def load_optimizer(self , new_opt_kwargs = None , new_lr_kwargs = None):
        if new_opt_kwargs is None:
            opt_kwargs = config.train_params['trainer']['optimizer']
        else:
            opt_kwargs = deepcopy(config.train_params['trainer']['optimizer'])
            opt_kwargs.update(new_opt_kwargs)
        
        if new_lr_kwargs is None:
            lr_kwargs = config.train_params['trainer']['learn_rate']
        else:
            lr_kwargs = deepcopy(config.train_params['trainer']['learn_rate'])
            lr_kwargs.update(new_lr_kwargs)
        base_lr = lr_kwargs['base'] * lr_kwargs['ratio']['attempt'][:self.attempt_i+1][-1] * lr_kwargs['ratio']['round'][:self.round_i+1][-1]
        
        return self.new_optimizer(self.net , opt_kwargs['name'] , base_lr , transfer = self.path['candidate'].get('transfer') , 
                                  encoder_lr_ratio = lr_kwargs['ratio']['transfer'], **opt_kwargs['param'])
    
    def load_scheduler(self , new_shd_kwargs = None):
        if new_shd_kwargs is None:
            shd_kwargs = config.train_params['trainer']['scheduler']
        else:
            shd_kwargs = deepcopy(config.train_params['trainer']['scheduler'])
            shd_kwargs.update(new_shd_kwargs)
        return self.new_scheduler(self.optimizer, shd_kwargs['name'], **shd_kwargs['param'])
    
    def reset_scheduler(self):
        rst_kwargs = config.train_params['trainer']['learn_rate']['reset']
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
            shd_kwargs = deepcopy(config.train_params['trainer']['scheduler'])
            for k in np.intersect1d(list(shd_kwargs['param'].keys()),['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']):
                shd_kwargs['param'][k] //= 2
        else:
            shd_kwargs = None
        self.scheduler = self.load_scheduler(shd_kwargs)
        self._prints('reset_learn_rate')
    
    @classmethod
    def new_model(cls , module , param = {} , state_dict = None , **kwargs):
        net = getattr(model , f'{module}')(**param)
        if state_dict: net.load_state_dict(state_dict)
        return net

    @classmethod
    def new_metricfunc(cls , params , **kwargs):
        return {
            'loss'    : loss_function(params['loss']) , 
            'penalty' : {k:penalty_function(k,v) for k,v  in params['penalty'].items()} ,
            'score'   : {k:score_function(v)     for k,v in params['score'].items()} ,
        }

    @classmethod
    def new_multiloss(cls , params , num_output = 1 , **kwargs):
        if num_output <= 1: return None
        return multiloss_calculator(params['type'],num_output,**params['param_dict'][params['type']])

    @classmethod
    def update_metricfunc(cls , mf : dict , model_param , config , **kwargs):
        mf['penalty'].get('hidden_orthogonality',{})['cond'] = config.tra_model or model_param.get('hidden_as_factors',False)
        mf['penalty'].get('tra_ot_penalty',{})['cond']       = config.tra_model
        return mf

    @classmethod
    def calculate_metrics(cls , key , metrics , label , pred , weight = None , multiloss = None , net = None ,
                        valid_sample = None , penalty_kwargs = {} , **kwargs):
        """
        Calculate loss(with gradient), penalty , score
        """
        if label.shape != pred.shape: # if more label than output
            label = label[...,:pred.shape[-1]]
            assert label.shape == pred.shape , (label.shape , pred.shape)
        if valid_sample is not None:
            label , pred = label[valid_sample] , pred[valid_sample]
            weight = None if weight is None else weight[valid_sample]

        label0 , pred0 = label.select(-1,0) , pred.select(-1,0)
        weight0 = None if weight is None else weight.select(-1,0)

        with torch.no_grad():
            losses = None
            loss_item = loss = penalty = 0.
            score = metrics['score'][key](label0 , pred0 , weight0).item()

        if key == 'train':
            # loss
            if multiloss is not None:
                losses = metrics['loss'](label , pred , weight , dim = 0)[:multiloss.num_task]
                if hasattr(net , 'get_multiloss_params'):
                    loss = multiloss.calculate_multi_loss(losses , net.get_multiloss_params())  # type:ignore
                else:
                    raise Exception('net has no attr: get_multiloss_params')
            else:
                losses = None
                loss = metrics['loss'](label0 , pred0 , weight0)
            loss_item = loss.item()
            # penalty
            for _pen_dict in metrics['penalty'].values():
                if _pen_dict['lamb'] <= 0 or not _pen_dict['cond']: continue
                penalty = penalty + _pen_dict['lamb'] * _pen_dict['func'](**penalty_kwargs)  
            
        return {'loss' : loss , 'loss_item' : loss_item , 'score' : score , 'penalty' : penalty , 'losses': losses}

    @classmethod
    def new_scheduler(cls , optimizer, key = 'cycle', **kwargs):
        if key == 'cos':
            scheduler = lr_cosine_scheduler(optimizer, **kwargs)
        elif key == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif key == 'cycle':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=[pg['lr_param'] for pg in optimizer.param_groups],cycle_momentum=False,mode='triangular2',**kwargs)
        return scheduler

    @classmethod
    def new_optimizer(cls , net , key ='Adam', base_lr = 0.005, transfer = False , encoder_lr_ratio = 1., decoder_lr_ratio = 1., **kwargs):
        if transfer:
            # define param list to train with different learn rate
            p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in net.named_parameters() if 'encoder' in x.split('.')[:3]]
            p_dec = [p for x,p in net.named_parameters() if 'encoder' not in x.split('.')[:3]]
            net_param_groups = [{'params': p_dec , 'lr': base_lr * decoder_lr_ratio , 'lr_param': base_lr * decoder_lr_ratio},
                                {'params': p_enc , 'lr': base_lr * encoder_lr_ratio , 'lr_param': base_lr * encoder_lr_ratio}]
        else:
            net_param_groups = [{'params': [p for p in net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

        optimizer = {
            'Adam': torch.optim.Adam ,
            'SGD' : torch.optim.SGD ,
        }[key](net_param_groups , **kwargs)
        return optimizer

    @classmethod
    def model_params_filler(cls , x_data = {} , data_type_list = None):
        if data_type_list is None: data_type_list = list(x_data.keys())

        filler = {}
        inday_dim_dict = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}
        input_dim , inday_dim = [] , []
        for mdt in data_type_list:
            x = x_data.get(mdt)
            input_dim.append(x.shape[-1] if x else 6)
            inday_dim.append(x.shape[-2] if x else inday_dim_dict.get(mdt , 1))
        if len(data_type_list) > 1:
            filler.update({'input_dim':tuple(input_dim), 'inday_dim':tuple(inday_dim)})
        elif len(data_type_list) == 1:
            filler.update({'input_dim':input_dim[0] , 'inday_dim':inday_dim[0]})
        else:
            filler.update({'input_dim':1            , 'inday_dim':1           })
        return filler

    @classmethod
    def new_random_input(cls , module = 'tra_lstm2' , model_data_type = 'day' , model_data = None):

        config = train_config(override_config = {'model_module' : module , 'model_data_type' : model_data_type} , do_process=False)
        data_type_list = config.data_type_list
        for smp in config.model_params: smp.update(cls.model_params_filler(data_type_list = data_type_list))
        model_data = ModelData(data_type_list , config)
        model_data.create_dataloader(*model_data.get_dataloader_param('train','train',20170104,config.model_params[0]))

        #inday_dim_dict = {'15m' : 16 , '30m' : 8 , '60m' : 4 , '120m' : 2}
        for i , batch_data in enumerate(model_data.dataloaders['train']):
            batch_data = batch_data
            break
        net = cls.new_model(module , param = config.model_params[0])
        metrics   = cls.update_metricfunc(cls.new_metricfunc(config.train_params['criterion']) , config.model_params[0] , config)
        multiloss = cls.new_multiloss(config.train_params['multitask'] , config.model_params[0]['num_output'])

        return config , net , batch_data , model_data , metrics , multiloss

"""
from run_model import RunModel
from copy import deepcopy
import torch
import torch.nn as nn

config , net , batch_data , model_data , metrics , multiloss = RunModel.new_random_input('simple_lstm' , 'day')
optimizer = RunModel.new_optimizer(net)

#pipeline 1

sd = deepcopy(net.state_dict())
optimizer = RunModel.new_optimizer(net)
optimizer.zero_grad()
if hasattr(net , 'dynamic_data_assign'): net.dynamic_data_assign(batch_data , model_data)
pred , hidden = net(batch_data['x'])
penalty_kwargs = {'net' : net , 'hidden' : hidden , 'label' : batch_data['y']}
metric = RunModel.calculate_metrics('train' , metrics, label=batch_data['y'], pred=pred,
                            weight=batch_data['w'], multiloss=multiloss,net=net,valid_sample=None,
                            penalty_kwargs=penalty_kwargs)
print(metric)
(metric['loss'] + metric['penalty']).backward()
print(net.get_parameter('fc.weight').grad)
clip_value = config.train_params['trainer']['gradient'].get('clip_value')
if clip_value is not None : nn.utils.clip_grad_value_(net.parameters(), clip_value = clip_value)
optimizer.step()
sd_step = deepcopy(net.state_dict())

"""

if __name__ == '__main__':
    try:
        parser = trainer_parser().parse_args()
    except:
        parser = trainer_parser().parse_args(args=[])

    config = train_config(parser = parser , do_process = True)
    set_trainer_environment(config)

    if not config.short_test:
        logger.warning('Model Specifics:')
        pretty_print_dict(config.get_dict([
            'verbosity' , 'storage_type' , 'device' , 'precision' , 'batch_size' , 'model_name' , 'model_module' , 
            'model_data_type' , 'model_num' , 'beg_date' , 'end_date' , 'interval' , 
            'input_step_day' , 'test_step_day' , 'MODEL_PARAM' , 'train_params' , 'compt_params'
        ]))

    app = RunModel(timer = do_timer , storage_type = config.storage_type , device = config.device)
    app.main_process()
    app.summary()
 