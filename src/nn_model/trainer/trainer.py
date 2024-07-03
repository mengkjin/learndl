#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --stage=0 --resume=0 --checkname=1 
import torch

from .data_module import NetDataModule , BoosterDataModule
from ..classes import BaseTrainer , BoosterData
from ..callback import CallBackManager
from ..ensemble import Booster , ModelEnsembler
from ..util import (Checkpoint , Deposition , Device , Logger , Metrics , Optimizer , TrainConfig)

from ...basic import BOOSTER_MODULE , REG_MODELS , THIS_IS_SERVER
from ...func import BigTimer

class Trainer(BaseTrainer):
    '''run through the whole process of training'''
    def init_config(self , config_path = None , **kwargs) -> None:
        self.config : TrainConfig = TrainConfig.load(config_path = config_path , do_parser = True , par_args = kwargs)
        self.stage_queue = self.config.stage_queue
    def init_utilities(self , **kwargs) -> None: 
        self.logger     = Logger()
        self.device     : Device = Device()
        self.checkpoint : Checkpoint = Checkpoint(self.config)
        self.deposition : Deposition = Deposition(self.config)
        self.metrics    : Metrics = Metrics(self.config)
        self.callbacks  : CallBackManager= CallBackManager.setup(self)
        self.model      : ModelEnsembler = ModelEnsembler.setup(self)
    
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    @property
    def model_types(self): return self.config.model_types
    @property
    def if_transfer(self): return bool(self.config.train_param['transfer'])
    @property
    def model_iter(self): return self.deposition.model_iter(self.status.stage , self.data.model_date_list)

    @classmethod
    def initialize(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        '''
        state:     [-1,choose] , [0,fit+test] , [1,fit] , [2,test]
        resume:    [-1,choose] , [0,no] , [1,yes]
        checkname: [-1,choose] , [0,default] , [1,yes]
        '''
        module_name = TrainConfig.guess_module()
        use_trainer = BoosterTrainer if module_name in BOOSTER_MODULE else NetTrainer
        app = use_trainer(stage = stage , resume = resume , checkname = checkname , **kwargs)
        return app
    
    def go(self):
        with BigTimer(self.logger.critical , 'Main Process'):
            self.main_process()
        return self

    @classmethod
    def update_models(cls):
        if not THIS_IS_SERVER:
            print('This is not server! Will not update models!')
            return
        for model in REG_MODELS:
            config_path = TrainConfig.get_config_path(model.name)
            cls.initialize(stage = 0 , resume = 1 , checkname = 0 , config_path = config_path).go()

class NetTrainer(Trainer):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data : NetDataModule = NetDataModule(self.config)
    def batch_forward(self) -> None: 
        self.batch_output = self(self.batch_data)
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.status.dataset , self.batch_data, self.batch_output, self.net, assert_nan = True)
        self.metrics.collect_batch_metric()
    def batch_backward(self) -> None:
        assert self.status.dataset == 'train' , self.status.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.output)
        self.on_after_backward()

    def fit_model(self):
        self.status.fit_model_start()
        self.on_fit_model_start()
        while not self.status.end_of_loop:
            self.status.fit_epoch_start()
            self.on_fit_epoch_start()

            self.status.dataset_train()
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()

            self.status.dataset_validation()
            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()

            self.on_before_fit_epoch_end()
            self.status.fit_epoch_end()
            self.on_fit_epoch_end()
        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        for self.status.model_type in self.model_types:
            self.status.dataset_test()
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    def on_configure_model(self):  self.config.set_config_environment()
    def on_fit_model_start(self):
        self.data.setup('fit' , self.model_param , self.model_date)
        self.load_model(True)
    def on_fit_model_end(self): self.save_model()
    
    def on_train_epoch_start(self):
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data.train_dataloader()
        self.metrics.new_epoch_metric('train' , self.status)
    
    def on_train_epoch_end(self): 
        self.metrics.collect_epoch_metric('train')
        self.optimizer.scheduler_step(self.status.epoch)
    
    def on_validation_epoch_start(self):
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data.val_dataloader()
        self.metrics.new_epoch_metric('valid' , self.status)
    
    def on_validation_epoch_end(self):
        self.metrics.collect_epoch_metric('valid')
        self.model.assess(self.status.epoch , self.metrics)
        torch.set_grad_enabled(True)
    
    def on_test_model_start(self):
        if not self.deposition.exists(self.model_date , self.model_num , self.model_type): self.fit_model()
        self.data.setup('test' , self.model_param , self.model_date)
        torch.set_grad_enabled(False)

    def on_test_model_end(self):
        torch.set_grad_enabled(True)
    
    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.dataloader = self.data.test_dataloader()
        self.assert_equity(len(self.dataloader) , len(self.batch_dates))
        self.metrics.new_epoch_metric('test' , self.status)
    
    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')

    def on_test_batch(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        self.model.override()
        # before this is warmup stage , only forward
        if self.batch_idx < self.batch_warm_up: return
        self.batch_metrics()

    def on_before_save_model(self):
        self.net = self.net.cpu()
    
    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1.):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_date = (self.prev_model_date if self.if_transfer else 0) if training else self.model_date
        model_file = self.deposition.load_model(model_date , self.model_num , model_type)
        self.transferred = training and self.if_transfer and model_file.exists()
        self.net : torch.nn.Module = self.model.new_model(training , model_file).model(model_file['state_dict'])
        self.metrics.new_model(self.model_param)
        if training:
            self.optimizer : Optimizer = Optimizer(self.net , self.config , self.transferred , lr_multiplier ,
                                                   model_module = self)
            self.checkpoint.new_model(self.model_param , self.model_date)
        else:
            assert model_file.exists() , str(model_file)
            self.net.eval()

    def stack_model(self):
        self.on_before_save_model()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        if self.metrics.better_attempt(self.status.best_attempt_metric): self.stack_model()
        [self.deposition.dump_model(self.model_date , self.model_num , model_type) for model_type in self.model_types]

class BoosterTrainer(Trainer):
    '''run through the whole process of training'''
    def init_data(self , **kwargs): 
        self.data : BoosterDataModule = BoosterDataModule(self.config)

    def batch_forward(self) -> None: 
        if self.status.dataset == 'train': self.booster.fit()
        self.pred  = self.booster.predict(self.status.dataset)
        self.label = self.booster.label(self.status.dataset)

    def batch_metrics(self) -> None:
        self.metrics.calculate_from_tensor(self.status.dataset , self.label , self.pred, assert_nan = True)
        self.metrics.collect_batch_metric()

    def batch_backward(self) -> None: ...

    def fit_model(self):
        self.on_fit_model_start()
        for self.batch_idx , self.batch_data in enumerate(zip(self.data.train_dataloader() , self.data.val_dataloader())):
            self.status.dataset_train()
            self.on_train_batch_start()
            self.on_train_batch()
            self.on_train_batch_end()

            self.status.dataset_validation()
            self.on_validation_batch_start()
            self.on_validation_batch()
            self.on_validation_batch_end()

        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        self.status.dataset_test()
        for self.status.model_type in self.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.data.test_dataloader()):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    def on_configure_model(self):  self.config.set_config_environment()
    def on_fit_model_start(self):
        self.data.setup('fit' , self.model_param , self.model_date)
        self.load_model(True)
    def on_fit_model_end(self): self.save_model()
    
    def on_train_batch_start(self):
        self.metrics.new_epoch_metric('train' , self.status)

    def on_train_batch_end(self): 
        self.metrics.collect_epoch_metric('train')

    def on_validation_batch_start(self):
        self.metrics.new_epoch_metric('valid' , self.status)

    def on_validation_batch_end(self): 
        self.metrics.collect_epoch_metric('valid')

    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.metrics.new_epoch_metric('test' , self.status)

    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')

    def on_test_model_start(self):
        if not self.deposition.exists(self.model_date , self.model_num , self.model_type): self.fit_model()
        self.data.setup('test' , self.model_param , self.model_date)

    def on_test_batch(self):
        if self.batch_idx < self.batch_warm_up: return
        self.batch_forward()
        self.batch_metrics()
    
    def load_model(self , training : bool , *args , **kwargs):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_file = self.deposition.load_model(self.model_date , self.model_num)
        self.booster : Booster = self.model.new_model(training , model_file).model()

    def stack_model(self):
        self.on_before_save_model()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        self.stack_model()
        for model_type in self.model_types:
            self.deposition.dump_model(self.model_date , self.model_num , model_type) 
    
    def __call__(self , input : BoosterData): raise Exception('Undefined call')

    