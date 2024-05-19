#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --stage=0 --resume=0 --checkname=1 
import numpy as np
import torch

from .data import DataModule
from ..classes import BaseModelModule
from ..func import BigTimer
from ..util import (
    CallBackManager , Checkpoint , Deposition , Device , 
    Logger , Metrics , Model , Optimizer , TrainConfig)

class ModelTrainer(BaseModelModule):
    '''run through the whole process of training'''
    def init_config(self , **kwargs) -> None:
        self.config = TrainConfig.load(do_parser = True , par_args = kwargs)
        self.stage_queue = self.config.stage_queue
    def init_utilities(self , **kwargs) -> None: 
        self.logger     = Logger()
        self.device     = Device()
        self.checkpoint = Checkpoint(self.config)
        self.deposition = Deposition(self.config)
        self.metrics    = Metrics(self.config)
        self.callbacks  = CallBackManager.setup(self)
        self.model      = Model.setup(self)
    def init_data(self , **kwargs): 
        self.data = DataModule(self.config)
    def batch_forward(self) -> None: 
        self.batch_output = self(self.batch_data)
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.status.dataset , self.batch_data, self.batch_output, self.net, assert_nan = True)
        self.metrics.collect_batch_metric()
    def batch_backward(self) -> None:
        assert self.status.dataset == 'train' , self.status.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.loss)
        self.on_after_backward()
    
    @property
    def model_param(self): return self.config.model_param[self.model_num]
    @property
    def model_types(self): return self.config.model_types
    @property
    def batch_dates(self): return np.concatenate([self.data.early_test_dates , self.data.model_test_dates])
    @property
    def batch_warm_up(self): return len(self.data.early_test_dates)
    @property
    def if_transfer(self) -> bool: return self.config.train_param['transfer']
    @property
    def prev_model_date(self): return self.data.prev_model_date(self.model_date)
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_type(self): return self.status.model_type
    @property
    def model_iter(self): return self.deposition.model_iter(self.status.stage , self.data.model_date_list)

    def on_configure_model(self): 
        self.config.set_config_environment()

    def on_fit_model_start(self):
        self.data.setup('fit' , self.model_param , self.model_date)
        self.load_model(True)
    
    def on_fit_model_end(self):
        self.save_model()

    def on_test_batch(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        if self.batch_idx < self.batch_warm_up: return  # before this is warmup stage , only forward
        if self.config.lgbm_ensembler:
            pred = self.model.lgbm_ensembler.predict()
            assert pred is not None
            self.batch_output.override_pred(pred)
        self.batch_metrics()
    
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
    
    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.dataloader = self.data.test_dataloader()
        self.assert_equity(len(self.dataloader) , len(self.batch_dates))
        self.metrics.new_epoch_metric('test' , self.status)
    
    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')
    
    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1.):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_date = (self.prev_model_date if self.if_transfer else 0) if training else self.model_date
        model_file = self.deposition.load_model(model_date , self.model_num , model_type)
        self.transferred = training and self.if_transfer and model_file.exists()
        self.net = self.model.new_model(training , model_file).net(model_file['state_dict'])
        self.metrics.new_model(self.model_param)
        if training:
            self.optimizer = Optimizer(self.net , self.config , self.transferred , lr_multiplier)
            self.checkpoint.new_model(self.model_param , self.model_date)
        else:
            assert model_file.exists() , str(model_file)
            self.net.eval()

    def stack_model(self):
        if not self.metrics.better_attempt(self.status.best_attempt_metric): return
        self.status.best_attempt_metric = self.metrics.best_metric
        self.on_before_save_model()
        self.net = self.net.cpu()
        for model_type in self.model_types:
            model_dict = self.model.collect(model_type)
            self.deposition.stack_model(model_dict , self.model_date , self.model_num , model_type) 

    def save_model(self):
        self.stack_model()
        for model_type in self.model_types:
            self.deposition.dump_model(self.model_date , self.model_num , model_type) 

    @classmethod
    def main(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        app = cls(stage = stage , resume = resume , checkname = checkname , **kwargs)
        with BigTimer(app.logger.critical , 'Main Process'):
            app.main_process()