#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --stage=0 --resume=0 --checkname=1 
import itertools , os
import numpy as np
import pandas as pd
import torch

from typing import Any , ClassVar , Literal , Optional

from ..environ import DIR
from .. import util as U
from .. import callback as CB
from ..util import Logger , PTimer , TrainConfig
from ..util.classes import BatchData , BatchOutput
from .DataModule import DataModule
from ..model import model as MODEL

hooker = CB.ModelHook()
    
class ModelTrainer():
    '''run through the whole process of training'''
    default_model_type = 'best'

    def __init__(self ,  **kwargs):
        self.config     = TrainConfig.load(do_parser = True , par_args = kwargs)
        self.ptimer     = PTimer(True)
        self.logger     = Logger()
        hooker.update_timer(self.ptimer)
        
        self.device     = U.Device()
        self.pipe       = U.Pipeline(self)
        self.data_mod   = DataModule(self.config)
        self.checkpoint = U.Checkpoint(self.config)
        self.deposition = U.Deposition(self.config)
        self.metrics    = U.Metrics(self.config)

        self.callbacks  = [self.pipe , self.checkpoint , self.deposition , self.metrics ,
                           CB.model.DynamicDataLink() , CB.display.LoaderDisplay() , CB.control.CudaEmptyCache()] 

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.configure_model()
        for self.stage in self.config.stage_queue: 
            getattr(self , f'stage_{self.stage}')()
        self.ptimer.summarize()

    @hooker.hook
    def configure_model(self): pass
    
    def stage_data(self):
        '''stage of loading model data'''
        self.on_data_start()
        self.data_mod.load_data()
        self.on_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.on_fit_start()
        for self.model_date , self.model_num in self.model_iter:
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.on_test_start()
        for self.model_date , self.model_num in self.model_iter:
            self.test_model()
        self.on_test_end()

    def fit_model(self):
        self.on_fit_model_start()
        while self.fit_loop:
            self.on_before_fit_epoch()

            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()
            if self.pipe.nanloss: return

            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()
            self.on_after_fit_epoch()
            if self.pipe.loop_terminate: self.save_model()
        self.on_fit_model_end()

    def test_model(self):
        self.on_test_model_start()
        for self.model_type in self.config.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

    def batch_forward(self) -> None:
        if self.batch_data.is_empty:
            self.net_output = BatchOutput.empty()
        else:
            self.net_output = BatchOutput(self.net(self.batch_data.x))
    
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.dataset, self.batch_data.y, self.net_output.pred, 
                               self.batch_data.w, self.net, self.penalty_kwargs, assert_nan = True)
        self.pipe.record_metric(self.metrics)

    def batch_backward(self) -> None:
        assert self.dataset == 'train' , self.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.loss)
        self.on_after_backward()

    
    @property
    def fit_loop(self): 
        return self.pipe.loop_continue
    @property
    def penalty_kwargs(self): 
        '''self customed penalty kwargs'''
        return {'net':self.net,'hidden':self.net_output.hidden,'label':self.batch_data.y}
    @property
    def model_param(self) -> dict: 
        '''current model param'''
        return self.config.model_param[self.model_num]
    @property
    def model_iter(self) -> Any:
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(self.data_mod.model_date_list , self.config.model_num_list))
        if self.config.resume_training and self.stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(self.model_path(model_date , model_num = model_num)):
                    models_trained[max(i-1,0):] = False
                    break
            new_iter = U.Filtered(new_iter , ~models_trained)
        return new_iter
    @hooker.hook
    def on_data_start(self): pass
    @hooker.hook
    def on_data_end(self): pass
    @hooker.hook
    def on_fit_start(self): pass
    @hooker.hook
    def on_fit_end(self): pass
    @hooker.hook
    def on_test_start(self): pass
    @hooker.hook
    def on_test_end(self): pass
    @hooker.hook
    def on_fit_model_start(self):
        self.data_mod.setup('fit' , self.model_param , self.model_date)
    @hooker.hook
    def on_fit_model_end(self): pass
    @hooker.hook
    def on_train_batch(self):
        self.batch_forward()
        self.batch_metrics()
        self.batch_backward()
    @hooker.hook
    def on_validation_batch(self):
        self.batch_forward()
        self.batch_metrics()
    @hooker.hook
    def on_test_batch(self):
        self.assert_equity(self.test_dates[self.batch_idx] , self.data_mod.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        if self.batch_idx < self.test_warm_up: return  # before this is warmup stage , only forward
        self.batch_metrics()
    @hooker.hook
    def on_before_fit_epoch(self):
        if self.pipe.loop_new_attempt: 
            self.pipe.new_attempt()
            self.checkpoint.new_model(self.model_param , self.model_date)
            self.models = U.FittestModel.get_models(self.config.model_types , self.checkpoint)
            self.load_model(True)
            self.optimizer = U.Optimizer(self.net , self.config , self.transferred , self.pipe.attempt)
    @hooker.hook
    def on_after_fit_epoch(self):
        for fittest_model in self.models.values():
            fittest_model.assess(self.net , self.pipe.epoch , self.pipe.valid_score , self.pipe.valid_loss)
    @hooker.hook
    def on_train_epoch_start(self):
        self.dataset = 'train'
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data_mod.train_dataloader()
    @hooker.hook 
    def on_train_epoch_end(self): pass
    @hooker.hook 
    def on_validation_epoch_start(self):
        self.dataset = 'valid'
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data_mod.val_dataloader()
    @hooker.hook 
    def on_validation_epoch_end(self):
        torch.set_grad_enabled(True)
    @hooker.hook 
    def on_test_model_start(self):
        self.dataset = 'test'
        if not self.deposition.exists(self.model_path(self.model_date)): self.fit_model()
        self.data_mod.setup('test' , self.model_param , self.model_date)
        self.test_dates = np.concatenate([self.data_mod.early_test_dates , self.data_mod.model_test_dates])
        self.test_warm_up = len(self.data_mod.early_test_dates)
        self.net.eval()
        torch.set_grad_enabled(False)
    @hooker.hook 
    def on_test_model_end(self):
        torch.set_grad_enabled(True)
    @hooker.hook 
    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.dataloader = self.data_mod.test_dataloader()
        self.assert_equity(len(self.dataloader) , len(self.test_dates))
    @hooker.hook 
    def on_test_model_type_end(self): pass
    @hooker.hook 
    def on_train_batch_start(self): pass
    @hooker.hook 
    def on_train_batch_end(self): pass
    @hooker.hook
    def on_validation_batch_start(self): pass
    @hooker.hook
    def on_validation_batch_end(self): pass
    @hooker.hook
    def on_test_batch_start(self): pass
    @hooker.hook
    def on_test_batch_end(self): pass
    @hooker.hook 
    def on_before_backward(self): pass
    @hooker.hook 
    def on_after_backward(self): pass
    @hooker.hook 
    def on_before_save_model(self): pass
    @hooker.hook
    def load_model(self , training : bool , model_type = default_model_type) -> None:
        '''load model state dict, return net and a sign of whether it is transferred'''
        if training and self.config.train_param['transfer']:         
            model_path = self.model_path(self.data_mod.prev_model_date(self.model_date))
        elif training:
            model_path = self.model_path()
        else:
            model_path = self.model_path(self.model_date , model_type)
        self.transferred = training and self.config.train_param['transfer'] and self.deposition.exists(model_path)
        self.net = MODEL.new(self.config.model_module , self.model_param , self.deposition.load(model_path) , self.device)

    def save_model(self):
        '''save model state dict to deposition'''
        self.on_before_save_model()
        self.net = self.net.cpu()
        for model_type , fittest_model in self.models.items():
            sd = fittest_model.state_dict(self.net , self.data_mod.train_dataloader())
            self.deposition.save_state_dict(sd , self.model_path(self.model_date , model_type)) 
    
    def model_path(self , model_date = -1, model_type = default_model_type , base_path = None , model_num = None):
        '''get model path of deposition giving model date/type/base_path/num'''
        if model_num is None:
            model_dir = self.model_param.get('path')
        elif base_path is None:
            model_dir = f'{self.config.model_base_path}/{model_num}'
        else:
            model_dir = f'{base_path}/{model_num}'
        return '{}/{}.{}.pt'.format(model_dir , model_date , model_type)



    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)

    @classmethod
    def main(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        app = cls(stage = stage , resume = resume , checkname = checkname , **kwargs)
        app.main_process()
