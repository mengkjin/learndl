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
from ..util.classes import BatchOutput , TrainerStatus
from .DataModule import DataModule
from ..model import model as MODEL

class ModelTrainer:
    '''run through the whole process of training'''
    default_model_type = 'best'

    def __init__(self ,  **kwargs):
        self.config     = U.TrainConfig.load(do_parser = True , par_args = kwargs)
        self.logger     = U.Logger()
        self.ptimer     = U.PTimer(True)
        self.device     = U.Device()
        self.pipe       = U.Pipeline(self)
        self.checkpoint = U.Checkpoint(self.config)
        self.deposition = U.Deposition(self.config)
        self.metrics    = U.Metrics(self.config)
        self.data_mod   = DataModule(self.config)
        self.status     = TrainerStatus()

        self.callbacks = CB.CallBackManager(
            self.pipe , 
            CB.model.DynamicDataLink(self)    , # 29s
            CB.display.LoaderDisplay(self)    , # 1.5s for 2250 batches
            # CB.control.CudaEmptyCache(self)   , # 2.5s for 86 epochs
            CB.control.ProcessTimer(self) ,
        )

    def main_process(self):
        '''Main stage of data & fit & test'''
        with self.ptimer('everything'):
            self.on_configure_model()
            for self.stage in self.config.stage_queue: 
                getattr(self , f'stage_{self.stage}')()
            self.on_summarize_model()
        self.ptimer.summarize()
    
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
        while not self.end_of_loop: # self.fit_loop_continue:
            self.on_fit_epoch_start()
            self.on_train_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_train_batch_start()
                self.on_train_batch()
                self.on_train_batch_end()
            self.on_train_epoch_end()

            self.on_validation_epoch_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_validation_batch_start()
                self.on_validation_batch()
                self.on_validation_batch_end()
            self.on_validation_epoch_end()
            self.on_fit_epoch_end()

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
            self.batch_output = BatchOutput.empty()
        else:
            self.batch_output = BatchOutput(self.net(self.batch_data.x))
    
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.dataset, self.batch_data, self.batch_output, self.net, assert_nan = True)
        self.metrics.collect_batch_metric()
        self.pipe.record_metric(self.metrics)

    def batch_backward(self) -> None:
        assert self.dataset == 'train' , self.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.loss)
        self.on_after_backward()
    
    @property
    def fit_loop_continue(self): 
        return self.pipe.loop_continue
    @property
    def penalty_kwargs(self): 
        '''self customed penalty kwargs , except for net , hidden and label'''
        return {}
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

    def on_configure_model(self): 
        with self.callbacks: pass
    
    def on_summarize_model(self):
        with self.callbacks: pass
    
    def on_data_start(self):
        with self.callbacks: pass
    
    def on_data_end(self):
        with self.callbacks: pass
    
    def on_fit_start(self):
        with self.callbacks: pass
    
    def on_fit_end(self):
        with self.callbacks: pass
    
    def on_test_start(self): 
        with self.callbacks: pass
    
    def on_test_end(self): 
        with self.callbacks: pass
    
    def on_fit_model_start(self):
        with self.callbacks:
            self.fit_status : list[Literal['unfit' , 'failed' , 'fitted']] = []
            self.data_mod.setup('fit' , self.model_param , self.model_date)
            self.metrics.new_model(self.model_param , self.config)
            self.status.epoch   = -1
            self.status.attempt = -1
            self.end_of_loop = U.EndofLoop()
    
    def on_fit_model_end(self):
        with self.callbacks:
            self.save_model()
            self.checkpoint.del_all()
    
    def on_train_batch(self):
        with self.callbacks:
            self.batch_forward()
            self.batch_metrics()
            self.batch_backward()
    
    def on_validation_batch(self):
        with self.callbacks:
            self.batch_forward()
            self.batch_metrics()

    def on_test_batch(self):
        with self.callbacks:
            self.assert_equity(self.test_dates[self.batch_idx] , self.data_mod.y_date[self.batch_data.i[0,1]]) 
            self.batch_forward()
            if self.batch_idx < self.test_warm_up: return  # before this is warmup stage , only forward
            self.batch_metrics()
    
    def on_fit_epoch_start(self):
        with self.callbacks:
            if self.pipe.loop_new_attempt: self.fit_new_attempt()
            self.status.epoch += 1
    
    def on_fit_epoch_end(self):
        with self.callbacks:
            for fittest_model in self.fitmodels.values():
                fittest_model.assess(self.net , self.pipe.epoch , self.pipe.valid_score , self.pipe.valid_loss)
            self.end_of_loop.loop_end(self.status.epoch)
    
    def on_train_epoch_start(self):
        with self.callbacks:
            self.dataset = 'train'
            self.net.train()
            torch.set_grad_enabled(True)
            self.dataloader = self.data_mod.train_dataloader()
            self.metrics.new_epoch_metric('train' , model_num = self.model_num , model_date = self.model_date , epoch = self.status.epoch , model_type = self.model_type)
    
    def on_train_epoch_end(self): 
        with self.callbacks: 
            self.metrics.collect_epoch_metric('train')
            pass
    
    def on_validation_epoch_start(self):
        with self.callbacks:
            self.dataset = 'valid'
            self.net.eval()
            torch.set_grad_enabled(False)
            self.dataloader = self.data_mod.val_dataloader()
            self.metrics.new_epoch_metric('valid' , model_num = self.model_num , model_date = self.model_date , epoch = self.status.epoch , model_type = self.model_type)
    
    def on_validation_epoch_end(self):
        with self.callbacks: 
            self.metrics.collect_epoch_metric('valid')
            torch.set_grad_enabled(True)
    
    def on_test_model_start(self):
        with self.callbacks:
            self.dataset = 'test'
            if not self.deposition.exists(self.model_path(self.model_date)): self.fit_model()
            self.data_mod.setup('test' , self.model_param , self.model_date)
            self.metrics.new_model(self.model_param , self.config)
            self.test_dates = np.concatenate([self.data_mod.early_test_dates , self.data_mod.model_test_dates])
            self.test_warm_up = len(self.data_mod.early_test_dates)
            self.net.eval()
            torch.set_grad_enabled(False)
    
    def on_test_model_end(self):
        with self.callbacks: torch.set_grad_enabled(True)
    
    def on_test_model_type_start(self):
        with self.callbacks:
            self.load_model(False , self.model_type)
            self.dataloader = self.data_mod.test_dataloader()
            self.assert_equity(len(self.dataloader) , len(self.test_dates))
            self.metrics.new_epoch_metric('test' , model_num = self.model_num , model_date = self.model_date , epoch = self.status.epoch , model_type = self.model_type)
    
    def on_test_model_type_end(self): 
        with self.callbacks: 
            self.metrics.collect_epoch_metric('test')
            pass
    
    def on_train_batch_start(self): 
        with self.callbacks: pass
    
    def on_train_batch_end(self): 
        with self.callbacks: pass
    
    def on_validation_batch_start(self): 
        with self.callbacks: pass
    
    def on_validation_batch_end(self): 
        with self.callbacks: pass
    
    def on_test_batch_start(self): 
        with self.callbacks: pass
    
    def on_test_batch_end(self): 
        with self.callbacks: pass
    
    def on_before_backward(self): 
        with self.callbacks: pass
    
    def on_after_backward(self): 
        with self.callbacks: pass
    
    def on_before_save_model(self): 
        with self.callbacks: pass

    def fit_new_attempt(self):
        self.status.attempt += 1
        self.pipe.new_attempt()
        self.metrics.new_attempt()
        self.checkpoint.new_model(self.model_param , self.model_date)
        self.load_model(True)

    def load_model(self , training : bool , model_type = default_model_type):
        '''load model state dict, return net and a sign of whether it is transferred'''
        if training and self.config.train_param['transfer']:         
            model_path = self.model_path(self.data_mod.prev_model_date(self.model_date))
        elif training:
            model_path = self.model_path()
        else:
            model_path = self.model_path(self.model_date , model_type)
        self.transferred = training and self.config.train_param['transfer'] and self.deposition.exists(model_path)
        self.net = MODEL.new(self.config.model_module , self.model_param , self.deposition.load(model_path) , self.device)
        if training: 
            self.fitmodels = U.FittestModel.get_models(self.config.model_types , self.checkpoint)
            self.optimizer = U.Optimizer(self.net , self.config , self.transferred , self.pipe.attempt)

    def save_model(self):
        '''save model state dict to deposition'''
        self.on_before_save_model()
        self.net = self.net.cpu()
        for model_type , fittest_model in self.fitmodels.items():
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