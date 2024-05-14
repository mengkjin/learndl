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
from ..model import model as MODEL
from ..util import (
    BigTimer , CallBackManager , Checkpoint , Deposition , Device , EnsembleModels ,
    Logger , Metrics , Optimizer , TrainConfig)

class ModelTrainer(BaseModelModule):
    '''run through the whole process of training'''
    def init_config(self , **kwargs) -> None:
        self.config = TrainConfig.load(do_parser = True , par_args = kwargs)
        self.stage_queue = self.config.stage_queue
    def init_utilities(self , **kwargs) -> None: 
        self.logger     = Logger()
        self.device     = Device()
        self.checkpoint = Checkpoint(self.config)
        self.deposition = Deposition(self.config.model_base_path)
        self.metrics    = Metrics(self.config)
        self.callbacks  = CallBackManager.setup(self.config , self)
    def init_data(self , **kwargs): 
        self.data_mod = DataModule(self.config)
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
    def batch_dates(self): return np.concatenate([self.data_mod.early_test_dates , self.data_mod.model_test_dates])
    @property
    def batch_warm_up(self): return len(self.data_mod.early_test_dates)
    @property
    def if_transfer(self) -> bool: return self.config.train_param['transfer']
    @property
    def prev_model_date(self): return self.data_mod.prev_model_date(self.model_date)
    @property
    def model_date(self): return self.status.model_date
    @property
    def model_num(self): return self.status.model_num
    @property
    def model_type(self): return self.status.model_type
    @property
    def model_iter(self): return self.deposition.model_iter(
        self.data_mod.model_date_list , self.config.model_num_list , self.status.stage , self.config.resume_training)

    def on_configure_model(self): 
        self.config.set_config_environment()

    def on_fit_model_start(self):
        self.data_mod.setup('fit' , self.model_param , self.model_date)
        self.metrics.new_model(self.model_param , self.config)
        self.checkpoint.new_model(self.model_param , self.model_date)
        self.load_model(True)
    
    def on_fit_model_end(self):
        self.save_model()
        self.checkpoint.del_all()

    def on_test_batch(self):
        self.assert_equity(self.batch_dates[self.batch_idx] , self.data_mod.y_date[self.batch_data.i[0,1]]) 
        self.batch_forward()
        if self.batch_idx < self.batch_warm_up: return  # before this is warmup stage , only forward
        # if self.status.dataset == 'test' and self.config.lgbm_ensembler:
        self.batch_metrics()
    
    def on_fit_epoch_end(self):
        self.ensembles.assess(self.net , self.status.epoch , self.metrics.valid_scores[-1] , self.metrics.valid_losses[-1])
    
    def on_train_epoch_start(self):
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data_mod.train_dataloader()
        self.metrics.new_epoch_metric('train' , self.status)
    
    def on_train_epoch_end(self): 
        self.metrics.collect_epoch_metric('train')
        self.optimizer.scheduler_step(self.status.epoch)
    
    def on_validation_epoch_start(self):
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data_mod.val_dataloader()
        self.metrics.new_epoch_metric('valid' , self.status)
    
    def on_validation_epoch_end(self):
        self.metrics.collect_epoch_metric('valid')
        torch.set_grad_enabled(True)
    
    def on_test_model_start(self):
        if not self.deposition.exists(self.model_date , self.model_num , self.model_type): self.fit_model()

        self.data_mod.setup('test' , self.model_param , self.model_date)
        self.metrics.new_model(self.model_param , self.config)
        self.net.eval()
        torch.set_grad_enabled(False)
    
    def on_test_model_type_start(self):
        self.load_model(False , self.model_type)
        self.dataloader = self.data_mod.test_dataloader()
        self.assert_equity(len(self.dataloader) , len(self.batch_dates))
        self.metrics.new_epoch_metric('test' , self.status)
    
    def on_test_model_type_end(self): 
        self.metrics.collect_epoch_metric('test')
    
    def load_model(self , training : bool , model_type = 'best' , lr_multiplier = 1.):
        '''load model state dict, return net and a sign of whether it is transferred'''
        model_date = (self.prev_model_date if self.if_transfer else 0) if training else self.model_date
        model_dict = self.deposition.load_model(model_date , self.model_num , model_type)

        self.transferred = training and self.if_transfer and model_dict.exists()
        self.net = MODEL.new(self.config.model_module , self.model_param , model_dict.state_dict() , self.device)
        if training:
            self.ensembles = EnsembleModels(self.net , self.config , self.data_mod , self.checkpoint , device=self.device)
            self.optimizer = Optimizer(self.net , self.config , self.transferred , lr_multiplier)
        else:
            assert model_dict.exists() , str(model_dict)

    def save_model(self):
        '''save model state dict to deposition'''
        self.on_before_save_model()
        self.net = self.net.cpu()
        for model_type in self.model_types:
            model_dict = self.ensembles.collect(model_type)
            self.deposition.save_model(self.model_date , self.model_num , model_type , model_dict) 

    @classmethod
    def main(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        app = cls(stage = stage , resume = resume , checkname = checkname , **kwargs)
        with BigTimer(app.logger.critical , 'Main Process'):
            app.main_process()