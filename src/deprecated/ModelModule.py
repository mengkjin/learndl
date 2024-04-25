#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : ${2023-6-27} ${21:05}
# @Author : Mathew Jin
# @File : ${run_model.py}
# chmod +x run_model.py
# python3 scripts/run_model3.py --stage=0 --resume=0 --checkname=1 
import itertools , os
import numpy as np
import torch

from dataclasses import dataclass
from inspect import currentframe
from typing import ClassVar , Literal , Optional

from .environ import DIR
from . import util as U
from .util import callback as CB
from .module.DataModule import DataModule
from .model import model as MODEL

logger = U.Logger()
config = U.TrainConfig()
ptimer = U.time.PTimer(True)
hooker = CB.ModelHook(ptimer)

class ModelTrainer():
    '''run through the whole process of training'''
    default_model_type = 'best'

    def __init__(self ,  **kwargs):
        self.config     = config
        self.ptimer     = ptimer
        self.logger     = logger
        config.reload(do_parser = True , par_args = config.parser_args(**kwargs))
        
        self.device     = U.Device()
        self.data_mod   = DataModule(config)

        self.pipe       = U.Pipeline(config , logger)
        self.checkpoint = U.Checkpoint('mem' if config.mem_storage else 'disk')
        self.deposition = U.Deposition()
        self.metrics    = U.Metrics(config.train_param['criterion'])

        self.callbacks  = [self.pipe , self.checkpoint , self.deposition , self.metrics ,
                           CB.DynamicDataLink()]

    def main_process(self):
        '''Main stage of data & fit & test'''
        self.configure_model()
        for self.stage in config.stage_queue: 
            getattr(self , f'stage_{self.stage}')()
        ptimer.summarize()

    @hooker.hook
    def configure_model(self):
        pass
    
    def stage_data(self):
        '''stage of loading model data'''
        self.on_data_start()
        self.data_mod.load_data()
        self.on_data_end()
        
    def stage_fit(self):
        '''stage of fitting'''
        self.on_fit_start()
        for self.model_date , self.model_num in self.model_iter():
            self.fit_model()
        self.on_fit_end()

    def stage_test(self):
        '''stage of testing'''
        self.on_test_start()
        for self.model_date , self.model_num in self.model_iter():
            self.test_model()
        self.on_test_end()

    def model_iter(self):
        '''iter of model_date and model_num , considering resume_training'''
        new_iter = list(itertools.product(self.data_mod.model_date_list , config.model_num_list))
        if config.resume_training and self.stage == 'fit':
            models_trained = np.full(len(new_iter) , True , dtype = bool)
            for i , (model_date , model_num) in enumerate(new_iter):
                if not os.path.exists(self.model_path(model_date,model_num = model_num)):
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
        '''to do when starting fit of one model'''
        self.param : dict = config.model_param[self.model_num]
        self.data_mod.setup('fit' , self.config.model_param[self.model_num] , self.model_date)
        pass

    @hooker.hook
    def fit_model(self):
        self.on_fit_model_start()
        while self.fit_loop:   
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
            
            if self.pipe.loop_new_attempt:  self.print_progress('new_attempt')
            if self.pipe.loop_terminate: self.save_model()
        self.on_fit_model_end()

    @hooker.hook
    def test_model(self):
        self.on_test_model_start()
        for self.model_type in config.model_types:
            self.on_test_model_type_start()
            for self.batch_idx , self.batch_data in enumerate(self.dataloader):
                self.on_test_batch_start()
                self.on_test_batch()
                self.on_test_batch_end()
            self.on_test_model_type_end()
        self.on_test_model_end()

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
   
    def print_progress(self , key : Optional[str] = None , sdout = None):
        printers = [logger.info] if (config.verbosity > 2 or self.pipe.initial_models) else [logger.debug]
        if key == 'epoch_step':
            if self.pipe.epoch_print: sdout = '{attempt} {epoch} : {status}'.format(**self.pipe.texts)
        elif key == 'reset_learn_rate':
            sdout = f'Reset learn rate and scheduler at the end of epoch {self.pipe.epoch} ,' + \
                f' effective at epoch {self.pipe.epoch+1}' + \
                ', and will speedup2x' * config.train_param['trainer']['learn_rate']['reset']['speedup2x']
        if sdout is not None: [prt(sdout) for prt in printers]

    @hooker.hook 
    def on_train_epoch_start(self):
        self.dataset = 'train'
        if self.pipe.loop_new_attempt: 
            self.pipe.new_attempt()
            self.checkpoint.new_model(self.param , self.model_date)
            self.models = U.FittestModel.get_models(config.model_types , self.checkpoint)
            self.load_model(True)
            self.optimizer = U.Optimizer(self.net , config , self.transferred , self.pipe.attempt)
        self.pipe.new_epoch()
        self.pipe.new_metric(self.dataset , 'best')
        self.net.train()
        torch.set_grad_enabled(True)
        self.dataloader = self.data_mod.train_dataloader().init_tqdm('Train Ep#{ep:3d} loss : {ls:.5f}')

    @hooker.hook 
    def on_train_epoch_end(self):
        self.pipe.collect_metric()
        self.pipe.collect_lr(self.optimizer.last_lr)
        self.print_progress(self.optimizer.scheduler_step(self.pipe.epoch))

    @hooker.hook 
    def on_validation_epoch_start(self):
        self.dataset = 'valid'
        self.pipe.new_metric(self.dataset , 'best')
        self.net.eval()
        torch.set_grad_enabled(False)
        self.dataloader = self.data_mod.val_dataloader().init_tqdm('Valid Ep#{ep:3d} score : {sc:.5f}')
        
    @hooker.hook 
    def on_validation_epoch_end(self):
        self.pipe.collect_metric()
        self.pipe.assess_terminate()
        
        for fittest_model in self.models.values():
            fittest_model.assess(self.net , self.pipe.epoch , self.pipe.valid_score , self.pipe.valid_loss)
        self.print_progress('epoch_step')
        torch.set_grad_enabled(True)

    @hooker.hook 
    def on_test_model_start(self) -> None:
        if not self.deposition.exists(self.model_path(self.model_date)): 
            self.fit_model()
        
        self.param = config.model_param[self.model_num]
        self.data_mod.setup('test' , self.param , self.model_date)
        self.pipe.new_test_model(self.model_num , self.model_date , self.data_mod.model_test_dates)
        self.metrics.new_model(self.param , config)

        self.dataset = 'test'
        self.test_dates = np.concatenate([self.data_mod.early_test_dates , self.data_mod.model_test_dates])
        self.test_warm_up = len(self.data_mod.early_test_dates)
        self.net.eval()
        torch.set_grad_enabled(False)

    @hooker.hook 
    def on_test_model_end(self) -> None:
        torch.set_grad_enabled(True)
        self.pipe.end_test_model()

    @hooker.hook 
    def on_test_model_type_start(self) -> None:
        self.load_model(False , self.model_type)
        self.pipe.new_metric(self.dataset , self.model_type)
        self.dataloader = self.data_mod.test_dataloader().init_tqdm('Test {mt} {dt} score : {sc:.5f}')
        self.assert_equity(len(self.dataloader) , len(self.test_dates))

    @hooker.hook 
    def on_test_model_type_end(self):
        self.pipe.collect_metric()
        self.pipe.update_test_score(self.model_type)

    @hooker.hook 
    def on_train_batch_start(self): pass
    
    @hooker.hook 
    def on_train_batch_end(self):
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(ep=self.pipe.epoch, ls=self.pipe.aggloss)
    
    @hooker.hook
    def on_validation_batch_start(self): pass
    @hooker.hook
    def on_validation_batch_end(self):
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(ep=self.pipe.epoch, sc=self.pipe.aggscore)
    
    @hooker.hook
    def on_test_batch_start(self): pass
    @hooker.hook
    def on_test_batch_end(self) -> None:
        #if (self.batch_idx + 1) % 20 == 0 : torch.cuda.empty_cache()
        self.dataloader.display(dt=self.test_dates[self.batch_idx] , mt=self.model_type , sc=self.pipe.aggscore)

    def batch_forward(self) -> None:
        if self.batch_data.is_empty:
            self.net_output = U.trainer.BatchOutput.empty()
        else:
            self.net_output = U.trainer.BatchOutput(self.net(self.batch_data.x))
    
    def batch_metrics(self) -> None:
        self.metrics.calculate(self.dataset , self.batch_data.y , self.net_output.pred , 
                               self.batch_data.w , self.net , self.penalty_kwargs , assert_nan = True)
        self.pipe.record_metric(self.metrics)
    
    @hooker.hook 
    def on_before_backward(self): pass
    @hooker.hook 
    def on_after_backward(self): pass

    def batch_backward(self) -> None:
        assert self.dataset == 'train' , self.dataset
        self.on_before_backward()
        self.optimizer.backward(self.metrics.loss)
        self.on_after_backward()

    @hooker.hook
    def load_model(self , training : bool , model_type = default_model_type) -> None:
        '''load model state dict, return net and a sign of whether it is transferred'''
        if training and config.train_param['transfer']:         
            model_path = self.model_path(self.data_mod.prev_model_date(self.model_date))
        elif training:
            model_path = self.model_path()
        else:
            model_path = self.model_path(self.model_date , model_type)
        self.transferred = training and config.train_param['transfer'] and self.deposition.exists(model_path)
        self.net = MODEL.new(config.model_module , self.param , self.deposition.load(model_path) , self.device)

    @hooker.hook 
    def on_before_save_model(self): pass
     
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
            model_dir = self.param.get('path')
        elif base_path is None:
            model_dir = f'{config.model_base_path}/{model_num}'
        else:
            model_dir = f'{base_path}/{model_num}'
        return '{}/{}.{}.pt'.format(model_dir , model_date , model_type)

    @property
    def fit_loop(self): return self.pipe.loop_continue

    @property
    def penalty_kwargs(self): return {'net':self.net,'hidden':self.net_output.hidden,'label':self.batch_data.y}
    
    @staticmethod
    def assert_equity(a , b): assert a == b , (a , b)
    
    @staticmethod
    def hook_name(): return getattr(currentframe(),'f_back').f_code.co_name

    @classmethod
    def fit(cls , stage = -1 , resume = -1 , checkname = -1 , **kwargs):
        app = cls(stage = stage , resume = resume , checkname = checkname , **kwargs)
        app.main_process()

if __name__ == '__main__': 
    ModelTrainer.fit(**config.parser_args().__dict__)
 