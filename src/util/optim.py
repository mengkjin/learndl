import math
import torch

from copy import deepcopy
from torch import nn , optim , Tensor
from torch.nn.utils.clip_grad import clip_grad_value_
from typing import Any , Optional

from .config import TrainConfig

class Optimizer:
    reset_speedup_param_list = ['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']

    def __init__(self , net : nn.Module , config : TrainConfig , transfer : bool = False , attempt : int = 0 , 
                 add_opt_param : Optional[dict] = None , 
                 add_lr_param : Optional[dict] = None , 
                 add_shd_param : Optional[dict] = None) -> None:
        self.net = net
        self.config = config
        self.opt_param = deepcopy(config.train_param['trainer']['optimizer'])
        self.lr_param  = deepcopy(config.train_param['trainer']['learn_rate'])
        self.shd_param = deepcopy(config.train_param['trainer']['scheduler'])
        if add_opt_param: self.opt_param.update(add_opt_param)
        if add_lr_param:  self.lr_param.update(add_lr_param)
        if add_shd_param: self.shd_param.update(add_shd_param)

        self.optimizer = self.load_optimizer(net , self.opt_param , self.lr_param , transfer , attempt)
        self.scheduler = self.load_scheduler(self.optimizer , self.shd_param)
        self.clip_value = config.clip_value

    @classmethod
    def load_optimizer(cls , net : nn.Module , opt_param : dict , lr_param : dict  , 
                       transfer : bool = False , attempt : int = 0):
        base_lr = lr_param['base'] * lr_param['ratio']['attempt'][:attempt+1][-1]
        return cls.base_optimizer(net , opt_param['name'] , base_lr , transfer = transfer , 
                                  encoder_lr_ratio = lr_param['ratio']['transfer'], **opt_param['param'])
    
    @classmethod
    def load_scheduler(cls , optimizer , shd_param : dict):
        return cls.base_scheduler(optimizer, shd_param['name'], **shd_param['param'])
    
    def backward(self , loss : Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_value is not None : clip_grad_value_(self.net.parameters(), clip_value = self.clip_value) 
        self.optimizer.step()

    def step(self , epoch : int) -> str | None:
        self.scheduler.step()
        reset_param = self.lr_param.get('reset')
        if not reset_param: return
        if reset_param['num_reset'] <= 0 or (epoch + 1) < reset_param['trigger']: return
        
        trigger_intvl = reset_param['trigger'] // 2 if reset_param['speedup2x'] else reset_param['trigger']
        if (epoch + 1 - reset_param['trigger']) % trigger_intvl != 0: return
        
        trigger_times = ((epoch + 1 - reset_param['trigger']) // trigger_intvl) + 1
        if trigger_times > reset_param['num_reset']: return
        
        # confirm reset : change back optimizor learn rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr_param']  * reset_param['recover_level']
        
        # confirm reset : reassign scheduler
        shd_param = deepcopy(self.shd_param)
        if reset_param['speedup2x']:
            for key in shd_param['param'].keys():
                if key in self.reset_speedup_param_list: shd_param['param'][key] //= 2

        self.scheduler = self.load_scheduler(self.optimizer , shd_param)
        return 'reset_learn_rate'

    @property
    def last_lr(self) -> float: return self.scheduler.get_last_lr()[0]    

    @staticmethod
    def base_optimizer(net : nn.Module , key = 'Adam', base_lr = 0.005, transfer = False , 
                       encoder_lr_ratio = 1., decoder_lr_ratio = 1., **kwargs) -> torch.optim.Optimizer:
        if transfer:
            # define param list to train with different learn rate
            p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in net.named_parameters() if 'encoder' in x.split('.')[:3]]
            p_dec = [p for x,p in net.named_parameters() if 'encoder' not in x.split('.')[:3]]
            param_groups = [{'params': p_dec , 'lr': base_lr * decoder_lr_ratio , 'lr_param': base_lr * decoder_lr_ratio},
                            {'params': p_enc , 'lr': base_lr * encoder_lr_ratio , 'lr_param': base_lr * encoder_lr_ratio}]
        else:
            param_groups = [{'params': [p for p in net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

        return {
            'Adam': optim.Adam ,
            'SGD' : optim.SGD ,
        }[key](param_groups , **kwargs)

    @staticmethod
    def base_scheduler(optimizer, key = 'cycle', **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        if key == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif key == 'cycle':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, max_lr=[pg['lr_param'] for pg in optimizer.param_groups],cycle_momentum=False,mode='triangular2',**kwargs)
        return scheduler

class CosineScheduler:
    def __init__(self , optimizer , warmup_stage = 10 , anneal_stage = 40 , initial_lr_div = 10 , final_lr_div = 1e4):
        self.warmup_stage= warmup_stage
        self.anneal_stage= anneal_stage
        self.optimizer = optimizer
        self.base_lrs = [x['lr'] for x in optimizer.param_groups]
        self.initial_lr= [x / initial_lr_div for x in self.base_lrs]
        self.final_lr= [x / final_lr_div for x in self.base_lrs]
        self.last_epoch = 0
        self._step_count= 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
        self._last_lr= self.initial_lr
        
    def get_last_lr(self):
        #Return last computed learning rate by current scheduler.
        return self._last_lr

    def state_dict(self):
        #Returns the state of the scheduler as a dict.
        return self.__dict__
    
    def step(self):
        self.last_epoch += 1
        if self._step_count <= self.warmup_stage:
            self._last_lr = [y+(x-y)*self._linear_phase for x,y in zip(self.base_lrs,self.initial_lr)]
        elif self._step_count <= self.warmup_stage + self.anneal_stage:
            self._last_lr = [y+(x-y)*math.cos(self._cos_phase) for x,y in zip(self.base_lrs,self.final_lr)]
        else:
            self._last_lr = self.final_lr
        for x , param_group in zip(self._last_lr,self.optimizer.param_groups):
            param_group['lr'] = x
        self._step_count += 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
                