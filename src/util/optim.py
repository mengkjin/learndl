import math
import numpy as np
import torch

from dataclasses import dataclass
from torch import nn , optim , Tensor
from torch.utils.data import Sampler
from typing import Any

def new_optimizer(net : nn.Module , key = 'Adam', base_lr = 0.005, transfer = False , encoder_lr_ratio = 1., decoder_lr_ratio = 1., **kwargs):
    if transfer:
        # define param list to train with different learn rate
        p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in net.named_parameters() if 'encoder' in x.split('.')[:3]]
        p_dec = [p for x,p in net.named_parameters() if 'encoder' not in x.split('.')[:3]]
        net_param_groups = [{'params': p_dec , 'lr': base_lr * decoder_lr_ratio , 'lr_param': base_lr * decoder_lr_ratio},
                            {'params': p_enc , 'lr': base_lr * encoder_lr_ratio , 'lr_param': base_lr * encoder_lr_ratio}]
    else:
        net_param_groups = [{'params': [p for p in net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

    optimizer = {
        'Adam': optim.Adam ,
        'SGD' : optim.SGD ,
    }[key](net_param_groups , **kwargs)
    return optimizer

def new_scheduler(optimizer, key = 'cycle', **kwargs):
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
                