import math , torch

from copy import deepcopy
from torch import nn , optim
from torch.nn.utils.clip_grad import clip_grad_value_
from typing import Any

from src.proj import Logger
from src.res.algo.nn.optimizer import sam

from .metrics import BatchMetric
from .config import TrainConfig

NAN_GRADS_HALT = False
NAN_GRADS_IGNORE = False

class Optimizer:
    '''specify trainer optimizer and scheduler'''
    # reset_speedup_param_list = ['step_size' , 'warmup_stage' , 'anneal_stage' , 'step_size_up' , 'step_size_down']

    def __init__(
            self , 
            net : nn.Module , 
            config : TrainConfig , 
            transfer : bool = False , 
            lr_multiplier : float = 1. , 
            add_opt_param : dict | None = None , 
            add_lr_param : dict | None = None , 
            add_shd_param : dict | None = None ,
            trainer = None) -> None:
        self.net = net
        self.config = config
        self.trainer = trainer

        self.opt_param : dict[str,Any] = deepcopy(self.config.train_trainer_optimizer)
        self.shd_param : dict[str,Any] = deepcopy(self.config.train_trainer_scheduler)
        self.lr_param  : dict[str,Any] = deepcopy(self.config.train_trainer_learn_rate)
        self.clip_value = self.config.train_trainer_gradient_clip_value
      
        if add_opt_param: 
            self.opt_param.update(add_opt_param)
        if add_shd_param: 
            self.shd_param.update(add_shd_param)
        if add_lr_param:  
            self.lr_param.update(add_lr_param)
        
        self.optimizer = self.load_optimizer(net , self.opt_param , self.lr_param , transfer , lr_multiplier)
        self.scheduler = self.load_scheduler(self.optimizer , self.shd_param)

    @classmethod
    def load_optimizer(cls , net : nn.Module , opt_param : dict , lr_param : dict  , 
                       transfer : bool = False , lr_multiplier : float = 1.):
        assert lr_multiplier > 0 , f'lr_multiplier must be positive, but got {lr_multiplier}'
        return cls.base_optimizer(net , opt_param['name'] , lr_param['base'] * lr_multiplier , transfer = transfer , 
                                  encoder_lr_multiplier = lr_param['transfer_multiplier'].get('encoder',1.) , 
                                  decoder_lr_multiplier = lr_param['transfer_multiplier'].get('decoder',1.) ,
                                  **opt_param['param'])
    
    @classmethod
    def load_scheduler(cls , optimizer , shd_param : dict):
        return cls.base_scheduler(optimizer, shd_param['name'], **shd_param['param'])

    def backward(self , batch_metric : BatchMetric):
        '''BP of optimizer.parameters'''
        self.optimizer.zero_grad()
        batch_metric.loss.backward(retain_graph = NAN_GRADS_HALT)
        self.check_nan_gradients(batch_metric)
        self.clip_gradients()
        self.optimizer.step()

    def scheduler_step(self , epoch : int = 0) -> str | None:
        '''scheduler step on learn rate , reset learn rate to base_lr on conditions'''
        self.scheduler.step()
        
    @property
    def last_lr(self) -> float: return self.scheduler.get_last_lr()[0]    

    @staticmethod
    def base_optimizer(net : nn.Module , key = 'Adam', base_lr = 0.005, transfer = False , 
                       encoder_lr_multiplier = 1., decoder_lr_multiplier = 1., **kwargs) -> torch.optim.Optimizer:
        if transfer:
            # define param list to train with different learn rate
            p_enc = [(p if p.dim()<=1 else nn.init.xavier_uniform_(p)) for x,p in net.named_parameters() if 'encoder' in x.split('.')[:3]]
            p_dec = [p for x,p in net.named_parameters() if 'encoder' not in x.split('.')[:3]]
            param_groups = [{'params': p_dec , 'lr': base_lr * decoder_lr_multiplier , 'lr_param': base_lr * decoder_lr_multiplier},
                            {'params': p_enc , 'lr': base_lr * encoder_lr_multiplier , 'lr_param': base_lr * encoder_lr_multiplier}]
        else:
            param_groups = [{'params': [p for p in net.parameters()] , 'lr' : base_lr , 'lr_param' : base_lr} ]

        return {
            'Adam': optim.Adam ,
            'SGD' : optim.SGD ,
            'SAM' : sam.SAM ,
            'ASAM': sam.ASAM ,  
            'GSAM': sam.GSAM ,
            'GAM' : sam.GAM ,
            'FSAM': sam.FriendlySAM ,
        }[key](param_groups , **kwargs)

    @staticmethod
    def base_scheduler(optimizer, key = 'cycle', **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        if key == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif key == 'cycle':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, max_lr=[pg['lr_param'] for pg in optimizer.param_groups],cycle_momentum=False,mode='triangular2',**kwargs)
        return scheduler
    
    def check_nan_gradients(self , metric : BatchMetric):
        if not NAN_GRADS_HALT: 
            return
        for param in self.net.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                return True
            
        from src import api
        setattr(api , 'mod', self.trainer)
        Logger.stdout('total loss has nan gradients: ' , metric.loss)

        for key , loss in metric.losses.items():
            Logger.stdout(key , loss)
            self.optimizer.zero_grad()
            # if loss.grad_fn is None: continue
            loss.backward(retain_graph = True)
            for name , param in self.net.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    Logger.stdout(name , param , param.grad)

        raise KeyError

    def clip_gradients(self):
        if self.clip_value is not None:
            clip_grad_value_(self.net.parameters(), clip_value = self.clip_value)
        if NAN_GRADS_IGNORE:
            grads = [p.grad for p in self.net.parameters() if p.grad is not None]
            for grad in grads: 
                grad.nan_to_num_(0.)

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
                