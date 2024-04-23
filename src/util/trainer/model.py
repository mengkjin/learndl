import numpy as np

from abc import abstractmethod
from copy import deepcopy
from torch import nn
from torch.optim.swa_utils import AveragedModel , update_bn
from typing import Literal 

from .ckpt import Checkpoint

class FittestModel:
    '''abstract class of fittest model, e.g. model with the best score, swa model of best scores or last ones'''
    def __init__(self, ckpt : Checkpoint , use : Literal['loss','score'] = 'score') -> None:
        self.ckpt , self.use = ckpt , use

    @abstractmethod
    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        '''use score or loss to update assessment'''
        pass
    @abstractmethod
    def state_dict(self , *args , device = None) -> nn.Module | dict: 
        '''output the final fittest model state dict'''
        pass
    @classmethod
    def get_models(cls , model_types , *args , **kwargs):
        '''get a dict of FittestModel'''
        return {model_type:cls.choose(model_type)(*args , **kwargs) for model_type in model_types}
    @staticmethod
    def choose(model_type):
        '''get a subclass of FittestModel'''
        if model_type == 'best': return BestModel
        elif model_type == 'swabest': return SWABest
        elif model_type == 'swalast': return SWALast
        else: raise KeyError(model_type)

class SWAModel:
    def __init__(self , module : nn.Module) -> None:
        self.template = deepcopy(module)
        self.avgmodel = AveragedModel(self.template)

    def update_sd(self , state_dict):
        self.template.load_state_dict(state_dict)
        self.avgmodel.update_parameters(self.template) 
        return self
    
    def update_bn(self , data_loader , device = None):
        self.avgmodel = device(self.avgmodel) if callable(device) else self.avgmodel.to(device)
        update_bn(self.bn_loader(data_loader) , self.avgmodel) 
        return self
     
    def bn_loader(self , data_loader):
        for batch_data in data_loader: 
            yield (batch_data.x , batch_data.y , batch_data.w)

    @property
    def module(self) -> nn.Module: return self.avgmodel.module

class BestModel(FittestModel):
    def __init__(self, ckpt : Checkpoint , use : Literal['loss','score'] = 'score') -> None:
        super().__init__(ckpt , use)
        self.epoch_fix  = -1
        self.metric_fix = None

    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        if self.metric_fix is None or (self.metric_fix < value if self.use == 'score' else self.metric_fix > value):
            self.ckpt.disjoin(self , self.epoch_fix)
            self.epoch_fix = epoch
            self.metric_fix = value
            self.ckpt.join(self , epoch , net)

    def state_dict(self , *args , device = None , **kwargs):
        return self.ckpt.load_epoch(self.epoch_fix)

class SWABest(FittestModel):
    def __init__(self, ckpt : Checkpoint , n_best = 5 , use : Literal['loss','score'] = 'score') -> None:
        super().__init__(ckpt , use)
        assert n_best > 0, n_best
        self.n_best      = n_best
        self.metric_list = []
        self.candidates  = []
        
    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        if len(self.metric_list) == self.n_best :
            arg = np.argmin(self.metric_list) if self.use == 'score' else np.argmax(self.metric_list)
            if (self.metric_list[arg] < value if self.use == 'score' else self.metric_list[arg] > value):
                self.metric_list.pop(arg)
                candid = self.candidates.pop(arg)
                self.ckpt.disjoin(self , candid)

        if len(self.metric_list) < self.n_best:
            self.metric_list.append(value)
            self.candidates.append(epoch)
            self.ckpt.join(self , epoch , net)

    def state_dict(self , net , data_loader , *args , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(data_loader , getattr(data_loader , 'device' , None))
        return swa.module.cpu().state_dict()
    
class SWALast(FittestModel):
    def __init__(self, ckpt : Checkpoint , n_last = 5 , interval = 3 ,
                 use : Literal['loss','score'] = 'score') -> None:
        super().__init__(ckpt , use)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      = n_last
        self.interval    = interval
        self.left_epochs = (n_last // 2) * interval
        self.epoch_fix   = -1
        self.metric_fix  = None
        self.candidates  = []

    def assess(self , net : nn.Module , epoch : int , score = 0. , loss = 0.):
        value = loss if self.use == 'loss' else score
        if self.metric_fix is None or (self.metric_fix < value if self.use == 'score' else self.metric_fix > value):
            self.epoch_fix , self.metric_fix = epoch , value
        candidates = self._new_candidates(epoch)
        [self.ckpt.disjoin(self , candid) for candid in self.candidates if candid < min(candidates)]
        if epoch in candidates: self.ckpt.join(self , epoch , net)
        self.candidates = candidates

    def _new_candidates(self , epoch):
        epochs  = np.arange(self.interval , epoch + 1 , self.interval)
        left    = epochs[epochs < self.epoch_fix]
        right   = epochs[epochs > self.epoch_fix]
        return [*left[-((self.n_last - 1) // 2):] , self.epoch_fix , *right][:self.n_last]

    def state_dict(self , net , data_loader , *args , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: 
            swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(data_loader , getattr(data_loader , 'device' , None))
        return swa.module.cpu().state_dict()