import numpy as np

from abc import ABC , abstractmethod
from copy import deepcopy
from torch import nn
from torch.optim.swa_utils import AveragedModel , update_bn

from ..metric import Metrics
from ..store import Checkpoint
from ...classes import BaseDataModule

def choose_ensembler(model_type):
    '''get a subclass of _BaseEnsembler'''
    if model_type == 'best': return EnsembleBest
    elif model_type == 'swabest': return EnsembleSWABest
    elif model_type == 'swalast': return EnsembleSWALast
    else: raise KeyError(model_type)

class Ensembler(ABC):
    '''abstract class of fittest model, e.g. model with the best score, swa model of best scores or last ones'''
    def __init__(self, ckpt : Checkpoint , use_score = True , **kwargs) -> None:
        self.ckpt , self.use_score = ckpt , use_score
        self.reset()
    def __bool__(self): return True
    @abstractmethod
    def reset(self): ...
    @abstractmethod
    def assess(self , net , epoch : int , score = 0. , loss = 0.): '''score or loss to update assessment'''
    @abstractmethod
    def collect(self , net , *args , device = None , **kwargs) -> nn.Module: '''output the final fittest model state dict'''

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

class EnsembleBest(Ensembler):
    '''state dict of epoch with best score or least loss'''
    def __init__(self, ckpt : Checkpoint , use_score = True , **kwargs) -> None:
        super().__init__(ckpt , use_score)

    def reset(self):
        self.epoch_fix  = -1
        self.metric_fix = None

    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        # value = score if self.use_score else loss
        if metrics.better_epoch(self.metric_fix):
        #if self.metric_fix is None or (self.metric_fix < value if self.use_score else self.metric_fix > value):
            self.ckpt.disjoin(self , self.epoch_fix)
            self.epoch_fix = epoch
            self.metric_fix = metrics.last_metric # value
            self.ckpt.join(self , epoch , net)

    def collect(self , net : nn.Module , data : BaseDataModule , *args , device = None , **kwargs):
        #return self.ckpt.load_epoch(self.epoch_fix)
        net = deepcopy(net)
        net.load_state_dict(self.ckpt.load_epoch(self.epoch_fix))
        return net

class EnsembleSWABest(Ensembler):
    '''state dict of n_best epochs with best score or least loss'''
    def __init__(self, ckpt : Checkpoint , use_score = True , n_best = 5 , **kwargs) -> None:
        super().__init__(ckpt , use_score)
        assert n_best > 0, n_best
        self.n_best      = n_best

    def reset(self):
        self.metric_list = []
        self.candidates  = []
        
    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        # value = score if self.use_score else loss
        if len(self.metric_list) == self.n_best :
            arg = np.argmin(self.metric_list) if metrics.use_metric == 'score' else np.argmax(self.metric_list)
            # arg = np.argmin(self.metric_list) if self.use_score else np.argmax(self.metric_list)
            #if (self.metric_list[arg] < value if self.use_score else self.metric_list[arg] > value):
            if metrics.better_epoch(self.metric_list[arg]):
                self.metric_list.pop(arg)
                candid = self.candidates.pop(arg)
                self.ckpt.disjoin(self , candid)

        if len(self.metric_list) < self.n_best:
            # self.metric_list.append(value)
            self.metric_list.append(metrics.last_metric)
            self.candidates.append(epoch)
            self.ckpt.join(self , epoch , net)

    def collect(self , net : nn.Module , data : BaseDataModule , *args , device = None , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: swa.update_sd(self.ckpt.load_epoch(epoch))
        loader = data.train_dataloader()
        swa.update_bn(loader , getattr(loader , 'device' , device))
        return swa.module.cpu()
    
class EnsembleSWALast(Ensembler):
    '''state dict of n_last epochs around best score or least loss'''
    def __init__(self, ckpt : Checkpoint , use_score = True , n_last = 5 , interval = 3 , **kwargs) -> None:
        super().__init__(ckpt , use_score)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      = n_last
        self.interval    = interval
        self.left_epochs = (n_last // 2) * interval

    def reset(self):
        self.epoch_fix   = -1
        self.metric_fix  = None
        self.candidates  = []

    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        # value = score if self.use_score else loss
        if metrics.better_epoch(self.metric_fix):
        #if self.metric_fix is None or (self.metric_fix < value if self.use_score else self.metric_fix > value):
            self.epoch_fix = epoch
            self.metric_fix = metrics.last_metric # value
            # self.epoch_fix , self.metric_fix = epoch , value
        candidates = self._full_candidates(epoch)
        [self.ckpt.disjoin(self , candid) for candid in self.candidates if candid < min(candidates)]
        if epoch in candidates: self.ckpt.join(self , epoch , net)
        self.candidates = candidates[:self.n_last]

    def _full_candidates(self , epoch):
        epochs  = np.arange(self.interval , epoch + 1 , self.interval)
        left    = epochs[epochs < self.epoch_fix]
        right   = epochs[epochs > self.epoch_fix]
        return [*left[-((self.n_last - 1) // 2):] , self.epoch_fix , *right]

    def collect(self , net : nn.Module , data : BaseDataModule , *args , device = None , **kwargs):
        swa = SWAModel(net)
        for epoch in self.candidates: 
            swa.update_sd(self.ckpt.load_epoch(epoch))
        loader = data.train_dataloader()
        swa.update_bn(loader , getattr(loader , 'device' , device))
        return swa.module.cpu()
