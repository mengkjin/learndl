import numpy as np

from abc import ABC , abstractmethod
from copy import deepcopy
from torch import nn , no_grad
from torch.optim.swa_utils import AveragedModel
from typing import Any , Literal

from src.res.model.util import BaseTrainer , BatchData , Checkpoint , Metrics

def choose_swa_method(submodel : Literal['best' , 'swabest' , 'swalast'] | Any):
    '''get a subclass of _BaseEnsembler'''
    if submodel == 'best': 
        return EnsembleBestOne
    elif submodel == 'swabest': 
        return EnsembleSWABest
    elif submodel == 'swalast': 
        return EnsembleSWALast
    else: 
        raise KeyError(submodel)

class SWAEnsembler(ABC):
    '''abstract class of fittest model, e.g. model with the best score, swa model of best scores or last ones'''
    def __init__(self, ckpt : Checkpoint ,  *args , **kwargs) -> None:
        self.ckpt = ckpt
        self.reset()
    def __bool__(self): return True
    @abstractmethod
    def reset(self): ...
    @abstractmethod
    def assess(self , module , epoch : int , score = 0. , loss = 0.): '''score or loss to update assessment'''
    @abstractmethod
    def collect(self , module , *args , **kwargs) -> nn.Module: '''output the final fittest model state dict'''

class SWAModel:
    def __init__(self , module : nn.Module) -> None:
        self.template = deepcopy(module)
        self.avgmodel = AveragedModel(self.template)

    def update_sd(self , state_dict):
        self.template.load_state_dict(state_dict)
        self.avgmodel.update_parameters(self.template) 
        return self
    
    def update_bn(self , trainer : BaseTrainer):
        device = trainer.device
        self.avgmodel = device(self.avgmodel) if callable(device) else self.avgmodel.to(device)
        update_swa_bn(self.bn_loader(trainer) , self.avgmodel) 
        return self
     
    def bn_loader(self , trainer : BaseTrainer):
        for batch_data in trainer.data.train_dataloader(): 
            assert isinstance(batch_data, BatchData) , f'{type(batch_data)} is not a BatchData'
            trainer.on_train_batch_start()
            yield (batch_data.x , batch_data.kwargs)

    @property
    def module(self) -> nn.Module: return self.avgmodel.module

@no_grad()
def update_swa_bn(loader , model : AveragedModel):
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
    if not momenta: 
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():  
        module.momentum = None
    for x , kwargs in loader: 
        model(x , **kwargs)
    for bn_module in momenta.keys(): 
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

class EnsembleBestOne(SWAEnsembler):
    '''state dict of epoch with best score or least loss'''
    def __init__(self, ckpt : Checkpoint , *args , **kwargs) -> None:
        super().__init__(ckpt , *args , **kwargs)

    def reset(self):
        self.epoch_fix  = -1
        self.metric_fix = None

    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        if metrics.better_epoch(self.metric_fix):
            self.ckpt.disjoin(self , self.epoch_fix)
            self.epoch_fix = epoch
            self.metric_fix = metrics.last_metric # value
            self.ckpt.join(self , epoch , net)

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        #return self.ckpt.load_epoch(self.epoch_fix)
        net : nn.Module = deepcopy(getattr(trainer.model , 'net'))
        net.load_state_dict(self.ckpt.load_epoch(self.epoch_fix))
        return net

class EnsembleSWABest(SWAEnsembler):
    '''state dict of n_best epochs with best score or least loss'''
    def __init__(self, ckpt : Checkpoint , n_best = 5 ,  *args , **kwargs) -> None:
        super().__init__(ckpt ,  *args , **kwargs)
        assert n_best > 0, n_best
        self.n_best      = n_best

    def reset(self):
        self.metric_list = []
        self.candidates  = []
        
    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        if len(self.metric_list) == self.n_best :
            arg = np.argmin(self.metric_list) if metrics.VAL_METRIC == 'score' else np.argmax(self.metric_list)
            if metrics.better_epoch(self.metric_list[arg]):
                self.metric_list.pop(arg)
                candid = self.candidates.pop(arg)
                self.ckpt.disjoin(self , candid)

        if len(self.metric_list) < self.n_best:
            # self.metric_list.append(value)
            self.metric_list.append(metrics.last_metric)
            self.candidates.append(epoch)
            self.ckpt.join(self , epoch , net)

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        swa = SWAModel(getattr(trainer.model , 'net'))
        for epoch in self.candidates: 
            swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(trainer)
        return swa.module.cpu()
    
class EnsembleSWALast(SWAEnsembler):
    '''state dict of n_last epochs around best score or least loss'''
    def __init__(self, ckpt : Checkpoint , n_last = 5 , interval = 3 ,  *args , **kwargs) -> None:
        super().__init__(ckpt , *args , **kwargs)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      = n_last
        self.interval    = interval
        self.left_epochs = (n_last // 2) * interval

    def reset(self):
        self.epoch_fix   = -1
        self.metric_fix  = None
        self.candidates  = []

    def assess(self , net , epoch : int , metrics : Metrics , score = 0. , loss = 0.):
        if metrics.better_epoch(self.metric_fix):
            self.epoch_fix = epoch
            self.metric_fix = metrics.last_metric 

        candidates = self._full_candidates(epoch)
        [self.ckpt.disjoin(self , candid) for candid in self.candidates if candid < min(candidates)]
        if epoch in candidates: 
            self.ckpt.join(self , epoch , net)
        self.candidates = candidates[:self.n_last]

    def _full_candidates(self , epoch):
        epochs  = np.arange(self.interval , epoch + 1 , self.interval)
        left    = epochs[epochs < self.epoch_fix]
        right   = epochs[epochs > self.epoch_fix]
        return [*left[-((self.n_last - 1) // 2):] , self.epoch_fix , *right]

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        swa = SWAModel(getattr(trainer.model , 'net'))
        for epoch in self.candidates:  
            swa.update_sd(self.ckpt.load_epoch(epoch))
        swa.update_bn(trainer)
        return swa.module.cpu()