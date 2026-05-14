from __future__ import annotations
import numpy as np

from abc import ABC , abstractmethod
from copy import deepcopy
from torch import nn , no_grad
from torch.optim.swa_utils import AveragedModel
from typing import Any , Literal

from src.res.model.util import BaseTrainer , BatchInput , Checkpoint , TrainerMetrics , EpochMetricResult , TrainerStatus

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
    '''abstract class of fittest model, e.g. model with the best accuracy, swa model of best accuracies or last ones'''
    def __init__(self, ckpt : Checkpoint ,  *args , **kwargs) -> None:
        self.ckpt = ckpt
        self.reset()
    def __bool__(self): return True
    @abstractmethod
    def reset(self): ...
    @abstractmethod
    def assess(self , status : TrainerStatus , metrics : TrainerMetrics): 
        '''accuracy or loss to update assessment'''
    @abstractmethod
    def collect(self , trainer : BaseTrainer , *args , **kwargs) -> nn.Module: 
        '''output the final fittest model state dict'''

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
        for batch_input in trainer.data.train_dataloader(): 
            assert isinstance(batch_input, BatchInput) , f'{type(batch_input)} is not a BatchInput'
            trainer.on_train_batch_start()
            yield (batch_input.x , batch_input.kwargs)

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
    '''state dict of epoch with best accuracy or least loss'''
    def __init__(self, ckpt : Checkpoint , *args , **kwargs) -> None:
        super().__init__(ckpt , *args , **kwargs)

    def reset(self):
        self.best_epoch : EpochMetricResult | None = None

    def assess(self , status : TrainerStatus , metrics : TrainerMetrics):
        latest_epoch = metrics.attempt_metrics.latest_epoch()
        if latest_epoch and metrics.compare_epochs(latest_epoch, self.best_epoch):
            if self.best_epoch:
                self.ckpt.disjoin(self , self.best_epoch.epoch , self.best_epoch.phase)
            self.best_epoch = latest_epoch
            self.ckpt.join(self , self.best_epoch.epoch , self.best_epoch.phase)

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        #return self.ckpt.load_epoch(self.epoch_fix)
        net : nn.Module = deepcopy(getattr(trainer.model , 'net'))
        if not self.best_epoch:
            return net
        net.load_state_dict(self.ckpt.load(self.best_epoch.epoch , self.best_epoch.phase)['net'])
        return net

class EnsembleSWABest(SWAEnsembler):
    '''state dict of n_best epochs with best accuracy or least loss'''
    def __init__(self, ckpt : Checkpoint , n_best = 5 ,  *args , **kwargs) -> None:
        super().__init__(ckpt ,  *args , **kwargs)
        assert n_best > 0, n_best
        self.n_best      : int = n_best

    def reset(self):
        self.top_epochs : list[EpochMetricResult] = []
        
    def assess(self , status : TrainerStatus , metrics : TrainerMetrics):
        latest_epoch = metrics.attempt_metrics.latest_epoch()
        if not latest_epoch:
            return
        if len(self.top_epochs) < self.n_best:
            self.top_epochs.append(latest_epoch)
            self.ckpt.join(self , latest_epoch.epoch , latest_epoch.phase)
        else:
            arg = metrics.argmin_epochs(self.top_epochs)
            if metrics.compare_epochs(latest_epoch, self.top_epochs[arg]):
                drop_epoch = self.top_epochs.pop(arg)
                self.ckpt.disjoin(self , drop_epoch.epoch , drop_epoch.phase)

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        swa = SWAModel(getattr(trainer.model , 'net'))
        for epoch in self.top_epochs: 
            swa.update_sd(self.ckpt.load(epoch.epoch , epoch.phase)['net'])
        swa.update_bn(trainer)
        return swa.module.cpu()
    
class EnsembleSWALast(SWAEnsembler):
    '''state dict of n_last epochs around best accuracy or least loss'''
    def __init__(self, ckpt : Checkpoint , n_last = 5 , interval = 3 ,  *args , **kwargs) -> None:
        super().__init__(ckpt , *args , **kwargs)
        assert n_last > 0 and interval > 0, (n_last , interval)
        self.n_last      : int = n_last
        self.interval    : int = interval
        self.left_epochs : int = (n_last // 2) * interval

    def reset(self):
        self.best_epoch  : EpochMetricResult | None = None
        self.adjacent_epochs  : list[tuple[int,int]] = []

    def assess(self , status : TrainerStatus , metrics : TrainerMetrics):
        latest_epoch = metrics.attempt_metrics.latest_epoch()
        if not latest_epoch:
            return
        if not self.best_epoch or latest_epoch.phase > self.best_epoch.phase:
            for ep , ph in self.adjacent_epochs:
                self.ckpt.disjoin(self , ep , ph)
            self.best_epoch = latest_epoch
        elif metrics.compare_epochs(latest_epoch, self.best_epoch):
            self.best_epoch = latest_epoch
        if not self.best_epoch:
            return

        adjacent_epochs = self.interval_epochs(self.best_epoch.epoch , latest_epoch.epoch , latest_epoch.phase)
        epochs = [ep for ep , _ in adjacent_epochs]
        if epochs:
            [self.ckpt.disjoin(self , ep , ph) for ep , ph in self.adjacent_epochs if ep < min(epochs)]
       
        if latest_epoch.epoch in epochs: 
            self.ckpt.join(self , latest_epoch.epoch , latest_epoch.phase)
        self.adjacent_epochs = adjacent_epochs

    def interval_epochs(self , fix_epoch : int , epoch : int , phase : int = 0):
        epochs  = np.arange(self.interval , epoch + 1 , self.interval)
        left    = epochs[epochs < fix_epoch]
        right   = epochs[epochs > fix_epoch]
        epochs = [*left[-((self.n_last - 1) // 2):] , fix_epoch , *right]
        return [(ep , phase) for ep in epochs]

    def collect(self , trainer : BaseTrainer , *args , **kwargs):
        swa = SWAModel(getattr(trainer.model , 'net'))
        for ep , ph in self.adjacent_epochs[:self.n_last]:  
            swa.update_sd(self.ckpt.load(ep , ph)['net'])
        swa.update_bn(trainer)
        return swa.module.cpu()