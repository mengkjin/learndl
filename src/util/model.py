import torch
import numpy as np

from abc import ABC , abstractmethod
from copy import deepcopy
from torch import nn
from torch.optim.swa_utils import AveragedModel , update_bn
from typing import Any , Iterator , Optional

from .. import nn as NN
from .config import TrainConfig
from .metric import Metrics
from .store import Checkpoint
from ..classes import (
    BaseDataModule , BaseModelModule , BatchData , BatchOutput , 
    BoosterData , ModelDict , ModelFile)
from ..algo.boost.lgbm import Lgbm

class ModelManager:
    '''a group of ensemble models , of same net structure'''
    def __init__(self, model_module : BaseModelModule , use_score = True , **kwargs) -> None:
        self.module = model_module
        self.use_score = use_score
        self.ensembles = {model_type:self.choose(model_type)(self.ckpt , use_score , **kwargs) for model_type in self.config.model_types}
        if self.config.lgbm_ensembler:
            self.lgbm_ensembler = LgbmEnsembler(model_module)

    @property
    def config(self) -> TrainConfig: return self.module.config
    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def ckpt(self) -> Checkpoint: return self.module.checkpoint
    @property
    def device(self): return self.module.device

    @classmethod
    def setup(cls , model_module : BaseModelModule):
        return cls(model_module)

    def new_model(self , training : bool , model_file : ModelFile):
        for model in self.ensembles.values(): model.reset()
        if not training and self.config.lgbm_ensembler:
            self.lgbm_ensembler.load(model_file['lgbm_string'])
        return self
    
    @staticmethod
    def get_net(module_name : str , param = {} , state_dict = None , device = None , **kwargs):
        net = getattr(NN , module_name)(**param)
        assert isinstance(net , nn.Module) , net.__class__
        if state_dict: net.load_state_dict(state_dict)
        return device(net) if callable(device) else net.to(device)

    def net(self , use_state_dict = None):
        return self.get_net(self.config.model_module , self.module.model_param , use_state_dict , self.device)

    def override(self):
        if self.config.lgbm_ensembler:
            pred = self.lgbm_ensembler.predict()
            assert pred is not None
            self.module.batch_output.override_pred(pred)

    def assess(self , epoch : int , metrics : Metrics): 
        for model in self.ensembles.values(): 
            model.assess(self.module.net , epoch , metrics) # , metrics.latest['valid.score'] , metrics.latest['valid.loss'])

    def collect(self , model_type , *args):
        # if model_type == 'best' and self.ensembles[model_type].epoch_fix < 0: print(self.module.status)
        net = self.ensembles[model_type].collect(self.module.net , self.data , *args , device=self.device)
        model_dict = ModelDict(net.state_dict())
        if self.config.lgbm_ensembler:
            model_dict.lgbm_string = self.lgbm_ensembler.fit(net).model_string
        return model_dict

    @staticmethod
    def choose(model_type):
        '''get a subclass of _BaseEnsembler'''
        if model_type == 'best': return EnsembleBest
        elif model_type == 'swabest': return EnsembleSWABest
        elif model_type == 'swalast': return EnsembleSWALast
        else: raise KeyError(model_type)

class _BaseEnsembler(ABC):
    '''abstract class of fittest model, e.g. model with the best score, swa model of best scores or last ones'''
    def __init__(self, ckpt : Checkpoint , use_score = True , **kwargs) -> None:
        self.ckpt , self.use_score = ckpt , use_score
        self.reset()
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

class EnsembleBest(_BaseEnsembler):
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

class EnsembleSWABest(_BaseEnsembler):
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
    
class EnsembleSWALast(_BaseEnsembler):
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
    
class LgbmEnsembler:
    '''load booster data and fit'''
    def __init__(self , model_module : BaseModelModule) -> None:
        self.module = model_module

    @property
    def data(self) -> BaseDataModule: return self.module.data
    @property
    def train_dl(self): return self.data.train_dataloader()
    @property
    def valid_dl(self): return self.data.val_dataloader()
    @property
    def test_dl(self): return self.data.test_dataloader()
    @property
    def y_secid(self) -> np.ndarray | torch.Tensor: return self.data.y_secid
    @property
    def y_date(self) -> np.ndarray | torch.Tensor: return self.data.y_date
    @property
    def model_string(self): return self.model.model_to_string()
    @property
    def is_cuda(self) -> bool: return self.module.device.device.type == 'cuda'

    def booster_data(self , net : nn.Module , loader : Iterator[BatchData | Any]) -> BoosterData:
        hh , yy , ii = [] , [] , []
        net.eval()
        with torch.no_grad():
            for batch_data in loader:
                hidden = BatchOutput(net(batch_data.x)).hidden
                assert hidden is not None , f'hidden must not be none when using LgbmEnsembler'
                hh.append(hidden.detach().cpu())
                yy.append(batch_data.y.detach().cpu())
                ii.append(batch_data.i.detach().cpu())
        hh , yy , ii = torch.vstack(hh).numpy() , torch.vstack(yy).numpy() , torch.vstack(ii).numpy()
        secid_i , secid_j = np.unique(ii[:,0] , return_inverse=True)
        date_i  , date_j  = np.unique(ii[:,1] , return_inverse=True)
        hh_values = np.full((len(secid_i) , len(date_i) , hh.shape[-1]) , fill_value = np.nan)
        yy_values = np.full((len(secid_i) , len(date_i)) , fill_value = np.nan)
        
        hh_values[secid_j , date_j] = hh[:]
        yy_values[secid_j , date_j] = yy[...,0]

        return BoosterData(hh_values , yy_values , self.y_secid[secid_i] , self.y_date[date_i])

    def fit(self , net : nn.Module):
        net = self.module.device(net)
        train_data = self.booster_data(net , self.train_dl)
        valid_data = self.booster_data(net , self.valid_dl)
        self.model = Lgbm(train_data , valid_data , cuda=self.is_cuda).fit()
        # self.model.plot.training()
        return self
    
    def load(self , model_str):
        self.model = Lgbm.model_from_string(model_str , cuda=self.is_cuda)
        self.loaded = True
        return self
    
    def predict(self):
        assert self.loaded
        hidden = self.module.batch_output.hidden
        if hidden is None: return
        hidden = hidden.detach().cpu().numpy()
        label = self.module.batch_data.y.detach().cpu().numpy()
        pred  = torch.tensor(self.model.predict(BoosterData(hidden , label) , reform = False))
        return pred
            
    