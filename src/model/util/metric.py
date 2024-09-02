import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from abc import ABC , abstractmethod
from torch import nn , no_grad , Tensor
from typing import Any , Literal , Optional

from ..classes import BatchData , BatchMetric , BatchOutput , MetricList , TrainConfig , TrainerStatus
from ..nn import get_multiloss_params
from ...func import mse , pearson , ccc , spearman

DISPLAY_CHECK  = True # will display once if true
DISPLAY_RECORD = {'loss' : {} , 'score' : {} , 'penalty' : {}}
METRIC_FUNC    = {'mse':mse ,'pearson':pearson,'ccc':ccc,'spearman':spearman,}

class Metrics:
    '''calculator of batch output'''
    def __init__(self , config : TrainConfig , use_dataset  : Literal['train','valid'] = 'valid' , 
                 use_metric : Literal['loss' , 'score'] = 'score' , **kwargs) -> None:
        self.config      = config
        self.use_dataset = use_dataset
        self.use_metric  = use_metric
        
        self.loss_calc    = LossCalculator(self.config.train_criterion_loss)
        self.score_calc   = ScoreCalculator(self.config.train_criterion_score)
        self.penalty_calc = {k:PenaltyCalculator(k,v) for k,v in self.config.train_criterion_penalty.items()}
        
        self.output    = BatchMetric()
        self.metric_batchs = MetricsAggregator()
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}
        self.latest : dict[str,Any] = {}
        self.last_metric : Any = None
        self.best_metric : Any = None

    @property
    def loss(self): return self.output.loss
    @property
    def score(self): return self.output.score
    @property
    def loss_item(self): return self.output.loss_item
    @property
    def losses(self): return self.metric_batchs.losses
    @property
    def scores(self): return self.metric_batchs.scores
    
    def new_model(self , model_param : dict[str,Any] , **kwargs):
        self.model_param  = model_param
        self.num_output   = model_param.get('num_output' , 1)
        self.which_output = model_param.get('which_output' , 0)
        self.multi_losses = MultiHeadLosses(
            self.num_output , self.config.train_multilosses_type, **self.config.train_multilosses_param)

        self.update_penalty_calc()
        self.new_attempt()
        return self
    
    def update_penalty_calc(self):
        if self.penalty_calc.get('hidden_corr'): 
            cond_hidden_corr = (self.config.nn_category == 'tra') or self.model_param.get('hidden_as_factors',False)
            self.penalty_calc['hidden_corr'].cond = cond_hidden_corr

    def new_attempt(self):
        self.best_metric = None
        self.metric_epochs = {f'{ds}.{mt}':[] for ds in ['train','valid','test'] for mt in ['loss','score']}

    def new_epoch_metric(self , dataset : Literal['train','valid','test'] , status : TrainerStatus):
        self.metric_batchs.new(dataset , status.model_num , status.model_date , status.epoch , status.model_type)

    def collect_batch_metric(self):
        self.metric_batchs.record(self.output)

    def collect_epoch_metric(self , dataset : Literal['train','valid','test']):
        self.metric_batchs.collect()
        loss , score = self.metric_batchs.loss , self.metric_batchs.score
        self.latest[f'{dataset}.loss']  = loss
        self.latest[f'{dataset}.score'] = score
        self.metric_epochs[f'{dataset}.loss'].append(loss) 
        self.metric_epochs[f'{dataset}.score'].append(score)

        if dataset == self.use_dataset:
            metric = loss if self.use_metric == 'loss' else score
            self.last_metric = metric
            if (self.best_metric is None or
                (self.use_metric == 'score' and self.best_metric < metric) or 
                (self.use_metric == 'loss' and self.best_metric > metric)): 
                self.best_metric = metric
    
    def better_epoch(self , old_best_epoch : Any) -> bool:
        if old_best_epoch is None:
            return True
        elif self.use_metric == 'score':
            return self.last_metric > old_best_epoch 
        else:
            return self.last_metric < old_best_epoch
        
    def better_attempt(self , old_best_attempt : Any) -> bool:
        if old_best_attempt is None:
            return True
        elif self.use_metric == 'score':
            return self.best_metric > old_best_attempt 
        else:
            return self.best_metric < old_best_attempt

    def calculate(self , dataset , batch_data : BatchData , batch_output : BatchOutput , net : Optional[nn.Module] = None , 
                  assert_nan = False , **kwargs):
        label  = batch_data.y
        pred   = batch_output.pred
        weight = batch_data.w
        other  = batch_output.other
        penalty_kwargs = {'net':net,'pre':pred,'label':label,**kwargs,**other}
        mt_param = get_multiloss_params(net)
        self.calculate_from_tensor(dataset, label, pred, weight, penalty_kwargs, mt_param, assert_nan)

    def calculate_from_tensor(self , dataset , label : Tensor , pred : Tensor , weight : Optional[Tensor] = None , 
                              penalty_kwargs : dict[str,Any] = {} , multiloss_param = {} , assert_nan = False):
        '''Calculate loss(with gradient), penalty , score'''
        assert dataset in ['train','valid','test']
        
        if label.shape != pred.shape: # if more label than output
            label = label[...,:pred.shape[-1]]
        if self.config.module_type != 'nn':
            label , pred = label.flatten() , pred.flatten()
        if label.ndim == 1:
            label , pred = label.reshape(-1, 1) , pred.reshape(-1, 1)
        assert label.shape == pred.shape , (label.shape , pred.shape)
        assert label.shape[-1] == self.num_output , (label.shape[-1] , self.num_output)

        with no_grad():
            score = self.score_calc(label , pred , weight , nan_check = True , which_head = self.which_output , training = False).item()
            self.output = BatchMetric(score = score)

        if dataset == 'train' and self.config.module_type == 'nn':
            multiheadlosses = self.loss_calc(label , pred , weight , nan_check = False , training = True)
            loss = self.multi_losses(multiheadlosses , multiloss_param) 
            self.output.add_loss(self.loss_calc.criterion , loss)

            for key , value in penalty_kwargs.items():
                if key.startswith('loss_') or key.startswith('penalty_'): 
                    self.output.add_loss(key , value)

            for key , pen_cal in self.penalty_calc.items():
                value = pen_cal(**penalty_kwargs)
                self.output.add_loss(key , value)

        #if assert_nan and self.output.loss.isnan():
        #    print(self.output)
        #    raise Exception('nan loss here')
        

class _MetricCalculator(ABC):
    KEY : Literal['loss' , 'score' , 'penalty'] = 'loss'
    def __init__(self , criterion : str):
        '''
        criterion : metric function
        collapse  : aggregate last dimension if which_col < -1
        '''
        self.criterion = criterion

    def __call__(self, label : Tensor ,pred : Tensor , weight : Optional[Tensor] = None , 
                 dim : int = 0 , nan_check : bool = False , which_head : Optional[int] = None, 
                 training = False , **kwargs) -> Tensor:
        '''calculate the resulting metric'''
        label , pred , weight = self.slice_data(label , pred , weight , nan_check , which_head , training)
        v = self.forward(label , pred , weight, dim , **kwargs)
        self.display()
        return v
    
    @abstractmethod
    def forward(self, label : Tensor ,pred : Tensor , weight : Optional[Tensor] = None , dim : int = 0 , **kwargs) -> Tensor:
        '''calculate the metric'''
        return METRIC_FUNC[self.criterion](label , pred , weight, dim , **kwargs)

    def display(self):
        if DISPLAY_CHECK and not DISPLAY_RECORD[self.KEY].get(self.criterion , False):
            print(f'{self.KEY} function of [{self.criterion}] calculated and success!')
            DISPLAY_RECORD[self.KEY][self.criterion] = True
    
    @classmethod
    def slice_data(cls , label , pred , weight = None , nan_check : bool = False , 
                   which_head : Optional[int] = None , training = False) -> tuple[Tensor , Tensor , Optional[Tensor]]:
        '''each element return ith column, if negative then return raw element'''
        if nan_check: label , pred , weight = cls.slice_data_nonnan(label , pred , weight)
        label  = cls.slice_data_col(label , None if training else 0)
        pred   = cls.slice_data_col(pred  , None if training else which_head)
        weight = cls.slice_data_col(weight, None if training else which_head)

        if pred.ndim > label.ndim:
            pred = pred.nanmean(dim = -1)
            if weight is not None: weight = weight.nanmean(dim = -1)
        assert label.shape == pred.shape , (label.shape , pred.shape)
        return label , pred , weight
    
    @staticmethod
    def slice_data_col(data : Optional[Tensor] , col : Optional[int] = None) -> Any:
        return data if data is None or col is None else data[...,col]
        
    @staticmethod
    def slice_data_nonnan(*args):
        nanpos = False
        for arg in args:
            if arg is not None: nanpos = arg.isnan() + nanpos
        if isinstance(nanpos , Tensor) and nanpos.any():
            if nanpos.ndim > 1: nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0
            if nanpos.all(): [print(arg) for arg in args]
            args = [None if arg is None else arg[~nanpos] for arg in args]
        return args
    
class LossCalculator(_MetricCalculator):
    KEY = 'loss'
    def __init__(self , criterion : Literal['mse', 'pearson', 'ccc']):
        ''' 'mse', 'pearson', 'ccc' , will not collapse columns '''
        assert criterion in ('mse' , 'pearson' , 'ccc')
        super().__init__(criterion)

    def forward(self, label: Tensor, pred: Tensor, weight: Tensor | None = None, dim: int = 0, **kwargs):
        v = METRIC_FUNC[self.criterion](label , pred , weight, dim , **kwargs)
        if self.criterion != 'mse': v = torch.exp(-v)
        return v
    
class ScoreCalculator(_MetricCalculator):
    KEY = 'score'
    def __init__(self , criterion : Literal['mse', 'pearson', 'ccc' , 'spearman']):
        ''' 'mse', 'pearson', 'ccc' , 'spearman', will collapse columns if multi-labels '''
        assert criterion in ('mse' , 'pearson' , 'ccc' , 'spearman')
        super().__init__(criterion)
    
    def forward(self, label: Tensor, pred: Tensor, weight: Tensor | None = None, dim: int = 0, **kwargs):
        v = METRIC_FUNC[self.criterion](label , pred , weight, dim , **kwargs)
        if self.criterion == 'mse' : v = -v
        return v
    
class PenaltyCalculator(_MetricCalculator):
    KEY = 'penalty'
    def __init__(self , criterion : Literal['hidden_corr'] , param : dict[str,Any]):
        ''' 'hidden_corr' '''
        super().__init__(criterion)
        self.param = param
        if self.criterion == 'hidden_corr':
            self.penalty = self.hidden_corr
        else: 
            raise KeyError(self.criterion)
        self.lamb = param['lamb']
        self.cond = True
        
    def __call__(self, **kwargs) -> Tensor:
        if self.lamb <= 0 or not self.cond: return torch.Tensor([0.])
        v = self.penalty(**kwargs)
        self.display()
        return self.lamb * v
    
    def forward(self, *args , **kwargs): return None
    def hidden_corr(self , hidden : Tensor | list | tuple , **kwargs) -> Tensor:
        '''if kwargs containse hidden, calculate 2nd-norm of hTh'''
        if isinstance(hidden,(tuple,list)): hidden = torch.cat(hidden,dim=-1)
        h = (hidden - hidden.mean(dim=0,keepdim=True)) / (hidden.std(dim=0,keepdim=True) + 1e-6)
        # pen = h.T.cov().norm().square() / (h.shape[-1] ** 2)
        pen = h.T.cov().square().mean()
        return pen

class MetricsAggregator:
    '''record a list of batch metric and perform agg operations, usually used in an epoch'''
    def __init__(self) -> None:
        self.table : Optional[pd.DataFrame] = None
        self.new('init',0,0)
    def __len__(self):  return len(self._record['loss'].values)
    def new(self , dataset , model_num , model_date , epoch = 0 , model_type = 'best'):
        self._params = [dataset , model_num , model_date , model_type , epoch]
        self._record = {m:MetricList(f'{dataset}.{model_num}.{model_date}.{model_type}.{epoch}.{m}',m) for m in ['loss','score']}
    def record(self , metrics): 
        [self._record[m].record(metrics) for m in ['loss','score']]
    def collect(self):
        df = pd.DataFrame([[*self._params , self.loss , self.score]] , columns=['dataset','model_num','model_date','model_type','epoch','loss','score'])
        self.table = df if self.table is None else pd.concat([self.table , df]).reindex()
    @property
    def nanloss(self): return self._record['loss'].any_nan()
    @property
    def loss(self):  return self._record['loss'].mean()
    @property
    def score(self): return self._record['score'].mean()
    @property
    def losses(self): return self._record['loss'].values
    @property
    def scores(self): return self._record['score'].values


class MultiHeadLosses:
    '''some realization of multi-losses combine method'''
    def __init__(self , num_head = -1 , multi_type = None , **kwargs):
        '''
        example:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            
            ml = MultiHeadLosses(2)
            ml.view_plot(2 , 'dwa')
            ml.view_plot(2 , 'ruw')
            ml.view_plot(2 , 'gls')
            ml.view_plot(2 , 'rws')
        '''
        self.num_head = num_head
        self.multi_type = multi_type
        self.kwargs = kwargs
        self.reset_multi_type()

    def reset_multi_type(self):
        if self.num_head > 1 and self.multi_type is not None: 
            self.multi_class = {
                'ewa':self.EWA,
                'hybrid':self.Hybrid,
                'dwa':self.DWA,
                'ruw':self.RUW,
                'gls':self.GLS,
                'rws':self.RWS,
            }[self.multi_type](self.num_head , **self.kwargs)
        else:
            self.multi_class = self.single_loss
        return self
    
    def single_loss(self, loss , *args , **kwargs): return loss
    def __call__(self , losses : Tensor , mt_param : dict[str,Any] , **kwargs): 
        '''calculate combine loss of multiple output losses'''
        return self.multi_class(losses , mt_param , **kwargs)
    
    class _BaseMHL():
        '''base class of multi_head_losses class'''
        def __init__(self , num_head , **kwargs):
            self.num_head = num_head
            self.record_num = 0 
            self.record_losses = []
            self.record_weight = []
            self.record_penalty = []
            self.kwargs = kwargs
            self.reset(**kwargs)
        def __call__(self , losses , mt_param , **kwargs):
            weight , penalty = self.weight(losses , mt_param) , self.penalty(losses , mt_param)
            self.record(losses , weight , penalty)
            return self.total_loss(losses , weight , penalty)
        def reset(self , **kwargs): ...
        def record(self , losses , weight , penalty):
            self.record_num += 1
            self.record_losses.append(losses.detach() if isinstance(losses,Tensor) else losses)
            self.record_weight.append(weight.detach() if isinstance(weight,Tensor) else weight)
            self.record_penalty.append(penalty.detach() if isinstance(penalty,Tensor) else penalty)
        def weight(self , losses , mt_param : dict = {}):
            return torch.ones_like(losses)
        def penalty(self , losses , mt_param : dict = {}): 
            return 0.
        def total_loss(self , losses , weight , penalty):
            return (losses * weight).sum() + penalty
    
    class EWA(_BaseMHL):
        '''Equal weight average'''
        def penalty(self , losses , mt_param : dict = {}): 
            return 0.
    
    class Hybrid(_BaseMHL):
        '''Hybrid of DWA and RUW'''
        def reset(self , **kwargs):
            self.tau = kwargs['tau']
            self.phi = kwargs['phi']
        def weight(self , losses , mt_param : dict = {}):
            if self.record_num < 2:
                weight = torch.ones_like(losses)
            else:
                weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
                weight = weight / weight.sum() * weight.numel()
            return weight + 1 / mt_param['alpha'].square()
        def penalty(self , losses , mt_param : dict = {}): 
            penalty = (mt_param['alpha'].log().square()+1).log().sum()
            if self.phi is not None: 
                penalty = penalty + (self.phi - mt_param['alpha'].log().abs().sum()).abs()
            return penalty
    
    class DWA(_BaseMHL):
        '''dynamic weight average , https://arxiv.org/pdf/1803.10704.pdf'''
        def reset(self , **kwargs):
            self.tau = kwargs['tau']
        def weight(self , losses , mt_param : dict = {}):
            if self.record_num < 2:
                weight = torch.ones_like(losses)
            else:
                weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
                weight = weight / weight.sum() * weight.numel()
            return weight
        
    class RUW(_BaseMHL):
        '''Revised Uncertainty Weighting (RUW) Loss , https://arxiv.org/pdf/2206.11049v2.pdf (RUW + DWA)'''
        def reset(self , **kwargs):
            self.phi = kwargs['phi']
        def weight(self , losses , mt_param : dict = {}):
            return 1 / mt_param['alpha'].square()
        def penalty(self , losses , mt_param : dict = {}): 
            penalty = (mt_param['alpha'].log().square()+1).log().sum()
            if self.phi is not None: 
                penalty = penalty + (self.phi - mt_param['alpha'].log().abs().sum()).abs()
            return penalty

    class GLS(_BaseMHL):
        '''geometric loss strategy , Chennupati etc.(2019)'''
        def total_loss(self , losses , weight , penalty):
            return losses.pow(weight).prod().pow(1/weight.sum()) + penalty
    class RWS(_BaseMHL):
        '''random weight loss, RW , Lin etc.(2021) , https://arxiv.org/pdf/2111.10603.pdf'''
        def weight(self , losses , mt_param : dict = {}): 
            return nn.functional.softmax(torch.rand_like(losses),-1)

    @classmethod
    def view_plot(cls , num_head = 2 , multi_type = 'ruw'):
        if multi_type == 'ruw':
            if num_head > 2 : num_head = 2
            x,y = torch.rand(100,num_head),torch.rand(100,1)
            ls = (x - y).sqrt().sum(dim = 0)
            alpha = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            ruw = cls.RUW(num_head)
            l = torch.stack([torch.stack([ruw(ls,{'alpha':Tensor([s1[i,j],s2[i,j]])})[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('alpha-1')
            ax.set_ylabel('alpha-2')
            ax.set_zlabel('loss') #type:ignore
            ax.set_title(f'RUW Loss vs alpha ({num_head}-D)')
        elif multi_type == 'gls':
            ls = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([torch.tensor([s1[i,j],s2[i,j]]).prod().sqrt() for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('gls_loss') #type:ignore
            ax.set_title(f'GLS Loss vs sub-Loss ({num_head}-D)')
        elif multi_type == 'rws':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([(torch.tensor([s1[i,j],s2[i,j]])*nn.functional.softmax(torch.rand(num_head),-1)).sum() for j in range(s1.shape[1])]) 
                             for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('rws_loss') #type:ignore
            ax.set_title(f'RWS Loss vs sub-Loss ({num_head}-D)')
        elif multi_type == 'dwa':
            nepoch = 100
            s = np.arange(nepoch)
            l1 = 1 / (4 + s) + 0.1 + np.random.rand(nepoch) *0.05
            l2 = 1 / (4 + 2*s) + 0.15 + np.random.rand(nepoch) *0.03
            tau = 2
            w1 = np.exp(np.concatenate((np.array([1,1]),l1[2:]/l1[1:-1]))/tau)
            w2 = np.exp(np.concatenate((np.array([1,1]),l2[2:]/l1[1:-1]))/tau)
            w1 , w2 = num_head * w1 / (w1+w2) , num_head * w2 / (w1+w2)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(s, l1, color='blue', label='task1')
            ax1.plot(s, l2, color='red', label='task2')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss for Epoch')
            ax1.legend()
            ax2.plot(s, w1, color='blue', label='task1')
            ax2.plot(s, w2, color='red', label='task2')
            ax1.set_xlabel('Epoch')
            ax2.set_ylabel('Weight')
            ax2.set_title('Weight for Epoch')
            ax2.legend()
        else:
            print(f'Unknow multi_type : {multi_type}')
            
        plt.show()