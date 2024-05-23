import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..classes import BatchData , BatchMetric , BatchOutput , MetricList , TrainerStatus
from torch import nn , no_grad , Tensor
from typing import Any , Literal , Optional

from .config import TrainConfig
from ..func import mse , pearson , ccc , spearman

class Metrics:
    '''calculator of batch output'''
    DISPLAY_CHECK  = True # will display once if true
    DISPLAY_RECORD = {'loss' : {} , 'score' : {} , 'penalty' : {}}
    METRIC_FUNC = {'mse':mse ,'pearson':pearson,'ccc':ccc,'spearman':spearman,}

    def __init__(self , config : TrainConfig , use_dataset  : Literal['train','valid'] = 'valid' , 
                 use_metric : Literal['loss' , 'score'] = 'score' , **kwargs) -> None:
        self.config = config
        self.criterion = config.train_param['criterion']
        self.use_dataset = use_dataset
        self.use_metric  = use_metric
        self.f_loss    = self.loss_function(self.criterion['loss'])
        self.f_score   = self.score_function(self.criterion['score'])
        self.f_pen     = {k:self.penalty_function(k,v) for k,v in self.criterion['penalty'].items()}
        self.multiloss = None
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
    def penalty(self): return self.output.penalty

    @property
    def losses(self): return self.metric_batchs.losses
    @property
    def scores(self): return self.metric_batchs.scores
    
    '''
    @property
    def aggloss(self): return self.metric_batchs.loss
    @property
    def aggscore(self): return self.metric_batchs.score
    @property
    def train_scores(self): return self.metric_epochs['train.score']
    @property
    def train_losses(self):  return self.metric_epochs['train.loss']
    @property
    def valid_scores(self): return self.metric_epochs['valid.score']
    @property
    def valid_losses(self):  return self.metric_epochs['valid.loss']
    '''
    
    def new_model(self , model_param , **kwargs):
        if model_param['num_output'] > 1:
            multi_param = self.config.train_param['multilosses']
            self.multiloss = MultiLosses(multi_param['type'], model_param['num_output'] , **multi_param['param_dict'][multi_param['type']])
    
        self.f_pen.get('hidden_corr',{})['cond']       = self.config.tra_model or model_param.get('hidden_as_factors',False)
        self.f_pen.get('tra_opt_transport',{})['cond'] = self.config.tra_model
        self.new_attempt()
        return self
    
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
        penalty_kwargs = {'net':net,'pre':pred,'hidden':batch_output.hidden,'label':label , **kwargs}
        mt_param = getattr(net , 'get_multiloss_params')() if net and hasattr(net , 'get_multiloss_params') else {}
        self.calculate_from_tensor(dataset, label, pred, weight, penalty_kwargs, mt_param, assert_nan)

    def calculate_from_tensor(self , dataset , label : Tensor , pred : Tensor , weight : Optional[Tensor] = None , 
                              penalty_kwargs = {} , multiloss_param = {} , assert_nan = False):
        '''Calculate loss(with gradient), penalty , score'''
        assert dataset in ['train','validation','test']
        if label.shape != pred.shape: # if more label than output
            label = label[...,:pred.shape[-1]]
            assert label.shape == pred.shape , (label.shape , pred.shape)
        elif label.ndim == 1:
            label , pred = label.reshape(-1, 1) , pred.reshape(-1, 1)

        with no_grad():
            score = self.f_score(label , pred , weight , nan_check = (dataset == 'test') , first_col = True).item()

        if dataset == 'train':
            if self.multiloss is not None:
                losses = self.f_loss(label , pred , weight)[:self.multiloss.num_task]
                loss = self.multiloss.calculate_multi_loss(losses , multiloss_param)    
            else:
                losses = self.f_loss(label , pred , weight , first_col = True)
                loss = losses

            penalty = 0.
            for pen in self.f_pen.values():
                if pen['lamb'] <= 0 or not pen['cond']: continue
                penalty = penalty + pen['lamb'] * pen['func'](label , pred , weight , **penalty_kwargs)  
            loss = loss + penalty
            self.output = BatchMetric(loss = loss , score = score , penalty = penalty , losses = losses)
        else:
            self.output = BatchMetric(score = score)

        if assert_nan and self.output.loss.isnan():
            print(self.output)
            raise Exception('nan loss here')
        
    @classmethod
    def decorator_display(cls , func , mtype , mkey):
        def metric_display(mtype , mkey):
            if not cls.DISPLAY_RECORD[mtype].get(mkey , False):
                print(f'{mtype} function of [{mkey}] calculated and success!')
                cls.DISPLAY_RECORD[mtype][mkey] = True
        def wrapper(*args, **kwargs):
            v = func(*args, **kwargs)
            metric_display(mtype , mkey)
            return v
        return wrapper if cls.DISPLAY_CHECK else func
    
    @classmethod
    def firstC(cls , *args):
        new_args = [None if arg is None else arg[:,0] for arg in args]
        return new_args
    
    @classmethod
    def nonnan(cls , *args):
        nanpos = False
        for arg in args:
            if arg is not None: nanpos = arg.isnan() + nanpos
        if isinstance(nanpos , Tensor) and nanpos.any():
            if nanpos.ndim > 1: nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0
            if nanpos.all(): 
                for arg in args:
                    print(arg.shape)
                    print(arg)
            new_args = [None if arg is None else arg[~nanpos] for arg in args]
        else:
            new_args = args
        return new_args

    @classmethod
    def loss_function(cls , key):
        '''loss function , pearson/ccc should * -1.'''
        assert key in ('mse' , 'pearson' , 'ccc')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False,**kwargs):
                if first_col: label , pred , weight = cls.firstC(label , pred , weight)
                if nan_check: label , pred , weight = cls.nonnan(label , pred , weight)
                v = func(label , pred , weight, dim , **kwargs)
                if key != 'mse': v = torch.exp(-v)
                return v
            return wrapper
        new_func = decorator(cls.METRIC_FUNC[key])
        new_func = cls.decorator_display(new_func , 'loss' , key)
        return new_func

    @classmethod
    def score_function(cls , key):
        assert key in ('mse' , 'pearson' , 'ccc' , 'spearman')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False,**kwargs):
                if first_col: label , pred , weight = cls.firstC(label , pred , weight)
                if nan_check: label , pred , weight = cls.nonnan(label , pred , weight)
                v = func(label , pred , weight , dim , **kwargs)
                if key == 'mse' : v = -v
                return v
            return wrapper
        new_func = decorator(cls.METRIC_FUNC[key])
        new_func = cls.decorator_display(new_func , 'score' , key)
        return new_func
    
    @classmethod
    def penalty_function(cls , key , param):
        assert key in ('hidden_corr' , 'tra_opt_transport')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False, **kwargs):
                if first_col: label , pred , weight = cls.firstC(label , pred , weight)
                if nan_check: label , pred , weight = cls.nonnan(label , pred , weight)
                return func(label , pred , weight, dim , param = param , **kwargs)
            return wrapper
        new_func = decorator(getattr(cls , key , cls.null))
        new_func = cls.decorator_display(new_func , 'penalty' , key)
        return {'lamb': param['lamb'] , 'cond' : True , 'func' : new_func}
    
    @staticmethod
    def null(*args, **kwargs):
        return 0.

    @staticmethod
    def hidden_corr(*args , param , **kwargs):
        hidden = kwargs.get('hidden')
        assert isinstance(hidden,Tensor)
        if hidden.shape[-1] == 1: return 0
        if isinstance(hidden,(tuple,list)): hidden = torch.cat(hidden,dim=-1)
        pen = hidden.T.corrcoef().triu(1).nan_to_num().square().sum()
        return pen
    
    @staticmethod
    def tra_opt_transport(*args , param , **kwargs):
        tra_pdata = kwargs['net'].penalty_data
        pen = 0.
        if kwargs['net'].training and tra_pdata['probs'] is not None and tra_pdata['num_states'] > 1:
            square_error = (kwargs['hidden'] - kwargs['label']).square()
            square_error -= square_error.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            P = _sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
            lamb = (param['rho'] ** tra_pdata['global_steps'])
            reg = (tra_pdata['probs'] + 1e-4).log().mul(P).sum(dim=-1).mean()
            pen = - lamb * reg
        return pen

def _shoot_infs(inp_tensor):
    '''Replaces inf by maximum of tensor'''
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def _sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with no_grad():
        Q = _shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

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


class MultiLosses:
    '''some realization of multi-losses combine method'''
    def __init__(self , multi_type = None , num_task = -1 , **kwargs):
        '''
        example:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            
            ml = multiloss(2)
            ml.view_plot(2 , 'dwa')
            ml.view_plot(2 , 'ruw')
            ml.view_plot(2 , 'gls')
            ml.view_plot(2 , 'rws')
        '''
        self.multi_type = multi_type
        self.reset_multi_type(num_task,**kwargs)

    def reset_multi_type(self, num_task , **kwargs):
        self.num_task   = num_task
        self.num_output = num_task
        if num_task > 0 and self.multi_type is not None: 
            self.multi_class = {
                'ewa':self.EWA,
                'hybrid':self.Hybrid,
                'dwa':self.DWA,
                'ruw':self.RUW,
                'gls':self.GLS,
                'rws':self.RWS,
            }[self.multi_type](num_task , **kwargs)
        return self
    
    def calculate_multi_loss(self , losses , mt_param , **kwargs):
        return self.multi_class(losses , mt_param , **kwargs)
    
    class _BaseMultiLossesClass():
        '''base class of multi_class class'''
        def __init__(self , num_task , **kwargs):
            self.num_task = num_task
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
    
    class EWA(_BaseMultiLossesClass):
        '''Equal weight average'''
        def penalty(self , losses , mt_param : dict = {}): 
            return 0.
    
    class Hybrid(_BaseMultiLossesClass):
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
    
    class DWA(_BaseMultiLossesClass):
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
        
    class RUW(_BaseMultiLossesClass):
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

    class GLS(_BaseMultiLossesClass):
        '''geometric loss strategy , Chennupati etc.(2019)'''
        def total_loss(self , losses , weight , penalty):
            return losses.pow(weight).prod().pow(1/weight.sum()) + penalty
    class RWS(_BaseMultiLossesClass):
        '''random weight loss, RW , Lin etc.(2021) , https://arxiv.org/pdf/2111.10603.pdf'''
        def weight(self , losses , mt_param : dict = {}): 
            return nn.functional.softmax(torch.rand_like(losses),-1)

    @classmethod
    def view_plot(cls , multi_type = 'ruw'):
        num_task = 2
        if multi_type == 'ruw':
            if num_task > 2 : num_task = 2
            x,y = torch.rand(100,num_task),torch.rand(100,1)
            ls = (x - y).sqrt().sum(dim = 0)
            alpha = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            ruw = cls.RUW(num_task)
            l = torch.stack([torch.stack([ruw(ls,{'alpha':Tensor([s1[i,j],s2[i,j]])})[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('alpha-1')
            ax.set_ylabel('alpha-2')
            ax.set_zlabel('loss') #type:ignore
            ax.set_title(f'RUW Loss vs alpha ({num_task}-D)')
        elif multi_type == 'gls':
            ls = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([torch.tensor([s1[i,j],s2[i,j]]).prod().sqrt() for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('gls_loss') #type:ignore
            ax.set_title(f'GLS Loss vs sub-Loss ({num_task}-D)')
        elif multi_type == 'rws':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([(torch.tensor([s1[i,j],s2[i,j]])*nn.functional.softmax(torch.rand(num_task),-1)).sum() for j in range(s1.shape[1])]) 
                             for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('rws_loss') #type:ignore
            ax.set_title(f'RWS Loss vs sub-Loss ({num_task}-D)')
        elif multi_type == 'dwa':
            nepoch = 100
            s = np.arange(nepoch)
            l1 = 1 / (4 + s) + 0.1 + np.random.rand(nepoch) *0.05
            l2 = 1 / (4 + 2*s) + 0.15 + np.random.rand(nepoch) *0.03
            tau = 2
            w1 = np.exp(np.concatenate((np.array([1,1]),l1[2:]/l1[1:-1]))/tau)
            w2 = np.exp(np.concatenate((np.array([1,1]),l2[2:]/l1[1:-1]))/tau)
            w1 , w2 = num_task * w1 / (w1+w2) , num_task * w2 / (w1+w2)
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