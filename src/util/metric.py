import matplotlib.pyplot as plt
import numpy as np
import torch

from dataclasses import dataclass
from typing import Literal

from ..func import basic as B

class Metrics:
    display_check  = True # will display once if true
    display_record = {'loss' : {} , 'score' : {} , 'penalty' : {}}
    default_metric = {'mse':B.mse ,'pearson':B.pearson,'ccc':B.ccc,'spearman':B.spearman,}

    def __init__(self , criterion , **kwargs) -> None:
        self.criterion = criterion
        self.loss    = self.loss_function(criterion['loss'])
        self.score   = self.score_function(criterion['score'])
        self.penalty = {k:self.penalty_function(k,v) for k,v in criterion['penalty'].items()}
        self.multiloss = None
    
    def model_update(self , model_param , config , **kwargs):
        if model_param['num_output'] > 1:
            multi_param = config.train_param['multitask']
            self.multiloss = MultiLosses(multi_param['type'], model_param['num_output'] , **multi_param['param_dict'][multi_param['type']])
    
        self.penalty.get('hidden_corr',{})['cond']       = config.tra_model or model_param.get('hidden_as_factors',False)
        self.penalty.get('tra_opt_transport',{})['cond'] = config.tra_model
        return self
    
    def calculate(self , key : Literal['train' , 'valid' , 'test'] , 
                  label , pred , weight = None , net = None , 
                  penalty_kwargs = {} , **kwargs):
        '''
        Calculate loss(with gradient), penalty , score
        '''
        if label.shape != pred.shape: # if more label than output
            label = label[...,:pred.shape[-1]]
            assert label.shape == pred.shape , (label.shape , pred.shape)

        with torch.no_grad():
            score = self.score(label , pred , weight , nan_check = (key == 'test') , first_col = True).item()

        if key == 'train':
            if self.multiloss is not None:
                losses = self.loss(label , pred , weight)[:self.multiloss.num_task]
                mt_param = {}
                if net and hasattr(net , 'get_multiloss_params'):
                    mt_param = getattr(net , 'get_multiloss_params')()
                loss = self.multiloss.calculate_multi_loss(losses , mt_param)    
            else:
                losses = self.loss(label , pred , weight , first_col = True)
                loss = losses

            penalty = 0.
            for pen in self.penalty.values():
                if pen['lamb'] <= 0 or not pen['cond']: continue
                penalty = penalty + pen['lamb'] * pen['func'](label , pred , weight , **penalty_kwargs)  
            loss = loss + penalty
            return self.MetricOutput(loss = loss , score = score , penalty = penalty , losses = losses)
        else:
            return self.MetricOutput(score = score)
    
    @dataclass
    class MetricOutput:
        loss      : torch.Tensor = torch.Tensor([0.])
        score     : float = 0.
        loss_item : float = 0.
        penalty   : torch.Tensor | float = 0.
        losses    : torch.Tensor = torch.Tensor([0.])

        def __post_init__(self):
            self.loss_item = self.loss.item()

    @classmethod
    def decorator_display(cls , func , mtype , mkey):
        def metric_display(mtype , mkey):
            if not cls.display_record[mtype].get(mkey , False):
                print(f'{mtype} function of [{mkey}] calculated and success!')
                cls.display_record[mtype][mkey] = True
        def wrapper(*args, **kwargs):
            v = func(*args, **kwargs)
            metric_display(mtype , mkey)
            return v
        return wrapper if cls.display_check else func
    
    @classmethod
    def firstC(cls , *args):
        new_args = [None if arg is None else arg[:,0] for arg in args]
        return new_args
    
    @classmethod
    def nonnan(cls , *args):
        nanpos = False
        for arg in args:
            if arg is not None: nanpos = arg.isnan() + nanpos
        if isinstance(nanpos , torch.Tensor) and nanpos.any():
            if nanpos.ndim > 1: nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0
            new_args = [None if arg is None else arg[~nanpos] for arg in args]
        else:
            new_args = args
        return new_args

    @classmethod
    def loss_function(cls , key):
        """
        loss function , pearson/ccc should * -1.
        """
        assert key in ('mse' , 'pearson' , 'ccc')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False,**kwargs):
                if first_col: label , pred , weight = cls.firstC(label , pred , weight)
                if nan_check: label , pred , weight = cls.nonnan(label , pred , weight)
                v = func(label , pred , weight, dim , **kwargs)
                if key != 'mse': v = torch.exp(-v)
                return v
            return wrapper
        new_func = decorator(cls.default_metric[key])
        new_func = cls.decorator_display(new_func , 'loss' , key)
        return new_func

    @classmethod
    def score_function(cls , key):
        assert key in ('mse' , 'pearson' , 'ccc' , 'spearman')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False,**kwargs):
                if first_col: label , pred , weight = cls.firstC(label , pred , weight)
                if nan_check: label , pred , weight = cls.nonnan(label , pred , weight)
                v = func(label , pred , weight, dim , **kwargs)
                if key == 'mse' : v = -v
                return v
            return wrapper
        new_func = decorator(cls.default_metric[key])
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
        assert isinstance(hidden,torch.Tensor)
        if hidden.shape[-1] == 1: return 0
        if isinstance(hidden,(tuple,list)): hidden = torch.cat(hidden,dim=-1)
        pen = hidden.T.corrcoef().triu(1).nan_to_num().square().sum()
        return pen
    
    @staticmethod
    def tra_opt_transport(*args , param , **kwargs):
        tra_pdata = kwargs['net'].penalty_data_access()
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
    """Replaces inf by maximum of tensor"""
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
    with torch.no_grad():
        Q = _shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

class MultiLosses:
    def __init__(self , multi_type = None , num_task = -1 , **kwargs):
        """
        example:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            
            ml = multiloss(2)
            ml.view_plot(2 , 'dwa')
            ml.view_plot(2 , 'ruw')
            ml.view_plot(2 , 'gls')
            ml.view_plot(2 , 'rws')
        """
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
        """
        base class of multi_class class
        """
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
        def reset(self , **kwargs):
            pass
        def record(self , losses , weight , penalty):
            self.record_num += 1
            self.record_losses.append(losses.detach() if isinstance(losses,torch.Tensor) else losses)
            self.record_weight.append(weight.detach() if isinstance(weight,torch.Tensor) else weight)
            self.record_penalty.append(penalty.detach() if isinstance(penalty,torch.Tensor) else penalty)
        def weight(self , losses , mt_param : dict = {}):
            return torch.ones_like(losses)
        def penalty(self , losses , mt_param : dict = {}): 
            return 0.
        def total_loss(self , losses , weight , penalty):
            return (losses * weight).sum() + penalty
    
    class EWA(_BaseMultiLossesClass):
        """
        Equal weight average
        """
        def penalty(self , losses , mt_param : dict = {}): 
            return 0.
    
    class Hybrid(_BaseMultiLossesClass):
        """
        Hybrid of DWA and RUW
        """
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
        """
        dynamic weight average
        https://arxiv.org/pdf/1803.10704.pdf
        https://github.com/lorenmt/mtan/tree/master/im2im_pred
        """
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
        """
        Revised Uncertainty Weighting (RUW) Loss
        https://arxiv.org/pdf/2206.11049v2.pdf (RUW + DWA)
        """
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
        """
        geometric loss strategy , Chennupati etc.(2019)
        """
        def total_loss(self , losses , weight , penalty):
            return losses.pow(weight).prod().pow(1/weight.sum()) + penalty
    class RWS(_BaseMultiLossesClass):
        """
        random weight loss, RW , Lin etc.(2021)
        https://arxiv.org/pdf/2111.10603.pdf
        """
        def weight(self , losses , mt_param : dict = {}): 
            return torch.nn.functional.softmax(torch.rand_like(losses),-1)

    @classmethod
    def view_plot(cls , multi_type = 'ruw'):
        num_task = 2
        if multi_type == 'ruw':
            if num_task > 2 : num_task = 2
            x,y = torch.rand(100,num_task),torch.rand(100,1)
            ls = (x - y).sqrt().sum(dim = 0)
            alpha = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            ruw = cls.RUW(num_task)
            l = torch.stack([torch.stack([ruw(ls,{'alpha':torch.tensor([s1[i,j],s2[i,j]])})[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('alpha-1')
            ax.set_ylabel('alpha-2')
            ax.set_zlabel('loss') #type:ignore
            ax.set_title(f'RUW Loss vs alpha ({num_task}-D)')
        elif multi_type == 'gls':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
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
            l = torch.stack([torch.stack([(torch.tensor([s1[i,j],s2[i,j]])*torch.nn.functional.softmax(torch.rand(num_task),-1)).sum() for j in range(s1.shape[1])]) 
                             for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('rws_loss') #type:ignore
            ax.set_title(f'RWS Loss vs sub-Loss ({num_task}-D)')
        elif multi_type == 'dwa':
            nepoch = 100
            s = np.arange(nepoch)
            l1 = 1 / (4+s) + 0.1 + np.random.rand(nepoch) *0.05
            l2 = 1 / (4+2*s) + 0.15 + np.random.rand(nepoch) *0.03
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