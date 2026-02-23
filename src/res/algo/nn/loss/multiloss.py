import torch
from torch import nn , Tensor
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.proj import Logger

class MultiHeadLosses:
    '''some realization of multi-losses combine method'''
    def __init__(self , num_head = -1 , name : str | None = None , params : dict[str,Any] | None = None , mt_param : dict[str,Any] | None = None , **kwargs):
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
        self.name = name
        self.params = params or {}
        self.mt_param = mt_param or {}
        self.reset_multi_type()

    @staticmethod
    def add_params(module : nn.Module , num_of_heads : int):
        if num_of_heads > 1:
            assert not hasattr(module , 'multiloss_alpha') , f'{module} already has multiloss_alpha'
            module.multiloss_alpha = nn.Parameter((torch.ones(num_of_heads) + 1e-4).requires_grad_())

    @staticmethod
    def get_params(module : nn.Module | Any):
        if isinstance(module , nn.Module) and hasattr(module , 'multiloss_alpha'):
            return {'alpha':module.multiloss_alpha}
        else:
            return {}

    def reset_multi_type(self):
        if self.num_head > 1 and self.name is not None: 
            self.multiloss = {
                'ewa':EWA,
                'hybrid':Hybrid,
                'dwa':DWA,
                'ruw':RUW,
                'gls':GLS,
                'rws':RWS,
            }[self.name](self.num_head , **self.params)
        else:
            self.multiloss = None
        return self
    
    def __call__(self , losses : Tensor | dict[str,Tensor] , mt_param : dict[str,Any] | None = None , **kwargs) -> Tensor | dict[str,Tensor]: 
        '''calculate combine loss of multiple output losses'''
        if self.multiloss is not None:
            return self.multiloss(losses , self.mt_param | (mt_param or {}) , **kwargs)
        else:
            return losses
    def __bool__(self):
        return self.num_head > 1 and self.name is not None

    @classmethod
    def view_plot(cls , num_head = 2 , multi_type = 'ruw'):
        if multi_type == 'ruw':
            num_head = min(num_head, 2)
            x,y = torch.rand(100,num_head),torch.rand(100,1)
            ls = (x - y).sqrt().sum(dim = 0)
            alpha = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            _,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            ruw = RUW(num_head)
            loss = torch.stack([torch.stack([ruw.total_loss(ls,{'alpha':Tensor([s1[i,j],s2[i,j]])})[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, loss, cmap='viridis') #type:ignore
            ax.set_xlabel('alpha-1')
            ax.set_ylabel('alpha-2')
            ax.set_zlabel('loss') #type:ignore
            ax.set_title(f'RUW Loss vs alpha ({num_head}-D)')
        elif multi_type == 'gls':
            ls = Tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            _,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            loss = torch.stack([torch.stack([torch.tensor([s1[i,j],s2[i,j]]).prod().sqrt() for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, loss, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('gls_loss') #type:ignore
            ax.set_title(f'GLS Loss vs sub-Loss ({num_head}-D)')
        elif multi_type == 'rws':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            _,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            loss = torch.stack([torch.stack([(torch.tensor([s1[i,j],s2[i,j]])*nn.functional.softmax(torch.rand(num_head),-1)).sum() for j in range(s1.shape[1])]) 
                             for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, loss, cmap='viridis') #type:ignore
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
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
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
            Logger.error(f'Unknow multi_type : {multi_type}')
        plt.show()
        #plt.close(fig)

class BaseMHLCalculator():
    '''base class of multi_head_losses class'''
    def __init__(self , num_head : int , **kwargs):
        self.num_head = num_head
        self.record_num = 0 
        self.record_losses : list[Tensor] = []
        self.kwargs = kwargs
        self.reset(**kwargs)
    def __call__(self , losses : Tensor , mt_param : dict[str,Any] | None = None , **kwargs) -> dict[str,Tensor]:
        multihead_losses = self.multihead_losses(losses , mt_param or {})
        penalty = self.penalty(losses , mt_param)
        losses_dict = {
            f'multiloss.{self.name}.head.{i}':multihead_losses[i] for i in range(self.num_head)
        }
        if penalty is not None:
            losses_dict[f'multiloss.{self.name}.penalty'] = penalty
        return losses_dict
        
    def reset(self , **kwargs): ...
    def record_loss(self , losses : Tensor):
        self.record_num += 1
        self.record_losses.append(losses.detach())
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        return torch.ones_like(losses)
    def penalty(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor | None: 
        return None
    def multihead_losses(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        losses = losses * self.weight(losses , mt_param)
        self.record_loss(losses)
        return losses
    def total_loss(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        multi_losses = self(losses , mt_param)
        return torch.concatenate([v.detach().flatten() for v in multi_losses.values()]).sum()
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

class EWA(BaseMHLCalculator):
    '''Equal weight average'''

class Hybrid(BaseMHLCalculator):
    '''Hybrid of DWA and RUW'''
    def reset(self , **kwargs):
        self.tau : float = kwargs['tau']
        self.phi : float = kwargs['phi']
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        assert mt_param is not None , f'mt_param should be provided'
        alpha : Tensor = mt_param['alpha']
        if self.record_num < 2:
            weight = torch.ones_like(losses)
        else:
            weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
            weight = weight / weight.sum() * weight.numel()
        return weight + 1 / alpha.square()
    def penalty(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor: 
        assert mt_param is not None , f'mt_param should be provided'
        alpha : Tensor = mt_param['alpha']
        penalty = (alpha.log().square()+1).log().sum()
        if self.phi is not None: 
            penalty = penalty + (self.phi - alpha.log().abs().sum()).abs()
        return penalty

class DWA(BaseMHLCalculator):
    '''dynamic weight average , https://arxiv.org/pdf/1803.10704.pdf'''
    def reset(self , **kwargs):
        self.tau : float = kwargs['tau']
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        if self.record_num < 2:
            weight = torch.ones_like(losses)
        else:
            weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
            weight = weight / weight.sum() * weight.numel()
        return weight
    
class RUW(BaseMHLCalculator):
    '''Revised Uncertainty Weighting (RUW) Loss , https://arxiv.org/pdf/2206.11049v2.pdf (RUW + DWA)'''
    def reset(self , **kwargs):
        self.phi : float = kwargs['phi']
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        assert mt_param is not None , f'mt_param should be provided'
        alpha : Tensor = mt_param['alpha']
        return 1 / alpha.square()
    def penalty(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor: 
        assert mt_param is not None , f'mt_param should be provided'
        alpha : Tensor = mt_param['alpha']
        penalty = (alpha.log().square()+1).log().sum()
        if self.phi is not None: 
            penalty = penalty + (self.phi - alpha.log().abs().sum()).abs()
        return penalty

class GLS(BaseMHLCalculator):
    '''geometric loss strategy , Chennupati etc.(2019)'''
    def multi_losses(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        weight = self.weight(losses , mt_param)
        return losses.pow(weight).prod().pow(1 / weight.sum())
class RWS(BaseMHLCalculator):
    '''random weight loss, RW , Lin etc.(2021) , https://arxiv.org/pdf/2111.10603.pdf'''
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor: 
        return nn.functional.softmax(torch.rand_like(losses),-1)