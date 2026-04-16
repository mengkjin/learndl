"""Multi-task loss combination strategies for multi-head NN training.

Strategies available (``name`` argument to MultiHeadLosses):
    'ewa'    — Equal Weight Average (no reweighting)
    'dwa'    — Dynamic Weight Average (Liu et al. 2019)
    'ruw'    — Revised Uncertainty Weighting (Groenendijk et al. 2022)
    'gls'    — Geometric Loss Strategy (Chennupati et al. 2019)
    'rws'    — Random Weight Loss (Lin et al. 2021)
    'hybrid' — Hybrid of DWA epoch weights + RUW uncertainty weights
"""
import torch
from torch import nn , Tensor
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.proj import Logger

class MultiHeadLosses:
    """Orchestrator for multi-task / multi-head loss combination.

    When ``num_head > 1`` and ``name`` is provided, wraps a concrete
    ``BaseMHLCalculator`` strategy that re-weights per-head losses.  When
    disabled (``num_head <= 1`` or ``name=None``), passes the loss tensor
    through unchanged.

    Args:
        num_head: Number of output heads.  ``-1`` or ``1`` disables
                  multi-head combination.
        name:     Strategy key — one of ``'ewa'``, ``'dwa'``, ``'ruw'``,
                  ``'gls'``, ``'rws'``, ``'hybrid'``.
        params:   Hyper-parameters forwarded to the strategy constructor
                  (e.g. ``{'tau': 2.0}`` for DWA).
        mt_param: Default per-step parameters forwarded at call time
                  (e.g. ``{'alpha': module.multiloss_alpha}`` for RUW/Hybrid).
    """
    def __init__(self , num_head = -1 , name : str | None = None , params : dict[str,Any] | None = None , mt_param : dict[str,Any] | None = None , **kwargs):
        """
        example:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt

            ml = MultiHeadLosses(2)
            ml.view_plot(2 , 'dwa')
            ml.view_plot(2 , 'ruw')
            ml.view_plot(2 , 'gls')
            ml.view_plot(2 , 'rws')
        """
        self.num_head = num_head
        self.name = name
        self.params = params or {}
        self.mt_param = mt_param or {}
        self.reset_multi_type()

    @staticmethod
    def add_params(module : nn.Module , num_of_heads : int):
        """Inject a learnable ``multiloss_alpha`` parameter into a model.

        Must be called inside the model's ``__init__`` so that the parameter
        is registered with the optimizer.  Only injects when
        ``num_of_heads > 1``.

        Args:
            module:       The ``nn.Module`` to inject the parameter into.
            num_of_heads: Number of output heads; if ``<= 1`` this is a no-op.
        """
        if num_of_heads > 1:
            assert not hasattr(module , 'multiloss_alpha') , f'{module} already has multiloss_alpha'
            module.multiloss_alpha = nn.Parameter((torch.ones(num_of_heads) + 1e-4).requires_grad_())

    @staticmethod
    def get_params(module : nn.Module | Any):
        """Extract ``multiloss_alpha`` from a module for use as ``mt_param``.

        Returns:
            ``{'alpha': module.multiloss_alpha}`` if the attribute exists,
            otherwise an empty dict.
        """
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
        """Apply the multi-head loss combination strategy.

        Args:
            losses:   Per-head loss tensor of shape ``(num_head,)`` or a dict
                      of named sub-losses passed through when strategy is
                      disabled.
            mt_param: Per-step parameters (e.g. ``{'alpha': ...}`` for
                      RUW/Hybrid).  Merged with the instance-level
                      ``self.mt_param``.

        Returns:
            When strategy is active: a ``dict`` mapping
            ``'multiloss.<name>.head.<i>'`` keys to per-head loss tensors.
            When disabled: returns ``losses`` unchanged.
        """
        if self.multiloss is not None:
            return self.multiloss(losses , self.mt_param | (mt_param or {}) , **kwargs)
        else:
            return losses
    def __bool__(self):
        return self.num_head > 1 and self.name is not None

    @classmethod
    def view_plot(cls , num_head = 2 , multi_type = 'ruw'):
        """Visualize the loss surface of a given multi-head strategy.

        Plotting helper for offline analysis — not used during training.

        Args:
            num_head:   Number of task heads to visualize (2 for 3-D plots).
            multi_type: Strategy to visualize — one of ``'ruw'``, ``'gls'``,
                        ``'rws'``, ``'dwa'``.
        """
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
            loss = torch.stack([torch.stack([torch.Tensor([s1[i,j],s2[i,j]]).prod().sqrt() for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, loss, cmap='viridis') #type:ignore
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('gls_loss') #type:ignore
            ax.set_title(f'GLS Loss vs sub-Loss ({num_head}-D)')
        elif multi_type == 'rws':
            ls = torch.Tensor(np.repeat(np.linspace(0.2, 10, 40),num_head).reshape(-1,num_head))
            _,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            loss = torch.stack([torch.stack([(torch.Tensor([s1[i,j],s2[i,j]])*nn.functional.softmax(torch.rand(num_head),-1)).sum() for j in range(s1.shape[1])]) 
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
    """Abstract base class for multi-head loss calculators.

    Tracks a rolling history of per-head loss tensors (``record_losses``) to
    support epoch-relative weighting strategies like DWA.

    Subclasses override ``weight()`` and/or ``penalty()`` to implement their
    specific reweighting logic.  ``multihead_losses()`` applies the weight and
    records for history; ``total_loss()`` sums the dict for scalar reporting.
    """
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
    """Equal Weight Average — applies no reweighting (all weights = 1)."""

class Hybrid(BaseMHLCalculator):
    """Hybrid of DWA epoch-relative weights and RUW uncertainty weights.

    Requires ``params={'tau': float, 'phi': float}`` and a learnable
    ``alpha`` parameter passed in ``mt_param`` at each step.

    Weight formula: ``DWA_weight + 1/alpha²``
    Penalty: RUW uncertainty penalty + optional L1 constraint on ``|log alpha|``
    """
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
    """Dynamic Weight Average (Liu et al. 2019).

    Weights each head by the relative change in its loss over the last epoch:
    ``w_i = exp(L_{t-1,i} / L_{t-2,i} / tau)``, then normalizes so weights
    sum to ``num_head``.

    Requires ``params={'tau': float}`` (temperature; larger tau → flatter
    weights).

    Reference: https://arxiv.org/pdf/1803.10704.pdf
    """
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
    """Revised Uncertainty Weighting (Groenendijk et al. 2022).

    Weight formula: ``w_i = 1 / alpha_i²`` where ``alpha`` is a learnable
    parameter injected via ``MultiHeadLosses.add_params()``.
    Penalty: ``sum(log(log(alpha²) + 1)) + |phi - sum|log(alpha)||``
    when ``phi`` is provided.

    Requires ``params={'phi': float | None}`` and ``mt_param={'alpha': Tensor}``.

    Reference: https://arxiv.org/pdf/2206.11049v2.pdf
    """
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
    """Geometric Loss Strategy (Chennupati et al. 2019).

    Combines losses as their geometric mean:
    ``(prod(L_i^w_i))^(1/sum(w_i))``

    NOTE: The override method is named ``multi_losses`` but the base class
    method is ``multihead_losses``.  This naming mismatch means ``GLS`` never
    actually overrides the base class and falls back to equal weighting.
    This is a known bug — see ``TODO_res_algo.md``.
    """
    def multi_losses(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        weight = self.weight(losses , mt_param)
        return losses.pow(weight).prod().pow(1 / weight.sum())

class RWS(BaseMHLCalculator):
    """Random Weight Loss (Lin et al. 2021).

    At each step, samples random softmax weights to combine the per-head
    losses, providing implicit multi-task loss diversification.

    Reference: https://arxiv.org/pdf/2111.10603.pdf
    """
    def weight(self , losses : Tensor , mt_param : dict[str,Any] | None = None) -> Tensor:
        return nn.functional.softmax(torch.rand_like(losses),-1)