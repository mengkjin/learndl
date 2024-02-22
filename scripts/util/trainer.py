import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from .logger import *

use_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Device:
    def __init__(self , device = None) -> None:
        if device is None: device = use_device
        self.device = device
    def __call__(self, *args):
        if len(args) == 0: 
            return None
        elif len(args) == 1:
            return self._to(args[0])
        else:
            return self._to(args)
    def _to(self , x):
        if isinstance(x , (list,tuple)):
            return type(x)(self._to(v) for v in x)
        elif isinstance(x , (dict)):
            for k in x.keys(): x[k] = self._to(x[k])
            return x
        elif isinstance(x , (torch.Tensor , torch.nn.Module , torch.nn.ModuleList , torch.nn.ModuleDict)): # maybe modulelist ... should be included
            return x.to(self.device)
        else:
            return x
    @classmethod
    def cpu(cls , x):
        if isinstance(x , (list,tuple)):
            return type(x)(cls.cpu(v) for v in x)
        elif isinstance(x , (dict)):
            return {k:cls.cpu(v) for k,v in x.items()}
        elif isinstance(x , (torch.Tensor , torch.nn.Module , torch.nn.ModuleList , torch.nn.ModuleDict)): # maybe modulelist ... should be included
            return x.cpu()
        else:
            return x
    def torch_nans(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs).fill_(torch.nan)
    def torch_zeros(self,*args , **kwargs):
        return torch.zeros(*args , device = self.device , **kwargs)
    def torch_ones(self,*args,**kwargs):
        return torch.ones(*args , device = self.device , **kwargs)
    def torch_arange(self,*args,**kwargs):
        return torch.arange(*args , device = self.device , **kwargs)
    def print_cuda_memory(self):
        print(f'Allocated {torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G, '+\
              f'Reserved {torch.cuda.memory_reserved(self.device) / 1024**3:.1f}G')
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
        def weight(self , losses , mt_param):
            return torch.ones_like(losses)
        def penalty(self , losses , mt_param): 
            return 0.
        def total_loss(self , losses , weight , penalty):
            return (losses * weight).sum() + penalty
    
    class EWA(_BaseMultiLossesClass):
        """
        Equal weight average
        """
        def penalty(self , losses , mt_param): 
            return 0.
    
    class Hybrid(_BaseMultiLossesClass):
        """
        Hybrid of DWA and RUW
        """
        def reset(self , **kwargs):
            self.tau = kwargs['tau']
            self.phi = kwargs['phi']
        def weight(self , losses , mt_param):
            if self.record_num < 2:
                weight = torch.ones_like(losses)
            else:
                weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
                weight = weight / weight.sum() * weight.numel()
            return weight + 1 / mt_param['alpha'].square()
        def penalty(self , losses , mt_param): 
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
        def weight(self , losses , mt_param):
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
        def weight(self , losses , mt_param):
            return 1 / mt_param['alpha'].square()
        def penalty(self , losses , mt_param): 
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
        def weight(self , losses , mt_param): 
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

class CosineScheduler:
    def __init__(self , optimizer , warmup_stage = 10 , anneal_stage = 40 , initial_lr_div = 10 , final_lr_div = 1e4):
        self.warmup_stage= warmup_stage
        self.anneal_stage= anneal_stage
        self.optimizer = optimizer
        self.base_lrs = [x['lr'] for x in optimizer.param_groups]
        self.initial_lr= [x / initial_lr_div for x in self.base_lrs]
        self.final_lr= [x / final_lr_div for x in self.base_lrs]
        self.last_epoch = 0
        self._step_count= 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
        self._last_lr= self.initial_lr
        
    def get_last_lr(self):
        #Return last computed learning rate by current scheduler.
        return self._last_lr

    def state_dict(self):
        #Returns the state of the scheduler as a dict.
        return self.__dict__
    
    def step(self):
        self.last_epoch += 1
        if self._step_count <= self.warmup_stage:
            self._last_lr = [y+(x-y)*self._linear_phase for x,y in zip(self.base_lrs,self.initial_lr)]
        elif self._step_count <= self.warmup_stage + self.anneal_stage:
            self._last_lr = [y+(x-y)*math.cos(self._cos_phase) for x,y in zip(self.base_lrs,self.final_lr)]
        else:
            self._last_lr = self.final_lr
        for x , param_group in zip(self._last_lr,self.optimizer.param_groups):
            param_group['lr'] = x
        self._step_count += 1
        self._linear_phase = self._step_count / self.warmup_stage
        self._cos_phase = math.pi / 2 * (self._step_count - self.warmup_stage) / self.anneal_stage
                
class CustomBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler , batch_size_list , drop_res = True):
        self.sampler = sampler
        self.batch_size_list = np.array(batch_size_list).astype(int)
        assert (self.batch_size_list >= 0).all()
        self.drop_res = drop_res
        
    def __iter__(self):
        if (not self.drop_res) and (sum(self.batch_size_list) < len(self.sampler)):
            new_list = np.append(self.batch_size_list , len(self.sampler) - sum(self.batch_size_list))
        else:
            new_list = self.batch_size_list
        
        batch_count , sample_idx = 0 , 0
        while batch_count < len(new_list):
            if new_list[batch_count] > 0:
                batch = [0] * new_list[batch_count]
                idx_in_batch = 0
                while True:
                    batch[idx_in_batch] = self.sampler[sample_idx]
                    idx_in_batch += 1
                    sample_idx +=1
                    if idx_in_batch == new_list[batch_count]:
                        yield batch
                        break
            batch_count += 1
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

    def __len__(self):
        if self.batch_size_list.sum() < len(self.sampler):
            return len(self.batch_size_list) + 1 - self.drop_res
        else:
            return np.where(self.batch_size_list.cumsum() >= len(self.sampler))[0][0] + 1