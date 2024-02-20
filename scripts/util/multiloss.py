import torch
import numpy as np
import matplotlib.pyplot as plt
from ..function.basic import *
        
class multiloss_calculator:
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
            self.multi_class = self.multi_class_dict()[self.multi_type](num_task , **kwargs)
        return self
    
    def calculate_multi_loss(self , losses , mt_param , **kwargs):
        return self.multi_class(losses , mt_param)
    
    """
    def reset_loss_function(self, loss_type):
        self.loss_type = loss_type
        if isinstance(self.loss_type , (list,tuple)):
            # various loss function version, tasks can be of the same output but different loss function
            self.loss_functions = [loss_function(k) for k in self.loss_type]
        else:
            self.loss_functions = [loss_function(self.loss_type) for _ in range(self.num_task)]
            
    def losses(self , y , x , **kwargs):
        return torch.tensor([f(y[i] , x[i] , **kwargs) for i,f in enumerate(self.loss_functions)])
    
    def calculate_multi_loss(self , y , x , **kwargs):
        sub_losses = self.losses(y , x , **kwargs)
        multi_loss = self.multi_class.total_loss(sub_losses)
        return multi_loss , sub_losses
    """
    def multi_class_dict(self):
        return {
            'ewa':self.EWA,
            'hybrid':self.Hybrid,
            'dwa':self.DWA,
            'ruw':self.RUW,
            'gls':self.GLS,
            'rws':self.RWS,
        }
    
    class _base_class():
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
    
    class EWA(_base_class):
        """
        Equal weight average
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
    
    class Hybrid(_base_class):
        """
        Hybrid of DWA and RUW
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
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
    
    class DWA(_base_class):
        """
        dynamic weight average
        https://arxiv.org/pdf/1803.10704.pdf
        https://github.com/lorenmt/mtan/tree/master/im2im_pred
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
        def reset(self , **kwargs):
            self.tau = kwargs['tau']
        def weight(self , losses , mt_param):
            if self.record_num < 2:
                weight = torch.ones_like(losses)
            else:
                weight = (self.record_losses[-1] / self.record_losses[-2] / self.tau).exp()
                weight = weight / weight.sum() * weight.numel()
            return weight
        
    class RUW(_base_class):
        """
        Revised Uncertainty Weighting (RUW) Loss
        https://arxiv.org/pdf/2206.11049v2.pdf (RUW + DWA)
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
        def reset(self , **kwargs):
            self.phi = kwargs['phi']
        def weight(self , losses , mt_param):
            return 1 / mt_param['alpha'].square()
        def penalty(self , losses , mt_param): 
            penalty = (mt_param['alpha'].log().square()+1).log().sum()
            if self.phi is not None: 
                penalty = penalty + (self.phi - mt_param['alpha'].log().abs().sum()).abs()
            return penalty

    class GLS(_base_class):
        """
        geometric loss strategy , Chennupati etc.(2019)
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
        def total_loss(self , losses , weight , penalty):
            return losses.pow(weight).prod().pow(1/weight.sum()) + penalty
    
    class RWS(_base_class):
        """
        random weight loss, RW , Lin etc.(2021)
        https://arxiv.org/pdf/2111.10603.pdf
        """
        def __init__(self , num_task , **kwargs):
            super().__init__(num_task , **kwargs)
        def weight(self , losses , mt_param): 
            return torch.nn.functional.softmax(torch.rand_like(losses),-1)

    def view_plot(self , multi_type = 'ruw'):
        num_task = 2
        if multi_type == 'ruw':
            if num_task > 2 : num_task = 2
            x,y = torch.rand(100,num_task),torch.rand(100,1)
            ls = (x - y).sqrt().sum(dim = 0)
            alpha = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            ruw = self.RUW(num_task)
            l = torch.stack([torch.stack([ruw(ls,{'alpha':torch.tensor([s1[i,j],s2[i,j]])})[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis')
            ax.set_xlabel('alpha-1')
            ax.set_ylabel('alpha-2')
            ax.set_zlabel('loss')
            ax.set_title(f'RUW Loss vs alpha ({num_task}-D)')
        elif multi_type == 'gls':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([torch.tensor([s1[i,j],s2[i,j]]).prod().sqrt() for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis')
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('gls_loss')
            ax.set_title(f'GLS Loss vs sub-Loss ({num_task}-D)')
        elif multi_type == 'rws':
            ls = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(ls[:,0].numpy(), ls[:,1].numpy())
            l = torch.stack([torch.stack([(torch.tensor([s1[i,j],s2[i,j]])*torch.nn.functional.softmax(torch.rand(num_task),-1)).sum() for j in range(s1.shape[1])]) 
                             for i in range(s1.shape[0])]).numpy()
            ax.plot_surface(s1, s2, l, cmap='viridis')
            ax.set_xlabel('loss-1')
            ax.set_ylabel('loss-2')
            ax.set_zlabel('rws_loss')
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