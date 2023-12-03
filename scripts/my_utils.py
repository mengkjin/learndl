import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import gc
from torch.utils.data.dataset import IterableDataset , Dataset
from mpl_toolkits import mplot3d
from copy import deepcopy
from function import *

__all__ = [
    'FilteredIterator' ,
    'lr_cosine_scheduler' , # learn rate scheduler
    'Mydataset' , 'MyIterdataset' , 'Mydataloader_basic' , 'Mydataloader_saved' , 'DesireBatchSampler' , # dataset , dataloader , sampler
    'multiloss_calculator' , 'versatile_storage'
]    

class FilteredIterator:
    def __init__(self, iterable, condition):
        self.iterable = iterable
        self.condition = condition

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            item = next(self.iterable)
            cond = self.condition(item) if callable(self.condition) else next(self.condition)
            if cond: return item

class lr_cosine_scheduler:
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
        
class Mydataset(Dataset):
    def __init__(self, data1 , label , weight = None) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
            self.weight = weight
    def __len__(self):
        return len(self.data1)
    def __getitem__(self , ii):
        if self.weight is None:
            return self.data1[ii], self.label[ii]
        else:
            return self.data1[ii], self.label[ii], self.weight[ii]

class MyIterdataset(IterableDataset):
    def __init__(self, data1 , label) -> None:
            super().__init__()
            self.data1 = data1
            self.label = label
    def __len__(self):
        return len(self.data1)
    def __iter__(self):
        for ii in range(len(self.data1)):
            yield self.data1[ii], self.label[ii]
            
class Mydataloader_basic:
    def __init__(self, x_set , y_set , batch_size = 1, num_worker = 0, set_name = '', batch_num = None):
        self.dataset = Mydataset(x_set, y_set)
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.dataloader = torch.utils.data.DataLoader(self.dataset , batch_size = batch_size , num_workers = num_worker)
        self.set_name = set_name
        self.batch_num = math.ceil(len(y_set)/batch_size)
    def __iter__(self):
        for d in self.dataloader: 
            yield d

class Mydataloader_saved:
    def __init__(self, set_name , batch_num , batch_folder):
        self.set_name = set_name
        self.batch_num = batch_num
        self.batch_folder = batch_folder
        self.batch_path = [f'{self.batch_folder}/{self.set_name}.{ii}.pt' for ii in range(self.batch_num)]
    def __iter__(self):
        for ii in range(self.batch_num): 
            yield torch.load(self.batch_path[ii])
                
class DesireBatchSampler(torch.utils.data.Sampler):
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
        
class multiloss_calculator:
    def __init__(self , multi_type = None):
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
        
    def reset_multi_type(self, num_task , **kwargs):
        self.num_task   = num_task
        self.multi_class = self.multi_class_dict()[self.multi_type](num_task , **kwargs)
    
    def calculate_multi_loss(self , losses , mt_param , **kwargs):
        return self.multi_class.forward(losses , mt_param)
    
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
        def reset(self , **kwargs):
            pass
        def record(self , losses , weight , penalty):
            self.record_num += 1
            self.record_losses.append(losses.detach() if isinstance(losses,torch.Tensor) else losses)
            self.record_weight.append(weight.detach() if isinstance(weight,torch.Tensor) else weight)
            self.record_penalty.append(penalty.detach() if isinstance(penalty,torch.Tensor) else penalty)
        def forward(self , losses , mt_param , **kwargs):
            weight , penalty = self.weight(losses , mt_param) , self.penalty(losses , mt_param)
            self.record(losses , weight , penalty)
            return self.total_loss(losses , weight , penalty)
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
            alpha = torch.tensor(np.repeat(np.linspace(0.2, 10, 40),num_task).reshape(-1,num_task))
            fig,ax = plt.figure(),plt.axes(projection='3d')
            s1, s2 = np.meshgrid(alpha[:,0].numpy(), alpha[:,1].numpy())
            l = torch.stack([torch.stack([self.RUW(y,x,[s1[i,j],s2[i,j]])[0] for j in range(s1.shape[1])]) for i in range(s1.shape[0])]).numpy()
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
        
class versatile_storage():
    def __init__(self , *args):
        if len(args) > 0 and args[0] in ['disk' , 'mem']:
            self.activate(args[0])

    def activate(self , default = 'mem'):
        assert default in ['disk' , 'mem']
        self.default = default
        self.mem_disk = dict()
        self.file_record = list()
        self.file_group = dict()
    
    def save(self , obj , paths , to_disk = False , group = 'default'):
        for p in self._pathlist(paths): 
            self._saveone(obj , p , self.default == 'disk' or to_disk)
            self._addrecord(p , group)
            
    def load(self , path , from_disk = False):
        return torch.load(path) if self.default == 'disk' or from_disk else self.mem_disk[path]

    def _pathlist(self , p):
        if p is None: return []
        return [p] if isinstance(p , str) else p
    
    def _saveone(self , obj , p , to_disk = False):
        if to_disk:
            torch.save(obj , p)
        else:
            self.mem_disk[p] = deepcopy(obj)
    
    def _addrecord(self , p , group):
        self.file_record = np.union1d(self.file_record , p)
        if group not in self.file_group.keys(): 
            self.file_group[group] = [p]
        else:
            self.file_group[group] = np.union1d(self.file_group[group] , [p])
    
    def save_model_state(self , model , paths , to_disk = False , group = 'default'):
        sd = model.state_dict() if (self.default == 'disk' or to_disk) else deepcopy(model).cpu().state_dict()
        self.save(sd , paths , to_disk , group)
        
    def load_model_state(self , model , path , from_disk = False):
        sd = self.load(path , from_disk)
        model.load_state_dict(sd)
        return model
            
    def valid_paths(self , paths):
        return np.intersect1d(self._pathlist(paths) ,  self.file_record).tolist()
    
    def del_path(self , *args):
        for paths in args:
            if self.default == 'disk':
                [os.remove(p) for p in self._pathlist(paths) if os.path.exists(p)]
            else:
                [self.mem_disk.__delitem__(p) for p in np.intersect1d(self._pathlist(paths) , list(self.mem_disk.keys()))]
            self.file_record = np.setdiff1d(self.file_record , paths)
        gc.collect()
        
    def del_group(self , clear_groups = []):
        for g in self._pathlist(clear_groups):
            paths = self.file_group.get(g)
            if paths is not None:
                self.del_path(paths)
                del self.file_group[g]