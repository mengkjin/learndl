import torch

from typing import Literal
from . import basic as B

class Metrics:
    display_check  = True # will display once if true
    display_record = {'loss' : {} , 'score' : {} , 'penalty' : {}}
    default_metric = {'mse':B.mse ,'pearson':B.pearson,'ccc':B.ccc,'spearman':B.spearman,}

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
        nanpos = True
        for arg in args:
            if arg is not None: nanpos = arg.isnan() + nanpos
        if isinstance(nanpos , torch.Tensor) and arg.any():
            nanpos = nanpos.sum(tuple(range(1 , nanpos.ndim))) > 0
            new_args = [None if arg is None else arg[~nanpos] for arg in args]
        else:
            new_args = args
        return new_args

    @classmethod
    def loss(cls , key):
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
    def score(cls , key):
        assert key in ('mse' , 'pearson' , 'ccc' , 'spearman')
        def decorator(func):
            def wrapper(label,pred,weight=None,dim=0,nan_check=False,first_col=False,**kwargs):
                with torch.no_grad():
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
    def penalty(cls , key , param):
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