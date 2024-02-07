import torch
from .basic import mse,pearson,ccc,spearman

_metric_function = {
    'mse':mse ,
    'pearson':pearson,
    'ccc':ccc,
    'spearman':spearman,
}
_display_check  = True # will display once if true
_display_record = {'loss' : {} , 'score' : {} , 'penalty' : {}}

def decorator_display(func , mtype , mkey):
    def metric_display(mtype , mkey):
        if not _display_record[mtype].get(mkey , False):
            print(f'{mtype} function of [{mkey}] calculated and success!')
            _display_record[mtype][mkey] = True
    def wrapper(*args, **kwargs):
        v = func(*args, **kwargs)
        metric_display(mtype , mkey)
        return v
    return wrapper if _display_check else func

def loss_function(key):
    """
    loss function , pearson/ccc should * -1.
    """
    assert key in ('mse' , 'pearson' , 'ccc')
    def decorator(func , key):
        def wrapper(*args, **kwargs):
            v = func(*args, **kwargs)
            if key != 'mse': v = torch.exp(-v)
            return v
        return wrapper
    new_func = decorator(_metric_function[key] , key)
    new_func = decorator_display(new_func , 'loss' , key)
    return new_func

def score_function(key):
    assert key in ('mse' , 'pearson' , 'ccc' , 'spearman')
    def decorator(func , key):
        def wrapper(*args, **kwargs):
            with torch.no_grad():
                v = func(*args, **kwargs)
            if key == 'mse' : v = -v
            return v
        return wrapper
    new_func = decorator(_metric_function[key] , key)
    new_func = decorator_display(new_func , 'score' , key)
    return new_func
    
def penalty_function(key , param):
    def null_penalty(**kwargs):
        return 0.
    def hidden_orthogonality(**kwargs):
        hidden = kwargs['hidden']
        if hidden.shape[-1] == 1: return 0
        if isinstance(hidden,(tuple,list)): hidden = torch.cat(hidden,dim=-1)
        pen = hidden.T.corrcoef().triu(1).nan_to_num().square().sum()
        return pen
    def tra_ot_penalty(**kwargs):
        net = kwargs['net']
        pen = 0.
        if net.training and net.probs is not None and net.num_states > 1:
            square_error = (kwargs['hidden'] - kwargs['label']).square()
            square_error -= square_error.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            P = sinkhorn(-square_error, epsilon=0.01)  # sample assignment matrix
            lamb = (param['rho'] ** net.global_steps)
            reg = (net.probs + 1e-4).log().mul(P).sum(dim=-1).mean()
            net.global_steps += 1
            pen = - lamb * reg
        return pen
    new_func = locals().get(key , null_penalty)
    new_func = decorator_display(new_func , 'penalty' , key)
    return {'lamb': param['lamb'] , 'cond' : True , 'func' : new_func}

def shoot_infs(inp_tensor):
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

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q