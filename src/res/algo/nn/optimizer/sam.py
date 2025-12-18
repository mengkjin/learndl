#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# File    :   sharpness_aware_minimization.py
# Time    :   2023/07/04 14:37:18
# Author  :   Pu Yanheng
'''

# here put the import lib
import contextlib
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed

from src.proj import Logger

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            setattr(module, 'backup_momentum', module.momentum)
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = getattr(module, 'backup_momentum')
    model.apply(_enable)

class SAM(optim.Optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), \
            f"base_optimizer must be an `Optimizer`, but got {type(base_optimizer)}"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        super(SAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step()
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2
        )
        return norm


class SSAMF(SAM):

    def __init__(
        self,
        params,
        base_optimizer,
        device,
        rho=0.05,
        sparsity=0.5,
        num_samples=64,
        update_freq=1,
        **kwargs
    ) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), \
            f"base_optimizer must be an `Optimizer`, but got {type(base_optimizer)}"
        self.base_optimizer = base_optimizer
        self.device = device

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 1.0 <= num_samples, f"num_samples should be greater than 1: {num_samples}"
        assert 1.0 <= update_freq, f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        super(SSAMF, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["num_samples"] = num_samples
            group["update_freq"] = update_freq

        self.init_mask()

    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).to(p)

    @torch.no_grad()
    def update_mask(self, model, train_data, **kwargs):
        fisher_value_dict = {}
        fisher_mask_dict = {}
        for group in self.param_groups:
            for p in group['params']:
                fisher_value_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)
                fisher_mask_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)

        sampled_idxs = np.random.choice(
            range(len(train_data)),
            size=self.num_samples,
            replace=False
        )

        # cal fisher value
        criterion = torch.nn.CrossEntropyLoss()
        with torch.enable_grad():
            for idx, sample_idx in enumerate(sampled_idxs):
                if idx % (self.num_samples // 10) == 0:
                    Logger.stdout(f'Updating Mask: [{idx}/{self.num_samples}]..')

                (feature, label) = train_data.dataset[sample_idx]
                feature = torch.from_numpy(feature).to(self.device).float()
                label = torch.from_numpy(label).to(self.device).float()

                output = model(feature)
                loss = criterion(output, label.squeeze())
                loss.backward()

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        fisher_value_dict[id(p)] += torch.square(p.grad).data
                model.zero_grad()

        # topk fisher value
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])

        keep_num = int(len(fisher_value_list) * (1 - self.sparsity))
        _value, _index = torch.topk(fisher_value_list, keep_num)

        mask_list = torch.zeros_like(fisher_value_list)
        mask_list.scatter_(0, _index, torch.ones_like(_value))

        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index:start_index +
                                                  p.numel()].reshape(p.shape)
                self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0  , self.state[p]['mask'].max()
                assert self.state[p]['mask'].min() >= 0.0 , self.state[p]['mask'].min()
        assert start_index == len(mask_list) , (start_index , len(mask_list))

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, model=None, epoch=None, batch_idx=None, train_data=None, **kwargs):
        super().step(closure, **kwargs)
        assert model is not None , f'{self.__class__.__name__} model is None'
        assert train_data is not None , f'{self.__class__.__name__} train_data is None'
        assert epoch is not None , f'{self.__class__.__name__} epoch is None'
        assert batch_idx is not None , f'{self.__class__.__name__} batch_idx is None'
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            Logger.stdout('\nUpdate Mask!')
            self.update_mask(model, train_data)
            Logger.stdout(f'Mask Lived Weight: {self.mask_info():.4f}')

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item()
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num


class ASAM(optim.Optimizer):

    def __init__(self, params, optimizer, model, rho=0.5, eta=0.01, **kwargs):

        assert isinstance(optimizer, torch.optim.Optimizer), "base_optimizer must be an `Optimizer`"
        self.optimizer = optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0 <= eta, f"eta should be non-negative:{eta}"
        self.rho = rho
        self.eta = eta
        super(ASAM, self).__init__(params, dict(rho=rho, eta=eta))

        self.param_groups = self.optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["eta"] = eta

        self.model = model
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "ASAM requires closure, which is not provided."

        self.ascent_step()
        with torch.enable_grad():
            closure()
        self.descent_step()


class GSAM(optim.Optimizer):

    def __init__(
        self,
        params,
        base_optimizer,
        model,
        gsam_alpha=0.2,
        rho_t=0.05,
        adaptive=False,
        perturb_eps=1e-12,
        grad_reduce='mean',
        **kwargs
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha
        self.rho_t = rho_t

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                inner_prod += torch.sum(self.state[p]['old_g'] * p.grad.data)

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.data.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack(
                    [
                        ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack(
                    [
                        ((torch.abs(p.data) if weight_adaptive else 1.0) *
                         self.state[p][by]).norm(p=2)
                        for group in self.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ]
                ),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None, **kwargs):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            # outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # disable running stats for second pass
            disable_running_stats(self.model)

            # get gradient at perturbed weights
            with torch.enable_grad():
                get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)


class GAM(optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer,
        model,
        adaptive=False,
        perturb_eps=1e-12,
        args={
            'grad_rho': 0.02,
            'grad_norm_rho': 0.2,
            'grad_beta_1': 1.,
            'grad_beta_2': -1.,
            'grad_beta_3': 1.,
            'grad_gamma': 0.03,
        },
        grad_reduce='mean',
        **kwargs
    ):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GAM, self).__init__(params, defaults)
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.args = args
        self.get_grad_reduce(grad_reduce)

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.args['grad_rho'] / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.args['grad_norm_rho'] / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):

        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_decompose(self, args={
            'grad_rho': 0.02,
            'grad_norm_rho': 0.2,
            'grad_beta_1': 1.,
            'grad_beta_2': -1.,
            'grad_beta_3': 1.,
            'grad_gamma': 0.03,
        }):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # update the weighted sum of grads
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(args['grad_beta_2']
                                                                   ) * self.state[p]['g_2']
                p.grad.data = args['grad_beta_1'] * self.state[p]["g_1"] + args[
                    'grad_beta_3'] * p.grad.data.detach().clone()
                inner_prod += torch.sum(self.state[p]['pro_m'] * p.grad.data)

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                    new_grad_norm + self.perturb_eps
                )
                p.grad.data.add_(vertical, alpha=-args['grad_gamma'])

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data))**2)
                else:
                    norm += torch.sum(g**2)

        return torch.sqrt(norm) if isinstance(norm, torch.Tensor) else torch.tensor(norm).sqrt()

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def step(self, closure=None, **kwargs):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            # outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            disable_running_stats(self.model)
            # model 1
            with torch.enable_grad():
                get_grad()
            # grad 1

            self.unperturb(perturb_key="e_w_0")
            # model 0
            self.grad_norm_ascent()
            # model 2
            with torch.enable_grad():
                get_grad()
            # grad 2

            self.perturb_weights(perturb_idx=1)
            # model 3
            with torch.enable_grad():
                get_grad()
            # grad 3
            # decompose and get new update direction
            self.gradient_decompose(args=self.args)

            # unperturb
            self.unperturb(perturb_key="e_w_1_2")

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'


class FriendlySAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        base_optimizer,
        rho=0.05,
        sigma=1,
        lmbda=0.9,
        adaptive=False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FriendlySAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        self.adaptive = adaptive
        self.rho = rho
        Logger.stdout('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                if "momentum" not in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = self.state[p][
                        "momentum"] * self.lmbda + grad * (1 - self.lmbda)

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self, weight_adaptive=False):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if weight_adaptive else 1.0) * p.grad).norm(p=2
                                                                              ).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
