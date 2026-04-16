"""SAM-family sharpness-aware minimizers.

All optimizers in this module wrap a base optimizer and add an extra forward-
backward pass to perturb parameters toward sharper loss regions, then update
from the perturbed gradient.  This improves generalization by seeking flat
minima.

Helper utilities:
    disable_running_stats / enable_running_stats — freeze BatchNorm momentum
    during the second forward pass to avoid polluting running statistics.
"""

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
    """Freeze BatchNorm running statistics by setting momentum to 0.

    Called before the second (perturbed) forward pass in SAM variants so that
    the perturbed pass does not update the BatchNorm running mean/variance.
    The original momentum is saved as ``backup_momentum`` and restored by
    :func:`enable_running_stats`.
    """
    def _disable(module):
        if isinstance(module, _BatchNorm):
            setattr(module, 'backup_momentum', module.momentum)
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    """Restore BatchNorm momentum from ``backup_momentum`` after the SAM step.

    Must be called after the SAM update (i.e. after the second forward/backward
    pass) to resume normal BatchNorm tracking.
    """
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, 'backup_momentum'):
            module.momentum = getattr(module, 'backup_momentum')
    model.apply(_enable)

class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (SAM).

    Two-step optimization loop:
    1. ``first_step``  — perturbs weights to the local loss maximum
       ``w + e(w)`` where ``e(w) = rho * grad / ||grad||₂``
    2. ``second_step`` — reverts to ``w`` and performs the true gradient step
       using the perturbed gradient.

    Usage requires a closure for the second forward-backward pass::

        optimizer.first_step(zero_grad=True)
        loss = model(inputs)
        loss.backward()
        optimizer.second_step(zero_grad=True)

    Args:
        params:         Model parameters (same as any ``optim.Optimizer``).
        base_optimizer: Instantiated inner optimizer (e.g. ``torch.optim.SGD``).
        rho:            Neighborhood size for perturbation (default ``0.05``).

    Reference: Foret et al. (2021) "Sharpness-Aware Minimization for
    Efficiently Improving Generalization."
    """
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
        """Perturb parameters to the local loss maximum ``w + e(w)``.

        Computes perturbation ``e(w) = rho * grad / ||grad||₂`` and adds it to
        each parameter.  Saves ``e_w`` in optimizer state for reverting later.

        Args:
            zero_grad: If ``True``, call ``self.zero_grad()`` after perturbing.
        """
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
        """Revert perturbation and perform the base optimizer step.

        Subtracts the saved ``e_w`` from each parameter (returning to ``w``),
        then calls ``base_optimizer.step()`` with the current gradients.

        Args:
            zero_grad: If ``True``, call ``self.zero_grad()`` after the step.
        """
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
        """Perform a full SAM step using a closure for the second forward-backward pass.

        Args:
            closure: A callable that performs a forward and backward pass and
                     returns the loss.  Required for SAM.
        """
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step()
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        """Compute the L2 gradient norm across all parameter groups.

        Places all per-parameter norms on a shared device to handle model
        parallelism safely.
        """
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
    """Sparse SAM with Fisher Information mask (SSAM-F).

    Extends SAM by masking the perturbation to the top-k most important
    parameters as measured by their Fisher Information (squared gradient).
    Reduces compute and can improve generalization by focusing sharpness
    reduction on the most sensitive parameters.

    Args:
        sparsity:    Fraction of parameters to *mask out* (set perturbation
                     to zero).  Default ``0.5`` keeps top 50%.
        num_samples: Number of training samples used to estimate Fisher
                     Information when ``update_mask`` is called.
        update_freq: Epoch frequency at which to recompute the Fisher mask.
                     Mask is updated at the first batch of each epoch that is
                     a multiple of ``update_freq``.

    NOTE: ``update_mask`` uses ``CrossEntropyLoss`` internally, which is only
    appropriate for classification tasks.  For regression tasks this is
    incorrect; see ``TODO_res_algo.md``.

    Reference: Liu et al. (2022) "Sparse and Imperceptible Adversarial Attack
    via a Hessian-based Method."
    """
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
        """Recompute the Fisher Information mask from a random subset of data.

        Accumulates squared gradients over ``num_samples`` random training
        examples to estimate per-parameter Fisher Information.  The top
        ``(1 - sparsity) * total_params`` parameters by Fisher Information
        receive a mask value of 1 (live); the rest are set to 0 (pruned).

        Args:
            model:      The model being trained.
            train_data: PyTorch DataLoader or Dataset with ``dataset`` attribute.
        """
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
        """Return the fraction of live (unmasked / active) parameters.

        Returns:
            Float in ``[0.0, 1.0]``.  ``1.0`` means all parameters are active.
        """
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item()
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num


class ASAM(optim.Optimizer):
    """Adaptive SAM (ASAM) — weight-adaptive perturbation.

    Scales the perturbation ``e(w)`` by ``|w| + eta``, making the neighborhood
    size adaptive to the magnitude of each weight.  This addresses the scale-
    invariance issue in the original SAM where large-weight parameters dominate
    the norm calculation.

    Args:
        params:    Model parameters.
        optimizer: Base optimizer instance.
        model:     The model (needed for named_parameters iteration).
        rho:       Neighborhood radius (default ``0.5``).
        eta:       Minimum adaptive scale to prevent zero perturbation on
                   very small weights (default ``0.01``).

    Reference: Kim et al. (2021) "ASAM: Adaptive Sharpness-Aware Minimization
    for Scale-Invariant Learning of Deep Neural Networks."
    """
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
    """Gradient SAM (GSAM) — surrogate gap minimization via gradient decomposition.

    After perturbing weights, decomposes the perturbed gradient into:
    * Vertical component: orthogonal to the original gradient direction
    * Parallel component: along the original gradient direction

    Subtracts ``alpha * vertical`` from the gradient before the base optimizer
    step, pushing optimization toward directions that simultaneously reduce
    loss and sharpness.

    Args:
        gsam_alpha:  Coefficient for subtracting the vertical gradient
                     component (default ``0.2``).
        rho_t:       Perturbation radius (default ``0.05``).
        adaptive:    If ``True``, scale perturbation by ``|w|²`` (ASAM-style).
        grad_reduce: How to reduce gradients across distributed workers:
                     ``'mean'`` (default) or ``'sum'``.

    Reference: Zhuang et al. (2022) "Surrogate Gap Minimization Improves
    Sharpness-Aware Training."
    """
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
    """Gradient Agreement Maximization (GAM).

    Uses three forward-backward passes per step:
    1. Computes gradient at ``w`` (``g_0``)
    2. Perturbs to ``w + e_0``; computes gradient there (``g_1``)
    3. Applies grad-norm ascent perturbation; computes gradient (``g_2``)
    4. Decomposes gradients to maximize agreement between the three estimates

    The update direction combines ``g_1`` and ``g_2`` with a vertical
    decomposition to reduce sharpness.

    Args:
        args: Dict with keys ``grad_rho``, ``grad_norm_rho``, ``grad_beta_1``,
              ``grad_beta_2``, ``grad_beta_3``, ``grad_gamma``.
        grad_reduce: ``'mean'`` or ``'sum'``.
    """
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

        return torch.sqrt(norm) if isinstance(norm, torch.Tensor) else torch.Tensor(norm).sqrt()

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
    """Friendly SAM — momentum-corrected perturbation direction.

    Before computing the perturbation norm, subtracts a momentum term
    (``sigma * m``) from the gradient, where ``m`` is an EMA of past gradients.
    This reduces the perturbation noise caused by stochastic gradients.

    Args:
        rho:      Perturbation radius (default ``0.05``).
        sigma:    Momentum subtraction coefficient (default ``1``).
        lmbda:    EMA decay for the momentum estimate (default ``0.9``).
        adaptive: If ``True``, use ASAM-style weight-adaptive scaling.

    Reference: Zhang et al. (2023) "Friendly Sharpness-Aware Minimization."
    """
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
