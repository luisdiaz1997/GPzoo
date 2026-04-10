import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
from torch.distributions import Poisson
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from .knn_utilities import calculate_knn


def create_sequential_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    base_lr: float,
    warmup_fraction: float = 0.2,
    start_factor: float = 0.1,
    min_lr: float = 3e-5,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    Create a OneCycleLR scheduler for aggressive training.

    Args:
        optimizer: The optimizer to schedule
        total_steps: Total training steps
        base_lr: Base learning rate (maximum LR during cycle) - only used as reference
        warmup_fraction: Fraction of total steps for warmup (default: 0.2)
        start_factor: Starting LR factor relative to each group's initial LR (default: 0.1)
        min_lr: Minimum learning rate (default: 3e-5)

    Returns:
        OneCycleLR scheduler with:
        - Warmup to max_lr over warmup_fraction of steps
        - Cosine decay to min_lr over remaining steps
        - Respects per-parameter-group learning rates
    """
    # Get max_lr for each param group from their initial 'lr' setting
    # This preserves the different LRs set for different param groups (e.g., Lu vs others)
    max_lrs = [group['lr'] for group in optimizer.param_groups]

    # Create OneCycleLR scheduler with per-group max_lr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,  # Use list to respect different LRs per group
        total_steps=total_steps,
        # pct_start controls warmup phase (fraction of total steps)
        pct_start=warmup_fraction,
        # anneal_strategy: 'cos' for cosine, 'linear' for linear
        anneal_strategy='linear',
        # For Adam/AdamW, we should not cycle momentum (they use adaptive moments, not SGD momentum)
        cycle_momentum=False,
        # Final learning rate
        div_factor=1.0 / start_factor,  # max_lr / start_lr for each group
        final_div_factor=max(max_lrs) / min_lr,  # Use max of group LRs for final decay
    )

    return scheduler


def _get_log_likelihood(pY, data):
    """Return element-wise log likelihood, with Poisson handled manually."""
    if isinstance(pY, Poisson):
        rate = torch.clamp(pY.rate, min=torch.finfo(pY.rate.dtype).tiny)
        return data * torch.log(rate) - rate
    return pY.log_prob(data)


def _log_factor_images(
    writer: SummaryWriter,
    tag: str,
    factors: torch.Tensor,
    coords: torch.Tensor,
    step: int,
    max_factors: int = 10,
    max_points: int = 2000,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Log a small grid of factor scatter plots to TensorBoard."""

    import matplotlib.pyplot as plt
    import numpy as np

    factors_np = factors.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()

    # Downsample points for plotting to limit size.
    if coords_np.shape[0] > max_points:
        idx = np.random.choice(coords_np.shape[0], size=max_points, replace=False)
        coords_np = coords_np[idx]
        factors_np = factors_np[:, idx]

    L = min(factors_np.shape[0], max_factors)
    if L == 0:
        return

    if vmin is None:
        vmin = np.percentile(factors_np, 5)
    if vmax is None:
        vmax = np.percentile(factors_np, 95)

    # Use 2x5 grid for 10 factors (rectangular layout)
    cols = 5
    rows = (L + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False, tight_layout=True)

    for i in range(L):
        ax = axes[i // cols, i % cols]
        ax.scatter(
            coords_np[:, 0],
            coords_np[:, 1],
            c=factors_np[i],
            vmin=vmin,
            vmax=vmax,
            alpha=0.9,
            cmap="turbo",
            s=2,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("xkcd:gray")
        ax.set_title(f"Factor {i}", fontsize="small")

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


def train_svgp_batched_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "SVGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
) -> Tuple[List[float], List, List, List]:
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0  # True if no scale params to manage
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            # Unfreeze scale parameters and update their learning rate
            for param in scale_params_list:
                param.requires_grad = True
            # Find the param group containing scale params and update its lr
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(Xb.squeeze(), idx=idx, idy=idy)

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence(qZ))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        # Apply parameter projection if using projected gradient mode
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            # Log qF.scale statistics to debug constraint violations
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            # Check for invalid values
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            # Log Lu diagonal (after softplus/exp transformation)
            if hasattr(model.prior, "Lu"):
                Lu_raw = model.prior.Lu.detach()
                Lu_diag = torch.exp(torch.diagonal(Lu_raw, dim1=-2, dim2=-1))
                writer.add_histogram("Lu/diag", Lu_diag, it)
                writer.add_scalar("Lu/diag_mean", Lu_diag.mean().item(), it)
            # Log W loadings (raw values before transformation)
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                # Count negative values
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs


def _is_param_in_optimizer(param, optimizer) -> bool:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is param:
                return True
    return False


def train_vnngp_batched_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "VNNGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
) -> Tuple[List[float], List, List, List]:
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0  # True if no scale params to manage
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    full_knn_idx = calculate_knn(model.prior, X, strategy="knn").clone()[:, :-1]

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            ls_param.requires_grad = True
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            # Unfreeze scale parameters and update their learning rate
            for param in scale_params_list:
                param.requires_grad = True
            # Find the param group containing scale params and update its lr
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        knn_idx = full_knn_idx[idx].to(device)
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(Xb.squeeze(), idx=idx, idy=idy)

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence_full(qZ, idx=idx))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        # Apply parameter projection if using projected gradient mode
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            if kernel is not None and hasattr(kernel, "group_diff_param"):
                writer.add_scalar("kernel/group_diff_param", kernel.group_diff_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            # Log qF.scale statistics to debug constraint violations
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            # Check for invalid values
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            # Log Lu diagonal (after softplus/exp transformation)
            if hasattr(model.prior, "Lu"):
                Lu_raw = model.prior.Lu.detach()
                Lu_diag = torch.exp(torch.diagonal(Lu_raw, dim1=-2, dim2=-1))
                writer.add_histogram("Lu/diag", Lu_diag, it)
                writer.add_scalar("Lu/diag_mean", Lu_diag.mean().item(), it)
            # Log W loadings (raw values before transformation)
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                # Count negative values
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, knn_idx
        model.prior.knn_idx = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs


def train_mggp_svgp_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    groupsX: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "MGGP-SVGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
):
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0  # True if no scale params to manage
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            # Unfreeze scale parameters and update their learning rate
            for param in scale_params_list:
                param.requires_grad = True
            # Find the param group containing scale params and update its lr
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        gb = groupsX[idx].to(device)

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(Xb.squeeze(), idx=idx, groupsX=gb, idy=idy)

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence(qZ))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        # Apply parameter projection if using projected gradient mode
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            if kernel is not None and hasattr(kernel, "group_diff_param"):
                writer.add_scalar("kernel/group_diff_param", kernel.group_diff_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            # Log qF.scale statistics to debug constraint violations
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            # Check for invalid values
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            # Log Lu diagonal (after softplus/exp transformation)
            if hasattr(model.prior, "Lu"):
                Lu_raw = model.prior.Lu.detach()
                Lu_diag = torch.exp(torch.diagonal(Lu_raw, dim1=-2, dim2=-1))
                writer.add_histogram("Lu/diag", Lu_diag, it)
                writer.add_scalar("Lu/diag_mean", Lu_diag.mean().item(), it)
            # Log W loadings (raw values before transformation)
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                # Count negative values
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, gb
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs


def train_mggp_vnngp_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    groupsX: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    e_samples: int = 1,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "MGGP-VNNGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
):
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0  # True if no scale params to manage
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    full_knn_idx = calculate_knn(model.prior, X, strategy="knn").clone()[:, :-1]

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            ls_param.requires_grad = True
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            # Unfreeze scale parameters and update their learning rate
            for param in scale_params_list:
                param.requires_grad = True
            # Find the param group containing scale params and update its lr
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        gb = groupsX[idx].to(device)
        knn_idx = full_knn_idx[idx].to(device)
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(
            Xb.squeeze(), idx=idx, groupsX=gb, idy=idy, E=e_samples
        )

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence_full(qZ, idx=idx))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        # Apply parameter projection if using projected gradient mode
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            if kernel is not None and hasattr(kernel, "group_diff_param"):
                writer.add_scalar("kernel/group_diff_param", kernel.group_diff_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            # Log qF.scale statistics to debug constraint violations
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            # Check for invalid values
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            # Log Lu diagonal (after softplus/exp transformation)
            if hasattr(model.prior, "Lu"):
                Lu_raw = model.prior.Lu.detach()
                Lu_diag = torch.exp(torch.diagonal(Lu_raw, dim1=-2, dim2=-1))
                writer.add_histogram("Lu/diag", Lu_diag, it)
                writer.add_scalar("Lu/diag_mean", Lu_diag.mean().item(), it)
            # Log W loadings (raw values before transformation)
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                # Count negative values
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, gb, knn_idx
        model.prior.knn_idx = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs


def train_lcgp_batched_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    e_samples: int = 1,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "LCGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
) -> Tuple[List[float], List, List, List]:
    """
    Training loop for LCGP (Locally Conditioned GP) models.
    
    Similar to VNNGP training but adapted for LCGP's LowRankPlusDiagonal 
    variational covariance parameterization.
    """
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Precompute KNN indices for all points
    full_knn_idx = calculate_knn(model.prior, X, strategy="knn").clone()[:, :-1]

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            ls_param.requires_grad = True
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            for param in scale_params_list:
                param.requires_grad = True
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        knn_idx = full_knn_idx[idx].to(device)
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(Xb.squeeze(), idx=idx, idy=idy, E=e_samples)

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence_full(qZ, idx=idx))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            
            # Log LowRankPlusDiagonal components
            if hasattr(model.prior, "Lu") and hasattr(model.prior.Lu, "D"):
                D = model.prior.Lu.D
                writer.add_histogram("Lu/diag", D, it)
                writer.add_scalar("Lu/diag_mean", D.mean().item(), it)
                writer.add_scalar("Lu/diag_min", D.min().item(), it)
                
                V = model.prior.Lu.V.data
                writer.add_histogram("Lu/V_norm", (V ** 2).sum(dim=-1).sqrt(), it)
            
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, knn_idx
        model.prior.knn_idx = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs


def freeze_svgp_kernel_and_inputs(model) -> None:
    """Freeze kernel hyperparameters and inducing inputs for SVGP-style models."""

    kernel = getattr(model.prior, "kernel", None)
    if kernel is not None:
        if hasattr(kernel, "lengthscale"):
            kernel.lengthscale.requires_grad = False
        if hasattr(kernel, "sigma"):
            kernel.sigma.requires_grad = False

    if hasattr(model.prior, "Z"):
        model.prior.Z.requires_grad = False

    if hasattr(model.prior, "mu"):
        model.prior.mu.requires_grad = True
    if hasattr(model.prior, "Lu"):
        model.prior.Lu.requires_grad = True

    if hasattr(model, "W"):
        model.W.requires_grad = True
    if hasattr(model, "V"):
        model.V.requires_grad = False


def freeze_mggp_kernel_and_inputs(model) -> None:
    """Freeze kernel hyperparameters and inducing data for MGGP models."""

    freeze_svgp_kernel_and_inputs(model)

    kernel = getattr(model.prior, "kernel", None)
    if kernel is not None and hasattr(kernel, "group_diff_param"):
        kernel.group_diff_param.requires_grad = False

    if hasattr(model.prior, "groupsZ"):
        model.prior.groupsZ.requires_grad = False


def freeze_kernel_and_inputs(model) -> None:
    """Backward-compatible wrapper used by legacy scripts."""

    freeze_mggp_kernel_and_inputs(model)


def prepare_frozen_scale_and_lengthscale_params(model):
    """
    Prepare parameter groups with frozen scale (Lu) and lengthscale parameters.

    This function:
    1. Freezes the lengthscale parameter if it exists
    2. Separates scale parameters (Lu) from other parameters
    3. Freezes scale parameters initially
    4. Returns the separated parameter lists for optimizer setup

    Args:
        model: The model whose parameters to prepare

    Returns:
        tuple: (scale_params, other_params, lengthscale_param)
            - scale_params: List of Lu parameters (frozen)
            - other_params: List of other trainable parameters
            - lengthscale_param: The lengthscale parameter (frozen) or None
    """
    # Freeze lengthscale parameter
    lengthscale_param = getattr(model.prior.kernel, "lengthscale", None)
    if lengthscale_param is not None:
        lengthscale_param.requires_grad = False

    # Separate scale parameters (Lu) from other parameters
    scale_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "Lu" in name:
            param.requires_grad = False  # Freeze initially
            scale_params.append(param)
        else:
            other_params.append(param)

    return scale_params, other_params, lengthscale_param


def prepare_frozen_params(model):
    """
    Prepare parameter groups with separate learning rates for scale (Lu), lengthscale, mean (mu), and loadings (W).

    This function separates parameters into groups that can have different learning rates:
    1. Freezes the lengthscale parameter if it exists
    2. Separates scale parameters (Lu) and freezes them initially
    3. Separates mean parameters (mu)
    4. Separates loading parameters (W)
    5. Collects remaining parameters

    Args:
        model: The model whose parameters to prepare

    Returns:
        dict: Dictionary containing:
            - 'scale_params': List of Lu parameters (frozen)
            - 'mean_params': List of mu parameters
            - 'loading_params': List of W parameters
            - 'other_params': List of other trainable parameters
            - 'lengthscale_param': The lengthscale parameter (frozen) or None
    """
    # Freeze lengthscale parameter
    lengthscale_param = getattr(model.prior.kernel, "lengthscale", None)
    if lengthscale_param is not None:
        lengthscale_param.requires_grad = False

    # Separate parameters into groups
    scale_params = []
    mean_params = []
    loading_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "Lu" in name:
            param.requires_grad = False  # Freeze initially
            scale_params.append(param)
        elif "mu" in name:
            mean_params.append(param)
        elif "W" in name:
            loading_params.append(param)
        else:
            other_params.append(param)

    return {
        'scale_params': scale_params,
        'mean_params': mean_params,
        'loading_params': loading_params,
        'other_params': other_params,
        'lengthscale_param': lengthscale_param
    }


def train_mggp_lcgp_with_tracking(
    optimizer,
    model,
    X: torch.Tensor,
    groupsX: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 200,
    start_step: int = 0,
    x_batch_size: int = 1000,
    y_batch_size: int = 1000,
    e_samples: int = 1,
    lengthscale_unfreeze_step: Optional[int] = None,
    lengthscale_param=None,
    lengthscale_lr: Optional[float] = None,
    scale_unfreeze_step: Optional[int] = None,
    scale_params=None,
    scale_lr: Optional[float] = None,
    writer: Optional[SummaryWriter] = None,
    progress_desc: str = "MGGP-LCGP",
    image_log_every: Optional[int] = None,
    scheduler=None,
):
    losses, means, scales, idxs = [], [], [], []
    N = len(X)
    J = len(y)

    kernel = getattr(model.prior, "kernel", None)
    ls_param = lengthscale_param if lengthscale_param is not None else getattr(kernel, "lengthscale", None)
    ls_added = ls_param is None or _is_param_in_optimizer(ls_param, optimizer)
    base_lr = lengthscale_lr if lengthscale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    # Handle scale parameter unfreezing
    scale_params_list = scale_params if scale_params is not None else []
    scale_unfrozen = len(scale_params_list) == 0  # True if no scale params to manage
    scale_target_lr = scale_lr if scale_lr is not None else optimizer.param_groups[0].get("lr", 1e-3)

    full_knn_idx = calculate_knn(model.prior, X, strategy="knn").clone()[:, :-1]

    for it in trange(start_step, start_step + steps, desc=progress_desc):
        if lengthscale_unfreeze_step is not None and not ls_added and it >= lengthscale_unfreeze_step:
            ls_param.requires_grad = True
            optimizer.add_param_group({"params": [ls_param], "lr": base_lr})
            ls_added = True

        if scale_unfreeze_step is not None and not scale_unfrozen and it >= scale_unfreeze_step:
            # Unfreeze scale parameters and update their learning rate
            for param in scale_params_list:
                param.requires_grad = True
            # Find the param group containing scale params and update its lr
            for group in optimizer.param_groups:
                if scale_params_list and any(p is scale_params_list[0] for p in group['params']):
                    group['lr'] = scale_target_lr
                    break
            scale_unfrozen = True

        idx = torch.multinomial(torch.ones(N), num_samples=min(x_batch_size, N), replacement=False)
        idy = torch.multinomial(torch.ones(J), num_samples=min(y_batch_size, J), replacement=False)

        Xb = X[idx].to(device)
        yb = y[idy][:, idx].to(device)
        gb = groupsX[idx].to(device)
        knn_idx = full_knn_idx[idx].to(device)
        model.prior.knn_idx = knn_idx

        optimizer.zero_grad()
        pY, qF, qZ, _ = model.forward_batched_train(
            Xb.squeeze(), idx=idx, groupsX=gb, idy=idy, E=e_samples
        )

        logpY = _get_log_likelihood(pY, yb)
        L1 = logpY.mean(dim=0).sum()
        L2 = torch.sum(model.prior.kl_divergence_full(qZ, idx=idx))
        loss = -(L1 - L2)

        loss.backward()
        optimizer.step()
        # Apply parameter projection if using projected gradient mode
        if hasattr(model, 'project_parameters'):
            model.project_parameters()
        if scheduler is not None and scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        losses.append(loss.item())

        if writer is not None and it % 10 == 0:
            writer.add_scalar("loss/train", loss.item(), it)
            if ls_param is not None:
                writer.add_scalar("kernel/lengthscale", ls_param.detach().mean().item(), it)
            if kernel is not None and hasattr(kernel, "group_diff_param"):
                writer.add_scalar("kernel/group_diff_param", kernel.group_diff_param.detach().mean().item(), it)
            writer.add_histogram("qF/mean", qF.mean, it)
            writer.add_histogram("qF/scale", qF.scale, it)
            # Log qF.scale statistics to debug constraint violations
            writer.add_scalar("qF/scale_min", qF.scale.min().item(), it)
            writer.add_scalar("qF/scale_max", qF.scale.max().item(), it)
            writer.add_scalar("qF/scale_mean", qF.scale.mean().item(), it)
            # Check for invalid values
            num_nan = torch.isnan(qF.scale).sum().item()
            num_inf = torch.isinf(qF.scale).sum().item()
            num_nonpositive = (qF.scale <= 0).sum().item()
            writer.add_scalar("qF/scale_num_nan", num_nan, it)
            writer.add_scalar("qF/scale_num_inf", num_inf, it)
            writer.add_scalar("qF/scale_num_nonpositive", num_nonpositive, it)
            
            # Log LowRankPlusDiagonal components (LCGP Specific)
            if hasattr(model.prior, "Lu") and hasattr(model.prior.Lu, "D"):
                D = model.prior.Lu.D
                writer.add_histogram("Lu/diag", D, it)
                writer.add_scalar("Lu/diag_mean", D.mean().item(), it)
                writer.add_scalar("Lu/diag_min", D.min().item(), it)
                
                V = model.prior.Lu.V.data
                writer.add_histogram("Lu/V_norm", (V ** 2).sum(dim=-1).sqrt(), it)

            # Log W loadings (raw values before transformation)
            if hasattr(model, 'W'):
                W_raw = model.W.detach()
                writer.add_histogram("W/raw", W_raw, it)
                writer.add_scalar("W/mean", W_raw.mean().item(), it)
                writer.add_scalar("W/min", W_raw.min().item(), it)
                writer.add_scalar("W/max", W_raw.max().item(), it)
                # Count negative values
                num_negative = (W_raw < 0).sum().item()
                writer.add_scalar("W/num_negative", num_negative, it)

        if it % 10 == 0:
            idxs.append(idx.detach().cpu().numpy())
            means.append(qF.mean.detach().cpu().numpy())
            scales.append(qF.scale.detach().cpu().numpy())

        if writer is not None and image_log_every is not None and it % image_log_every == 0:
            _log_factor_images(writer, "factors/mean", qF.mean, Xb, it)
            _log_factor_images(writer, "factors/scale", qF.scale, Xb, it, vmin=0, vmax=1)

        del pY, qF, qZ, logpY, L1, L2, loss, Xb, yb, gb, knn_idx
        model.prior.knn_idx = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return losses, means, scales, idxs
