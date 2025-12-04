from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.models import MGGP_VNNGP_NSF
from torch.utils.tensorboard import SummaryWriter


def _infer_k(state_dict: Mapping[str, torch.Tensor]) -> Optional[int]:
    lu = state_dict.get("prior.Lu") if state_dict is not None else None
    if lu is None or lu.ndim < 2:
        return None
    return int(lu.shape[1])


def create_model(
    *,
    X: torch.Tensor,
    Y: torch.Tensor,
    groupsX: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    L: int = 12,
    lengthscale: float = 20.0,
    kernel_kwargs: Optional[Mapping[str, float]] = None,
    jitter: float = 1e-5,
    K: Optional[int] = None,
    inducing_points: Optional[torch.Tensor] = None,
    inducing_groups: Optional[torch.Tensor] = None,
    lu_reference_points: Optional[torch.Tensor] = None,
    lu_reference_groups: Optional[torch.Tensor] = None,
    subset_step: int = 10,
    precompute_knn: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
):
    """Build or resume an MGGP VNNGP NSF model for Slideseq."""

    if groupsX is None:
        raise ValueError("groupsX is required for MGGP models.")

    sigma = 1.0
    group_diff_param = 10.0
    if kernel_kwargs:
        sigma = kernel_kwargs.get("sigma", sigma)
        group_diff_param = kernel_kwargs.get("group_diff_param", group_diff_param)

    if state_dict is not None:
        if inducing_points is None:
            try:
                inducing_points = state_dict["prior.Z"].detach().clone()
            except KeyError as exc:
                raise KeyError("Checkpoint missing prior.Z for MGGP VNNGP.") from exc
        if inducing_groups is None:
            try:
                inducing_groups = state_dict["prior.groupsZ"].detach().clone()
            except KeyError as exc:
                raise KeyError("Checkpoint missing prior.groupsZ for MGGP VNNGP.") from exc
        if K is None:
            inferred = _infer_k(state_dict)
            if inferred is not None:
                K = inferred

    if K is None:
        K = 25

    return MGGP_VNNGP_NSF(
        X=X,
        Y=Y,
        groupsX=groupsX,
        V=V,
        L=L,
        K=K,
        lengthscale=lengthscale,
        sigma=sigma,
        group_diff_param=group_diff_param,
        jitter=jitter,
        inducing_points=inducing_points,
        inducing_groups=inducing_groups,
        subset_step=subset_step,
        precompute_knn=precompute_knn,
        device=device,
        seed=seed,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import freeze_mggp_kernel_and_inputs, train_mggp_vnngp_with_tracking
    from gpzoo.datasets.slideseq.common import (
        DEVICE,
        OUTPUT_DIR,
        load_slideseq_with_groups,
        run_training,
    )
    from gpzoo.datasets.slideseq.config import (
        VNNGP_MGGP_CHECKPOINT,
        GROUP_DIFF_PARAM,
        IMAGE_LOG_EVERY,
        JITTER,
        L_FACTORS,
        LENGTHSCALE,
        LENGTHSCALE_TRAIN_AFTER,
        LR,
        LR_LENGTHSCALE,
        LR_SCALE,
        SEED,
        STEPS,
        VNNGP_K,
        VNNGP_E_SAMPLES,
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, groupsX, _, V_init = load_slideseq_with_groups()

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / f"slideseq_mggp_vnngp_k={VNNGP_K}")

    checkpoint_path = OUTPUT_DIR / f"slideseq_mggp_vnngp_k={VNNGP_K}.pth"
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "slideseq/vnngp_mggp_nsf",
        checkpoint_path=checkpoint_arg,
        X=X,
        groupsX=groupsX,
        Y=Y,
        V=V_init,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={
            "sigma": 1.0,
            "group_diff_param": GROUP_DIFF_PARAM,
        },
        jitter=JITTER,
        K=VNNGP_K,
        subset_step=10,
        device=DEVICE,
        seed=SEED,
        precompute_knn=True,
    )


    freeze_mggp_kernel_and_inputs(model)
    lengthscale_param = getattr(model.prior.kernel, "lengthscale", None)
    if lengthscale_param is not None:
        lengthscale_param.requires_grad = False

    # Build param groups with separate LR for scale (Lu)
    scale_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "Lu" in name:
            scale_params.append(param)
        else:
            other_params.append(param)

    param_groups = [{"params": other_params, "lr": LR}]
    if scale_params:
        param_groups.append({"params": scale_params, "lr": LR_SCALE})

    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # Add CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR*0.01)

    save_path = checkpoint_path
    result = run_training(
        label=f"slideseq_mggp_vnngp_k={VNNGP_K}",
        model=model,
        optimizer=optimizer,
        train_fn=train_mggp_vnngp_with_tracking,
        save_path=save_path,
        X=X,
        groupsX=groupsX,
        y=Y,
        device=DEVICE,
        steps=STEPS,
        x_batch_size=X_BATCH,
        y_batch_size=Y_BATCH,
        e_samples=VNNGP_E_SAMPLES,
        lengthscale_unfreeze_step=LENGTHSCALE_TRAIN_AFTER,
        lengthscale_param=lengthscale_param,
        lengthscale_lr=LR_LENGTHSCALE,
        writer=writer,
        progress_desc=f"MGGP-VNNGP (slideseq)",
        image_log_every=IMAGE_LOG_EVERY,
        start_step=start_step,
        scheduler=scheduler,
    )

    writer.close()

    print("Training finished. Artifacts written to:")
    for key, value in result.items():
        if key.endswith("path") or key.endswith("json"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    _train_main()
