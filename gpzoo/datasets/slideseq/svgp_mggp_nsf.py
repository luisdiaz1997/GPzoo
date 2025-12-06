from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.models import MGGP_SVGP_NSF
from torch.utils.tensorboard import SummaryWriter


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
    num_inducing: Optional[int] = None,
    inducing_points: Optional[torch.Tensor] = None,
    inducing_groups: Optional[torch.Tensor] = None,
    inducing_init_method: str = "kmeans",
    inducing_allocation: str = "equal",
    inducing_seed: int = 123,
    scale_multiplier: float = 1e-6,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    loadings_mode: str = "softplus",
):
    """Build or resume an MGGP SVGP NSF model for Slideseq."""

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
                raise KeyError("Checkpoint is missing prior.Z needed for SVGP resume.") from exc
        if inducing_groups is None:
            try:
                inducing_groups = state_dict["prior.groupsZ"].detach().clone()
            except KeyError as exc:
                raise KeyError("Checkpoint missing prior.groupsZ for MGGP SVGP.") from exc

    return MGGP_SVGP_NSF(
        X=X,
        Y=Y,
        groupsX=groupsX,
        V=V,
        L=L,
        lengthscale=lengthscale,
        sigma=sigma,
        group_diff_param=group_diff_param,
        jitter=jitter,
        num_inducing=num_inducing,
        inducing_points=inducing_points,
        inducing_groups=inducing_groups,
        inducing_method=inducing_init_method,
        scale_multiplier=scale_multiplier,
        device=device,
        seed=seed or inducing_seed,
        loadings_mode=loadings_mode,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import (
        freeze_mggp_kernel_and_inputs,
        train_mggp_svgp_with_tracking,
        create_sequential_lr_scheduler,
    )
    from gpzoo.datasets.slideseq.common import (
        DEVICE,
        OUTPUT_DIR,
        load_slideseq_with_groups,
        run_training,
    )
    from gpzoo.datasets.slideseq.config import (
        SVGP_MGGP_CHECKPOINT,
        GROUP_DIFF_PARAM,
        IMAGE_LOG_EVERY,
        INDUCING_ALLOCATION,
        JITTER,
        L_FACTORS,
        LENGTHSCALE,
        LENGTHSCALE_TRAIN_AFTER,
        LOADINGS_MODE,
        LR,
        LR_LENGTHSCALE,
        LR_SCALE,
        SCALE_MULTIPLIER,
        SCALE_TRAIN_AFTER,
        SEED,
        STEPS,
        SVGP_INDUCING,
        USE_SCHEDULER,
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, groupsX, _, V_init = load_slideseq_with_groups()
    num_inducing = min(SVGP_INDUCING, X.shape[0])

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / "slideseq_mggp_svgp")

    checkpoint_path = OUTPUT_DIR / "slideseq_mggp_svgp.pth"
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "slideseq/svgp_mggp_nsf",
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
        num_inducing=num_inducing,
        inducing_allocation=INDUCING_ALLOCATION,
        inducing_seed=SEED,
        scale_multiplier=SCALE_MULTIPLIER,
        device=DEVICE,
        seed=SEED,
        loadings_mode=LOADINGS_MODE,
    )


    freeze_mggp_kernel_and_inputs(model)

    # Prepare frozen scale and lengthscale parameters
    from gpzoo.training_utilities import prepare_frozen_scale_and_lengthscale_params
    scale_params, other_params, lengthscale_param = prepare_frozen_scale_and_lengthscale_params(model)

    # Add all params to optimizer, but scale_params start with lr=0 (frozen)
    param_groups = [
        {"params": other_params, "lr": LR},
        {"params": scale_params, "lr": 0.0},  # Start frozen with lr=0
    ]

    # Use Adam optimizer
    optimizer = optim.Adam(param_groups)

    # Create SequentialLR scheduler: Linear warmup + Cosine decay (only if USE_SCHEDULER is True)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = create_sequential_lr_scheduler(
            optimizer=optimizer,
            total_steps=STEPS,
            base_lr=LR,
            warmup_fraction=0.2,
            start_factor=0.1,
            min_lr=3e-5,
        )

    save_path = checkpoint_path
    result = run_training(
        label="slideseq_mggp_svgp",
        model=model,
        optimizer=optimizer,
        train_fn=train_mggp_svgp_with_tracking,
        save_path=save_path,
        X=X,
        groupsX=groupsX,
        y=Y,
        device=DEVICE,
        steps=STEPS,
        x_batch_size=X_BATCH,
        y_batch_size=Y_BATCH,
        lengthscale_unfreeze_step=LENGTHSCALE_TRAIN_AFTER,
        lengthscale_param=lengthscale_param,
        lengthscale_lr=LR_LENGTHSCALE,
        scale_unfreeze_step=SCALE_TRAIN_AFTER,
        scale_params=scale_params,
        scale_lr=LR_SCALE,
        writer=writer,
        progress_desc="MGGP-SVGP (slideseq)",
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
