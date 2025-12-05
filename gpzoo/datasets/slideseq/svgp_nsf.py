from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.models import SVGP_NSF
from torch.utils.tensorboard import SummaryWriter


def create_model(
    *,
    X: torch.Tensor,
    Y: torch.Tensor,
    V: Optional[torch.Tensor] = None,
    L: int = 12,
    lengthscale: float = 20.0,
    kernel_kwargs: Optional[Mapping[str, float]] = None,
    jitter: float = 1e-5,
    num_inducing: Optional[int] = None,
    inducing_points: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    lu_rank: Optional[int] = None,
    lu_init_iters: int = 10,
    scale_multiplier: float = 1e-6,
    loadings_mode: str = "softplus",
):
    """Build or resume a vanilla SVGP NSF model for Slideseq."""

    sigma = 1.0
    if kernel_kwargs and "sigma" in kernel_kwargs:
        sigma = kernel_kwargs["sigma"]

    if state_dict is not None and inducing_points is None:
        try:
            inducing_points = state_dict["prior.Z"].detach().clone()
        except KeyError as exc:
            raise KeyError("Checkpoint is missing prior.Z needed for SVGP resume.") from exc

    return SVGP_NSF(
        X=X,
        Y=Y,
        V=V,
        L=L,
        lengthscale=lengthscale,
        sigma=sigma,
        jitter=jitter,
        num_inducing=num_inducing,
        inducing_points=inducing_points,
        lu_rank=lu_rank,
        lu_init_iters=lu_init_iters,
        scale_multiplier=scale_multiplier,
        device=device,
        seed=seed,
        loadings_mode=loadings_mode,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import (
        freeze_svgp_kernel_and_inputs,
        train_svgp_batched_with_tracking,
        create_sequential_lr_scheduler,
    )
    from gpzoo.datasets.slideseq.common import (
        DEVICE,
        OUTPUT_DIR,
        load_slideseq_with_groups,
        run_training,
    )
    from gpzoo.datasets.slideseq.config import (
        SVGP_CHECKPOINT,
        IMAGE_LOG_EVERY,
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
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, _, _, V_init = load_slideseq_with_groups()
    num_inducing = min(SVGP_INDUCING, X.shape[0])

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / "slideseq_svgp")

    checkpoint_path = OUTPUT_DIR / "slideseq_svgp.pth"
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "slideseq/svgp_nsf",
        checkpoint_path=checkpoint_arg,
        X=X,
        Y=Y,
        V=V_init,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={"sigma": 1.0},
        jitter=JITTER,
        num_inducing=num_inducing,
        scale_multiplier=SCALE_MULTIPLIER,
        device=DEVICE,
        seed=SEED,
        loadings_mode=LOADINGS_MODE,
    )


    freeze_svgp_kernel_and_inputs(model)

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

    # Create SequentialLR scheduler: Linear warmup + Cosine decay
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
        label="slideseq_svgp",
        model=model,
        optimizer=optimizer,
        train_fn=train_svgp_batched_with_tracking,
        save_path=save_path,
        X=X,
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
        progress_desc="SVGP (slideseq)",
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
