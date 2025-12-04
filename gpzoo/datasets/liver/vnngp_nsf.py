from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import optim

from gpzoo.kernels import batched_Matern32
from gpzoo.model_utilities import build_nsf_vnngp
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
    V: Optional[torch.Tensor] = None,
    L: int = 12,
    lengthscale: float = 10.0,
    kernel_kwargs: Optional[Mapping[str, float]] = None,
    jitter: float = 1e-5,
    K: Optional[int] = None,
    inducing_points: Optional[torch.Tensor] = None,
    lu_reference_points: Optional[torch.Tensor] = None,
    subset_step: int = 10,
    precompute_knn: bool = True,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    lu_rank: Optional[int] = None,
    lu_init_iters: int = 10,
):
    """Build or resume a vanilla VNNGP NSF model for liver MERFISH with Matern32 kernel."""

    kernel_cfg = {"sigma": 1.0, "lengthscale": lengthscale}
    if kernel_kwargs:
        kernel_cfg.update(dict(kernel_kwargs))
    kernel_cfg["lengthscale"] = lengthscale
    kernel = batched_Matern32(**kernel_cfg)

    if state_dict is not None and inducing_points is None:
        try:
            inducing_points = state_dict["prior.Z"].detach().clone()
        except KeyError as exc:
            raise KeyError("Checkpoint missing prior.Z for VNNGP resume.") from exc
        if K is None:
            inferred = _infer_k(state_dict)
            if inferred is not None:
                K = inferred

    if K is None:
        K = 50

    step = max(subset_step, 1)
    if lu_reference_points is None:
        lu_reference_points = X[::step]

    return build_nsf_vnngp(
        X=X,
        Y=Y,
        V=V,
        L=L,
        kernel=kernel,
        jitter=jitter,
        K=K,
        inducing_points=inducing_points,
        lu_reference_points=lu_reference_points,
        subset_step=step,
        lu_rank=lu_rank,
        lu_init_iters=lu_init_iters,
        device=device,
        seed=seed,
        precompute_knn=precompute_knn,
    )


def _train_main() -> None:
    from gpzoo.models import build_model
    from gpzoo.training_utilities import freeze_svgp_kernel_and_inputs, train_vnngp_batched_with_tracking
    from pathlib import Path

    from gpzoo.datasets.liver.common import (
        DEVICE,
        OUTPUT_DIR,
        load_liver_with_groups,
        run_training,
    )
    from gpzoo.datasets.liver.config import (
        JITTER,
        L_FACTORS,
        LENGTHSCALE,
        LENGTHSCALE_TRAIN_AFTER,
        LR,
        LR_LENGTHSCALE,
        IMAGE_LOG_EVERY,
        SEED,
        STEPS,
        VNNGP_K,
        X_BATCH,
        Y_BATCH,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X, Y, _, _, V_init = load_liver_with_groups()

    writer = SummaryWriter(log_dir=OUTPUT_DIR / "tb" / f"liver_healthy_vnngp_k={VNNGP_K}")

    checkpoint_path = OUTPUT_DIR / f"liver_healthy_vnngp_k={VNNGP_K}.pth"
    checkpoint_arg = str(checkpoint_path) if checkpoint_path.exists() else None
    start_step = 0

    model, _ = build_model(
        "liver/vnngp_nsf",
        checkpoint_path=checkpoint_arg,
        X=X,
        Y=Y,
        V=V_init,
        L=L_FACTORS,
        lengthscale=LENGTHSCALE,
        kernel_kwargs={"sigma": 1.0},
        jitter=JITTER,
        K=VNNGP_K,
        subset_step=10,
        device=DEVICE,
        seed=SEED,
        precompute_knn=True,
    )

    freeze_svgp_kernel_and_inputs(model)
    lengthscale_param = getattr(model.prior.kernel, "lengthscale", None)
    if lengthscale_param is not None:
        lengthscale_param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    save_path = checkpoint_path
    result = run_training(
        label=f"liver_healthy_vnngp_k={VNNGP_K}",
        model=model,
        optimizer=optimizer,
        train_fn=train_vnngp_batched_with_tracking,
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
        writer=writer,
        progress_desc=f"VNNGP (liver_healthy)",
        image_log_every=IMAGE_LOG_EVERY,
        start_step=start_step,
    )

    writer.close()

    print(f"Training finished for liver_healthy. Artifacts:")
    for key, value in result.items():
        if key.endswith("path") or key.endswith("json"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    _train_main()
