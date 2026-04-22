import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from psnn import datasets, nets
from psnn.config import cfg_get, resolve_path
from psnn.trainer import eval_phi_model, train_phi_model
from psnn.utils import infer_psnn_arch

from _gen_data import OMEGA, D, Phi_theta, U, _cartesian_grid, _grid_shape


def _set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_from_cfg(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        return torch.compile(model)  # type: ignore[attr-defined]
    except Exception:
        return model


def _write_training_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _evaluate_phi_approximation(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    theta_batch_size: int,
) -> dict[str, Any]:
    dg = cfg_get(cfg, "data_generation", {})
    theta_bounds = np.asarray(cfg_get(dg, "domain.theta_bounds", OMEGA), dtype=float)
    u_bounds = np.asarray(cfg_get(dg, "domain.u_bounds", D), dtype=float)
    approx_cfg = cfg_get(cfg, "training.approximation_eval", {})

    theta_counts = _grid_shape(cfg_get(approx_cfg, "theta_grid", [100, 100]), theta_bounds.shape[0], name="theta_grid")
    u_counts = _grid_shape(cfg_get(approx_cfg, "u_grid", [50, 50]), u_bounds.shape[0], name="u_grid")

    theta_grid = _cartesian_grid(theta_bounds, theta_counts)
    u_grid = _cartesian_grid(u_bounds, u_counts)
    device = next(model.parameters()).device

    sum_sq = 0.0
    total_points = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, theta_grid.shape[0], int(theta_batch_size)):
            theta_batch_np = theta_grid[start : start + int(theta_batch_size)]
            batch_size = theta_batch_np.shape[0]
            theta_repeated = np.repeat(theta_batch_np[:, None, :], u_grid.shape[0], axis=1).reshape(-1, theta_grid.shape[1])
            u_tiled = np.tile(u_grid[None, :, :], (batch_size, 1, 1)).reshape(-1, u_grid.shape[1])

            pred = model(
                torch.from_numpy(u_tiled).to(device),
                torch.from_numpy(theta_repeated).to(device),
            ).view(-1)

            pred_np = pred.detach().cpu().numpy().astype(np.float64).reshape(batch_size, u_grid.shape[0])
            true_np = np.stack([Phi_theta(u_grid, U(theta)) for theta in theta_batch_np], axis=0).astype(np.float64)
            diff = pred_np - true_np
            sum_sq += float(np.sum(diff * diff))
            total_points += int(diff.size)

    mse = sum_sq / max(1, total_points)
    rmse = float(np.sqrt(mse))

    step_sizes = [
        float(theta_bounds[idx, 1] - theta_bounds[idx, 0]) / float(max(1, theta_counts[idx] - 1))
        for idx in range(theta_bounds.shape[0])
    ] + [
        float(u_bounds[idx, 1] - u_bounds[idx, 0]) / float(max(1, u_counts[idx] - 1))
        for idx in range(u_bounds.shape[0])
    ]
    cell_volume = float(np.prod(step_sizes))
    l2_error = float(np.sqrt(sum_sq * cell_volume))

    return {
        "enabled": True,
        "theta_grid": theta_counts,
        "u_grid": u_counts,
        "num_theta_points": int(theta_grid.shape[0]),
        "num_u_points": int(u_grid.shape[0]),
        "num_total_points": int(total_points),
        "cell_volume": cell_volume,
        "mse": float(mse),
        "rmse": rmse,
        "l2_error": l2_error,
    }


def run_from_loaded_config(cfg: dict[str, Any], base_dir: str | Path) -> None:
    base_dir = Path(base_dir).expanduser().resolve()

    tr = cfg_get(cfg, "training", {})
    paths = cfg_get(tr, "paths", {})
    data_outputs = cfg_get(cfg, "data_generation.outputs", {})

    global_seed = int(cfg_get(cfg, "seed", cfg_get(tr, "seed", 123)))

    def seed_for(section: str) -> int:
        return int(cfg_get(tr, f"{section}.seed", global_seed))

    device = _device_from_cfg(cfg_get(tr, "device", "auto"))
    num_workers = int(cfg_get(tr, "num_workers", 0))
    compile_enabled = bool(cfg_get(tr, "compile", False))

    data_dir = resolve_path(base_dir, cfg_get(paths, "data_dir", cfg_get(data_outputs, "out_dir", "data")))
    out_dir = resolve_path(base_dir, cfg_get(paths, "out_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_npz = resolve_path(
        base_dir,
        cfg_get(paths, "train_npz", data_dir / cfg_get(data_outputs, "data_train_npz", "data_train.npz")),
    )
    test_npz = resolve_path(
        base_dir,
        cfg_get(paths, "test_npz", data_dir / cfg_get(data_outputs, "data_test_npz", "data_test.npz")),
    )
    out_phi = resolve_path(out_dir, cfg_get(paths, "phi_ckpt", "psnn_phi.pt"))
    out_compat_phi = cfg_get(paths, "compat_phi_ckpt", None)
    out_compat_phi = resolve_path(out_dir, out_compat_phi) if out_compat_phi else None
    approx_cfg = cfg_get(tr, "approximation_eval", {})
    approx_metric_path = resolve_path(out_dir, cfg_get(approx_cfg, "metric_path", "approximation_metrics.json"))

    summary: dict[str, Any] = {
        "experiment": str(cfg_get(cfg, "experiment", "gray-scott-toy")),
        "variant": str(cfg_get(cfg, "run.variant", out_dir.name)),
        "seed": int(global_seed),
        "device": str(device),
        "run_dir": str(out_dir),
        "shared_data_dir": str(data_dir),
        "models": {},
    }

    print(f"Base dir: {base_dir}")
    print(f"Device: {device}")
    print(f"Shared data dir: {data_dir}")

    if bool(cfg_get(tr, "phi.enabled", False)):
        if not train_npz.exists():
            raise FileNotFoundError(f"Missing train file: {train_npz}")
        if not test_npz.exists():
            raise FileNotFoundError(f"Missing test file: {test_npz}")

        print("Training Phi model...")
        phi_seed = seed_for("phi")
        _set_all_seeds(phi_seed)

        phi_batch_size = int(cfg_get(tr, "phi.batch_size", 256))
        phi_epochs = int(cfg_get(tr, "phi.epochs", 100))
        phi_lr = float(cfg_get(tr, "phi.lr", 1e-3))
        eta_scale = float(cfg_get(tr, "phi.eta.scale", 1.5))
        eta_cap = float(cfg_get(tr, "phi.eta.cap", 0.01))
        phi_compile = bool(cfg_get(tr, "phi.compile", compile_enabled))

        train_loader, test_loader = datasets.make_loaders(
            str(train_npz),
            str(test_npz),
            batch_size=phi_batch_size,
            num_workers=num_workers,
            device=device,
        )

        dim_theta = int(train_loader.dataset.Theta.shape[1])
        dim_u = int(train_loader.dataset.U.shape[1])
        phi_max = float(train_loader.dataset.Phi.max().item())
        eta = min(float(eta_cap), float(eta_scale) * (phi_max - 1.0))
        print(f"Using eta={eta:.3e} (Phi.max={phi_max:.3e})")

        model_phi = nets.PSNN(
            dim_theta=dim_theta,
            dim_u=dim_u,
            embed_dim=int(cfg_get(tr, "phi.model.embed_dim", 8)),
            width=cfg_get(tr, "phi.model.width", [30, 20]),
            depth=cfg_get(tr, "phi.model.depth", [4, 3]),
            eta=eta,
        ).to(device)
        model_phi = _maybe_compile(model_phi, phi_compile)

        train_phi_model(
            model_phi,
            train_loader,
            val_loader=test_loader,
            epochs=phi_epochs,
            lr=phi_lr,
            device=device,
        )

        mse = torch.nn.MSELoss()
        final_train_loss = float(eval_phi_model(model_phi, train_loader, mse))
        final_val_loss = float(eval_phi_model(model_phi, test_loader, mse))

        torch.save(
            {
                "format_version": 1,
                "kind": "phi",
                "state_dict": model_phi.state_dict(),
                "model": infer_psnn_arch(model_phi),
                "data": {
                    "train_npz": str(train_npz),
                    "test_npz": str(test_npz),
                    "phi_max": float(phi_max),
                },
                "train": {
                    "epochs": int(phi_epochs),
                    "lr": float(phi_lr),
                    "eta_scale": float(eta_scale),
                    "eta_cap": float(eta_cap),
                    "batch_size": int(phi_batch_size),
                    "device": str(device),
                    "seed": int(phi_seed),
                    "final_train_error": final_train_loss,
                    "final_val_error": final_val_loss,
                },
            },
            out_phi,
        )
        if out_compat_phi is not None:
            torch.save(model_phi.state_dict(), out_compat_phi)
        print(f"Saved Phi model to: {out_phi}")

        summary["models"]["phi"] = {
            "enabled": True,
            "metric_name": "mse",
            "final_train_error": final_train_loss,
            "final_val_error": final_val_loss,
        }

        if bool(cfg_get(approx_cfg, "enabled", False)):
            theta_batch_size = int(cfg_get(approx_cfg, "theta_batch_size", 64))
            print("Evaluating Phi approximation error on the reference grid...")
            approximation_metrics = _evaluate_phi_approximation(
                model_phi,
                cfg,
                theta_batch_size=theta_batch_size,
            )
            with approx_metric_path.open("w", encoding="utf-8") as handle:
                json.dump(approximation_metrics, handle, indent=2)
            print(f"Saved approximation metrics to: {approx_metric_path}")
            summary["approximation_eval"] = approximation_metrics

    _write_training_summary(out_dir / "training_summary.json", summary)
    print(f"Saved training summary to: {out_dir / 'training_summary.json'}")
