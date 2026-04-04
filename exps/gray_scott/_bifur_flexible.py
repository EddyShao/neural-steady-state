#!/usr/bin/env python3
"""Draw a Gray-Scott bifurcation diagram using the flexible locator."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from exps.gray_scott._gen_data import U as true_U
from locater.flexible import adaptive_peak_detection
from psnn.config import cfg_get, load_yaml, resolve_path
from psnn.loaders import load_inference_functions


_WORKER_PHI_FN = None
_WORKER_STABILITY_FN = None


def _init_worker(phi_ckpt: str, stability_ckpt: str) -> None:
    global _WORKER_PHI_FN, _WORKER_STABILITY_FN
    import torch as _torch

    _torch.set_grad_enabled(False)
    try:
        _torch.set_num_threads(1)
    except Exception:
        pass
    try:
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    phi_fn, _count_fn, stability_fn = load_inference_functions(
        phi_ckpt=phi_ckpt,
        stability_ckpt=stability_ckpt,
        device=_torch.device("cpu"),
    )
    if phi_fn is None or stability_fn is None:
        raise RuntimeError("Worker failed to load inference functions")
    _WORKER_PHI_FN = phi_fn
    _WORKER_STABILITY_FN = stability_fn


def _run_one_f(task: tuple) -> tuple:
    f_value, k_value, domain_bounds, apd_kwargs, stable_thresh = task
    theta = np.asarray([float(f_value), float(k_value)], dtype=np.float32)

    tru = true_U(theta)
    true_u: list[list[float]] = []
    true_stable: list[bool] = []
    for sol in tru:
        true_u.append([float(sol["u"][0]), float(sol["u"][1])])
        true_stable.append(bool(sol["stable"]))

    if _WORKER_PHI_FN is None or _WORKER_STABILITY_FN is None:
        raise RuntimeError("Worker inference functions are not initialized")
    phi_u = _WORKER_PHI_FN(theta)
    centers, _init_centers, _history, _layers = adaptive_peak_detection(phi_u, domain_bounds, **apd_kwargs)

    pred_u: list[list[float]] = []
    pred_stable: list[bool] = []
    if getattr(centers, "size", 0) != 0:
        p_stable = _WORKER_STABILITY_FN(theta, centers)
        for u, prob in zip(centers, p_stable):
            pred_u.append([float(u[0]), float(u[1])])
            pred_stable.append(bool(float(prob) >= float(stable_thresh)))

    return (float(f_value), true_u, true_stable, pred_u, pred_stable)


def _device_from_arg(device: str) -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=os.path.join(exp_dir, "configs", "complete.yaml"))
    pre_args, _ = pre.parse_known_args()

    cfg_path = resolve_path(exp_dir, pre_args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    path_cfg = cfg_get(cfg, "training.paths", {})
    u_bounds_cfg = cfg_get(cfg, "data_generation.domain.u_bounds", [[-0.2, 1.2], [-0.2, 1.2]])
    theta_bounds_cfg = cfg_get(cfg, "data_generation.domain.theta_bounds", [[0.0, 0.3], [0.0, 0.08]])
    flat_u_bounds = [
        float(u_bounds_cfg[0][0]),
        float(u_bounds_cfg[0][1]),
        float(u_bounds_cfg[1][0]),
        float(u_bounds_cfg[1][1]),
    ]

    p = argparse.ArgumentParser(description="Bifurcation diagram using adaptive_peak_detection + stability classifier")
    p.add_argument("--config", type=str, default=str(pre_args.config))
    p.add_argument("--phi-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))))
    p.add_argument("--stability-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "stability_ckpt", "psnn_stability_cls.pt"))))
    p.add_argument("--device", type=str, default=cfg_get(cfg, "training.device", "auto"), choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-procs", type=int, default=0, help="CPU processes for parallel sweep. 0=auto, 1=serial.")
    p.add_argument("--k", type=float, default=float(theta_bounds_cfg[1][1]) / 2.0)
    p.add_argument("--f-min", type=float, default=float(theta_bounds_cfg[0][0]))
    p.add_argument("--f-max", type=float, default=float(theta_bounds_cfg[0][1]))
    p.add_argument("--f-steps", type=int, default=101)
    p.add_argument("--u-bounds", type=float, nargs=4, default=flat_u_bounds, help="2D bounds for u as: u0_low u0_high u1_low u1_high")
    p.add_argument("--L-cut", type=float, default=0.48)
    p.add_argument("--N-global", type=int, default=3000)
    p.add_argument("--m-global", type=int, default=50)
    p.add_argument("--C-max", type=int, default=4)
    p.add_argument("--r-init", type=float, default=0.3)
    p.add_argument("--conv-steps", type=int, default=2)
    p.add_argument("--sample-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--ball-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--valley-ratio", type=float, default=0.9)
    p.add_argument("--sil-var-thres", type=float, default=4e-3)
    p.add_argument("--random-state", type=int, default=int(cfg_get(cfg, "seed", 0)))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out-root", type=str, default=os.path.join(exp_dir, "bifur_flexible_runs"))
    args = p.parse_args()

    device = _device_from_arg(args.device)
    domain_bounds = [[float(args.u_bounds[0]), float(args.u_bounds[1])], [float(args.u_bounds[2]), float(args.u_bounds[3])]]

    num_procs = int(args.num_procs)
    if num_procs == 0:
        cpu_n = os.cpu_count() or 1
        num_procs = min(8, max(1, cpu_n - 1))
    if num_procs > 1 and device.type != "cpu":
        warnings.warn("--num-procs>1 forces CPU; falling back to serial on the selected device")
        num_procs = 1
    if num_procs > 1 and args.verbose:
        warnings.warn("--verbose is disabled in parallel mode to keep logs readable")

    phi_fn = None
    stability_fn = None
    if num_procs <= 1:
        phi_fn, _count_fn, stability_fn = load_inference_functions(
            phi_ckpt=args.phi_ckpt,
            stability_ckpt=args.stability_ckpt,
            device=device,
        )
        if phi_fn is None:
            raise RuntimeError("Failed to load phi checkpoint")
        if stability_fn is None:
            raise RuntimeError("Failed to load stability checkpoint")

    f_grid = np.linspace(float(args.f_min), float(args.f_max), int(args.f_steps), dtype=np.float32)
    pred_centers_by_f: list[np.ndarray] = []
    pred_stability_by_f: list[np.ndarray] = []
    true_centers_by_f: list[np.ndarray] = []
    true_stability_by_f: list[np.ndarray] = []

    apd_kwargs = dict(
        L_cut=float(args.L_cut),
        N_global=int(args.N_global),
        m_global=int(args.m_global),
        C_max=int(args.C_max),
        r_init=float(args.r_init),
        conv_steps=int(args.conv_steps),
        sample_method=str(args.sample_method),
        ball_method=str(args.ball_method),
        valley_ratio=float(args.valley_ratio),
        sil_var_thres=float(args.sil_var_thres),
        random_state=int(args.random_state),
        verbose=bool(args.verbose) if num_procs <= 1 else False,
    )
    stable_thresh = 0.5

    if num_procs <= 1:
        for f_value in tqdm.tqdm(f_grid, desc="Processing f values"):
            theta = np.asarray([float(f_value), float(args.k)], dtype=np.float32)

            tru = true_U(theta)
            true_centers = np.asarray([[float(sol["u"][0]), float(sol["u"][1])] for sol in tru], dtype=np.float32).reshape(-1, 2)
            true_stability = np.asarray([bool(sol["stable"]) for sol in tru], dtype=bool)
            true_centers_by_f.append(true_centers)
            true_stability_by_f.append(true_stability)

            phi_u = phi_fn(theta)
            centers, _init_centers, _history, _layers = adaptive_peak_detection(phi_u, domain_bounds, **apd_kwargs)
            if centers.size == 0:
                pred_centers_by_f.append(np.empty((0, 2), dtype=np.float32))
                pred_stability_by_f.append(np.empty((0,), dtype=bool))
                continue

            p_stable = stability_fn(theta, centers)
            pred_centers_by_f.append(np.asarray(centers, dtype=np.float32))
            pred_stability_by_f.append(np.asarray(p_stable >= stable_thresh, dtype=bool))
    else:
        if not os.path.exists(args.phi_ckpt):
            raise FileNotFoundError(args.phi_ckpt)
        if not os.path.exists(args.stability_ckpt):
            raise FileNotFoundError(args.stability_ckpt)

        ctx = mp.get_context("spawn")
        tasks = (
            (
                float(f_value),
                float(args.k),
                domain_bounds,
                apd_kwargs,
                stable_thresh,
            )
            for f_value in f_grid
        )

        with ctx.Pool(
            processes=int(num_procs),
            initializer=_init_worker,
            initargs=(str(args.phi_ckpt), str(args.stability_ckpt)),
        ) as pool:
            for _f_value, t_u, t_stable, p_u, p_stable in tqdm.tqdm(
                pool.imap(_run_one_f, tasks, chunksize=1),
                total=len(f_grid),
                desc=f"Processing f values (x{num_procs})",
            ):
                true_centers_by_f.append(np.asarray(t_u, dtype=np.float32).reshape(-1, 2))
                true_stability_by_f.append(np.asarray(t_stable, dtype=bool))
                pred_centers_by_f.append(np.asarray(p_u, dtype=np.float32).reshape(-1, 2))
                pred_stability_by_f.append(np.asarray(p_stable, dtype=bool))

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"flexible_{str(args.sample_method).lower()}"
    fig_path = out_dir / f"bifur_{file_tag}.png"
    data_path = out_dir / f"bifur_{file_tag}_data.npz"

    def _flatten_component(
        f_values: np.ndarray,
        centers_by_f: list[np.ndarray],
        stability_by_f: list[np.ndarray],
        component_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_vals: list[float] = []
        y_vals: list[float] = []
        stable_vals: list[bool] = []
        for f_value, centers, stable in zip(f_values, centers_by_f, stability_by_f):
            for center, is_stable in zip(centers, stable):
                x_vals.append(float(f_value))
                y_vals.append(float(center[component_idx]))
                stable_vals.append(bool(is_stable))
        return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float), np.asarray(stable_vals, dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    for ax, component_idx, ylabel in zip(axes, (0, 1), ("u", "v")):
        x_true_np, y_true_np, stable_true_np = _flatten_component(f_grid, true_centers_by_f, true_stability_by_f, component_idx)
        x_pred_np, y_pred_np, stable_pred_np = _flatten_component(f_grid, pred_centers_by_f, pred_stability_by_f, component_idx)

        if x_true_np.size > 0:
            ax.scatter(x_true_np[stable_true_np], y_true_np[stable_true_np], s=10, c="0.75", marker="o", alpha=0.6, label="true stable")
            ax.scatter(x_true_np[~stable_true_np], y_true_np[~stable_true_np], s=10, c="0.85", marker="x", alpha=0.6, label="true unstable")
        if x_pred_np.size > 0:
            ax.scatter(x_pred_np[stable_pred_np], y_pred_np[stable_pred_np], s=16, c="tab:blue", marker="o", alpha=0.9, label="pred stable")
            ax.scatter(x_pred_np[~stable_pred_np], y_pred_np[~stable_pred_np], s=16, c="tab:orange", marker="x", alpha=0.9, label="pred unstable")

        ax.set_xlabel("f")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Gray-Scott bifurcation ({ylabel}, k={float(args.k):.4f})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    np.savez(
        data_path,
        metadata_json=np.asarray(json.dumps({"script": "_bifur_flexible.py", "args": vars(args), "stable_threshold": stable_thresh}, sort_keys=True)),
        f_grid=f_grid,
        true_centers=np.asarray(true_centers_by_f, dtype=object),
        true_stability=np.asarray(true_stability_by_f, dtype=object),
        pred_centers=np.asarray(pred_centers_by_f, dtype=object),
        pred_stability=np.asarray(pred_stability_by_f, dtype=object),
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {data_path}")
    print(f"Predicted slices: {sum(len(v) for v in pred_centers_by_f)} | True slices: {sum(len(v) for v in true_centers_by_f)}")


if __name__ == "__main__":
    main()
