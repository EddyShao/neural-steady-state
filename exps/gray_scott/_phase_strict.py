#!/usr/bin/env python3
"""Draw a Gray-Scott phase diagram using the strict locator."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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
from locater.strict import adaptive_peak_detection_amr
from psnn.config import cfg_get, load_yaml, resolve_path
from psnn.loaders import load_inference_functions


_WORKER_PHI_FN = None
_WORKER_COUNT_FN = None
_WORKER_STABILITY_FN = None


def _init_worker(phi_ckpt: str, count_ckpt: str, stability_ckpt: str) -> None:
    global _WORKER_PHI_FN, _WORKER_COUNT_FN, _WORKER_STABILITY_FN
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

    phi_fn, count_fn, stability_fn = load_inference_functions(
        phi_ckpt=phi_ckpt,
        count_ckpt=count_ckpt,
        stability_ckpt=stability_ckpt,
        device=_torch.device("cpu"),
    )
    if phi_fn is None or count_fn is None or stability_fn is None:
        raise RuntimeError("Worker failed to load inference functions")
    _WORKER_PHI_FN = phi_fn
    _WORKER_COUNT_FN = count_fn
    _WORKER_STABILITY_FN = stability_fn


def _device_from_arg(device: str) -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _theta_from_components(f_value: float, k_value: float) -> np.ndarray:
    return np.asarray([f_value, k_value], dtype=np.float32)


def _predict_centers(phi_fn, count_fn, theta: np.ndarray, domain_bounds: list[list[float]], locator_kwargs: dict) -> np.ndarray:
    num = max(0, int(count_fn(theta)))
    if num <= 0:
        return np.empty((0, len(domain_bounds)), dtype=np.float32)
    phi_u = phi_fn(theta)
    centers, _init_centers, _history, _layers = adaptive_peak_detection_amr(
        phi_u,
        domain_bounds,
        num=num,
        **locator_kwargs,
    )
    return centers


def _signature_from_stability(stability: np.ndarray | list[bool]) -> tuple[int, int]:
    stable_np = np.asarray(stability, dtype=bool).reshape(-1)
    return int(stable_np.size), int(np.count_nonzero(stable_np))


def _signature_label(signature: tuple[int, int]) -> str:
    n_sol, n_stable = map(int, signature)
    if n_sol == 0:
        return "0 sol"
    n_unstable = n_sol - n_stable
    sol_word = "sol" if n_sol == 1 else "sols"
    return f"{n_sol} {sol_word} ({n_stable} stable, {n_unstable} unstable)"


def _build_palette(n_categories: int) -> list[tuple[float, float, float, float]]:
    if n_categories <= 10:
        cmap = plt.get_cmap("tab10", max(n_categories, 1))
        return [cmap(i) for i in range(n_categories)]
    cmap = plt.get_cmap("tab20", max(n_categories, 1))
    return [cmap(i) for i in range(n_categories)]


def _run_one_theta(task: tuple) -> tuple:
    f_value, k_value, domain_bounds, locator_kwargs, stable_thresh = task
    theta = _theta_from_components(float(f_value), float(k_value))

    true_stability = np.asarray([bool(sol["stable"]) for sol in true_U(theta)], dtype=bool)
    true_signature = _signature_from_stability(true_stability)

    if _WORKER_PHI_FN is None or _WORKER_COUNT_FN is None or _WORKER_STABILITY_FN is None:
        raise RuntimeError("Worker inference functions are not initialized")

    centers = _predict_centers(_WORKER_PHI_FN, _WORKER_COUNT_FN, theta, domain_bounds, locator_kwargs)
    if centers.size == 0:
        pred_signature = (0, 0)
    else:
        p_stable = _WORKER_STABILITY_FN(theta, centers)
        pred_signature = _signature_from_stability(np.asarray(p_stable, dtype=float) >= float(stable_thresh))

    return float(f_value), float(k_value), true_signature, pred_signature


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

    p = argparse.ArgumentParser(description="Phase diagram using strict locator + count classifier")
    p.add_argument("--config", type=str, default=str(pre_args.config))
    p.add_argument("--phi-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))))
    p.add_argument("--count-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "count_ckpt", "psnn_numsol.pt"))))
    p.add_argument("--stability-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "stability_ckpt", "psnn_stability_cls.pt"))))
    p.add_argument("--device", type=str, default=cfg_get(cfg, "training.device", "auto"), choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-procs", type=int, default=0, help="CPU processes for parallel sweep. 0=auto, 1=serial.")
    p.add_argument("--f-min", type=float, default=float(theta_bounds_cfg[0][0]))
    p.add_argument("--f-max", type=float, default=float(theta_bounds_cfg[0][1]))
    p.add_argument("--f-steps", type=int, default=121)
    p.add_argument("--k-min", type=float, default=float(theta_bounds_cfg[1][0]))
    p.add_argument("--k-max", type=float, default=float(theta_bounds_cfg[1][1]))
    p.add_argument("--k-steps", type=int, default=81)
    p.add_argument("--u-bounds", type=float, nargs=4, default=flat_u_bounds, help="2D bounds for u as: u0_low u0_high u1_low u1_high")
    p.add_argument("--L-cut", type=float, default=0.48)
    p.add_argument("--N-global", type=int, default=3000)
    p.add_argument("--m-global", type=int, default=55)
    p.add_argument("--conv-th", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--sample-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--ball-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--merge-ratio", type=float, default=1.1)
    p.add_argument("--random-state", type=int, default=int(cfg_get(cfg, "seed", 0)))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out-root", type=str, default=os.path.join(exp_dir, "phase_strict_runs"))
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
    count_fn = None
    stability_fn = None
    if num_procs <= 1:
        phi_fn, count_fn, stability_fn = load_inference_functions(
            phi_ckpt=args.phi_ckpt,
            count_ckpt=args.count_ckpt,
            stability_ckpt=args.stability_ckpt,
            device=device,
        )
        if phi_fn is None or count_fn is None or stability_fn is None:
            raise RuntimeError("Failed to load inference functions")

    f_grid = np.linspace(float(args.f_min), float(args.f_max), int(args.f_steps), dtype=np.float32)
    k_grid = np.linspace(float(args.k_min), float(args.k_max), int(args.k_steps), dtype=np.float32)

    true_signatures = np.zeros((len(k_grid), len(f_grid), 2), dtype=np.int16)
    pred_signatures = np.zeros((len(k_grid), len(f_grid), 2), dtype=np.int16)

    locator_kwargs = dict(
        L_cut=float(args.L_cut),
        N_global=int(args.N_global),
        m_global=int(args.m_global),
        conv_th=float(args.conv_th),
        max_iter=int(args.max_iter),
        sample_method=str(args.sample_method),
        ball_method=str(args.ball_method),
        merge_ratio=float(args.merge_ratio),
        random_state=int(args.random_state),
        verbose=bool(args.verbose) if num_procs <= 1 else False,
    )
    stable_thresh = 0.5

    if num_procs <= 1:
        for k_idx, k_value in enumerate(tqdm.tqdm(k_grid, desc="Processing k values")):
            for f_idx, f_value in enumerate(f_grid):
                theta = _theta_from_components(float(f_value), float(k_value))

                true_stability = np.asarray([bool(sol["stable"]) for sol in true_U(theta)], dtype=bool)
                true_signatures[k_idx, f_idx] = np.asarray(_signature_from_stability(true_stability), dtype=np.int16)

                centers = _predict_centers(phi_fn, count_fn, theta, domain_bounds, locator_kwargs)
                if centers.size == 0:
                    pred_signatures[k_idx, f_idx] = np.asarray((0, 0), dtype=np.int16)
                else:
                    p_stable = stability_fn(theta, centers)
                    pred_signatures[k_idx, f_idx] = np.asarray(
                        _signature_from_stability(np.asarray(p_stable, dtype=float) >= stable_thresh),
                        dtype=np.int16,
                    )
    else:
        for ckpt in (args.phi_ckpt, args.count_ckpt, args.stability_ckpt):
            if not os.path.exists(ckpt):
                raise FileNotFoundError(ckpt)

        ctx = mp.get_context("spawn")
        tasks = (
            (
                float(f_value),
                float(k_value),
                domain_bounds,
                locator_kwargs,
                stable_thresh,
            )
            for k_value in k_grid
            for f_value in f_grid
        )

        with ctx.Pool(
            processes=int(num_procs),
            initializer=_init_worker,
            initargs=(str(args.phi_ckpt), str(args.count_ckpt), str(args.stability_ckpt)),
        ) as pool:
            for f_value, k_value, true_sig, pred_sig in tqdm.tqdm(
                pool.imap(_run_one_theta, tasks, chunksize=1),
                total=len(f_grid) * len(k_grid),
                desc=f"Processing phase grid (x{num_procs})",
            ):
                f_idx = int(np.argmin(np.abs(f_grid - np.float32(f_value))))
                k_idx = int(np.argmin(np.abs(k_grid - np.float32(k_value))))
                true_signatures[k_idx, f_idx] = np.asarray(true_sig, dtype=np.int16)
                pred_signatures[k_idx, f_idx] = np.asarray(pred_sig, dtype=np.int16)

    signature_set = {
        tuple(map(int, sig))
        for sig in np.concatenate(
            [true_signatures.reshape(-1, 2), pred_signatures.reshape(-1, 2)],
            axis=0,
        )
    }
    ordered_signatures = sorted(signature_set, key=lambda item: (item[0], item[1]))
    signature_to_code = {sig: idx for idx, sig in enumerate(ordered_signatures)}
    code_to_label = [_signature_label(sig) for sig in ordered_signatures]

    true_codes = np.asarray(
        [[signature_to_code[tuple(map(int, sig))] for sig in row] for row in true_signatures],
        dtype=np.int16,
    )
    pred_codes = np.asarray(
        [[signature_to_code[tuple(map(int, sig))] for sig in row] for row in pred_signatures],
        dtype=np.int16,
    )

    colors = _build_palette(len(ordered_signatures))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(ordered_signatures) + 1) - 0.5, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True, sharey=True)
    for ax, code_grid, title in (
        (axes[0], true_codes, "True phase diagram"),
        (axes[1], pred_codes, "Predicted phase diagram (strict locator)"),
    ):
        mesh = ax.pcolormesh(f_grid, k_grid, code_grid, cmap=cmap, norm=norm, shading="nearest")
        mesh.set_rasterized(True)
        ax.set_xlabel("f")
        ax.set_ylabel("k")
        ax.set_title(title)
        ax.grid(True, alpha=0.15)

    legend_handles = [mpatches.Patch(color=colors[idx], label=label) for idx, label in enumerate(code_to_label)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(3, max(1, len(legend_handles))), frameon=False)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_tag = f"strict_{str(args.sample_method).lower()}"
    fig_path = out_dir / f"phase_{file_tag}.png"
    data_path = out_dir / f"phase_{file_tag}_data.npz"

    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    np.savez(
        data_path,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "script": "_phase_strict.py",
                    "args": vars(args),
                    "stable_threshold": stable_thresh,
                    "signature_labels": code_to_label,
                },
                sort_keys=True,
            )
        ),
        f_grid=f_grid,
        k_grid=k_grid,
        true_signatures=true_signatures,
        pred_signatures=pred_signatures,
        true_codes=true_codes,
        pred_codes=pred_codes,
        signature_codes=np.asarray(ordered_signatures, dtype=np.int16),
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved data: {data_path}")
    print(f"Categories: {', '.join(code_to_label)}")


if __name__ == "__main__":
    main()
