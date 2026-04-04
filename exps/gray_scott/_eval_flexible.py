#!/usr/bin/env python3
"""Evaluate flexible clustering on the Gray-Scott observation test set."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from exps.gray_scott._eval_common import aggregate_evaluation_results, evaluate_observations, load_observations, load_true_solutions, save_outputs
from locater.flexible import adaptive_peak_detection
from psnn.config import cfg_get, load_yaml, resolve_path
from psnn.loaders import load_inference_functions


_WORKER_PHI_FN = None


def _init_worker(phi_ckpt: str) -> None:
    global _WORKER_PHI_FN

    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    phi_fn, _, _ = load_inference_functions(
        phi_ckpt=phi_ckpt,
        device=torch.device("cpu"),
    )
    if phi_fn is None:
        raise RuntimeError("Worker failed to load phi inference function")
    _WORKER_PHI_FN = phi_fn


def _run_one(task):
    idx, theta, true_solutions, domain_bounds, locator_kwargs = task
    if _WORKER_PHI_FN is None:
        raise RuntimeError("Worker inference function not initialized")

    locator_kwargs = dict(locator_kwargs)
    locator_kwargs["random_state"] = int(locator_kwargs["random_state"]) + idx
    phi_u = _WORKER_PHI_FN(theta)
    pred_solutions, _, _, _ = adaptive_peak_detection(
        phi_u,
        domain_bounds,
        **locator_kwargs,
    )
    return (
        np.asarray(theta, dtype=np.float32),
        np.asarray(true_solutions, dtype=np.float32),
        np.asarray(pred_solutions, dtype=np.float32),
    )


def _device_from_arg(device: str) -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=os.path.join(exp_dir, "configs", "complete.yaml"))
    pre_args, _ = pre.parse_known_args()

    cfg_path = resolve_path(exp_dir, pre_args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    path_cfg = cfg_get(cfg, "training.paths", {})
    data_outputs_cfg = cfg_get(cfg, "data_generation.outputs", {})
    u_bounds_cfg = cfg_get(cfg, "data_generation.domain.u_bounds", [[-0.2, 1.2], [-0.2, 1.2]])
    flat_u_bounds = [
        float(u_bounds_cfg[0][0]),
        float(u_bounds_cfg[0][1]),
        float(u_bounds_cfg[1][0]),
        float(u_bounds_cfg[1][1]),
    ]

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(pre_args.config))
    p.add_argument("--phi-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))))
    p.add_argument("--obs-path", type=str, default=str(resolve_path(exp_dir, cfg_get(data_outputs_cfg, "obs_test_pkl", "gray_scott_obs_test.pkl"))))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-procs", type=int, default=0)
    p.add_argument("--u-bounds", type=float, nargs=4, default=flat_u_bounds)
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
    p.add_argument("--limit", type=int)
    p.add_argument("--sample-size", type=int)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out-root", type=str, default=os.path.join(exp_dir, "test_observation_eval_flexible"))
    args = p.parse_args()

    device = _device_from_arg(args.device)
    domain_bounds = [
        [float(args.u_bounds[0]), float(args.u_bounds[1])],
        [float(args.u_bounds[2]), float(args.u_bounds[3])],
    ]

    num_procs = int(args.num_procs)
    if num_procs == 0:
        cpu_n = os.cpu_count() or 1
        num_procs = min(8, max(1, cpu_n - 1))
    if num_procs > 1 and device.type != "cpu":
        warnings.warn("Parallel mode only supported on CPU; switching to serial.")
        num_procs = 1

    obs = load_observations(
        args.obs_path,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )

    locator_kwargs = dict(
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
        verbose=False,
    )

    if num_procs <= 1:
        phi_fn, _, _ = load_inference_functions(
            phi_ckpt=args.phi_ckpt,
            device=device,
        )
        if phi_fn is None:
            raise RuntimeError("Failed to load phi inference function")

        def predict_fn(theta):
            phi_u = phi_fn(theta)
            centers, _, _, _ = adaptive_peak_detection(
                phi_u,
                domain_bounds,
                **locator_kwargs,
            )
            return centers

        metrics, details = evaluate_observations(
            obs,
            predict_fn=predict_fn,
            count_source="flexible_final_solution_count",
        )
    else:
        try:
            ctx = mp.get_context("fork")
            tasks = [
                (
                    i,
                    np.asarray(entry["Theta"], dtype=np.float32).reshape(-1),
                    load_true_solutions(entry),
                    domain_bounds,
                    locator_kwargs,
                )
                for i, entry in enumerate(obs)
            ]

            results = []
            with ctx.Pool(
                processes=num_procs,
                initializer=_init_worker,
                initargs=(str(args.phi_ckpt),),
            ) as pool:
                for item in tqdm.tqdm(
                    pool.imap(_run_one, tasks, chunksize=4),
                    total=len(tasks),
                    desc=f"Evaluating test observations (x{num_procs})",
                ):
                    results.append(item)

            metrics, details = aggregate_evaluation_results(
                results,
                count_source="flexible_final_solution_count",
            )
        except Exception as exc:
            warnings.warn(f"Parallel evaluation failed ({exc}); falling back to serial")

            phi_fn, _, _ = load_inference_functions(
                phi_ckpt=args.phi_ckpt,
                device=device,
            )
            if phi_fn is None:
                raise RuntimeError("Failed to load phi inference function")

            def predict_fn(theta):
                phi_u = phi_fn(theta)
                centers, _, _, _ = adaptive_peak_detection(
                    phi_u,
                    domain_bounds,
                    **locator_kwargs,
                )
                return centers

            metrics, details = evaluate_observations(
                obs,
                predict_fn=predict_fn,
                count_source="flexible_final_solution_count",
            )

    file_tag = f"flexible_{args.sample_method}"
    json_path, npz_path = save_outputs(
        out_root=args.out_root,
        stem=f"test_observation_{file_tag}",
        metrics=metrics,
        details=details,
    )

    print(f"Observation test set: {metrics['num_samples']} theta")
    print(f"Count accuracy: {metrics['count_accuracy']:.6f}")
    print(f"Mean L2 (correct): {metrics['mean_l2_correctly_counted_theta']:.6f}")
    print(f"Saved metrics to {json_path}")
    print(f"Saved details to {npz_path}")


if __name__ == "__main__":
    main()
