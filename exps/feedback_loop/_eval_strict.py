#!/usr/bin/env python3
"""Evaluate strict clustering on the feedback-loop observation test set."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import torch
import tqdm

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from exps.feedback_loop._eval_common import aggregate_evaluation_results, evaluate_observations, load_observations, load_true_solutions, save_outputs
from locater.strict import adaptive_peak_detection_amr
from psnn.config import cfg_get, load_yaml, resolve_path
from psnn.loaders import load_inference_functions


_WORKER_PHI_FN = None
_WORKER_COUNT_FN = None


def _init_worker(phi_ckpt: str, count_ckpt: str) -> None:
    global _WORKER_PHI_FN, _WORKER_COUNT_FN
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

    phi_fn, count_fn, _stability_fn = load_inference_functions(
        phi_ckpt=phi_ckpt,
        count_ckpt=count_ckpt,
        device=_torch.device("cpu"),
    )
    if phi_fn is None or count_fn is None:
        raise RuntimeError("Worker failed to load phi/count inference functions")
    _WORKER_PHI_FN = phi_fn
    _WORKER_COUNT_FN = count_fn


def _run_one(task: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, true_solutions, D, locator_kwargs = task
    if _WORKER_PHI_FN is None or _WORKER_COUNT_FN is None:
        raise RuntimeError("Worker inference functions are not initialized")

    pred_count = max(0, int(_WORKER_COUNT_FN(theta)))
    if pred_count <= 0:
        pred_solutions = np.empty((0, len(D)), dtype=np.float32)
    else:
        phi_u = _WORKER_PHI_FN(theta)
        pred_solutions, _init_centers, _history, _layers = adaptive_peak_detection_amr(
            phi_u,
            D,
            num=pred_count,
            **locator_kwargs,
        )
    return np.asarray(theta, dtype=np.float32), np.asarray(true_solutions, dtype=np.float32), np.asarray(pred_solutions, dtype=np.float32)


def _device_from_arg(device: str) -> torch.device:
    device = str(device).lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=os.path.join(exp_dir, "config.yaml"))
    pre_args, _ = pre.parse_known_args()

    cfg_path = resolve_path(exp_dir, pre_args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    path_cfg = cfg_get(cfg, "training.paths", {})
    data_outputs_cfg = cfg_get(cfg, "data_generation.outputs", {})
    u_bounds_cfg = cfg_get(cfg, "data_generation.domain.u_bounds", [[-1.0, 6.0], [-1.0, 6.0]])
    flat_u_bounds = [
        float(u_bounds_cfg[0][0]),
        float(u_bounds_cfg[0][1]),
        float(u_bounds_cfg[1][0]),
        float(u_bounds_cfg[1][1]),
    ]

    p = argparse.ArgumentParser(description="Evaluate strict clustering on the feedback-loop observation test set.")
    p.add_argument("--config", type=str, default=str(pre_args.config))
    p.add_argument("--phi-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "phi_ckpt", "psnn_phi.pt"))))
    p.add_argument("--count-ckpt", type=str, default=str(resolve_path(exp_dir, cfg_get(path_cfg, "count_ckpt", "psnn_numsol.pt"))))
    p.add_argument("--obs-path", type=str, default=str(resolve_path(exp_dir, cfg_get(data_outputs_cfg, "obs_test", "feedback_loop_obs_test.pkl"))))
    p.add_argument("--device", type=str, default=cfg_get(cfg, "training.device", "auto"), choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-procs", type=int, default=0, help="CPU processes for parallel evaluation. 0=auto, 1=serial.")
    p.add_argument("--u-bounds", type=float, nargs=4, default=flat_u_bounds)
    p.add_argument("--L-cut", type=float, default=0.35)
    p.add_argument("--N-global", type=int, default=3000)
    p.add_argument("--m-global", type=int, default=55)
    p.add_argument("--conv-th", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--sample-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--ball-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--random-state", type=int, default=int(cfg_get(cfg, "seed", 0)))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Evaluate the first N observations.")
    p.add_argument("--sample-size", type=int, default=None, help="Evaluate a random subset of N observations.")
    p.add_argument("--sample-seed", type=int, default=0, help="RNG seed used with --sample-size.")
    p.add_argument("--out-root", type=str, default=os.path.join(exp_dir, "test_observation_eval_strict"))
    args = p.parse_args()

    device = _device_from_arg(args.device)
    D = [[float(args.u_bounds[0]), float(args.u_bounds[1])], [float(args.u_bounds[2]), float(args.u_bounds[3])]]
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
    if num_procs <= 1:
        phi_fn, count_fn, _stability_fn = load_inference_functions(
            phi_ckpt=args.phi_ckpt,
            count_ckpt=args.count_ckpt,
            device=device,
        )
        if phi_fn is None or count_fn is None:
            raise RuntimeError("Failed to load phi/count inference functions")

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
        conv_th=float(args.conv_th),
        max_iter=int(args.max_iter),
        sample_method=str(args.sample_method),
        ball_method=str(args.ball_method),
        random_state=int(args.random_state),
        verbose=bool(args.verbose) if num_procs <= 1 else False,
    )
    if num_procs <= 1:
        def predict_fn(theta):
            pred_count = max(0, int(count_fn(theta)))
            if pred_count <= 0:
                return []
            phi_u = phi_fn(theta)
            centers, _init_centers, _history, _layers = adaptive_peak_detection_amr(
                phi_u,
                D,
                num=pred_count,
                **locator_kwargs,
            )
            return centers

        metrics, details = evaluate_observations(
            obs,
            predict_fn=predict_fn,
            count_source="strict_final_solution_count",
        )
    else:
        try:
            ctx = mp.get_context("spawn")
            tasks = (
                (
                    np.asarray(entry["Theta"], dtype=np.float32).reshape(-1),
                    load_true_solutions(entry),
                    D,
                    locator_kwargs,
                )
                for entry in obs
            )
            results = []
            with ctx.Pool(
                processes=int(num_procs),
                initializer=_init_worker,
                initargs=(str(args.phi_ckpt), str(args.count_ckpt)),
            ) as pool:
                for item in tqdm.tqdm(
                    pool.imap(_run_one, tasks, chunksize=1),
                    total=len(obs),
                    desc=f"Evaluating test observations (x{num_procs})",
                ):
                    results.append(item)
            metrics, details = aggregate_evaluation_results(results, count_source="strict_final_solution_count")
        except (PermissionError, OSError) as exc:
            warnings.warn(f"Parallel evaluation unavailable ({exc}); falling back to serial")

            phi_fn, count_fn, _stability_fn = load_inference_functions(
                phi_ckpt=args.phi_ckpt,
                count_ckpt=args.count_ckpt,
                device=device,
            )
            if phi_fn is None or count_fn is None:
                raise RuntimeError("Failed to load phi/count inference functions")

            def predict_fn(theta):
                pred_count = max(0, int(count_fn(theta)))
                if pred_count <= 0:
                    return []
                phi_u = phi_fn(theta)
                centers, _init_centers, _history, _layers = adaptive_peak_detection_amr(
                    phi_u,
                    D,
                    num=pred_count,
                    **locator_kwargs,
                )
                return centers

            num_procs = 1
            metrics, details = evaluate_observations(
                obs,
                predict_fn=predict_fn,
                count_source="strict_final_solution_count",
            )
    metrics.update(
        {
            "script": "_eval_strict.py",
            "obs_path": str(Path(args.obs_path).resolve()),
            "sampling": {
                "limit": args.limit,
                "sample_size": args.sample_size,
                "sample_seed": int(args.sample_seed),
            },
            "locator": {
                "L_cut": float(args.L_cut),
                "N_global": int(args.N_global),
                "m_global": int(args.m_global),
                "conv_th": float(args.conv_th),
                "max_iter": int(args.max_iter),
                "sample_method": str(args.sample_method),
                "ball_method": str(args.ball_method),
                "random_state": int(args.random_state),
            },
            "num_procs": int(num_procs),
        }
    )
    file_tag = f"strict_{str(args.sample_method).lower()}"
    json_path, npz_path = save_outputs(
        out_root=args.out_root,
        stem=f"test_observation_{file_tag}",
        metrics=metrics,
        details=details,
    )

    print(f"Observation test set: {metrics['num_samples']} theta")
    print(f"Count accuracy: {metrics['count_accuracy']:.6f}")
    print(f"Mean L2 (correctly counted theta): {metrics['mean_l2_correctly_counted_theta']:.6f}")
    print(f"Saved metrics to {json_path}")
    print(f"Saved details to {npz_path}")


if __name__ == "__main__":
    main()
