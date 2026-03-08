#!/usr/bin/env python3
"""Draw a bifurcation diagram for the feedback-loop experiment."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
import tqdm

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

import torch

try:
	from sklearn.exceptions import ConvergenceWarning

	warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
	pass

from psnn.loaders import load_inference_functions
from psnn.config import cfg_get, load_yaml, resolve_path

# Reuse the adaptive locator (kept as a single standalone file at repo root).
from locater.flexible import adaptive_peak_detection

from exps.feedback_loop.gen_data import U as true_U


_WORKER_PHI_FN = None
_WORKER_STABILITY_FN = None


def _init_worker(phi_ckpt: str, stability_ckpt: str) -> None:
	"""Initializer for multiprocessing workers.

	Loads models once per process (avoids pickling torch modules/callables).
	"""
	global _WORKER_PHI_FN, _WORKER_STABILITY_FN
	import torch as _torch

	_torch.set_grad_enabled(False)
	# Prevent each process from using many BLAS/OpenMP threads.
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


def _run_one_alpha1(task: tuple) -> tuple:
	"""Run one alpha1 value.

	Returns: (a1, true_u[][2], true_stable[], pred_u[][2], pred_stable[])
	"""
	(
		a1,
		alpha2,
		gamma1,
		gamma2,
		D,
		apd_kwargs,
		stable_thresh,
	) = task
	theta = _theta_from_components(float(a1), float(alpha2), float(gamma1), float(gamma2))

	# True branch (background)
	tru = true_U(theta)
	true_u: list[list[float]] = []
	true_stable: list[bool] = []
	if tru is not None:
		for sol in tru:
			true_u.append([float(sol["u"][0]), float(sol["u"][1])])
			true_stable.append(bool(sol["stable"]))

	# Predicted centers
	if _WORKER_PHI_FN is None or _WORKER_STABILITY_FN is None:
		raise RuntimeError("Worker inference functions are not initialized")
	phi_u = _WORKER_PHI_FN(theta)
	centers, _init_centers, _history, _layers = adaptive_peak_detection(phi_u, D, **apd_kwargs)

	pred_u: list[list[float]] = []
	pred_stable: list[bool] = []
	if getattr(centers, "size", 0) != 0:
		p_stable = _WORKER_STABILITY_FN(theta, centers)
		for u, prob in zip(centers, p_stable):
			pred_u.append([float(u[0]), float(u[1])])
			pred_stable.append(bool(float(prob) >= float(stable_thresh)))

	return (float(a1), true_u, true_stable, pred_u, pred_stable)


def _device_from_arg(device: str) -> torch.device:
	device = str(device).lower()
	if device == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device)


def _theta_from_components(alpha1: float, alpha2: float, gamma1: float, gamma2: float) -> np.ndarray:
	return np.asarray([alpha1, alpha2, gamma1, gamma2], dtype=np.float32)


def main() -> None:
	pre = argparse.ArgumentParser(add_help=False)
	pre.add_argument("--config", type=str, default=os.path.join(exp_dir, "config.yaml"))
	pre_args, _ = pre.parse_known_args()

	cfg_path = resolve_path(exp_dir, pre_args.config)
	cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
	path_cfg = cfg_get(cfg, "training.paths", {})
	u_bounds_cfg = cfg_get(cfg, "data_generation.domain.u_bounds", [[-1.0, 6.0], [-1.0, 6.0]])
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
	p.add_argument(
		"--num-procs",
		type=int,
		default=0,
		help="CPU processes for parallel sweep. 0=auto, 1=serial. Parallel mode forces CPU models.",
	)

	# theta = (alpha1, alpha2, gamma1, gamma2)
	# Bifurcating parameter: alpha1
	p.add_argument("--alpha2", type=float, default=1.9)
	p.add_argument("--gamma1", type=float, default=0.6)
	p.add_argument("--gamma2", type=float, default=0.2)
	p.add_argument("--alpha1-min", type=float, default=1.2)
	p.add_argument("--alpha1-max", type=float, default=4.8)
	p.add_argument("--alpha1-steps", type=int, default=51)

	# u=(p1,p2) domain
	p.add_argument(
		"--u-bounds",
		type=float,
		nargs=4,
		default=flat_u_bounds,
		help="2D bounds for u as: u0_low u0_high u1_low u1_high",
	)

	# adaptive_peak_detection hyperparams
	p.add_argument("--L-cut", type=float, default=0.35)
	p.add_argument("--N-global", type=int, default=3000)
	p.add_argument("--m-global", type=int, default=50)
	p.add_argument("--C-max", type=int, default=4)
	p.add_argument("--r-init", type=float, default=0.3)
	p.add_argument("--conv-steps", type=int, default=2)
	p.add_argument("--sample-method", type=str, default="grid", choices=["grid", "uniform"])
	p.add_argument("--ball-method", type=str, default="grid", choices=["grid", "uniform"])
	p.add_argument("--random-state", type=int, default=int(cfg_get(cfg, "seed", 0)))
	p.add_argument("--verbose", action="store_true")

	# output
	p.add_argument("--out-root", type=str, default=os.path.join(exp_dir, "bifur_flexible_runs"))

	args = p.parse_args()

	device = _device_from_arg(args.device)
	D = [[float(args.u_bounds[0]), float(args.u_bounds[1])], [float(args.u_bounds[2]), float(args.u_bounds[3])]]

	num_procs = int(args.num_procs)
	if num_procs == 0:
		cpu_n = os.cpu_count() or 1
		num_procs = min(8, max(1, cpu_n - 1))

	# Parallel mode is CPU-only to avoid multi-process CUDA contention.
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

	alpha1_grid = np.linspace(float(args.alpha1_min), float(args.alpha1_max), int(args.alpha1_steps), dtype=np.float32)

	pred_centers_by_alpha1: list[np.ndarray] = []
	pred_stability_by_alpha1: list[np.ndarray] = []
	true_centers_by_alpha1: list[np.ndarray] = []
	true_stability_by_alpha1: list[np.ndarray] = []

	apd_kwargs = dict(
		L_cut=float(args.L_cut),
		N_global=int(args.N_global),
		m_global=int(args.m_global),
		C_max=int(args.C_max),
		r_init=float(args.r_init),
		conv_steps=int(args.conv_steps),
		sample_method=str(args.sample_method),
		ball_method=str(args.ball_method),
		random_state=int(args.random_state),
		verbose=bool(args.verbose) if num_procs <= 1 else False,
	)
	stable_thresh = 0.5

	if num_procs <= 1:
		# Serial execution
		for a1 in tqdm.tqdm(alpha1_grid, desc="Processing alpha1 values"):
			theta = _theta_from_components(float(a1), args.alpha2, args.gamma1, args.gamma2)

			# True branch (background)
			tru = true_U(theta)
			true_centers = np.empty((0, 2), dtype=np.float32)
			true_stability = np.empty((0,), dtype=bool)
			if tru is not None:
				true_centers = np.asarray([[float(sol["u"][0]), float(sol["u"][1])] for sol in tru], dtype=np.float32)
				true_stability = np.asarray([bool(sol["stable"]) for sol in tru], dtype=bool)
			true_centers_by_alpha1.append(true_centers)
			true_stability_by_alpha1.append(true_stability)

			# Predicted centers
			phi_u = phi_fn(theta)
			centers, _init_centers, _history, _layers = adaptive_peak_detection(phi_u, D, **apd_kwargs)
			if centers.size == 0:
				pred_centers_by_alpha1.append(np.empty((0, 2), dtype=np.float32))
				pred_stability_by_alpha1.append(np.empty((0,), dtype=bool))
				continue

			p_stable = stability_fn(theta, centers)
			pred_centers_by_alpha1.append(np.asarray(centers, dtype=np.float32))
			pred_stability_by_alpha1.append(np.asarray(p_stable >= stable_thresh, dtype=bool))
	else:
		# Parallel execution on CPU
		if not os.path.exists(args.phi_ckpt):
			raise FileNotFoundError(args.phi_ckpt)
		if not os.path.exists(args.stability_ckpt):
			raise FileNotFoundError(args.stability_ckpt)

		ctx = mp.get_context("spawn")
		tasks = (
			(
				float(a1),
				float(args.alpha2),
				float(args.gamma1),
				float(args.gamma2),
				D,
				apd_kwargs,
				stable_thresh,
			)
			for a1 in alpha1_grid
		)

		with ctx.Pool(
			processes=int(num_procs),
			initializer=_init_worker,
			initargs=(str(args.phi_ckpt), str(args.stability_ckpt)),
		) as pool:
			for _a1, t_u, t_stable, p_u, p_stable in tqdm.tqdm(
				pool.imap(_run_one_alpha1, tasks, chunksize=1),
				total=len(alpha1_grid),
				desc=f"Processing alpha1 values (x{num_procs})",
			):
				true_centers_by_alpha1.append(np.asarray(t_u, dtype=np.float32).reshape(-1, 2))
				true_stability_by_alpha1.append(np.asarray(t_stable, dtype=bool))
				pred_centers_by_alpha1.append(np.asarray(p_u, dtype=np.float32).reshape(-1, 2))
				pred_stability_by_alpha1.append(np.asarray(p_stable, dtype=bool))

	out_dir = Path(args.out_root)
	out_dir.mkdir(parents=True, exist_ok=True)
	fig_path = out_dir / "bifur_flexible.png"
	data_path = out_dir / "bifur_flexible_data.npz"

	def _flatten_component(
		alpha1_values: np.ndarray,
		centers_by_alpha1: list[np.ndarray],
		stability_by_alpha1: list[np.ndarray],
		component_idx: int,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		x_vals: list[float] = []
		y_vals: list[float] = []
		stable_vals: list[bool] = []
		for alpha1, centers, stable in zip(alpha1_values, centers_by_alpha1, stability_by_alpha1):
			for center, is_stable in zip(centers, stable):
				x_vals.append(float(alpha1))
				y_vals.append(float(center[component_idx]))
				stable_vals.append(bool(is_stable))
		return (
			np.asarray(x_vals, dtype=float),
			np.asarray(y_vals, dtype=float),
			np.asarray(stable_vals, dtype=bool),
		)

	fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
	for ax, component_idx, ylabel in zip(axes, (0, 1), (r"$p_1$", r"$p_2$")):
		x_true_np, y_true_np, stable_true_np = _flatten_component(alpha1_grid, true_centers_by_alpha1, true_stability_by_alpha1, component_idx)
		x_pred_np, y_pred_np, stable_pred_np = _flatten_component(alpha1_grid, pred_centers_by_alpha1, pred_stability_by_alpha1, component_idx)

		if x_true_np.size > 0:
			ax.scatter(x_true_np[stable_true_np], y_true_np[stable_true_np], s=10, c="0.75", marker="o", alpha=0.6, label="true stable")
			ax.scatter(x_true_np[~stable_true_np], y_true_np[~stable_true_np], s=10, c="0.85", marker="x", alpha=0.6, label="true unstable")
		if x_pred_np.size > 0:
			ax.scatter(x_pred_np[stable_pred_np], y_pred_np[stable_pred_np], s=16, c="tab:blue", marker="o", alpha=0.9, label="pred stable")
			ax.scatter(x_pred_np[~stable_pred_np], y_pred_np[~stable_pred_np], s=16, c="tab:orange", marker="x", alpha=0.9, label="pred unstable")

		ax.set_xlabel(r"$\alpha_1$")
		ax.set_ylabel(ylabel)
		ax.set_title(f"Feedback-loop bifurcation ({ylabel})")
		ax.grid(True, alpha=0.25)
		ax.legend(loc="best", fontsize=9)

	fig.tight_layout()
	fig.savefig(fig_path, dpi=200)
	plt.close(fig)

	np.savez(
		data_path,
		metadata_json=np.asarray(
			json.dumps(
				{
					"script": "bifur_flexible.py",
					"args": vars(args),
					"stable_threshold": stable_thresh,
				},
				sort_keys=True,
			)
		),
		alpha1_grid=alpha1_grid,
		true_centers=np.asarray(true_centers_by_alpha1, dtype=object),
		true_stability=np.asarray(true_stability_by_alpha1, dtype=object),
		pred_centers=np.asarray(pred_centers_by_alpha1, dtype=object),
		pred_stability=np.asarray(pred_stability_by_alpha1, dtype=object),
	)

	print(f"Saved figure: {fig_path}")
	print(f"Saved data: {data_path}")
	print(f"Predicted slices: {sum(len(v) for v in pred_centers_by_alpha1)} | True slices: {sum(len(v) for v in true_centers_by_alpha1)}")


if __name__ == "__main__":
	main()
