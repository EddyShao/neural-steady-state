#!/usr/bin/env python3
"""Visualize generated feedback-loop observations.

This script loads the joblib-dumped observation lists produced by
[exps/feedback_loop/data/gen_feedback_loop.py](exps/feedback_loop/data/gen_feedback_loop.py)
(typically `feedback_loop_obs_train.pkl` / `feedback_loop_obs_test.pkl`).

Each observation is expected to be a dict with:
- "Theta": (4,) array = (alpha1, alpha2, gamma1, gamma2)
- "U": list of solution dicts, each with keys {"u": (2,), "stable": bool}

Plots:
- Histogram of number of solutions per observation.
- Scatter projections of theta colored by solution count.
- A grid of per-observation solution locations in u-space.

Examples:
  python exps/feedback_loop/visualize_observations.py --which train --save exps/feedback_loop/obs_train_viz.png
  python exps/feedback_loop/visualize_observations.py --which test --n-show 12
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/feedback_loop
repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psnn.config import cfg_get, load_yaml, resolve_path


def _load_obs(path: str) -> List[Dict]:
    try:
        import joblib
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required to load observation .pkl files") from e

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing observations file: {path}. "
            "Generate it with: python exps/feedback_loop/data/gen_feedback_loop.py"
        )
    obs_list = joblib.load(path)
    if not isinstance(obs_list, list):
        raise ValueError(f"Expected a list of observations in {path}, got: {type(obs_list)}")
    return obs_list


def _extract_arrays(obs_list: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Theta (N,4), counts (N,), stable_counts (N,)."""
    thetas = []
    counts = []
    stable_counts = []

    for obs in obs_list:
        theta = np.asarray(obs.get("Theta"), dtype=float).reshape(-1)
        if theta.shape[0] != 4:
            raise ValueError(f"Observation has Theta shape {theta.shape}, expected (4,)")

        sols = obs.get("U", [])
        if sols is None:
            sols = []
        if not isinstance(sols, list):
            raise ValueError(f"Observation has U type {type(sols)}, expected list")

        c = int(len(sols))
        s = 0
        for sol in sols:
            if isinstance(sol, dict) and bool(sol.get("stable", False)):
                s += 1

        thetas.append(theta)
        counts.append(c)
        stable_counts.append(s)

    return np.asarray(thetas, dtype=float), np.asarray(counts, dtype=int), np.asarray(stable_counts, dtype=int)


def _count_color(count: int) -> str:
    # Keep this simple and readable.
    if count <= 0:
        return "royalblue"
    if count == 1:
        return "black"
    if count == 2:
        return "darkorange"
    if count == 3:
        return "crimson"
    return "purple"


def plot_summary(obs_list: Sequence[Dict], *, title: str) -> plt.Figure:
    thetas, counts, stable_counts = _extract_arrays(obs_list)

    fig = plt.figure(figsize=(12.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    ax_hist = fig.add_subplot(gs[0, 0])
    max_count = int(max(counts.max(), 1))
    bins = np.arange(-0.5, max_count + 1.5, 1.0)
    ax_hist.hist(counts, bins=bins, color="slategray", edgecolor="white")
    ax_hist.set_xlabel("# solutions")
    ax_hist.set_ylabel("# observations")
    ax_hist.set_title("Solution count histogram")

    # alpha1 vs alpha2
    ax_a = fig.add_subplot(gs[0, 1])
    colors = [_count_color(int(c)) for c in counts]
    ax_a.scatter(thetas[:, 0], thetas[:, 1], c=colors, s=10, alpha=0.8)
    ax_a.set_xlabel("alpha1")
    ax_a.set_ylabel("alpha2")
    ax_a.set_title("(alpha1, alpha2) colored by #solutions")

    # gamma1 vs gamma2
    ax_g = fig.add_subplot(gs[1, 0])
    ax_g.scatter(thetas[:, 2], thetas[:, 3], c=colors, s=10, alpha=0.8)
    ax_g.set_xlabel("gamma1")
    ax_g.set_ylabel("gamma2")
    ax_g.set_title("(gamma1, gamma2) colored by #solutions")

    # stable fraction vs alpha2
    ax_sf = fig.add_subplot(gs[1, 1])
    denom = np.maximum(counts, 1)
    stable_frac = stable_counts / denom
    ax_sf.scatter(thetas[:, 1], stable_frac, c=colors, s=10, alpha=0.8)
    ax_sf.set_xlabel("alpha2")
    ax_sf.set_ylabel("stable fraction (stable/total)")
    ax_sf.set_title("Stability summary vs alpha2")
    ax_sf.set_ylim(-0.05, 1.05)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def plot_u_panels(
    obs_list: Sequence[Dict],
    *,
    indices: Sequence[int],
    u_bounds: Optional[np.ndarray] = None,
    title: str,
) -> plt.Figure:
    if len(indices) == 0:
        raise ValueError("No indices provided to plot_u_panels")

    u_bounds = np.asarray(u_bounds, dtype=float) if u_bounds is not None else None
    n = len(indices)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(4.2 * cols, 3.6 * rows))

    for i, idx in enumerate(indices, start=1):
        ax = fig.add_subplot(rows, cols, i)
        obs = obs_list[int(idx)]
        theta = np.asarray(obs.get("Theta"), dtype=float).reshape(-1)
        sols = obs.get("U", []) or []

        # Plot solution centers in u-space.
        if len(sols) > 0:
            pts = np.asarray([np.asarray(s["u"], dtype=float) for s in sols], dtype=float)
            stable_mask = np.asarray([bool(s.get("stable", False)) for s in sols], dtype=bool)
            if np.any(stable_mask):
                ax.scatter(pts[stable_mask, 0], pts[stable_mask, 1], c="royalblue", s=40, marker="o", label="stable")
            if np.any(~stable_mask):
                ax.scatter(pts[~stable_mask, 0], pts[~stable_mask, 1], c="hotpink", s=40, marker="x", label="unstable")
        else:
            ax.text(0.5, 0.5, "no solutions", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("p1")
        ax.set_ylabel("p2")
        if u_bounds is not None and u_bounds.shape == (2, 2):
            ax.set_xlim(float(u_bounds[0, 0]), float(u_bounds[0, 1]))
            ax.set_ylim(float(u_bounds[1, 0]), float(u_bounds[1, 1]))

        a1, a2, g1, g2 = theta.tolist()
        ax.set_title(f"idx={idx} | k={len(sols)}\n(a1,a2,g1,g2)=({a1:.2f},{a2:.2f},{g1:.2f},{g2:.2f})")
        if len(sols) > 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def _select_indices(n_total: int, *, n_show: int, seed: int, indices: Optional[List[int]]) -> List[int]:
    if indices is not None and len(indices) > 0:
        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise ValueError(f"Index {idx} out of range for obs_list length {n_total}")
        return [int(i) for i in indices]

    n_show = max(1, int(n_show))
    rng = np.random.default_rng(int(seed))
    n_show = min(n_show, n_total)
    return rng.choice(np.arange(n_total), size=n_show, replace=False).tolist()


def _save_or_show(fig: plt.Figure, *, path: Optional[str]):
    if path:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}")
    else:
        plt.show()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    pre_args, remaining = pre.parse_known_args()

    default_cfg = os.path.join(exp_dir, "config.yaml")
    cfg = {}
    cfg_path = pre_args.config or (default_cfg if os.path.exists(default_cfg) else None)
    if cfg_path:
        cfg = load_yaml(cfg_path)

    dg = cfg_get(cfg, "data_generation", {})
    out_cfg = cfg_get(dg, "outputs", {})

    parser = argparse.ArgumentParser(description="Visualize feedback-loop generated observations.", parents=[pre])
    parser.add_argument("--which", choices=["train", "test", "both"], default="train", help="Which observations to load")
    parser.add_argument("--obs-train", type=str, default=cfg_get(out_cfg, "obs_train_pkl", "feedback_loop_obs_train.pkl"), help="Train obs .pkl path")
    parser.add_argument("--obs-test", type=str, default=cfg_get(out_cfg, "obs_test_pkl", "feedback_loop_obs_test.pkl"), help="Test obs .pkl path")
    parser.add_argument("--out-dir", type=str, default=cfg_get(out_cfg, "out_dir", "data"), help="Base directory used to resolve relative obs paths")

    parser.add_argument("--n-show", type=int, default=12, help="How many random observations to show in u-panels")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random selection")
    parser.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit observation indices to visualize")

    parser.add_argument("--save", type=str, default=None, help="If set, saves figures with this base path (suffixes are added)")
    args = parser.parse_args(remaining)

    u_bounds = np.asarray(cfg_get(dg, "domain.u_bounds", [[0.0, 5.0], [0.0, 5.0]]), dtype=float)

    base_dir = resolve_path(exp_dir, args.out_dir)
    train_path = resolve_path(base_dir, args.obs_train)
    test_path = resolve_path(base_dir, args.obs_test)

    to_process: List[Tuple[str, str]] = []
    if args.which in ("train", "both"):
        to_process.append(("train", str(train_path)))
    if args.which in ("test", "both"):
        to_process.append(("test", str(test_path)))

    for split, path in to_process:
        obs_list = _load_obs(path)
        print(f"Loaded {len(obs_list)} observations from: {path}")

        fig1 = plot_summary(obs_list, title=f"Feedback-loop observations ({split})")
        save1 = None
        save2 = None
        if args.save:
            root, ext = os.path.splitext(args.save)
            ext = ext or ".png"
            save1 = f"{root}_{split}_summary{ext}"
            save2 = f"{root}_{split}_u_panels{ext}"
        _save_or_show(fig1, path=save1)

        idxs = _select_indices(len(obs_list), n_show=args.n_show, seed=args.seed, indices=args.indices)
        fig2 = plot_u_panels(obs_list, indices=idxs, u_bounds=u_bounds, title=f"Solution centers in u-space ({split})")
        _save_or_show(fig2, path=save2)


if __name__ == "__main__":
    main()
