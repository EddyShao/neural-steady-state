"""Flexible locater runner (single-theta) — AMR version with known #clusters.

This module used to live at the repository top-level as locater_strict.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from psnn.loaders import load_phi_model, make_phi_function

from sklearn.cluster import KMeans


# -------------------------
# Sampling helpers
# -------------------------
def _device_from_cfg(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg or "auto").lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grid_sampling(D: list[list[float]], m: int) -> np.ndarray:
    dim = len(D)
    grids = [np.linspace(D[i][0], D[i][1], m) for i in range(dim)]
    mesh = np.meshgrid(*grids)
    return np.stack(mesh, axis=-1).reshape(-1, dim).astype(np.float32)


def uniform_sampling(D: list[list[float]], N: int, rng: np.random.Generator) -> np.ndarray:
    dim = len(D)
    low = np.asarray([D[i][0] for i in range(dim)], dtype=np.float32)
    high = np.asarray([D[i][1] for i in range(dim)], dtype=np.float32)
    return rng.uniform(low=low, high=high, size=(int(N), dim)).astype(np.float32)


def grid_ball_sampling(center: np.ndarray, radius: float, m: int) -> np.ndarray:
    """Grid sampling within a box around a center point.

    This is actually an L_infty box: [c_i - r, c_i + r] in each dimension.
    """
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dim = int(center.shape[0])
    grids = [np.linspace(center[i] - radius, center[i] + radius, int(m)) for i in range(dim)]
    mesh = np.meshgrid(*grids)
    points = np.stack(mesh, axis=-1).reshape(-1, dim).astype(np.float32)
    return points


def uniform_ball_sampling(center: np.ndarray, radius: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sampling within an L2-ball in R^d."""
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dim = int(center.shape[0])
    N = int(N)
    directions = rng.normal(size=(N, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    radii = (rng.random(N).astype(np.float32) ** (1.0 / dim)).reshape(-1, 1)
    return center.reshape(1, -1) + directions * (radii * float(radius))


# -------------------------
# Clustering + radii
# -------------------------
def _cluster_fixed_k(points: np.ndarray, k: int, *, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run KMeans with fixed k and return (centers, per-center radii)."""
    points = np.asarray(points, dtype=np.float32)
    n = int(points.shape[0])
    dim = int(points.shape[1])

    if k <= 0 or n == 0:
        return np.empty((0, dim), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if n < k:
        # Not enough points to support k clusters.
        return np.empty((0, dim), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if k == 1:
        c = np.mean(points, axis=0, keepdims=True)
        r = float(np.max(np.linalg.norm(points - c[0], axis=1)))
        return c.astype(np.float32), np.asarray([r], dtype=np.float32)

    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(random_state)).fit(points)
    centers = km.cluster_centers_.astype(np.float32)
    labels = km.labels_

    radii = np.zeros((k,), dtype=np.float32)
    for j in range(k):
        mask = labels == j
        if not np.any(mask):
            radii[j] = 0.0
            continue
        pts = points[mask]
        radii[j] = float(np.max(np.linalg.norm(pts - centers[j], axis=1)))
    return centers, radii


# -------------------------
# AMR utilities
# -------------------------
def _lex_sort_centers(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers)
    if centers.size == 0:
        return centers
    keys = [centers[:, j] for j in reversed(range(centers.shape[1]))]
    order = np.lexsort(keys)
    return centers[order]


def _metric_dist(c1: np.ndarray, c2: np.ndarray, metric: str) -> float:
    if metric == "l2":
        return float(np.linalg.norm(c1 - c2))
    if metric == "linf":
        return float(np.max(np.abs(c1 - c2)))
    raise ValueError(f"Unknown metric={metric}")


def _merge_overlapping_regions(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy merge: if distance(c_i,c_j) < r_i + r_j, merge them."""
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)

    if centers.shape[0] <= 1:
        return centers, radii

    C = [centers[i].copy() for i in range(centers.shape[0])]
    R = [float(radii[i]) for i in range(radii.shape[0])]

    changed = True
    while changed:
        changed = False
        m = len(C)
        i = 0
        while i < m:
            j = i + 1
            while j < m:
                d = _metric_dist(C[i], C[j], metric)
                if d < (R[i] + R[j]):
                    # Merge into one region covering both conservatively.
                    new_center = 0.5 * (C[i] + C[j])
                    new_radius = 0.5 * d + max(R[i], R[j])

                    C[i] = new_center.astype(np.float32)
                    R[i] = float(new_radius)

                    C.pop(j)
                    R.pop(j)
                    m -= 1
                    changed = True
                    continue
                j += 1
            i += 1

    return np.asarray(C, dtype=np.float32), np.asarray(R, dtype=np.float32)


@dataclass(frozen=True)
class AdaptiveStep:
    layer: int
    centers: np.ndarray
    radii: np.ndarray
    n_collected: int
    move: float
    action: str


def adaptive_peak_detection_amr(
    f: Callable[[np.ndarray], np.ndarray],
    D: list[list[float]],
    *,
    L_cut: float = 0.35,
    N_global: int = 3000,
    m_global: int = 55,
    num: int = 3,
    conv_th: float = 1e-2,
    max_iter: int = 25,
    sample_method: str = "grid",
    ball_method: str = "grid",
    random_state: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[AdaptiveStep], list[int]]:
    """AMR with fixed number of clusters (no split), plus overlap merging."""
    sample_method = str(sample_method).lower()
    ball_method = str(ball_method).lower()
    dim = len(D)
    rng = np.random.default_rng(int(random_state))

    metric = "l2" if ball_method == "uniform" else "linf"

    def _fmt_budget(value: int, *, method: str, label: str) -> str:
        if method == "uniform":
            return f"{label}={value}"
        try:
            pts = int(value) ** int(dim)
        except Exception:
            pts = -1
        return f"m_{label}={value} (pts={pts})"

    # 1) global init
    if sample_method == "uniform":
        samples = uniform_sampling(D, int(N_global), rng)
    else:
        samples = grid_sampling(D, int(m_global))

    vals = f(samples)
    collected = samples[vals >= float(L_cut) * float(vals.max())]

    if collected.shape[0] == 0:
        if verbose:
            print("[init] collected=0 -> no centers")
        return np.empty((0, dim), np.float32), np.empty((0, dim), np.float32), [], []

    centers, radii = _cluster_fixed_k(collected, int(num), random_state=int(random_state))
    if centers.size == 0:
        if verbose:
            print(f"[init] collected={collected.shape[0]} but cannot form num={num} clusters -> no centers")
        return np.empty((0, dim), np.float32), np.empty((0, dim), np.float32), [], []

    init_centers = centers.copy()

    centers, radii = _merge_overlapping_regions(centers, radii, metric=metric)

    history: list[AdaptiveStep] = []
    prev_sorted = _lex_sort_centers(centers)

    if verbose:
        global_budget = _fmt_budget(int(N_global), method="uniform", label="global") if sample_method == "uniform" else _fmt_budget(int(m_global), method="grid", label="global")
        local_budget = _fmt_budget(int(N_global), method="uniform", label="local") if ball_method == "uniform" else _fmt_budget(int(m_global), method="grid", label="local")
        print(f"[init] collected={len(collected)} init_centers={init_centers.tolist()}")
        print(f"[init] centers(after-merge)={centers.tolist()} radii={radii.tolist()} metric={metric} ({global_budget}, {local_budget})")

    # 2-3) local refine + merge, until converged
    for it in range(1, int(max_iter) + 1):
        new_centers = []
        new_radii = []
        n_col_total = 0

        for c, r in zip(centers, radii):
            r_use = float(max(r, 1e-6))

            if ball_method == "uniform":
                local = uniform_ball_sampling(c, r_use, int(N_global), rng)
            else:
                local = grid_ball_sampling(c, r_use, int(m_global))

            v = f(local)
            col = local[v >= float(L_cut) * float(v.max())]
            n_col_total += int(col.shape[0])

            if col.shape[0] == 0:
                # Keep center; shrink to encourage escape from empty region
                new_centers.append(c.astype(np.float32))
                new_radii.append(np.float32(0.5 * r_use))
                continue

            c_new = np.mean(col, axis=0).astype(np.float32)
            r_new = float(np.max(np.linalg.norm(col - c_new, axis=1)))
            new_centers.append(c_new)
            new_radii.append(np.float32(r_new))

        centers = np.asarray(new_centers, dtype=np.float32)
        radii = np.asarray(new_radii, dtype=np.float32)

        # merge overlaps (may reduce count)
        centers, radii = _merge_overlapping_regions(centers, radii, metric=metric)

        # convergence check (lex-sort)
        curr_sorted = _lex_sort_centers(centers)
        if prev_sorted.shape == curr_sorted.shape and curr_sorted.size > 0:
            move = float(np.max(np.linalg.norm(curr_sorted - prev_sorted, axis=1)))
        else:
            move = float("inf")

        action = f"refine+merge(k={centers.shape[0]})"
        history.append(
            AdaptiveStep(
                layer=int(it),
                centers=centers.copy(),
                radii=radii.copy(),
                n_collected=int(n_col_total),
                move=float(move),
                action=action,
            )
        )

        if verbose:
            print(f"[iter {it}] {action} centers={centers.tolist()} radii={radii.tolist()} move={move:.3e} collected={n_col_total}")

        if move < float(conv_th):
            if verbose:
                print(f"[done] converged at iter {it}: move={move:.3e} < conv_th={conv_th}")
            break

        prev_sorted = curr_sorted

    layers = [step.layer for step in history]
    return centers, init_centers, history, layers


# -------------------------
# Plotting
# -------------------------
def _plot_phi_landscape(
    *,
    phi_u: Callable[[np.ndarray], np.ndarray],
    D: list[list[float]],
    centers: np.ndarray,
    init_centers: np.ndarray,
    center_layers: list[int] | None,
    save_path: Path,
    m: int = 220,
) -> None:
    if len(D) != 2:
        raise ValueError("Visualization currently supports 2D u only.")

    grid = grid_sampling(D, m)
    vals = phi_u(grid).reshape(m, m)

    x = np.linspace(D[0][0], D[0][1], m)
    y = np.linspace(D[1][0], D[1][1], m)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(6.5, 5.5))
    cs = plt.contourf(X, Y, vals, levels=50)
    plt.colorbar(cs, label=r"$\phi(u;\theta)$")

    if init_centers.size:
        plt.scatter(init_centers[:, 0], init_centers[:, 1], s=60, c="yellow", marker="o", edgecolors="k", label="init")
    if centers.size:
        plt.scatter(centers[:, 0], centers[:, 1], s=90, c="red", marker="*", edgecolors="k", label="final")

        if center_layers is not None and len(center_layers) == int(centers.shape[0]):
            for i in range(int(centers.shape[0])):
                plt.text(
                    float(centers[i, 0]),
                    float(centers[i, 1]),
                    str(int(center_layers[i])),
                    color="black",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

    plt.xlim(D[0][0], D[0][1])
    plt.ylim(D[1][0], D[1][1])
    plt.xlabel("u[0]")
    plt.ylabel("u[1]")
    plt.title("AMR peak detection (fixed K + merge)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -------------------------
# CLI
# -------------------------
def _parse_theta(vals: Iterable[str]) -> np.ndarray:
    return np.asarray([float(v) for v in vals], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Run AMR locater for a single theta (fixed num clusters).")
    p.add_argument("--phi-ckpt", type=str, required=True, help="Path to phi checkpoint (.pt)")
    p.add_argument("--theta", type=float, nargs="+", required=True, help="Theta parameters, e.g. --theta a1 a2 g1 g2")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")

    p.add_argument("--L-cut", type=float, default=0.35)
    p.add_argument("--N-global", type=int, default=3000)
    p.add_argument("--m-global", type=int, default=55)
    p.add_argument("--num", type=int, default=3, help="Known number of clusters for KMeans init.")
    p.add_argument("--conv-th", type=float, default=1e-2, help="Convergence threshold on center movement.")
    p.add_argument("--max-iter", type=int, default=25)
    p.add_argument("--random-state", type=int, default=0)

    p.add_argument("--sample-method", type=str, default="grid", choices=["grid", "uniform"])
    p.add_argument("--ball-method", type=str, default="grid", choices=["grid", "uniform"])

    p.add_argument("--u-bounds", type=float, nargs=4, default=[-1.0, 6.0, -1.0, 6.0], help="u0_lo u0_hi u1_lo u1_hi")
    p.add_argument("--out-root", type=str, default="locater_flexible_runs")
    p.add_argument("--no-plot", action="store_true")

    args = p.parse_args()

    theta = np.asarray(args.theta, dtype=np.float32).reshape(-1)
    D = [[float(args.u_bounds[0]), float(args.u_bounds[1])], [float(args.u_bounds[2]), float(args.u_bounds[3])]]

    device = _device_from_cfg(args.device)
    print(f"Device: {device}")
    print(f"Theta: {theta.tolist()}")
    print(f"Domain D: {D}")

    phi_model = load_phi_model(args.phi_ckpt, device)
    phi = make_phi_function(phi_model, device=device)
    phi_u = phi(theta)

    centers, init_centers, history, layers = adaptive_peak_detection_amr(
        phi_u,
        D,
        L_cut=float(args.L_cut),
        N_global=int(args.N_global),
        m_global=int(args.m_global),
        num=int(args.num),
        conv_th=float(args.conv_th),
        max_iter=int(args.max_iter),
        sample_method=str(args.sample_method),
        ball_method=str(args.ball_method),
        random_state=int(args.random_state),
        verbose=True,
    )

    print(f"Final centers: {centers.tolist()}")
    print(f"Num centers (after merge): {int(centers.shape[0])}")

    if not args.no_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_root) / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "locater_flexible.png"

        # For labeling: if converged, you can label by last iter; otherwise blank.
        label_layers = None
        if centers.size and history:
            label_layers = [int(history[-1].layer)] * int(centers.shape[0])

        _plot_phi_landscape(
            phi_u=phi_u,
            D=D,
            centers=centers,
            init_centers=init_centers,
            center_layers=label_layers,
            save_path=fig_path,
            m=220,
        )
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
