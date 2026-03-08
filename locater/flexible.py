"""Flexible locater runner (single-theta).

This module used to live at the repository top-level as locater_flexible.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from collections import deque

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from psnn.loaders import load_phi_model, make_phi_function


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    return np.stack(mesh, axis=-1).reshape(-1, dim)


def uniform_sampling(D: list[list[float]], N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sampling over a hyper-rectangle domain."""
    dim = len(D)
    low = np.asarray([D[i][0] for i in range(dim)], dtype=np.float32)
    high = np.asarray([D[i][1] for i in range(dim)], dtype=np.float32)
    return rng.uniform(low=low, high=high, size=(int(N), dim)).astype(np.float32)


def grid_ball_sampling(center: np.ndarray, radius: float, m: int) -> np.ndarray:
    r"""Grid sampling within a box-shaped (\ell^1) ball around a center point."""
    dim = int(center.shape[0])
    grids = [np.linspace(center[i] - radius, center[i] + radius, m) for i in range(dim)]
    mesh = np.meshgrid(*grids)
    points = np.stack(mesh, axis=-1).reshape(-1, dim)
    return points


def uniform_ball_sampling(center: np.ndarray, radius: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random sampling within an L2-ball in R^d."""
    center = np.asarray(center, dtype=np.float32).reshape(-1)
    dim = int(center.shape[0])
    N = int(N)
    directions = rng.normal(size=(N, dim)).astype(np.float32)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    # Radii for uniform volume distribution
    radii = (rng.random(N).astype(np.float32) ** (1.0 / dim)).reshape(-1, 1)
    return center.reshape(1, -1) + directions * (radii * float(radius))


def valley_between(c1, c2, f, n_samples=100, eps=1e-3):
    t = np.linspace(0, 1, n_samples + 2)[1:-1]

    pts = (1 - t)[:, None] * c1 + t[:, None] * c2
    vals = f(pts)

    peak_val = min(f(c1[None, :])[0], f(c2[None, :])[0])

    return np.mean(vals) < peak_val - eps


def merge_centers(points, centers, labels, valley_eps=1e-3, f=None):

    centers = centers.copy()
    labels = labels.copy()

    K = len(centers)

    keep = np.ones(K, dtype=bool)

    for i in range(K):
        if not keep[i]:
            continue

        for j in range(i + 1, K):
            if not keep[j]:
                continue

            if f is None:
                continue

            has_valley = valley_between(
                centers[i],
                centers[j],
                f,
                eps=valley_eps,
            )

            # merge if NO valley
            if not has_valley:
                labels[labels == j] = i
                keep[j] = False

    # compress labels
    unique = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique)}

    labels_new = np.array([label_map[l] for l in labels])

    # recompute centers and radii
    centers_new = []
    radii_new = []

    for j in range(len(unique)):
        mask = labels_new == j
        cluster_pts = points[mask]

        c = np.mean(cluster_pts, axis=0)
        r = np.max(np.linalg.norm(cluster_pts - c, axis=1))

        centers_new.append(c)
        radii_new.append(r)

    centers_new = np.asarray(centers_new)
    radii_new = np.asarray(radii_new)

    return centers_new, radii_new, labels_new


def cluster_points(points: np.ndarray, C_max: int = 5, random_state: int = 0, f=None) -> tuple[np.ndarray, np.ndarray]:
    """Cluster high-score points and return (centers, radii).

    Radii are per-center, computed as the max distance from the center to its assigned points.
    """

    n = len(points)
    if n <= 2:
        center = np.mean(points, axis=0, keepdims=True)
        radius = float(np.max(np.linalg.norm(points - center, axis=1))) if n > 0 else 0.0
        return center, np.asarray([radius], dtype=np.float32)

    candidate_C = list(range(2, min(C_max, n - 1) + 1))
    scores: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}

    for k in candidate_C:
        km = KMeans(n_clusters=k, n_init=10, random_state=int(random_state)).fit(points)

        labels = km.labels_
        centers = km.cluster_centers_

        if len(np.unique(labels)) < 2:
            continue

        try:
            s = silhouette_score(points, labels)
        except Exception:
            continue

        # Store silhouette, centers, labels
        scores[k] = (float(s), centers, labels)

    if len(scores) == 0:
        center = np.mean(points, axis=0, keepdims=True)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        return center, np.array([radius])

    sil_values = sorted([scores[k][0] for k in scores], reverse=True)
    if np.var(sil_values[: min(3, len(sil_values))]) < 4e-3:
        center = np.mean(points, axis=0, keepdims=True)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        return center, np.array([radius])

    best_k = max(scores, key=lambda kk: scores[kk][0])
    _, centers, labels = scores[best_k]

    centers, radii, labels = merge_centers(points, centers, labels, valley_eps=1e-3, f=f)

    return centers, radii


@dataclass(frozen=True)
class AdaptiveStep:
    step: int
    layer: int
    center: np.ndarray
    radius: float
    n_samples: int
    flags: list[bool]
    n_collected: int
    new_centers: np.ndarray
    action: str


def adaptive_peak_detection(
    f: Callable[[np.ndarray], np.ndarray],
    D: list[list[float]],
    *,
    L_cut: float = 0.45,
    N_global: int = 3000,
    m_global: int = 55,
    C_max: int = 5,
    r_init: float = 0.3,
    conv_steps: int = 2,
    sample_method: str = "grid",
    ball_method: str = "grid",
    random_state: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[AdaptiveStep], list[int]]:
    """Adaptive peak detection with per-step verbose output.

    Algorithm intentionally matches exps/feedback_loop/locate_sol_flexible.py,
    with additional logging.
    """
    sample_method = str(sample_method).lower()
    ball_method = str(ball_method).lower()
    dim = len(D)
    rng = np.random.default_rng(int(random_state))

    def _fmt_budget(value: int, *, method: str, label: str) -> str:
        if method == "uniform":
            return f"{label}={value}"
        # grid: value is m, points is m^dim
        try:
            pts = int(value) ** int(dim)
        except Exception:
            pts = -1
        return f"m_{label}={value} (pts={pts})"

    if sample_method == "uniform":
        samples = uniform_sampling(D, int(N_global), rng)
    else:
        samples = grid_sampling(D, int(m_global))
    vals = f(samples)
    collected = samples[vals >= L_cut]

    if len(collected) == 0:
        if verbose:
            print("[init] collected=0 -> no centers")
        return np.empty((0, dim), dtype=np.float32), np.empty((0, dim), dtype=np.float32), [], []

    init_centers, radii = cluster_points(collected, C_max, random_state=random_state, f=f)
    # Queue items are (center, radius, budget, flags, layer)
    queue: deque[tuple[np.ndarray, float, int, list[bool], int]] = deque()
    # Local sampling reuses the *global* budget by design.
    local_budget0 = int(N_global) if ball_method == "uniform" else int(m_global)
    for c, r in zip(init_centers, radii):
        queue.append((np.asarray(c, dtype=np.float32), float(r), local_budget0, [False] * int(conv_steps), 0))

    final_centers: list[np.ndarray] = []
    final_layers: list[int] = []
    history: list[AdaptiveStep] = []

    if verbose:
        if sample_method == "uniform":
            global_budget = _fmt_budget(int(N_global), method="uniform", label="global")
        else:
            global_budget = _fmt_budget(int(m_global), method="grid", label="global")
        local_budget = _fmt_budget(local_budget0, method=ball_method, label="local")
        print(
            f"[init] collected={len(collected)} init_centers={init_centers.tolist()} "
            f"({global_budget}, {local_budget}, sample_method={sample_method}, ball_method={ball_method})"
        )

    event = 0
    while queue:
        center, r, N, flags, layer = queue.popleft()
        event += 1

        if all(flags):
            final_centers.append(center)
            final_layers.append(int(layer))
            history.append(
                AdaptiveStep(
                    step=event,
                    layer=int(layer),
                    center=center,
                    radius=r,
                    n_samples=N,
                    flags=list(flags),
                    n_collected=0,
                    new_centers=np.asarray([center], dtype=np.float32),
                    action="finalize",
                )
            )
            if verbose:
                print(
                    f"[layer {layer} event {event}] finalize center={center.tolist()} r={r:.4g} "
                    f"{_fmt_budget(int(N), method=ball_method, label='local')} flags={flags}"
                )
            continue

        if ball_method == "uniform":
            ball_samples = uniform_ball_sampling(center, r, int(N), rng)
        else:
            ball_samples = grid_ball_sampling(center, r, int(N))
        vals = f(ball_samples)
        collected = ball_samples[vals >= L_cut]

        if len(collected) == 0:
            history.append(
                AdaptiveStep(
                    step=event,
                    layer=int(layer),
                    center=center,
                    radius=r,
                    n_samples=N,
                    flags=list(flags),
                    n_collected=0,
                    new_centers=np.empty((0, dim), dtype=np.float32),
                    action="drop(empty)",
                )
            )
            if verbose:
                print(
                    f"[layer {layer} event {event}] drop(empty) center={center.tolist()} r={r:.4g} "
                    f"{_fmt_budget(int(N), method=ball_method, label='local')} flags={flags}"
                )
            continue

        new_centers, new_radii = cluster_points(collected, C_max, random_state=random_state, f=f)

        if len(new_centers) == 1:
            for k in range(len(flags)):
                if not flags[k]:
                    flags[k] = True
                    break
            # Only increase sampling budget for uniform sampling. For grid, the grid resolution is fixed.
            next_N = (N * 2) if ball_method == "uniform" else N
            queue.append((np.asarray(new_centers[0], dtype=np.float32), float(new_radii[0]), int(next_N), flags, int(layer) + 1))
            action = "refine(single)"
        else:
            for c, r in zip(new_centers, new_radii):
                # the new radius is 1.1 times the largest distance from the new center to collected points
                queue.append((np.asarray(c, dtype=np.float32), float(r), N, [False] * int(conv_steps), int(layer) + 1))
            action = "split(multi)"

        history.append(
            AdaptiveStep(
                step=event,
                layer=int(layer),
                center=center,
                radius=r,
                n_samples=N,
                flags=list(flags),
                n_collected=int(len(collected)),
                new_centers=np.asarray(new_centers, dtype=np.float32),
                action=action,
            )
        )
        if verbose:
            print(
                f"[layer {layer} event {event}] {action} center={center.tolist()} r={r:.4g} "
                f"{_fmt_budget(int(N), method=ball_method, label='local')} flags={flags} "
                f"collected={len(collected)} new_centers={np.asarray(new_centers).tolist()}"
            )

    return np.array(final_centers, dtype=np.float32), np.asarray(init_centers, dtype=np.float32), history, final_layers


def _plot_phi_landscape(
    *,
    phi_u: Callable[[np.ndarray], np.ndarray],
    D: list[list[float]],
    centers: np.ndarray,
    init_centers: np.ndarray,
    center_layers: list[int] | None = None,
    L_cut: float,
    save_path: Path,
    m: int = 200,
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
    plt.contour(X, Y, vals, levels=[L_cut], colors=["white"], linewidths=1.0)

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
    plt.title("Adaptive peak detection")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _parse_theta(vals: Iterable[str]) -> np.ndarray:
    arr = np.asarray([float(v) for v in vals], dtype=np.float32)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run flexible locater for a single theta and save a figure.")
    parser.add_argument("--phi-ckpt", type=str, required=True, help="Path to phi checkpoint (.pt)")
    parser.add_argument(
        "--theta",
        type=float,
        nargs="+",
        required=True,
        help="Theta parameters as a flat list, e.g. --theta alpha1 alpha2 gamma1 gamma2",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")

    parser.add_argument("--L-cut", type=float, default=0.35)
    parser.add_argument("--N-global", type=int, default=3000)
    parser.add_argument(
        "--m-global",
        type=int,
        default=55,
        help="Global grid resolution per dimension when --sample-method=grid.",
    )
    parser.add_argument("--C-max", type=int, default=4)
    parser.add_argument("--r-init", type=float, default=0.3)
    parser.add_argument("--conv-steps", type=int, default=2)
    parser.add_argument(
        "--sample-method",
        type=str,
        default="grid",
        choices=["grid", "uniform"],
        help="Global sampling method (grid or uniform).",
    )
    parser.add_argument(
        "--ball-method",
        type=str,
        default="grid",
        choices=["grid", "uniform"],
        help="Local ball sampling method (grid or uniform).",
    )

    parser.add_argument(
        "--u-bounds",
        type=float,
        nargs=4,
        default=[-1.0, 6.0, -1.0, 6.0],
        help="2D bounds for u as: u0_low u0_high u1_low u1_high",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="locater_flexible_runs",
        help="Folder to create timestamped output under",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")

    args = parser.parse_args()

    theta = np.asarray(args.theta, dtype=np.float32).reshape(-1)
    D = [[float(args.u_bounds[0]), float(args.u_bounds[1])], [float(args.u_bounds[2]), float(args.u_bounds[3])]]

    device = _device_from_cfg(args.device)
    print(f"Device: {device}")
    print(f"Theta: {theta.tolist()}")
    print(f"Domain D: {D}")

    phi_model = load_phi_model(args.phi_ckpt, device)
    phi = make_phi_function(phi_model, device=device)
    phi_u = phi(theta)

    centers, init_centers, history, center_layers = adaptive_peak_detection(
        phi_u,
        D,
        L_cut=float(args.L_cut),
        N_global=int(args.N_global),
        m_global=int(args.m_global),
        C_max=int(args.C_max),
        r_init=float(args.r_init),
        conv_steps=int(args.conv_steps),
        sample_method=str(args.sample_method),
        ball_method=str(args.ball_method),
        verbose=True,
    )

    print(f"Final centers: {centers.tolist()}")
    print(f"Num centers: {int(centers.shape[0])}")

    if not args.no_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_root) / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = out_dir / "locater_flexible.png"
        _plot_phi_landscape(
            phi_u=phi_u,
            D=D,
            centers=centers,
            init_centers=init_centers,
            center_layers=center_layers,
            L_cut=float(args.L_cut),
            save_path=fig_path,
            m=220,
        )
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
