from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import tqdm

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from psnn.config import cfg_get, deep_merge_dicts, dump_yaml, load_yaml, resolve_path


# Domain for state u=(p1,p2)
D = np.array(
    [
        [-1.0, 6.0],
        [-1.0, 6.0],
    ],
    dtype=float,
)

# Parameter box (alpha1, alpha2, gamma1, gamma2)
OMEGA = np.array(
    [
        [1.0, 5.0],
        [1.0, 5.0],
        [0.1, 0.3],
        [0.1, 0.3],
    ],
    dtype=float,
)


def build_run_config(config_path: Path, seed: int, run_dir: Path) -> dict:
    cfg = load_yaml(config_path)
    data_dir = run_dir / "data"
    variant_name = str(cfg_get(cfg, "run.variant", config_path.stem))

    overrides = {
        "seed": int(seed),
        "data_generation": {
            "outputs": {
                "out_dir": str(data_dir),
            }
        },
        "training": {
            "paths": {
                "data_dir": str(data_dir),
                "out_dir": str(run_dir),
            }
        },
        "run": {
            "name": f"{variant_name}_seed_{seed}",
            "variant": variant_name,
            "seed": int(seed),
            "output_dir": str(run_dir),
        },
    }
    return deep_merge_dicts(cfg, overrides)


def _default_run_dir(variant: str, seed: int) -> Path:
    return repo_root / "runs" / "feedback_loop" / variant / f"seed_{seed}"


def _poly_coeffs(alpha1: float, alpha2: float, gamma1: float, gamma2: float) -> np.ndarray:
    a1, a2, g1, g2 = float(alpha1), float(alpha2), float(gamma1), float(gamma2)

    ag1 = a1 + g1
    ag2 = a2 + g2
    g2_2 = g2**2
    g2_3 = g2**3
    ag2_2 = ag2**2
    ag2_3 = ag2**3

    c10 = g2_3 + 1.0
    c9 = -g1 * g2_3 - ag1
    c7 = g2_2 * ag2 + 1.0
    c6 = g1 * g2_2 * ag2 + ag1
    c4 = g2 * ag2_2 + 1.0
    c3 = g1 * g2 * ag2_2 + ag1
    c1 = ag2_3 + 1.0
    c0 = -g1 * g2_3 - ag1

    return np.array(
        [
            c10,
            c9,
            0.0,
            3.0 * c7,
            -3.0 * c6,
            0.0,
            3.0 * c4,
            -3.0 * c3,
            0.0,
            c1,
            c0,
        ],
        dtype=float,
    )


def U(theta: np.ndarray) -> list[dict]:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.shape[0] != 4:
        raise ValueError("theta must be a 1D array of shape (4,)")

    a1, a2, g1, g2 = map(float, theta)
    coeffs = _poly_coeffs(a1, a2, g1, g2)
    roots = np.roots(coeffs)

    p1_list: list[float] = []
    for root in roots:
        if abs(root.imag) < 1e-10 and root.real > 0:
            p1_list.append(float(root.real))
    if not p1_list:
        return []

    p1 = np.array(sorted(p1_list), dtype=float)
    p2 = g2 + a2 / (1.0 + p1**3)

    out: list[dict] = []
    for idx in range(len(p1)):
        u = np.array([p1[idx], p2[idx]], dtype=np.float32)
        j12 = -3 * a1 * u[1] ** 2 / (1 + u[1] ** 3) ** 2
        j21 = -3 * a2 * u[0] ** 2 / (1 + u[0] ** 3) ** 2
        stable = (-1.0) * (-1.0) - j12 * j21 > 0
        out.append({"theta": theta.astype(np.float32), "u": u, "stable": stable})
    return out


def _per_center_deltas(centers: np.ndarray, delta_default: float = 0.25, delta_min: float = 0.1) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.shape[0] <= 1:
        return np.array([delta_default], dtype=float)

    deltas = np.full((centers.shape[0],), float(delta_default), dtype=float)
    for i in range(centers.shape[0]):
        dmin = np.inf
        for j in range(centers.shape[0]):
            if i == j:
                continue
            dmin = min(dmin, float(np.linalg.norm(centers[i] - centers[j])))
        deltas[i] = max(delta_min, min(0.25 * dmin, float(delta_default)))
    return deltas


def _per_theta_delta(centers: np.ndarray, delta_default: float = 0.25, delta_min: float = 0.1) -> float:
    centers = np.asarray(centers, dtype=float)
    if centers.shape[0] <= 1:
        return float(delta_default)

    dmin = np.inf
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            dmin = min(dmin, float(np.linalg.norm(centers[i] - centers[j])))
    return float(max(delta_min, min(0.25 * dmin, float(delta_default))))


def Phi_theta(
    u_input: np.ndarray,
    centers: np.ndarray,
    deltas: Optional[np.ndarray] = None,
    delta_default: float = 0.25,
) -> np.ndarray:
    u_input = np.asarray(u_input, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.zeros((u_input.shape[0],), dtype=np.float32)

    if deltas is None:
        deltas = _per_center_deltas(centers, delta_default=delta_default)
    deltas = np.asarray(deltas, dtype=float)

    diff = u_input[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    phi = np.sum(np.exp(-d2 / (deltas[None, :] ** 2)), axis=1)
    return phi.astype(np.float32)


def _sample_disk(rng: np.random.Generator, center: np.ndarray, radius: float, n: int) -> np.ndarray:
    r = radius * np.sqrt(rng.random((n, 1)))
    t = 2.0 * np.pi * rng.random((n, 1))
    pts = np.concatenate([r * np.cos(t), r * np.sin(t)], axis=1)
    return pts + center[None, :]


def _sample_far_points(
    rng: np.random.Generator,
    centers: np.ndarray,
    radii: np.ndarray,
    n: int,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    pts = []
    attempts = 0
    while len(pts) < n:
        attempts += 1
        if attempts > 200000:
            break
        cand = rng.uniform(lo, hi, size=(1, 2))
        ok = True
        for center, radius in zip(centers, radii):
            if np.linalg.norm(cand[0] - center) <= 2.0 * radius:
                ok = False
                break
        if ok:
            pts.append(cand[0])
    return np.asarray(pts, dtype=np.float32)


def _corrupt_multi_solution_observations(
    observations: list[dict],
    rng: np.random.Generator,
    corruption_rate: float,
) -> list[dict]:
    rate = float(corruption_rate)
    if rate <= 1e-5:
        return observations
    if rate > 1.0:
        raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate}")

    corrupted: list[dict] = []
    for obs in observations:
        sols = list(obs["U"])
        if len(sols) > 1 and rng.random() < rate:
            n_drop = min(len(sols), int(rng.integers(1, 3)))
            drop_idx = set(rng.choice(len(sols), size=n_drop, replace=False).tolist())
            sols = [sol for idx, sol in enumerate(sols) if idx not in drop_idx]
        corrupted.append({"Theta": obs["Theta"], "U": sols})
    return corrupted


def gen_data(
    N_obs: int,
    seed: int = 42,
    *,
    method_theta: str = "uniform",
    method_u: str = "resample",
    n_local_base: int = 60,
    n_far: int = 100,
    delta_default: float = 0.25,
    delta_min: float = 0.1,
    delta_mode: str = "per_center",
    theta_bounds: Optional[np.ndarray] = None,
    u_bounds: Optional[np.ndarray] = None,
    corruption_rate: float = 0.0,
):
    rng = np.random.default_rng(seed)

    theta_bounds = OMEGA if theta_bounds is None else np.asarray(theta_bounds, dtype=float)
    u_bounds = D if u_bounds is None else np.asarray(u_bounds, dtype=float)
    if method_theta != "uniform":
        raise NotImplementedError(f"method_theta={method_theta} not implemented")

    theta_list = rng.uniform(theta_bounds[:, 0], theta_bounds[:, 1], size=(N_obs, 4))
    observations: list[dict] = []
    for theta in theta_list:
        observations.append({"Theta": np.array(theta, dtype=np.float32), "U": U(theta)})
    observations = _corrupt_multi_solution_observations(observations, rng=rng, corruption_rate=corruption_rate)

    theta_out_list = []
    u_out_list = []
    phi_out_list = []
    lo, hi = u_bounds[:, 0], u_bounds[:, 1]

    for obs in tqdm.tqdm(observations, desc="Generating feedback-loop data"):
        centers = (
            np.array([sol["u"] for sol in obs["U"]], dtype=np.float32)
            if len(obs["U"]) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        if centers.shape[0] == 0:
            deltas = np.zeros((0,), dtype=float)
        elif delta_mode == "per_center":
            deltas = _per_center_deltas(centers, delta_default=delta_default, delta_min=delta_min)
        elif delta_mode == "per_theta":
            delta_theta = _per_theta_delta(centers, delta_default=delta_default, delta_min=delta_min)
            deltas = np.full((centers.shape[0],), delta_theta, dtype=float)
        else:
            raise NotImplementedError(f"delta_mode={delta_mode} not implemented")

        if method_u == "uniform":
            u_all = rng.uniform(lo, hi, size=(n_far, 2)).astype(np.float32)
        elif method_u == "resample":
            if centers.shape[0] == 0:
                u_all = rng.uniform(lo, hi, size=(n_far, 2)).astype(np.float32)
            else:
                local_pts = []
                for idx in range(centers.shape[0]):
                    ratio = float(deltas[idx] / delta_default)
                    n_local = max(int(math.ceil(n_local_base * ratio)), 20)
                    local_pts.append(_sample_disk(rng, centers[idx], radius=2.0 * deltas[idx], n=n_local))
                local_pts = np.concatenate(local_pts, axis=0).astype(np.float32)
                far_pts = _sample_far_points(rng, centers, deltas, n=n_far, lo=lo, hi=hi)
                u_all = np.concatenate([centers.astype(np.float32), local_pts, far_pts], axis=0)
        else:
            raise NotImplementedError(f"method_u={method_u} not implemented")

        phi = Phi_theta(u_all, centers, deltas=deltas, delta_default=delta_default)
        theta_rep = np.repeat(obs["Theta"][None, :], u_all.shape[0], axis=0)
        theta_out_list.append(theta_rep)
        u_out_list.append(u_all)
        phi_out_list.append(phi)

    data_phi = {
        "Theta": np.vstack(theta_out_list).astype(np.float32),
        "U": np.vstack(u_out_list).astype(np.float32),
        "Phi": np.hstack(phi_out_list).astype(np.float32),
    }
    return data_phi, observations


def generate_from_loaded_config(cfg: dict, base_dir: str | Path) -> None:
    base_dir = Path(base_dir).expanduser().resolve()

    global_seed = int(cfg_get(cfg, "seed", 123))
    dg = cfg_get(cfg, "data_generation", {})
    theta_bounds_raw = cfg_get(dg, "domain.theta_bounds", None)
    u_bounds_raw = cfg_get(dg, "domain.u_bounds", None)
    if theta_bounds_raw is None or u_bounds_raw is None:
        raise ValueError("config must define data_generation.domain.theta_bounds and data_generation.domain.u_bounds")

    theta_bounds = np.asarray(theta_bounds_raw, dtype=float)
    u_bounds = np.asarray(u_bounds_raw, dtype=float)
    if theta_bounds.shape != (4, 2):
        raise ValueError(f"theta_bounds must have shape (4,2), got {theta_bounds.shape}")
    if u_bounds.shape != (2, 2):
        raise ValueError(f"u_bounds must have shape (2,2), got {u_bounds.shape}")

    train_cfg = cfg_get(dg, "train", {})
    test_cfg = cfg_get(dg, "test", {})
    out_cfg = cfg_get(dg, "outputs", {})
    corruption_rate = float(cfg_get(dg, "corruption_rate", 0.0))

    out_dir = resolve_path(base_dir, cfg_get(out_cfg, "out_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_train_phi, obs_train = gen_data(
        cfg_get(train_cfg, "N_obs", 1200),
        seed=global_seed,
        method_theta=cfg_get(train_cfg, "method_theta", "uniform"),
        method_u=cfg_get(train_cfg, "method_u", "resample"),
        n_local_base=cfg_get(train_cfg, "n_local_base", 60),
        n_far=cfg_get(train_cfg, "n_far", 100),
        delta_default=cfg_get(train_cfg, "delta_default", 0.25),
        delta_min=cfg_get(train_cfg, "delta_min", 0.1),
        delta_mode=cfg_get(train_cfg, "delta_mode", "per_center"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        corruption_rate=corruption_rate,
    )
    data_test_phi, obs_test = gen_data(
        cfg_get(test_cfg, "N_obs", 600),
        seed=global_seed + 1,
        method_theta=cfg_get(test_cfg, "method_theta", "uniform"),
        method_u=cfg_get(test_cfg, "method_u", "resample"),
        n_local_base=cfg_get(test_cfg, "n_local_base", 60),
        n_far=cfg_get(test_cfg, "n_far", 100),
        delta_default=cfg_get(test_cfg, "delta_default", 0.25),
        delta_min=cfg_get(test_cfg, "delta_min", 0.1),
        delta_mode=cfg_get(test_cfg, "delta_mode", "per_center"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        corruption_rate=0.0,
    )

    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_train_npz", "feedback_loop_data_train.npz"), **data_train_phi)
    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_test_npz", "feedback_loop_data_test.npz"), **data_test_phi)
    joblib.dump(obs_train, out_dir / cfg_get(out_cfg, "obs_train_pkl", "feedback_loop_obs_train.pkl"))
    joblib.dump(obs_test, out_dir / cfg_get(out_cfg, "obs_test_pkl", "feedback_loop_obs_test.pkl"))


def generate_from_config(config_path: str | Path | None = None) -> None:
    default_cfg = exp_dir / "config.yaml"
    cfg_path = Path(config_path).expanduser().resolve() if config_path else (default_cfg if default_cfg.exists() else None)
    cfg = load_yaml(cfg_path) if cfg_path is not None else {}
    generate_from_loaded_config(cfg, base_dir=cfg_path.parent if cfg_path is not None else exp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feedback-loop data for a config/seed pair.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Variant config under exps/feedback_loop.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this run.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory. Defaults to runs/feedback_loop/<variant>/seed_<seed>.")
    parser.add_argument("--write-config", type=str, default=None, help="Optionally write the merged run config to a file.")
    args = parser.parse_args()

    config_path = (exp_dir / args.config).resolve()
    raw_cfg = load_yaml(config_path)
    seed = int(args.seed if args.seed is not None else cfg_get(raw_cfg, "seed", 42))
    variant = str(cfg_get(raw_cfg, "run.variant", config_path.stem))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(variant, seed)

    cfg = build_run_config(config_path, seed=seed, run_dir=run_dir)
    if args.write_config:
        dump_yaml(Path(args.write_config).resolve(), cfg)
    generate_from_loaded_config(cfg, base_dir=run_dir)


if __name__ == "__main__":
    main()
