from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tqdm

exp_dir = Path(__file__).resolve().parent
repo_root = exp_dir.parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from psnn.config import cfg_get, deep_merge_dicts, dump_yaml, load_yaml, resolve_path


# Domain for state u=(u, v)
D = np.array(
    [
        [-0.2, 1.2],
        [-0.2, 1.2],
    ],
    dtype=float,
)

# Parameter box (f, k)
OMEGA = np.array(
    [
        [0.0, 0.3],
        [0.0, 0.08],
    ],
    dtype=float,
)


def build_run_config(config_path: Path, seed: int, run_dir: Path) -> dict[str, Any]:
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
    return repo_root / "runs" / "gray_scott" / variant / f"seed_{seed}"


def U(theta: np.ndarray) -> list[dict[str, Any]]:
    """Gray-Scott homogeneous steady states for parameters (f, k)."""
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.shape[0] != 2:
        raise ValueError("theta must be a 1D array of shape (2,)")

    f, k = map(float, theta)
    fp = f + k
    delta_expr = f - 4.0 * (fp**2)
    if delta_expr <= 0.0:
        return []

    disc = f * delta_expr
    root = np.sqrt(disc)

    u_1 = np.array([(f - root) / (2.0 * f), (f + root) / (2.0 * fp)], dtype=np.float32)
    u_2 = np.array([(f + root) / (2.0 * f), (f - root) / (2.0 * fp)], dtype=np.float32)

    stability_discriminator = f * root + f**2 - 2.0 * (fp**3)
    theta_out = theta.astype(np.float32)
    if stability_discriminator > 0.0:
        return [
            {"theta": theta_out, "u": u_1, "stable": True},
            {"theta": theta_out, "u": u_2, "stable": False},
        ]
    return [
        {"theta": theta_out, "u": u_1, "stable": False},
        {"theta": theta_out, "u": u_2, "stable": False},
    ]


def delta_(centers: list[dict[str, Any]] | np.ndarray, delta_default: float = 1.0, delta_min: float = 1e-3) -> float:
    if len(centers) <= 1:
        return float(delta_default)

    if isinstance(centers, list):
        centers_np = np.vstack([np.asarray(sol["u"], dtype=float) for sol in centers])
    else:
        centers_np = np.asarray(centers, dtype=float)

    norms_squared = np.sum(centers_np**2, axis=1, keepdims=True)
    d2 = norms_squared + norms_squared.T - 2.0 * centers_np @ centers_np.T
    d2 = np.maximum(d2, 0.0)
    d2[np.arange(centers_np.shape[0]), np.arange(centers_np.shape[0])] = np.inf
    dist_min = float(np.sqrt(np.min(d2)))
    return float(max(float(delta_min), min(0.25 * dist_min, float(delta_default))))


def Phi_theta(
    u_input: np.ndarray,
    centers: list[dict[str, Any]] | np.ndarray,
    delta: float | None = None,
    delta_default: float = 1.0,
    delta_min: float = 1e-3,
) -> np.ndarray:
    u_input = np.asarray(u_input, dtype=float)
    if len(centers) == 0:
        return np.zeros((u_input.shape[0],), dtype=np.float32)

    centers_np = (
        np.concatenate([np.asarray(sol["u"], dtype=float)[None, :] for sol in centers], axis=0)
        if isinstance(centers, list)
        else np.asarray(centers, dtype=float)
    )
    if delta is None:
        delta = delta_(centers_np, delta_default=delta_default, delta_min=delta_min)

    dist = u_input[:, None, :] - centers_np[None, :, :]
    dist_squared = np.sum(dist**2, axis=-1)
    phi_vals = np.sum(np.exp(-dist_squared / float(delta) ** 2), axis=-1)
    return phi_vals.astype(np.float32)


def _corrupt_two_solution_observations(
    observations: list[dict[str, Any]],
    rng: np.random.Generator,
    incomplete_rate: float,
    incomplete_mode: str = "random_solution",
) -> list[dict[str, Any]]:
    rate = float(incomplete_rate)
    if rate <= 1e-5:
        return observations
    if rate > 1.0:
        raise ValueError(f"incomplete_rate must be in [0, 1], got {incomplete_rate}")

    mode = str(incomplete_mode).lower()
    if mode in {"random", "random_drop", "random_solution"}:
        branch_to_drop: int | None = None
    elif mode in {"branch", "single_branch", "branch_only"}:
        branch_to_drop = int(rng.integers(0, 2))
    else:
        raise ValueError(
            "incomplete_mode must be one of "
            "{'random_solution', 'single_branch'} "
            f"(got {incomplete_mode!r})"
        )

    corrupted: list[dict[str, Any]] = []
    for obs in observations:
        sols = list(obs["U"])
        if len(sols) == 2 and rng.random() < rate:
            if branch_to_drop is None:
                keep_idx = int(rng.integers(0, 2))
                sols = [sols[keep_idx]]
            else:
                sols = [sol for idx, sol in enumerate(sols) if idx != branch_to_drop]
        corrupted.append({"Theta": obs["Theta"], "U": sols})
    return corrupted


def gen_data(
    n_obs: int,
    n_random: int,
    seed: int = 42,
    method_theta: str = "uniform",
    method_u: str = "uniform",
    *,
    theta_bounds: np.ndarray | None = None,
    u_bounds: np.ndarray | None = None,
    delta_min: float = 1e-3,
    delta_default: float = 1.0,
    incomplete_rate: float = 0.0,
    incomplete_mode: str = "random_solution",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    theta_bounds = OMEGA if theta_bounds is None else np.asarray(theta_bounds, dtype=float)
    u_bounds = D if u_bounds is None else np.asarray(u_bounds, dtype=float)

    method_theta = str(method_theta).lower()
    method_u = str(method_u).lower()
    if method_theta == "grid":
        side = max(1, int(np.sqrt(int(n_obs))))
        f_values = np.linspace(theta_bounds[0, 0], theta_bounds[0, 1], side)
        k_values = np.linspace(theta_bounds[1, 0], theta_bounds[1, 1], side)
        theta_list = np.asarray([(f, k) for f in f_values for k in k_values], dtype=np.float32)[: int(n_obs)]
    elif method_theta == "uniform":
        theta_list = rng.uniform(theta_bounds[:, 0], theta_bounds[:, 1], size=(int(n_obs), 2)).astype(np.float32)
    else:
        raise NotImplementedError(f"method_theta={method_theta!r} not implemented")

    observations: list[dict[str, Any]] = []
    for theta in theta_list:
        observations.append(
            {
                "Theta": np.asarray(theta, dtype=np.float32),
                "U": U(np.asarray(theta, dtype=float)),
            }
        )
    observations = _corrupt_two_solution_observations(
        observations,
        rng=rng,
        incomplete_rate=incomplete_rate,
        incomplete_mode=incomplete_mode,
    )

    theta_out_list: list[np.ndarray] = []
    u_out_list: list[np.ndarray] = []
    phi_out_list: list[np.ndarray] = []

    for obs in tqdm.tqdm(observations, desc="Generating Gray-Scott data"):
        u_obs = np.asarray([sol["u"] for sol in obs["U"]], dtype=np.float32).reshape(-1, 2)
        if u_obs.size == 0:
            u_obs = np.empty((0, 2), dtype=np.float32)

        if method_u == "uniform":
            u_lo, u_hi = u_bounds[:, 0], u_bounds[:, 1]
            u_rand = rng.uniform(u_lo, u_hi, size=(int(n_random), 2)).astype(np.float32)
        else:
            raise NotImplementedError(f"method_u={method_u!r} not implemented")

        u_all = np.vstack((u_obs, u_rand))
        delta = None
        if len(obs["U"]) > 0:
            delta = delta_(obs["U"], delta_default=delta_default, delta_min=delta_min)
        phi_vals = Phi_theta(u_all, obs["U"], delta=delta, delta_default=delta_default, delta_min=delta_min)

        u_out_list.append(u_all)
        phi_out_list.append(phi_vals)
        theta_out_list.append(np.repeat(obs["Theta"][None, :], u_all.shape[0], axis=0).astype(np.float32))

    phi_data = {
        "Theta": np.vstack(theta_out_list).astype(np.float32),
        "U": np.vstack(u_out_list).astype(np.float32),
        "Phi": np.hstack(phi_out_list).astype(np.float32),
    }
    return phi_data, observations


def generate_from_loaded_config(cfg: dict[str, Any], base_dir: str | Path) -> None:
    base_dir = Path(base_dir).expanduser().resolve()

    global_seed = int(cfg_get(cfg, "seed", 123))
    dg = cfg_get(cfg, "data_generation", {})
    theta_bounds_raw = cfg_get(dg, "domain.theta_bounds", None)
    u_bounds_raw = cfg_get(dg, "domain.u_bounds", None)
    if theta_bounds_raw is None or u_bounds_raw is None:
        raise ValueError("config must define data_generation.domain.theta_bounds and data_generation.domain.u_bounds")

    theta_bounds = np.asarray(theta_bounds_raw, dtype=float)
    u_bounds = np.asarray(u_bounds_raw, dtype=float)
    if theta_bounds.shape != (2, 2):
        raise ValueError(f"theta_bounds must have shape (2,2), got {theta_bounds.shape}")
    if u_bounds.shape != (2, 2):
        raise ValueError(f"u_bounds must have shape (2,2), got {u_bounds.shape}")

    train_cfg = cfg_get(dg, "train", {})
    test_cfg = cfg_get(dg, "test", {})
    out_cfg = cfg_get(dg, "outputs", {})
    incomplete_rate = float(cfg_get(dg, "incomplete_rate", cfg_get(dg, "corruption_rate", 0.0)))
    incomplete_mode = str(cfg_get(dg, "incomplete_mode", "random_solution"))

    out_dir = resolve_path(base_dir, cfg_get(out_cfg, "out_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_seed = int(cfg_get(train_cfg, "seed", global_seed))
    test_seed = int(cfg_get(test_cfg, "seed", train_seed + 1))

    data_train_phi, obs_train = gen_data(
        cfg_get(train_cfg, "N_obs", 1000),
        cfg_get(train_cfg, "N_random", 200),
        seed=train_seed,
        method_theta=cfg_get(train_cfg, "method_theta", "uniform"),
        method_u=cfg_get(train_cfg, "method_u", "uniform"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        delta_min=float(cfg_get(dg, "delta.delta_min", cfg_get(dg, "delta.delta0", 1e-3))),
        delta_default=float(cfg_get(dg, "delta.delta_default", cfg_get(dg, "delta.delta1", 1.0))),
        incomplete_rate=incomplete_rate,
        incomplete_mode=incomplete_mode,
    )
    data_test_phi, obs_test = gen_data(
        cfg_get(test_cfg, "N_obs", 600),
        cfg_get(test_cfg, "N_random", 200),
        seed=test_seed,
        method_theta=cfg_get(test_cfg, "method_theta", "uniform"),
        method_u=cfg_get(test_cfg, "method_u", "uniform"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        delta_min=float(cfg_get(dg, "delta.delta_min", cfg_get(dg, "delta.delta0", 1e-3))),
        delta_default=float(cfg_get(dg, "delta.delta_default", cfg_get(dg, "delta.delta1", 1.0))),
        incomplete_rate=0.0,
    )

    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_train_npz", "gray_scott_data_train.npz"), **data_train_phi)
    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_test_npz", "gray_scott_data_test.npz"), **data_test_phi)
    joblib.dump(obs_train, out_dir / cfg_get(out_cfg, "obs_train_pkl", "gray_scott_obs_train.pkl"))
    joblib.dump(obs_test, out_dir / cfg_get(out_cfg, "obs_test_pkl", "gray_scott_obs_test.pkl"))


def generate_from_config(config_path: str | Path | None = None) -> None:
    default_cfg = exp_dir / "configs" / "complete.yaml"
    cfg_path = Path(config_path).expanduser().resolve() if config_path else (default_cfg if default_cfg.exists() else None)
    cfg = load_yaml(cfg_path) if cfg_path is not None else {}
    generate_from_loaded_config(cfg, base_dir=cfg_path.parent if cfg_path is not None else exp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gray-Scott data for a config/seed pair.")
    parser.add_argument("--config", type=str, default="configs/complete.yaml", help="Variant config under exps/gray_scott.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed for this run.")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory. Defaults to runs/gray_scott/<variant>/seed_<seed>.")
    parser.add_argument("--write-config", type=str, default=None, help="Optionally write the merged run config to a file.")
    args = parser.parse_args()

    config_path = (exp_dir / args.config).resolve()
    raw_cfg = load_yaml(config_path)
    seed = int(args.seed if args.seed is not None else cfg_get(raw_cfg, "seed", 123))
    variant = str(cfg_get(raw_cfg, "run.variant", config_path.stem))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(variant, seed)

    cfg = build_run_config(config_path, seed=seed, run_dir=run_dir)
    if args.write_config:
        dump_yaml(Path(args.write_config).resolve(), cfg)
    generate_from_loaded_config(cfg, base_dir=run_dir)


if __name__ == "__main__":
    main()
