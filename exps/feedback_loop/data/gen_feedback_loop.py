import argparse
import math
import os
import sys
import numpy as np
import tqdm
import joblib
from typing import Optional, Tuple

# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../exps/feedback_loop
repo_root = os.path.abspath(os.path.join(exp_dir, ".."))  # .../ (repo root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Domain for state u=(p1,p2) used throughout the collaborator notebook
D = np.array([
    [0.0, 5.0],
    [0.0, 5.0],
], dtype=float)

# Parameter box (alpha1, alpha2, gamma1, gamma2)
Omega = np.array([
    [2.0, 4.0],   # alpha1
    [2.0, 4.0],   # alpha2
    [0.1, 0.3],   # gamma1
    [0.1, 0.3],   # gamma2
], dtype=float)


def _poly_coeffs(alpha1: float, alpha2: float, gamma1: float, gamma2: float) -> np.ndarray:
    """Coefficient vector for the degree-10 polynomial in p1 used in the notebook.

    Returns coeffs ordered for np.roots: highest degree first.
    """
    a1, a2, g1, g2 = float(alpha1), float(alpha2), float(gamma1), float(gamma2)

    ag1 = a1 + g1
    ag2 = a2 + g2
    g2_2 = g2**2
    g2_3 = g2**3
    ag2_2 = ag2**2
    ag2_3 = ag2**3

    # Matches the collaborator notebook construction (with zeros for missing powers)
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
    """Homogeneous steady states for the 2D feedback-loop model.

    Parameters
    ----------
    theta : (4,) array
        (alpha1, alpha2, gamma1, gamma2)

    Returns
    -------
    solutions : list of dict
        Each dict has keys: {"theta": theta, "u": np.ndarray(2,), "stable": bool}

    Notes
    -----
    This is a cleaned implementation of the algebraic solver in NN_feedback_loop.ipynb.

    - Solve degree-10 polynomial for p1 > 0 roots.
    - For each p1, compute p2 = gamma2 + alpha2 / (1 + p1^3)
    - Use the same stability discriminator as the notebook.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.shape[0] != 4:
        raise ValueError("theta must be a 1D array of shape (4,) = (alpha1, alpha2, gamma1, gamma2)")

    a1, a2, g1, g2 = map(float, theta)

    coeffs = _poly_coeffs(a1, a2, g1, g2)
    roots = np.roots(coeffs)

    # Positive real roots
    p1_list: list[float] = []
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            p1_list.append(float(r.real))

    if len(p1_list) == 0:
        return []

    p1 = np.array(sorted(p1_list), dtype=float)
    p2 = g2 + a2 / (1.0 + p1 ** 3)



    # ----------------------
    # STABILITY COMPUTATION
    # ----------------------
    # The collaborator notebook assigns stability (+1 stable / -1 unstable) using a
    # simple scalar discriminator computed from the equilibrium values.
    #
    # Here we store stability per equilibrium as a boolean field `stable`.


    out: list[dict] = []
    for i in range(len(p1)):
        u = np.array([p1[i], p2[i]], dtype=np.float32)
        # notebook: value_r > val => -1 (unstable), else +1 (stable)
        J11 = -1.0
        J22 = -1.0

        J12 = -3*a1*u[1]**2 / (1+u[1]**3)**2
        J21 = -3*a2*u[0]**2 / (1+u[0]**3)**2

        detJ = J11*J22 - J12*J21

        stable = detJ > 0
        out.append({"theta": theta.astype(np.float32), "u": u, "stable": stable})

    return out


def _per_center_deltas(centers: np.ndarray, delta_default: float = 0.25) -> np.ndarray:
    """Notebook-style per-center Gaussian radii.

    For 3 solutions, each center gets its own delta based on nearest-neighbor distance,
    capped by delta_default.
    """
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
        deltas[i] = min(0.25 * dmin, float(delta_default))
    return deltas


def _per_theta_delta(centers: np.ndarray, delta_default: float = 0.25) -> float:
    """Single Gaussian radius shared by all centers for one theta."""
    centers = np.asarray(centers, dtype=float)
    K = centers.shape[0]
    if K <= 1:
        return float(delta_default)

    dmin = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            dmin = min(dmin, float(np.linalg.norm(centers[i] - centers[j])))

    return float(min(0.25 * dmin, float(delta_default)))





def Phi_theta(u_input: np.ndarray,
              centers: np.ndarray,
              deltas: Optional[np.ndarray] = None,
              delta_default: float = 0.25) -> np.ndarray:
    """Compute Phi(u) = sum_k exp(-||u-c_k||^2 / delta_k^2)."""
    u_input = np.asarray(u_input, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        return np.zeros((u_input.shape[0],), dtype=np.float32)

    if deltas is None:
        deltas = _per_center_deltas(centers, delta_default=delta_default)
    deltas = np.asarray(deltas, dtype=float)

    # (N, K, 2)
    diff = u_input[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)  # (N, K)

    # each center has its own delta
    phi_terms = np.exp(-d2 / (deltas[None, :] ** 2))
    phi = np.sum(phi_terms, axis=1)

    return phi.astype(np.float32)


def _sample_disk(rng: np.random.Generator, center: np.ndarray, radius: float, n: int) -> np.ndarray:
    """Uniform sampling in a disk (matches the notebook construction)."""
    r = radius * np.sqrt(rng.random((n, 1)))
    t = 2.0 * np.pi * rng.random((n, 1))
    pts = np.concatenate([r * np.cos(t), r * np.sin(t)], axis=1)
    return pts + center[None, :]


def _sample_far_points(rng: np.random.Generator,
                       centers: np.ndarray,
                       radii: np.ndarray,
                       n: int,
                       lo: np.ndarray,
                       hi: np.ndarray) -> np.ndarray:
    """Rejection sample points that are farther than 2*radius from each center."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)

    pts = []
    attempts = 0
    while len(pts) < n:
        attempts += 1
        # safeguard to avoid infinite loops in degenerate cases
        if attempts > 200000:
            break
        cand = rng.uniform(lo, hi, size=(1, 2))
        ok = True
        for c, r in zip(centers, radii):
            if np.linalg.norm(cand[0] - c) <= 2.0 * r:
                ok = False
                break
        if ok:
            pts.append(cand[0])
    return np.asarray(pts, dtype=np.float32)


def gen_data(
    N_obs: int,
    seed: int = 42,
    *,
    method_theta: str = "uniform",
    method_u: str = "resample",
    n_local_base: int = 60,
    n_far: int = 100,
    delta_default: float = 0.25,
    delta_mode: str = "per_center",
    theta_bounds: Optional[np.ndarray] = None,
    u_bounds: Optional[np.ndarray] = None,
):
    """Generate (Theta, U, Phi) for the feedback-loop example.

    This follows the collaborator notebook's idea:
    - Sample parameters theta in Omega
    - Compute steady states U(theta)
    - Generate labeled point clouds around each steady state (plus far negatives)

    Returns
    -------
    data : dict with keys {"Theta","U","Phi"}
        Phi is the solution-likelihood target (sum of Gaussians).
    observations : list of dict
        Each has keys {"Theta","U"} where U is a list of solution dicts.
    """

    rng = np.random.default_rng(seed)

    theta_bounds = Omega if theta_bounds is None else np.asarray(theta_bounds, dtype=float)
    u_bounds = D if u_bounds is None else np.asarray(u_bounds, dtype=float)

    if method_theta == "uniform":
        theta_list = rng.uniform(theta_bounds[:, 0], theta_bounds[:, 1], size=(N_obs, 4))
    else:
        raise NotImplementedError(f"method_theta={method_theta} not implemented")

    observations: list[dict] = []
    for theta in theta_list:
        sols = U(theta)
        observations.append({"Theta": np.array(theta, dtype=np.float32), "U": sols})

    Theta_out_list = []
    U_out_list = []
    Phi_out_list = []

    lo, hi = u_bounds[:, 0], u_bounds[:, 1]

    for obs in tqdm.tqdm(observations, desc="Generating feedback-loop data"):
        centers = np.array([s["u"] for s in obs["U"]], dtype=np.float32) if len(obs["U"]) > 0 else np.zeros((0, 2), dtype=np.float32)
        if centers.shape[0] == 0:
            deltas = np.zeros((0,), dtype=float)
        elif delta_mode == "per_center":
            deltas = _per_center_deltas(centers, delta_default=delta_default)
        elif delta_mode == "per_theta":
            delta_theta = _per_theta_delta(centers, delta_default=delta_default)
            deltas = np.full((centers.shape[0],), delta_theta, dtype=float)
        else:
            raise NotImplementedError(f"delta_mode={delta_mode} not implemented")

        if method_u == "uniform":
            U_rand = rng.uniform(lo, hi, size=(n_far, 2)).astype(np.float32)
            U_all = U_rand
        elif method_u == "resample":
            if centers.shape[0] == 0:
                # fallback to uniform samples if solver returns none
                U_all = rng.uniform(lo, hi, size=(n_far, 2)).astype(np.float32)
            else:
                local_pts = []
                for k in range(centers.shape[0]):
                    ratio = float(deltas[k] / delta_default)
                    n_local = max(int(math.ceil(n_local_base * ratio)), 20)
                    local_pts.append(_sample_disk(rng, centers[k], radius=2.0 * deltas[k], n=n_local))
                local_pts = np.concatenate(local_pts, axis=0).astype(np.float32)

                far_pts = _sample_far_points(rng, centers, deltas, n=n_far, lo=lo, hi=hi)

                # include true centers too
                U_all = np.concatenate([centers.astype(np.float32), local_pts, far_pts], axis=0)
        else:
            raise NotImplementedError(f"method_u={method_u} not implemented")

        phi = Phi_theta(U_all, centers, deltas=deltas, delta_default=delta_default)

        Theta_rep = np.repeat(obs["Theta"][None, :], U_all.shape[0], axis=0)

        Theta_out_list.append(Theta_rep)
        U_out_list.append(U_all)
        Phi_out_list.append(phi)

    data_phi = {
        "Theta": np.vstack(Theta_out_list).astype(np.float32),
        "U": np.vstack(U_out_list).astype(np.float32),
        "Phi": np.hstack(Phi_out_list).astype(np.float32),
    }

    return data_phi, observations


if __name__ == "__main__":
    from psnn.config import cfg_get, load_yaml, resolve_path

    parser = argparse.ArgumentParser(description="Generate feedback-loop synthetic data.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    default_cfg = os.path.join(exp_dir, "config.yaml")
    cfg = {}
    cfg_path = args.config or (default_cfg if os.path.exists(default_cfg) else None)
    if cfg_path:
        cfg = load_yaml(cfg_path)

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

    out_dir = resolve_path(exp_dir, cfg_get(out_cfg, "out_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_train_phi, Obs_train = gen_data(
        cfg_get(train_cfg, "N_obs", 1200),
        seed=global_seed,
        method_theta=cfg_get(train_cfg, "method_theta", "uniform"),
        method_u=cfg_get(train_cfg, "method_u", "resample"),
        n_local_base=cfg_get(train_cfg, "n_local_base", 60),
        n_far=cfg_get(train_cfg, "n_far", 100),
        delta_default=cfg_get(train_cfg, "delta_default", 0.25),
        delta_mode=cfg_get(train_cfg, "delta_mode", "per_center"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
    )
    data_test_phi, Obs_test = gen_data(
        cfg_get(test_cfg, "N_obs", 600),
        seed=global_seed + 1,
        method_theta=cfg_get(test_cfg, "method_theta", "uniform"),
        method_u=cfg_get(test_cfg, "method_u", "resample"),
        n_local_base=cfg_get(test_cfg, "n_local_base", 60),
        n_far=cfg_get(test_cfg, "n_far", 100),
        delta_default=cfg_get(test_cfg, "delta_default", 0.25),
        delta_mode=cfg_get(test_cfg, "delta_mode", "per_center"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
    )

    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_train_npz", "feedback_loop_data_train.npz"), **data_train_phi)
    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_test_npz", "feedback_loop_data_test.npz"), **data_test_phi)

    joblib.dump(Obs_train, out_dir / cfg_get(out_cfg, "obs_train_pkl", "feedback_loop_obs_train.pkl"))
    joblib.dump(Obs_test, out_dir / cfg_get(out_cfg, "obs_test_pkl", "feedback_loop_obs_test.pkl"))
