
import argparse
import os
import sys
import numpy as np
import tqdm
import joblib
### DEFINITION OF ODE SYSTEM FOR GRAY-SCOTT MODEL ###

# --- make repo root importable (so `import psnn` works no matter where you run from) ---
exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # .../exps/grey_scott
repo_root = os.path.abspath(os.path.join(exp_dir, ".."))  # .../ (repo root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)



# WILL BE IMPLEMENTED LATER IF NEEDED

D = np.array([
    [0., 1.],
    [0., 1.]
])

Omega = np.array([
    [0.0, 0.3],
    [0.0, 0.08]
])


def U(theta):
    """
    Gray–Scott homogeneous steady states for parameters (f,k).
    Returns a list of dicts: {"u": u, "v": v, "stable": bool, "label": str}
    """
    if theta.ndim != 1 or theta.shape[0] !=2:
        raise ValueError("theta must be a 1D array of shape (2,)")
    f, k = theta
    
    out = []
    # Trivial steady state (u,v) = (1,0):
    # Widely noted as linearly stable for positive f,k in the homogeneous ODE (no diffusion).
    # out.append({"u": 1.0, "v": 0.0, "stable": True, "label": "trivial"})
 
    # Nontrivial solutions exist iff f > 4 (f + k)^2
    fp = f + k
    Delta = f - 4.0*(fp**2)
    if Delta <= 0.0:
        # No nontrivial solutions in D=(0,1)^2 under the paper's setting
        return out
    
    Delta = f * Delta

    root = np.sqrt(Delta)
    # U1 and U2 from the paper (Eq. 4.2)
    # u1 = (f - root) / (2.0*f)
    # v1 = (f + root) / (2.0*fp)
    # u2 = (f + root) / (2.0*f)
    # v2 = (f - root) / (2.0*fp)
    u_1 = np.array([(f - root) / (2.0*f), (f + root) / (2.0*fp)])
    u_2 = np.array([(f + root) / (2.0*f), (f - root) / (2.0*fp)])

    # Stability discriminator (Eq. 4.4): S = f*sqrt(Delta) + f^2 - 2(f+k)^3
    S = f*root + f**2 - 2.0*(fp**3)

    if S > 0:
        # U1 stable, U2 unstable
        # out.append({"u": float(u1), "v": float(v1), "stable": True,  "label": "nontrivial_1"})
        # out.append({"u": float(u2), "v": float(v2), "stable": False, "label": "nontrivial_2"})
        out.extend(
            [
                {"theta": theta, "u": u_1, "stable": True},
                {"theta": theta, "u": u_2, "stable": False}
            ]
        )
    else:
        # both unstable
        # out.append({"u": float(u1), "v": float(v1), "stable": False, "label": "nontrivial_1"})
        # out.append({"u": float(u2), "v": float(v2), "stable": False, "label": "nontrivial_2"})
        out.extend(
            [
                {"theta": theta, "u": u_1, "stable": False},
                {"theta": theta, "u": u_2, "stable": False}
            ]
        )

    return out

def delta_(centers, delta_0=1e-3, delta_1=1.0):
    if len(centers) <=1:
        return delta_1

    if type(centers) is list:
        # make it a n,2 array
        centers = np.vstack([np.array(s["u"]) for s in centers])
    
    # compute pairwise squared distances
    N = centers.shape[0]
    norms_squared = np.sum(centers**2, axis=1, keepdims=True)          # (N,1)
    d2 = norms_squared + norms_squared.T - 2.0 * centers @ centers.T           # (N,N)
    d2[np.arange(N), np.arange(N)] = np.inf                    # ignore diagonal
    min_dist = float(np.min(d2)) ** 0.5
    return max(0.25 * min_dist, delta_0)

def Phi_theta(u_input, centers, delta=None, delta0=1e-3, delta1=1.0):
    if len(centers) == 0:
        return np.zeros((u_input.shape[0],), dtype=float)
    centers = np.concatenate([np.array(s["u"])[None, :] for s in centers], axis=0) if type(centers) is list else centers
    if delta is None:
        delta = delta_(centers, delta0, delta1)
    # assuming u_input is (N, dim) np.ndarray
    dist = u_input[:, None, :] - centers[None, :, :]  # (N, M, dim)
    dist_squared = np.sum(dist ** 2, axis=-1)  # (N, M)
    phi_vals = np.sum(np.exp(- dist_squared/ delta ** 2), axis=-1)  # (N,)
    return phi_vals


def gen_data(
    N_obs,
    N_random,
    seed=42,
    method_theta="uniform",
    method_u="uniform",
    *,
    theta_bounds=None,
    u_bounds=None,
    delta0=1e-3,
    delta1=1.0,
):
    """
    Generate data for Gray-Scott model over (f,k) in [0,0.3]\times[0,0.08], based on (2.10) in Zhang et al. (2023).
    Returns:
      observations: list of dicts with keys:
    """

    # generate observations randomly in Omega
    rng = np.random.default_rng(seed)
    theta_bounds = Omega if theta_bounds is None else np.asarray(theta_bounds, dtype=float)
    u_bounds = D if u_bounds is None else np.asarray(u_bounds, dtype=float)
    if method_theta == 'grid':
        f_values = np.linspace(theta_bounds[0, 0], theta_bounds[0, 1], int(np.sqrt(N_obs)))
        k_values = np.linspace(theta_bounds[1, 0], theta_bounds[1, 1], int(np.sqrt(N_obs)))
        theta_list = []
        for f in f_values:
            for k in k_values:
                theta_list.append( (f,k) )
        theta_list = theta_list[:N_obs]
    if method_theta == 'uniform':
        theta_list = rng.uniform(theta_bounds[:, 0], theta_bounds[:, 1], size=(N_obs, 2))
    
    observations = []
    for theta in theta_list:
        solutions = U(theta)
        observations.append({
            "Theta": np.array(theta, dtype=np.float32),
            "U": solutions
        })    
    
    # if method_u == 'uniform':
    #     Ulo, Uhi = D[:,0], D[:,1]
    #     U_rand = rng.uniform(Ulo, Uhi, size=(N_random, 2))
    
    # else:
    #     raise NotImplementedError(f"method_u={method_u} not implemented yet.")

    Theta_out_list = []
    U_out_list  = []
    Phi_out_list = []
    
    # for obs in observations:
    for obs in tqdm.tqdm(observations, desc="Generating Gray-Scott data"):
        U_obs = np.array([s["u"] for s in obs["U"]]) if len(obs["U"])>0 else np.zeros((0,2))
        if method_u == 'uniform':
            Ulo, Uhi = u_bounds[:, 0], u_bounds[:, 1]
            U_rand = rng.uniform(Ulo, Uhi, size=(N_random, 2))
        else:
            raise NotImplementedError(f"method_u={method_u} not implemented yet.")
        U_all = np.vstack((U_obs, U_rand))  
        delta = None
        if len(obs["U"]) > 0:
            delta = delta_(obs["U"], delta_0=delta0, delta_1=delta1)

        Phi_vals = Phi_theta(U_all, obs["U"], delta=delta, delta0=delta0, delta1=delta1)

        U_out_list.append(U_all)
        Phi_out_list.append(Phi_vals)
        Theta_out_list.append(np.repeat(obs["Theta"][None, :], len(obs["U"]) + N_random, axis=0))
    
    phi_data = {
        "Theta": np.vstack(Theta_out_list),
        "U": np.vstack(U_out_list),
        "Phi": np.hstack(Phi_out_list)
    }

    return phi_data, observations

if __name__ == "__main__":
    from psnn.config import cfg_get, load_yaml, resolve_path

    parser = argparse.ArgumentParser(description="Generate Gray-Scott synthetic data.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    args = parser.parse_args()

    exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_cfg = os.path.join(exp_dir, "config.yaml")
    cfg = {}
    cfg_path = args.config or (default_cfg if os.path.exists(default_cfg) else None)
    if cfg_path:
        cfg = load_yaml(cfg_path)

    dg = cfg_get(cfg, "data_generation", {})

    theta_bounds = np.asarray(cfg_get(dg, "domain.theta_bounds", Omega), dtype=float)
    u_bounds = np.asarray(cfg_get(dg, "domain.u_bounds", D), dtype=float)
    delta0 = float(cfg_get(dg, "delta.delta0", 1e-3))
    delta1 = float(cfg_get(dg, "delta.delta1", 1.0))

    train_cfg = cfg_get(dg, "train", {})
    test_cfg = cfg_get(dg, "test", {})
    out_cfg = cfg_get(dg, "outputs", {})

    out_dir = resolve_path(exp_dir, cfg_get(out_cfg, "out_dir", "data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    data_train, Obs_train = gen_data(
        cfg_get(train_cfg, "N_obs", 1000),
        cfg_get(train_cfg, "N_random", 200),
        seed=cfg_get(train_cfg, "seed", 123),
        method_theta=cfg_get(train_cfg, "method_theta", "uniform"),
        method_u=cfg_get(train_cfg, "method_u", "uniform"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        delta0=delta0,
        delta1=delta1,
    )
    data_test, Obs_test = gen_data(
        cfg_get(test_cfg, "N_obs", 600),
        cfg_get(test_cfg, "N_random", 200),
        seed=cfg_get(test_cfg, "seed", 456),
        method_theta=cfg_get(test_cfg, "method_theta", "uniform"),
        method_u=cfg_get(test_cfg, "method_u", "uniform"),
        theta_bounds=theta_bounds,
        u_bounds=u_bounds,
        delta0=delta0,
        delta1=delta1,
    )
    print(data_train["Theta"].shape, data_train["U"].shape, data_train["Phi"].shape)
    print(data_test["Phi"].max(), data_test["Phi"].min())

    # Save to configured directory (avoid CWD-dependent behavior)
    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_train_npz", "gray_scott_data_train.npz"), **data_train)
    np.savez_compressed(out_dir / cfg_get(out_cfg, "data_test_npz", "gray_scott_data_test.npz"), **data_test)
    # Save observations using joblib as pickle
    joblib.dump(Obs_train, out_dir / cfg_get(out_cfg, "obs_train_pkl", "gray_scott_obs_train.pkl"))
    joblib.dump(Obs_test, out_dir / cfg_get(out_cfg, "obs_test_pkl", "gray_scott_obs_test.pkl"))
