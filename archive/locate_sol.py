import os
import sys

import numpy as np
from typing import Optional


def _add_repo_root_to_syspath():
    exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/feedback_loop
    repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return exp_dir


def make_u_grid(m: int = 200, lo: float = 0.0, hi: float = 5.0):
    """Construct a uniform m x m grid on [lo,hi]^2 for u=(p1,p2)."""
    p1_axis = np.linspace(lo, hi, m, endpoint=False)
    p2_axis = np.linspace(lo, hi, m, endpoint=False)
    P1, P2 = np.meshgrid(p1_axis, p2_axis, indexing="ij")
    U_all = np.stack([P1.ravel(), P2.ravel()], axis=1)
    return U_all, p1_axis, p2_axis


def locate_solutions_for_theta(psnn, theta: np.ndarray, U_all: np.ndarray, L_cut: float, *, Cmax: int = 6, sil1: float = 0.3):
    """Evaluate PSNN on a grid, threshold, cluster candidate points."""
    import torch
    from psnn.locater import cluster_points

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnn.eval()
    U_all_np = np.asarray(U_all, dtype=float)

    U_t = torch.from_numpy(U_all_np).float().to(device)
    theta_np = np.asarray(theta, dtype=float)
    theta_t = torch.from_numpy(theta_np).float().to(device)
    theta_t = theta_t.unsqueeze(0).repeat(U_t.shape[0], 1)

    with torch.no_grad():
        scores_t = psnn(U_t, theta_t).flatten()
    scores = scores_t.cpu().numpy()

    mask = scores >= float(L_cut)
    U_collected = U_all_np[mask, :]

    centers = cluster_points(U_collected, Cmax=Cmax, sil1=sil1, random_state=0)
    return centers, scores


def predict_phi_profile(
    psnn,
    theta: np.ndarray,
    *,
    m: int = 200,
    L_cut: Optional[float] = 0.5,
    Cmax: int = 6,
    sil1: float = 0.3,
    O_search: Optional[list] = None,
):
    """Convenience wrapper mirroring exps/grey_scott/locate_sol.py."""
    from psnn.locater import pick_best_cut

    U_all, p1_axis, p2_axis = make_u_grid(m=m, lo=0.0, hi=5.0)

    if L_cut is None and O_search is not None:
        # Phi is in roughly [0,3], so scan a modest range.
        L_grid = np.linspace(0.05, 2.5, 26)
        L_cut_used, _errs = pick_best_cut(
            psnn=psnn,
            U_all=U_all,
            O_search=O_search,
            Cmax=Cmax,
            sil1=sil1,
            diam_D=5.0 * np.sqrt(2.0),
            L_grid=L_grid,
            random_state=0,
        )
    else:
        L_cut_used = 0.5 if L_cut is None else float(L_cut)

    centers, scores = locate_solutions_for_theta(psnn, theta, U_all, L_cut_used, Cmax=Cmax, sil1=sil1)
    Phi_grid = scores.reshape(m, m)

    return centers, Phi_grid, (p1_axis, p2_axis), L_cut_used


if __name__ == "__main__":
    exp_dir = _add_repo_root_to_syspath()

    import torch
    import joblib
    import matplotlib.pyplot as plt

    from psnn import nets, datasets

    # Paths
    data_dir = os.path.join(exp_dir, "data")
    state_dict_path = os.path.join(exp_dir, "psnn_phi.pt")
    obs_path = os.path.join(data_dir, "feedback_loop_obs_test.pkl")
    train_npz = os.path.join(data_dir, "feedback_loop_data_train.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"Missing trained model: {state_dict_path}")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Missing observations: {obs_path} (run data/gen_feedback_loop.py)")
    if not os.path.exists(train_npz):
        raise FileNotFoundError(f"Missing training data: {train_npz} (run data/gen_feedback_loop.py)")

    Obs_test = joblib.load(obs_path)
    theta0 = Obs_test[0]["Theta"]

    # Match the training-time eta heuristic
    train_loader, _test_loader = datasets.make_loaders(
        train_npz,
        train_npz,
        batch_size=1024,
        num_workers=0,
        device=device,
    )
    phi_max = float(train_loader.dataset.Phi.max().item())
    eta = max(0.01, 1.5 * (phi_max - 1.0))
    print(f"Using eta={eta:.3e} (Phi.max={phi_max:.3e})")

    model = nets.PSNN(dim_theta=4, dim_u=2, embed_dim=8, width=[30, 20], depth=[4, 3], eta=eta).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    centers, Phi_grid, (p1_axis, p2_axis), L_used = predict_phi_profile(
        psnn=model,
        theta=theta0,
        m=200,
        L_cut=0.5,
        Cmax=6,
        sil1=0.3,
        O_search=None,
    )

    print("Theta:", theta0)
    print("Located centers:\n", centers)
    print("Used cutoff L =", L_used)

    P1, P2 = np.meshgrid(p1_axis, p2_axis, indexing="ij")
    plt.figure(figsize=(6, 5))
    plt.contourf(P1, P2, Phi_grid, levels=50)
    if centers.shape[0] > 0:
        plt.scatter(centers[:, 0], centers[:, 1], c="red", s=60, marker="x", label="Predicted solutions")
        plt.legend()
    plt.xlabel("p1")
    plt.ylabel("p2")
    plt.title(rf"$\\Phi(\\theta,u)$ profile; $\\Theta=[{theta0[0]:.2f},{theta0[1]:.2f},{theta0[2]:.2f},{theta0[3]:.2f}]$")
    cbar = plt.colorbar()
    cbar.set_label(r"$\\Phi$")
    plt.show()
