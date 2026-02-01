import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import joblib
import os
import sys
exp_dir = os.path.dirname(os.path.abspath(__file__))          # .../exps/grey_scott
repo_root = os.path.abspath(os.path.join(exp_dir, "../.."))   # .../ (repo root)
print(repo_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from psnn import nets, datasets

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from psnn.locater import pick_best_cut, cluster_points, average_error
import matplotlib.pyplot as plt
import tqdm

data_dir = os.path.join(exp_dir, "data")
state_dict_phi_path = os.path.join(exp_dir, 'psnn_phi.pt')
state_dict_psi_path = os.path.join(exp_dir, 'psnn_psi.pt')
state_dict_compat_path = os.path.join(exp_dir, 'psnn_final.pt')
train_npz = os.path.join(data_dir, 'gray_scott_data_train.npz')
test_npz = os.path.join(data_dir, 'gray_scott_data_test.npz')
stab_train_npz = os.path.join(data_dir, 'gray_scott_stability_train.npz')
train_obs = os.path.join(data_dir, 'gray_scott_obs_train.pkl')
test_obs = os.path.join(data_dir, 'gray_scott_obs_test.pkl')

train_loader, test_loader = datasets.make_loaders(train_npz, test_npz, batch_size=200, num_workers=0, device=device)
# find the value of eta
eta = 1.5 * (train_loader.dataset.Phi.max() - 1.0)
print(f"Using eta={eta:.3e}")

model = nets.PSNN(
    dim_theta=2, 
    dim_u=2, 
    embed_dim=8, 
    width=[30, 20], 
    depth=[4, 3],
    eta=eta
).to(device)

phi_ckpt = state_dict_phi_path
if not os.path.exists(phi_ckpt) and os.path.exists(state_dict_compat_path):
    phi_ckpt = state_dict_compat_path
if not os.path.exists(phi_ckpt):
    raise FileNotFoundError(f"Missing Phi model checkpoint: {state_dict_phi_path} (or compat {state_dict_compat_path})")

model.load_state_dict(torch.load(phi_ckpt, map_location=device))

model_psi = None
if os.path.exists(state_dict_psi_path):
    eta_psi = eta
    if os.path.exists(stab_train_npz):
        try:
            stab_loader, _ = datasets.make_loaders(
                stab_train_npz,
                stab_train_npz,
                batch_size=1024,
                num_workers=0,
                device=device,
            )
            psi_max_abs = float(stab_loader.dataset.Phi.abs().max().item())
            eta_psi = min(0.01, psi_max_abs - 1.0)
            print(f"Using eta_psi={eta_psi:.3e} (psi_max_abs={psi_max_abs:.3e})")
        except Exception as e:
            print(f"Warning: failed to infer eta_psi from stability dataset: {e}")

    model_psi = nets.PSNN_stb(
        dim_theta=2,
        dim_u=2,
        embed_dim=8,
        width=[30, 20],
        depth=[4, 3],
        eta=eta_psi,
    ).to(device)
    model_psi.load_state_dict(torch.load(state_dict_psi_path, map_location=device))




# assuming you already defined these somewhere:



# ---------------------------------------------------------
# 1. Build U-grid on D = [0,1]^2 (matches her m=100 style)
# ---------------------------------------------------------
def make_u_grid(m: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a uniform m x m grid on [0,1]^2.

    Returns
    -------
    U_all : (m*m, 2) array of (v,u) points
    v_axis, u_axis : 1D arrays of grid coordinates along each axis
    """
    v_axis = np.arange(0.0, 1.0, 1.0 / m)
    u_axis = np.arange(0.0, 1.0, 1.0 / m)
    V, U = np.meshgrid(v_axis, u_axis, indexing="ij")  # shape (m,m)

    U_all = np.stack([V.ravel(), U.ravel()], axis=1)   # (m*m, 2)
    return U_all, v_axis, u_axis


# ---------------------------------------------------------
# 2. Locate solutions for a single theta
# ---------------------------------------------------------
def locate_solutions_for_theta(psnn,
                               theta: np.ndarray,
                               U_all: np.ndarray,
                               L_cut: float,
                               Cmax: int = 5,
                               sil1: float = 0.3):
    """
    Given a trained PSNN, a parameter theta, and a grid U_all, locate steady states:

    - Evaluate Φ(theta, u) on all u in U_all
    - Keep points where Φ >= L_cut
    - Cluster them with KMeans+silhouette (cluster_points)
    - Return cluster centers (estimated steady states) and full score field

    Parameters
    ----------
    psnn : torch.nn.Module
        Trained PSNN model with signature psnn(U, Theta).
    theta : (2,) array-like
        Parameter (f,k).
    U_all : (N,2) ndarray
        Grid of state points u = (v,u).
    L_cut : float
        Threshold on Φ for collecting candidate points.
    Cmax : int
        Maximum number of clusters to try (matches her Cmax=5).
    sil1 : float
        Silhouette cutoff (matches her 0.3).

    Returns
    -------
    centers : (k,2) ndarray
        Estimated steady states in state space.
    scores : (N,) ndarray
        Φ(theta, U_all[i]) values on the grid.
    """

    psnn.eval()

    U_all_np = np.asarray(U_all, dtype=float)
    U_all_t = torch.from_numpy(U_all_np).float().to(device)

    theta_np = np.asarray(theta, dtype=float)
    theta_t = torch.from_numpy(theta_np).float().to(device)   # (2,)
    theta_t = theta_t.unsqueeze(0).repeat(U_all_t.shape[0], 1)  # (N,2)

    with torch.no_grad():
        scores_t = psnn(U_all_t, theta_t).flatten()  # (N,)

    scores = scores_t.cpu().numpy()
    mask = scores >= L_cut
    U_collected = U_all_np[mask, :]

    # Use your cluster_points to decide #clusters + centers
    centers = cluster_points(U_collected, Cmax=Cmax, sil1=sil1, random_state=0)

    return centers, scores


def classify_centers_stability(psnn_psi, theta: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Classify located centers as stable (True) vs unstable (False) using Psi sign."""
    if psnn_psi is None or centers is None or centers.size == 0:
        return np.zeros((0,), dtype=bool)

    psnn_psi.eval()
    centers_t = torch.from_numpy(np.asarray(centers, dtype=float)).float().to(device)
    theta_t = torch.from_numpy(np.asarray(theta, dtype=float)).float().to(device)
    theta_t = theta_t.unsqueeze(0).repeat(centers_t.shape[0], 1)
    with torch.no_grad():
        psi = psnn_psi(centers_t, theta_t).flatten().detach().cpu().numpy()
    return psi >= 0.0


# ---------------------------------------------------------
# 3. Optional: wrap to get a "solution profile" on a grid
# ---------------------------------------------------------
def predict_phi_profile(psnn,
                        theta: np.ndarray,
                        m: int = 100,
                        L_cut: float = 0.48,
                        Cmax: int = 5,
                        sil1: float = 0.3,
                        O_search: list = [],
                        diam_D: float = 1.0):
    """ 
    Convenience function that:

    - Builds an m x m grid on [0,1]^2
    - Optionally chooses an optimal L_cut via pick_best_cut on O_search
    - Locates solution centers
    - Returns centers, scores reshaped on the grid, and L_cut used.

    Parameters
    ----------
    psnn : nn.Module
    theta : (2,) array-like
    m : int
        Grid resolution (her code uses m=100).
    L_cut : float or None
        If None and O_search is provided, will be chosen by pick_best_cut.
    Cmax, sil1 : as before.
    O_search : list[dict] or None
        Observation list (e.g. Obs_test) for tuning cutoff via pick_best_cut.
    diam_D : float
        Diameter of D for normalized errors (use 1.0 to match her raw distances).

    Returns
    -------
    centers : (k,2) ndarray
        Estimated steady states.
    Phi_grid : (m,m) ndarray
        Φ(theta, v,u) evaluated on the grid.
    L_cut_used : float
        Threshold that was actually used.
    """

    # 1) Build grid on D
    U_all, v_axis, u_axis = make_u_grid(m=m)

    # 2) If L_cut is not provided but O_search is, tune it
    if L_cut is None and O_search is not None:
        # Simple example grid around 0.48; you can widen/narrow as you like
        L_grid = np.linspace(0.3, 0.7, 10)
        L_cut_used, _errs = pick_best_cut(
            psnn=psnn,
            U_all=U_all,
            O_search=O_search,
            Cmax=Cmax,
            sil1=sil1,
            diam_D=diam_D,
            L_grid=L_grid,
            random_state=0,
        )
    else:
        # Use provided L_cut (or default her value 0.48)
        L_cut_used = L_cut if L_cut is not None else 0.48

    # 3) Locate solutions for this theta
    centers, scores = locate_solutions_for_theta(
        psnn=psnn,
        theta=theta,
        U_all=U_all,
        L_cut=L_cut_used,
        Cmax=Cmax,
        sil1=sil1,
    )

    # 4) Reshape scores back to m x m profile
    Phi_grid = scores.reshape(m, m)

    return centers, Phi_grid, (v_axis, u_axis), L_cut_used


if __name__ == "__main__":


    # 1) Load trained PSNN (adapt path / class as needed)
    # from your_psnn_module import PSNN
    # psnn = PSNN(...)
    # psnn.load_state_dict(torch.load("psnn_gray_scott.pt", map_location=device))
    # psnn.to(device)

    # For this example, we assume psnn is already constructed & loaded in memory.

    # 2) Load observation list (to optionally tune L_cut)
    Obs_test = joblib.load(test_obs)  # list of dicts: {"Theta": ..., "U": [...]}

    # Choose one theta to test
    theta0 = Obs_test[0]["Theta"]  # shape (2,)

    # 3) Predict solution profile and locate steady states
    centers, Phi_grid, (v_axis, u_axis), L_used = predict_phi_profile(
        psnn=model,
        theta=theta0,
        m=100,
        L_cut=0.48,          # let it tune around [0.3,0.7] using Obs_test
        Cmax=5,
        sil1=0.3,
        O_search=Obs_test[:100],   # used only if L_cut=None
        diam_D=1.0,
    )

    print("Theta:", theta0)
    print("Located centers:\n", centers)
    print("Used cutoff L =", L_used)

    stable_mask = classify_centers_stability(model_psi, theta0, centers)
    if model_psi is None:
        print("No Psi model found; skipping stability classification.")

    # 4) Visualization
    V, U = np.meshgrid(v_axis, u_axis, indexing="ij")
    plt.figure(figsize=(5, 4))
    plt.contourf(V, U, Phi_grid, levels=50)
    if centers.shape[0] > 0:
        if model_psi is None:
            plt.scatter(centers[:, 0], centers[:, 1], c="red", s=50, marker="x", label="Predicted solutions")
            plt.legend()
        else:
            stable = centers[stable_mask]
            unstable = centers[~stable_mask]
            if stable.shape[0] > 0:
                plt.scatter(stable[:, 0], stable[:, 1], c="lime", s=60, marker="o", edgecolors="k", label="Predicted stable")
            if unstable.shape[0] > 0:
                plt.scatter(unstable[:, 0], unstable[:, 1], c="red", s=60, marker="x", label="Predicted unstable")
            plt.legend()
    plt.xlabel("v")
    plt.ylabel("u")
    plt.title(fr"$\Phi(\theta, u)$ profile and located centers, $\Theta = [{theta0[0]:.2f}, {theta0[1]:.2f}]$")
    # add colorbar
    cbar = plt.colorbar()
    cbar.set_label(r"$\Phi$")
    plt.tight_layout()
    plt.savefig("predicted_phi_profile.png", dpi=300)

    print("plot saved to <predicted_phi_profile.png>")
    # clear figure
    plt.clf()
    # ---------------------------------------------------------

    

    Theta_f_space = np.linspace(0.00, 0.30, 30)
    Theta_k_space = np.linspace(0.00, 0.08, 20)
    Theta_grid = np.array(np.meshgrid(Theta_f_space, Theta_k_space)).T.reshape(-1, 2)

    if model_psi is None:
        raise FileNotFoundError(
            f"Missing stability model checkpoint: {state_dict_psi_path}. "
            "Train the stability model first to produce a stability-aware solution map."
        )

    pts_zero = []
    pts_two_both_stable = []
    pts_two_mixed = []
    pts_two_both_unstable = []
    pts_other = []

    for i in tqdm.tqdm(range(Theta_grid.shape[0]), desc="Building stability solution map"):
        theta_i = Theta_grid[i]
        centers, _Phi_grid, _axes, _L_used = predict_phi_profile(
            psnn=model,
            theta=theta_i,
            m=100,
            L_cut=0.48,
            Cmax=5,
            sil1=0.3,
            O_search=Obs_test,
            diam_D=1.0,
        )

        n = int(centers.shape[0])
        if n == 0:
            pts_zero.append(theta_i)
            continue

        if n == 2:
            stable_mask = classify_centers_stability(model_psi, theta_i, centers)
            stable_count = int(np.sum(stable_mask))
            if stable_count == 2:
                pts_two_both_stable.append(theta_i)
            elif stable_count == 1:
                pts_two_mixed.append(theta_i)
            else:
                pts_two_both_unstable.append(theta_i)
        else:
            pts_other.append(theta_i)

    def _scatter(points, *, color, label, marker='o', s=18):
        if len(points) == 0:
            return
        arr = np.asarray(points)
        plt.scatter(arr[:, 0], arr[:, 1], c=color, s=s, marker=marker, label=label)

    plt.figure(figsize=(7, 4.5))
    _scatter(pts_zero, color='royalblue', label='0 solutions')
    _scatter(pts_two_both_stable, color='limegreen', label='2 solutions: both stable')
    _scatter(pts_two_mixed, color='darkorange', label='2 solutions: 1 stable + 1 unstable', marker='s')
    _scatter(pts_two_both_unstable, color='crimson', label='2 solutions: both unstable', marker='x')
    _scatter(pts_other, color='black', label='other (#solutions != 0 or 2)', marker='.')

    f_space = np.linspace(0.00, 0.255, 200)
    k_dividing = np.sqrt(f_space) / 2.0 - f_space
    plt.plot(f_space, k_dividing, c='red', linewidth=1.5, label='analytic divider')

    plt.xlabel('f')
    plt.ylabel('k')
    plt.title('Solution map with stability (via Psi sign at located centers)')
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(exp_dir, 'solution_map_with_stability_info.png')
    plt.savefig(out_path, dpi=300)
    print(f"plot saved to <{out_path}>")